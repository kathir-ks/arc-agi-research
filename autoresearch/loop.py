"""Main autoresearch loop.

Runs forever (or until ``AUTORESEARCH_BUDGET_S`` seconds elapse, useful for
smoke tests). For each tick:

  1. Ask the governor for resource pressure; back off or abort accordingly.
  2. Pop the next ExperimentSpec from the queue (re-seed if empty).
  3. Dispatch it through the registry.
  4. Record the result; spawn any followups it requested.
  5. Optionally rotate disk; loop.

Restart-safe: the queue persists between runs, model state is rebuilt fresh
each restart, and the registry seeds always advance one epoch so the loop
never duplicates IDs already in done.jsonl.

Environment knobs (all optional):

  AUTORESEARCH_STATE_DIR  – override state dir (default: autoresearch/state)
  AUTORESEARCH_BUDGET_S   – exit after this many seconds (default: forever)
  AUTORESEARCH_RAM_GB     – RAM cap (default: 30)
  AUTORESEARCH_DISK_GB    – disk cap on state dir (default: 10)
  AUTORESEARCH_TYPES      – comma list of registered experiment types to run
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from .governor import GIB, Governor, GovernorConfig, Pressure
from .queue import ExperimentSpec, Queue
from .registry import (
    SEEDS_EPOCHAL,
    RunCtx,
    _import_default_experiments,
    child_specs,
    epochal_spec,
    run_experiment,
    seed_specs,
    types as registered_types,
)

PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_DIR = PHASE2_ROOT / "autoresearch" / "state"

# Hard exit code the wrapper script knows to restart on (vs. clean exit).
EXIT_RESTART = 42

log = logging.getLogger("autoresearch.loop")


class _ShutdownRequested(Exception):
    """Raised by signal handlers; caught at the loop boundary for graceful exit."""


def _install_signal_handlers() -> None:
    def _h(signum: int, _frame) -> None:
        log.warning("received signal %d; shutting down", signum)
        raise _ShutdownRequested(signum)
    signal.signal(signal.SIGTERM, _h)
    signal.signal(signal.SIGINT, _h)


def _build_governor(state_dir: Path) -> Governor:
    ram_gb = float(os.environ.get("AUTORESEARCH_RAM_GB", "30"))
    disk_gb = float(os.environ.get("AUTORESEARCH_DISK_GB", "10"))
    cfg = GovernorConfig(
        ram_cap_bytes=int(ram_gb * GIB),
        disk_cap_bytes=int(disk_gb * GIB),
    )
    return Governor(state_dir, cfg)


def _build_ctx(state_dir: Path) -> RunCtx:
    return RunCtx(
        state_dir=state_dir,
        results_dir=state_dir / "results",
    )


def _ensure_canonical(ctx: RunCtx) -> None:
    if ctx.canonical_ids is None:
        from interp.puzzle_id_map import get_canonical_200_puzzle_ids
        ctx.canonical_ids = get_canonical_200_puzzle_ids(strict=False)


def _maybe_seed(queue: Queue, ctx: RunCtx, epoch: int, types_filter: set[str] | None) -> int:
    if not queue.is_empty():
        return 0
    _ensure_canonical(ctx)
    specs = seed_specs(ctx, epoch=epoch, types_filter=types_filter)
    added = queue.push_many(specs)
    log.info("seeded epoch=%d added=%d (canonical_ids=%d, types=%s)",
             epoch, added, len(ctx.canonical_ids or {}), sorted(types_filter or registered_types()))
    return added


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("AUTORESEARCH_LOG", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _install_signal_handlers()
    _import_default_experiments()

    state_dir = Path(os.environ.get("AUTORESEARCH_STATE_DIR", str(DEFAULT_STATE_DIR)))
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "results").mkdir(parents=True, exist_ok=True)

    governor = _build_governor(state_dir)
    queue = Queue(state_dir)
    ctx = _build_ctx(state_dir)
    log.info("autoresearch starting | state=%s | types=%s", state_dir, registered_types())
    log.info("caps: ram=%.1f GiB disk=%.1f GiB",
             governor.config.ram_cap_bytes / GIB, governor.config.disk_cap_bytes / GIB)

    types_env = os.environ.get("AUTORESEARCH_TYPES", "").strip()
    types_filter: set[str] | None = {t for t in types_env.split(",") if t} or None

    budget_s = float(os.environ.get("AUTORESEARCH_BUDGET_S", "0"))
    deadline = (time.monotonic() + budget_s) if budget_s > 0 else None

    epoch = 0
    consecutive_failures = 0
    ran = 0
    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                log.info("budget elapsed; clean exit. ran=%d stats=%s", ran, queue.stats())
                return 0

            pressure = governor.check()
            if pressure == Pressure.HARD:
                log.error("HARD pressure (ram=%.2f GiB disk=%.2f GiB); requesting restart",
                          governor.ram_used_bytes() / GIB, governor.disk_used_bytes() / GIB)
                return EXIT_RESTART
            if pressure == Pressure.SOFT:
                governor.relieve()
                continue

            added = _maybe_seed(queue, ctx, epoch, types_filter)
            if added == 0 and queue.is_empty():
                # No canonical IDs or registry empty; back off so we don't busy-loop.
                log.warning("queue empty and reseed produced nothing; sleeping 5s")
                time.sleep(5.0)
                epoch += 1
                continue
            if added > 0:
                epoch += 1

            spec = queue.pop()
            if spec is None:
                continue

            t0 = time.monotonic()
            try:
                result = run_experiment(spec, ctx)
            except _ShutdownRequested:
                raise
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc(limit=8)
                log.exception("experiment %s failed", spec.id)
                queue.record_failed(spec, f"{exc!r}\n{tb}")
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    log.error("5 consecutive failures; requesting restart")
                    return EXIT_RESTART
                continue
            consecutive_failures = 0
            dt = time.monotonic() - t0
            log.info("ran %s in %.2fs -> %s", spec.id, dt, _short(result))
            queue.record_done(spec, {**result, "duration_s": round(dt, 3)})
            ran += 1

            if isinstance(result, dict):
                # Type-derived followups: rerun other types on the parent's task.
                for child in child_specs(spec, result.get("followups", [])):
                    queue.push(child)
                # Explicit child specs (e.g., from the LLM proposer).
                for entry in result.get("spawn", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    typ = entry.get("type")
                    params = entry.get("params") or {}
                    if not typ or not isinstance(params, dict):
                        continue
                    queue.push(ExperimentSpec.new(type=typ, params=params, parent_id=spec.id))

            # Periodic LLM proposer trigger: once every N successful core runs.
            # Skipped when proposer isn't registered (e.g., types_filter excludes it).
            cadence = int(os.environ.get("AUTORESEARCH_PROPOSER_CADENCE", "10"))
            if (
                cadence > 0
                and "exp_propose" in registered_types()
                and (types_filter is None or "exp_propose" in types_filter)
                and ran > 0 and ran % cadence == 0
                and spec.type != "exp_propose"  # don't recurse
            ):
                proposer_spec = epochal_spec("exp_propose", epoch=ran // cadence)
                if queue.push_front(proposer_spec):
                    log.info("triggered exp_propose (epoch=%d after ran=%d, front-of-queue)",
                             ran // cadence, ran)

            if ran % 10 == 0:
                governor.relieve()  # opportunistic GC + rotation
    except _ShutdownRequested:
        log.info("graceful shutdown. ran=%d stats=%s", ran, queue.stats())
        return 0


def _short(result: dict | None) -> str:
    if not isinstance(result, dict):
        return repr(result)
    keep = {k: v for k, v in result.items() if k != "followups"}
    s = repr(keep)
    return s if len(s) < 200 else s[:197] + "..."


if __name__ == "__main__":
    sys.exit(main())
