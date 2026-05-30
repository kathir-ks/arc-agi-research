"""Experiment: ask Sonnet 4.6 what to investigate next.

Reads the last 30 done.jsonl records, dumps them into a tight prompt, calls the
proposer, validates each suggestion against the registry + canonical_200, and
returns the validated child specs via ``result["spawn"]``. The loop will queue
them just like any organic followup.

Cost & cadence are governed elsewhere:
  * per-call dollar cap by claude_proposer (env: AUTORESEARCH_PROPOSER_BUDGET_USD, default $0.05)
  * cadence by registry.seed_specs / loop's "every 10 done" trigger
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..claude_proposer import build_prompt, run_proposer
from ..inference import ensure_canonical
from ..queue import ExperimentSpec
from ..registry import RunCtx, register, types as registered_types

log = logging.getLogger("autoresearch.exp_propose")

CORE_TYPES = ("exp_a_convergence", "exp_c_ablate", "exp_d_halt")


def _read_recent_done(state_dir: Path, n: int = 30) -> list[dict[str, Any]]:
    done = state_dir / "done.jsonl"
    if not done.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(done) as f:
        lines = f.readlines()
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@register("exp_propose")
def run(params: dict[str, Any], ctx: RunCtx) -> dict[str, Any]:
    ensure_canonical(ctx)
    if not ctx.canonical_ids:
        return {"status": "skipped", "reason": "no canonical_ids loaded"}

    recent = _read_recent_done(ctx.state_dir, n=30)
    if len(recent) < 3:
        # Cold-start: don't waste an API call when there's nothing to analyse.
        return {"status": "skipped", "reason": f"only {len(recent)} prior results"}

    # Filter the registered types we'll actually accept proposals for.
    accepted_types = [t for t in registered_types() if t in CORE_TYPES]
    canonical_stems = list(ctx.canonical_ids.keys())

    prompt = build_prompt(
        recent_done=recent,
        registered_types=accepted_types,
        canonical_sample=canonical_stems,
    )

    res = run_proposer(prompt)
    if not res.ok:
        log.warning("proposer call failed: %s", res.error)
        return {
            "status": "proposer_failed",
            "error": res.error,
            "cost_usd": res.cost_usd,
            "duration_s": round(res.duration_s, 3),
        }

    canonical_set = set(canonical_stems)
    accepted_set = set(accepted_types)
    spawn: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for p in res.proposals:
        typ = p.get("type")
        stem = p.get("stem")
        if typ not in accepted_set:
            rejected.append({"proposal": p, "reason": "unknown type"})
            continue
        if stem not in canonical_set:
            rejected.append({"proposal": p, "reason": "stem not in canonical_200"})
            continue
        key = (typ, stem)
        if key in seen:
            continue
        seen.add(key)
        spawn.append(
            {
                "type": typ,
                "params": {
                    "stem": stem,
                    "puzzle_id": int(ctx.canonical_ids[stem]),
                    "epoch": int(params.get("epoch", 0)) + 1,
                    "from_proposer": True,
                    "rationale": p.get("rationale", "")[:200],
                },
            }
        )

    # Persist the analysis prose alongside other results for later inspection.
    out_path = ctx.results_dir / "exp_propose" / f"propose_{int(params.get('epoch', 0)):04d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "analysis": res.analysis,
                "proposals_accepted": spawn,
                "proposals_rejected": rejected,
                "cost_usd": res.cost_usd,
                "duration_s": round(res.duration_s, 3),
            },
            f,
            indent=None,
        )

    log.info("proposer accepted=%d rejected=%d cost=$%.4f", len(spawn), len(rejected), res.cost_usd)
    return {
        "status": "ok",
        "accepted": len(spawn),
        "rejected": len(rejected),
        "cost_usd": round(res.cost_usd, 6),
        "spawn": spawn,
        "result_path": str(out_path.relative_to(ctx.state_dir)),
    }
