"""Experiment registry + seed generation.

Each experiment module registers a ``run(params, *, ctx) -> dict`` callable.
The loop dispatches on ``ExperimentSpec.type`` through :func:`run_experiment`.

Seed generation walks the canonical-200 puzzle IDs and queues one of each
experiment per task. When all are done, the loop reseeds with bumped epoch
counter so it never sits idle — repeated runs of the same params are fine
because the underlying experiments are deterministic-ish and we de-dup by ID.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .queue import ExperimentSpec

log = logging.getLogger("autoresearch.registry")

# Lazy-imported model + meta to avoid the 30s checkpoint load at import time.
@dataclass
class RunCtx:
    state_dir: Path
    results_dir: Path
    model: Any = None             # torch.nn.Module, loaded on first use
    meta: Any = None              # CheckpointMeta
    canonical_ids: dict[str, int] | None = None  # stem -> puzzle_id


Runner = Callable[[dict[str, Any], RunCtx], dict[str, Any]]
_REGISTRY: dict[str, Runner] = {}


def register(name: str) -> Callable[[Runner], Runner]:
    def deco(fn: Runner) -> Runner:
        if name in _REGISTRY:
            raise ValueError(f"duplicate experiment type: {name}")
        _REGISTRY[name] = fn
        return fn

    return deco


def has(name: str) -> bool:
    return name in _REGISTRY


def types() -> list[str]:
    return sorted(_REGISTRY)


def run_experiment(spec: ExperimentSpec, ctx: RunCtx) -> dict[str, Any]:
    runner = _REGISTRY.get(spec.type)
    if runner is None:
        raise KeyError(f"no runner registered for type={spec.type!r}; "
                       f"available: {types()}")
    return runner(spec.params, ctx)


def _import_default_experiments() -> None:
    """Import each experiment module so its @register decorator fires."""
    from .experiments import exp_a_convergence, exp_c_ablate, exp_d_halt, exp_propose  # noqa: F401


# Experiment types whose specs depend only on the canonical_200 walk
# (one spec per (task, type) pair). The bulk of the loop is these.
SEEDS_CANONICAL: set[str] = {"exp_a_convergence", "exp_c_ablate", "exp_d_halt"}

# Experiment types seeded at coarser cadence (once per "trigger" event the loop
# fires — e.g., every 10 completed canonical experiments). Currently the LLM
# proposer; could later include longer-horizon syntheses.
SEEDS_EPOCHAL: set[str] = {"exp_propose"}


def seed_specs(ctx: RunCtx, epoch: int = 0, *, types_filter: Iterable[str] | None = None) -> list[ExperimentSpec]:
    """Build initial experiment specs covering canonical_200 × each canonical type.

    Epochal types (e.g. the LLM proposer) are deliberately NOT included here —
    they're queued separately by the loop based on triggers, not bulk walks.

    ``epoch`` is folded into params so re-seeding produces fresh spec IDs and the
    loop can sweep again with the (possibly updated) model.
    """
    _import_default_experiments()
    if ctx.canonical_ids is None:
        raise RuntimeError("RunCtx.canonical_ids must be populated before seeding")
    if types_filter is not None:
        selected = set(types_filter) & set(_REGISTRY)
    else:
        selected = set(_REGISTRY) & SEEDS_CANONICAL
    specs: list[ExperimentSpec] = []
    for stem, pid in ctx.canonical_ids.items():
        for typ in sorted(selected):
            specs.append(
                ExperimentSpec.new(
                    type=typ,
                    params={"stem": stem, "puzzle_id": int(pid), "epoch": epoch},
                )
            )
    return specs


def epochal_spec(typ: str, epoch: int) -> ExperimentSpec:
    """Build one spec for an epochal (non-canonical-walk) type."""
    if typ not in _REGISTRY:
        raise KeyError(f"unknown experiment type: {typ}")
    return ExperimentSpec.new(type=typ, params={"epoch": epoch, "kind": "epochal"})


def child_specs(parent: ExperimentSpec, kinds: Iterable[str]) -> list[ExperimentSpec]:
    """Spawn followup experiments referencing the parent's params."""
    out: list[ExperimentSpec] = []
    for typ in kinds:
        if not has(typ):
            log.warning("child_specs: unknown type %r; skipping", typ)
            continue
        params = {**parent.params, "epoch": parent.params.get("epoch", 0) + 1}
        out.append(ExperimentSpec.new(type=typ, params=params, parent_id=parent.id))
    return out


__all__ = [
    "RunCtx",
    "Runner",
    "register",
    "has",
    "types",
    "run_experiment",
    "seed_specs",
    "epochal_spec",
    "child_specs",
    "SEEDS_CANONICAL",
    "SEEDS_EPOCHAL",
]
