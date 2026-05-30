"""Experiment C — task-id ablation.

Runs the same input twice — once with the real ``puzzle_id``, once with
``puzzle_identifiers=0`` (the blank id) — and records per-iteration KL divergence
between the two logit distributions. Tells us when (and whether) the model
consumes the task-id signal.

The blank id corresponds to padding/no-task and is the natural "null prior" the
model was trained against, so KL ≈ 0 throughout means the task-id is unused for
this task; a large early-iteration KL means it's used as initialisation.
"""
from __future__ import annotations

import json
from typing import Any

import torch

from ..inference import (
    ensure_canonical,
    ensure_model,
    kl_div_per_position,
    make_batch,
    run_iterations,
)
from ..registry import RunCtx, register


@register("exp_c_ablate")
def run(params: dict[str, Any], ctx: RunCtx) -> dict[str, Any]:
    ensure_canonical(ctx)
    ensure_model(ctx)
    pid = int(params["puzzle_id"])
    stem = params["stem"]

    real = make_batch(pid)
    blank = make_batch(pid, override_pid=0)
    if real is None or blank is None:
        return {"status": "skipped", "reason": "no train-split row"}

    real_steps = run_iterations(ctx.model, real, max_steps=ctx.meta.halt_max_steps)
    blank_steps = run_iterations(ctx.model, blank, max_steps=ctx.meta.halt_max_steps)
    n = min(len(real_steps), len(blank_steps))
    if n == 0:
        return {"status": "skipped", "reason": "no steps captured"}

    kls = []
    for i in range(n):
        kl = kl_div_per_position(real_steps[i]["logits"], blank_steps[i]["logits"])
        kls.append(float(kl.mean().item()))

    result_path = ctx.results_dir / f"exp_c/{stem}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "stem": stem,
        "puzzle_id": pid,
        "n_steps": n,
        "kl_real_vs_blank_per_step": [round(k, 6) for k in kls],
        "peak_kl_step": int(max(range(n), key=lambda i: kls[i])),
        "peak_kl_value": round(max(kls), 6) if kls else 0.0,
    }
    with open(result_path, "w") as f:
        json.dump(record, f)

    return {
        "status": "ok",
        "peak_kl_step": record["peak_kl_step"],
        "peak_kl_value": record["peak_kl_value"],
        "result_path": str(result_path.relative_to(ctx.state_dir)),
    }
