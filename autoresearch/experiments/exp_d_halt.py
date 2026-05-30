"""Experiment D (lightweight v1) — ACT halting dynamics.

For one canonical task, record the q_halt_logits trajectory across supervision
steps. In eval mode TRM does NOT early-exit (it always runs ``halt_max_steps``)
so this gives us a clean look at *what the model would do* if the q-head were
trusted: the iteration where ``sigmoid(q_halt) ≥ 0.5`` first triggers.

This is a stepping stone toward the real Exp D — sampling the per-iteration
residual stream for the SAE handoff. We start here because it requires no
hook plumbing, only the ``q_halt_logits`` field already in outputs.
"""
from __future__ import annotations

import json
from typing import Any

import torch

from ..inference import ensure_canonical, ensure_model, make_batch, run_iterations
from ..registry import RunCtx, register


@register("exp_d_halt")
def run(params: dict[str, Any], ctx: RunCtx) -> dict[str, Any]:
    ensure_canonical(ctx)
    ensure_model(ctx)
    pid = int(params["puzzle_id"])
    stem = params["stem"]

    batch = make_batch(pid)
    if batch is None:
        return {"status": "skipped", "reason": "no train-split row"}

    steps = run_iterations(ctx.model, batch, max_steps=ctx.meta.halt_max_steps)
    if not steps:
        return {"status": "skipped", "reason": "no steps captured"}

    halt_probs: list[float] = []
    for s in steps:
        if "q_halt_logits" not in s:
            continue
        halt_probs.append(float(torch.sigmoid(s["q_halt_logits"]).mean().item()))
    if not halt_probs:
        return {"status": "skipped", "reason": "model exposes no q_halt_logits"}

    threshold = 0.5
    first_would_halt = next((i for i, p in enumerate(halt_probs) if p >= threshold), -1)

    result_path = ctx.results_dir / f"exp_d/{stem}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "stem": stem,
        "puzzle_id": pid,
        "halt_prob_per_step": [round(p, 6) for p in halt_probs],
        "first_would_halt_step": first_would_halt,
        "threshold": threshold,
    }
    with open(result_path, "w") as f:
        json.dump(record, f)

    return {
        "status": "ok",
        "first_would_halt_step": first_would_halt,
        "result_path": str(result_path.relative_to(ctx.state_dir)),
    }
