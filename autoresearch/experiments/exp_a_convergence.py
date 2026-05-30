"""Experiment A — per-iteration convergence curves.

For one canonical task, run the model and record:
  * argmax change rate (Hamming distance / seq_len) between consecutive iterations
  * mean softmax entropy per iteration
  * whether the prediction is correct (after crop) at each iteration

Output JSON (small, < 5 KB per task). Spawns followup C + D specs for the
same task so the queue compounds organically.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..inference import (
    SEQ_LEN,
    ensure_canonical,
    ensure_model,
    make_batch,
    run_iterations,
    softmax_entropy,
)
from ..registry import RunCtx, register


def _crop_to_grid(seq: np.ndarray) -> np.ndarray:
    """Mirror of tests/test_trm_baseline._crop_to_grid (without numba)."""
    grid = seq.reshape(30, 30)
    max_area, max_size = 0, (0, 0)
    num_c = 30
    for num_r in range(1, 31):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) or (x > 11):
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    return (grid[: max_size[0], : max_size[1]] - 2).astype(np.uint8)


def _exact_match(pred: np.ndarray, label: np.ndarray) -> bool:
    pg = _crop_to_grid(pred)
    lg = _crop_to_grid(label)
    if pg.shape != lg.shape:
        return False
    return bool(np.array_equal(pg, lg))


@register("exp_a_convergence")
def run(params: dict[str, Any], ctx: RunCtx) -> dict[str, Any]:
    ensure_canonical(ctx)
    ensure_model(ctx)
    pid = int(params["puzzle_id"])
    stem = params["stem"]

    batch = make_batch(pid)
    if batch is None:
        return {"status": "skipped", "reason": "no train-split row for puzzle_id"}

    steps = run_iterations(ctx.model, batch, max_steps=ctx.meta.halt_max_steps)
    if not steps:
        return {"status": "skipped", "reason": "model returned no steps"}

    label = batch["labels"][0].cpu().numpy()
    argmaxes = [s["logits"].argmax(-1).cpu().numpy()[0] for s in steps]  # list of (S,)
    entropies = [float(softmax_entropy(s["logits"]).mean().item()) for s in steps]
    hammings = []
    for i in range(1, len(argmaxes)):
        diff = float(np.mean(argmaxes[i] != argmaxes[i - 1]))
        hammings.append(diff)
    correct_per_iter = [bool(_exact_match(am, label)) for am in argmaxes]
    first_correct = next((i for i, c in enumerate(correct_per_iter) if c), -1)
    # Crude "slow convergence" flag: most change happens in the second half.
    second_half_total = float(sum(hammings[len(hammings) // 2 :])) if hammings else 0.0
    total = float(sum(hammings)) if hammings else 1e-9
    slow_convergence = total > 0 and (second_half_total / total) > 0.4

    result_path = ctx.results_dir / f"exp_a/{stem}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "stem": stem,
        "puzzle_id": pid,
        "n_steps": len(steps),
        "argmax_hamming_per_step": [round(h, 6) for h in hammings],
        "mean_entropy_per_step": [round(e, 6) for e in entropies],
        "correct_per_step": correct_per_iter,
        "first_correct_step": first_correct,
        "slow_convergence": slow_convergence,
    }
    with open(result_path, "w") as f:
        json.dump(record, f)

    # Followups: same task through Exp C (task-id ablation) and Exp D (halt dynamics).
    followups = ["exp_c_ablate", "exp_d_halt"]
    return {
        "status": "ok",
        "first_correct_step": first_correct,
        "slow_convergence": slow_convergence,
        "n_steps": len(steps),
        "result_path": str(result_path.relative_to(ctx.state_dir)),
        "followups": followups,
    }
