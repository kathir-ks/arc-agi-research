"""Shared inference helpers for experiment runners.

Centralises the model-load + per-iteration capture loop so each experiment is
just a thin wrapper that decides what to record. Saves us from copy-pasting
the carry/halt loop that already lives in :mod:`tests.test_trm_baseline`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from interp.canonical_tasks import get_canonical_200
from interp.puzzle_id_map import get_canonical_200_puzzle_ids
from interp.trm_runner import load_model

log = logging.getLogger("autoresearch.inference")

DEFAULT_BATCH_SIZE = 1  # CPU bound; per-experiment latency matters more than throughput
SEQ_LEN = 900


def ensure_model(ctx) -> None:  # type: RunCtx
    """Lazy-load the model into the ctx on first use. Idempotent."""
    if ctx.model is None or ctx.meta is None:
        log.info("loading TRM checkpoint (one-time, ~30s) …")
        model, meta = load_model(seq_len=SEQ_LEN, batch_size=DEFAULT_BATCH_SIZE)
        ctx.model = model
        ctx.meta = meta


def ensure_canonical(ctx) -> None:
    if ctx.canonical_ids is None:
        ctx.canonical_ids = get_canonical_200_puzzle_ids(strict=False)
        _ = get_canonical_200()  # sanity touch so failures surface early


def _train_arrays() -> tuple[Path, Path, Path]:
    from interp.trm_runner import PHASE2_ROOT
    train_dir = PHASE2_ROOT / "third_party" / "TinyRecursiveModels" / "data" / "arc1concept-aug-1000" / "train"
    return (
        train_dir / "all__inputs.npy",
        train_dir / "all__labels.npy",
        train_dir / "all__puzzle_identifiers.npy",
    )


_PID_TO_ROW_CACHE: dict[int, int] | None = None


def pid_to_row(puzzle_id: int) -> int | None:
    """Return the first row index in train split for ``puzzle_id``. Memoised."""
    global _PID_TO_ROW_CACHE
    if _PID_TO_ROW_CACHE is None:
        _, _, pids_path = _train_arrays()
        pids = np.load(pids_path)
        cache: dict[int, int] = {}
        for i, pid in enumerate(pids):
            cache.setdefault(int(pid), i)
        _PID_TO_ROW_CACHE = cache
    return _PID_TO_ROW_CACHE.get(puzzle_id)


def make_batch(puzzle_id: int, *, override_pid: int | None = None) -> dict[str, torch.Tensor] | None:
    """Build a length-1 batch for ``puzzle_id`` from the train split. Returns
    ``None`` if the puzzle_id has no train-split row."""
    row = pid_to_row(puzzle_id)
    if row is None:
        return None
    inputs_path, labels_path, _ = _train_arrays()
    inp = np.load(inputs_path, mmap_mode="r")[row].copy()
    lbl = np.load(labels_path, mmap_mode="r")[row].copy()
    pid_for_batch = override_pid if override_pid is not None else puzzle_id
    return {
        "inputs": torch.from_numpy(inp[None, :]).to(torch.int32),
        "labels": torch.from_numpy(lbl[None, :]).to(torch.int32),
        "puzzle_identifiers": torch.tensor([pid_for_batch], dtype=torch.int32),
    }


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-position softmax entropy in nats. ``logits`` shape (B, S, V)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)  # (B, S)


def kl_div_per_position(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) per position. Inputs (B, S, V), output (B, S)."""
    logp = torch.log_softmax(logits_p.float(), dim=-1)
    logq = torch.log_softmax(logits_q.float(), dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def run_iterations(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    max_steps: int,
) -> list[dict[str, torch.Tensor]]:
    """Run the ACT loop, collecting outputs from each supervision step.

    Each entry contains the cpu-detached tensors ``logits`` and ``q_halt_logits``.
    Stops early if ``carry.halted`` is all True.
    """
    out: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        carry = model.initial_carry(batch)
        for _ in range(max_steps):
            carry, outputs = model(carry, batch)
            rec = {
                "logits": outputs["logits"].detach().to(torch.float32).cpu(),
            }
            if "q_halt_logits" in outputs:
                rec["q_halt_logits"] = outputs["q_halt_logits"].detach().to(torch.float32).cpu()
            out.append(rec)
            if bool(carry.halted.all()):
                break
    return out


__all__ = [
    "ensure_model",
    "ensure_canonical",
    "make_batch",
    "pid_to_row",
    "run_iterations",
    "softmax_entropy",
    "kl_div_per_position",
    "SEQ_LEN",
]
