"""Baseline gate: TRM loaded checkpoint produces ≥40% exact-match accuracy
on canonical_200 (training tasks).

Canonical_200 are *training* tasks — the model has trained on them, so accuracy
should be well above the 41% pass@2-on-eval that the checkpoint card reports.
This is the "model loaded correctly" smoke test for everything downstream.

Skipped when the ARC dataset hasn't been built yet (no identifiers.json).

Env vars:
    TRM_BASELINE_FULL=1  → run all 200 canonical tasks (slow on CPU, ~10-15 min)
                            default: first 20 (quick, ~1-2 min)
    TRM_BASELINE_THRESH=0.40  → override the accuracy threshold
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from interp.canonical_tasks import get_canonical_200
from interp.puzzle_id_map import DEFAULT_IDENTIFIERS_JSON, get_canonical_200_puzzle_ids
from interp.trm_runner import PHASE2_ROOT, load_model

DATA_DIR = PHASE2_ROOT / "third_party" / "TinyRecursiveModels" / "data" / "arc1concept-aug-1000"
TRAIN_DIR = DATA_DIR / "train"

# Tokens: 0=PAD, 1=EOS, 2..11 = digits 0..9.
PAD_ID = 0
EOS_ID = 1
GRID_H = GRID_W = 30
SEQ_LEN = GRID_H * GRID_W  # 900


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _crop_to_grid(seq: np.ndarray) -> np.ndarray:
    """Find the largest top-left rectangle without EOS/PAD inside. Mirrors
    evaluators/arc.py::_crop."""
    grid = seq.reshape(GRID_H, GRID_W)
    max_area, max_size = 0, (0, 0)
    num_c = GRID_W
    for num_r in range(1, GRID_H + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) or (x > 11):  # PAD or EOS or out-of-range
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    return (grid[: max_size[0], : max_size[1]] - 2).astype(np.uint8)


def _load_canonical_examples(canonical_ids: dict[str, int]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """For each canonical (stem, puzzle_id), return one (puzzle_id, input, label)
    triple drawn from the train split. The unaugmented row exists for each task
    because the builder includes the identity augmentation."""
    inputs = np.load(TRAIN_DIR / "all__inputs.npy", mmap_mode="r")
    labels = np.load(TRAIN_DIR / "all__labels.npy", mmap_mode="r")
    pids = np.load(TRAIN_DIR / "all__puzzle_identifiers.npy")
    # Map pid → first row index in the train split.
    first_idx_for_pid: dict[int, int] = {}
    for i, pid in enumerate(pids):
        if int(pid) not in first_idx_for_pid:
            first_idx_for_pid[int(pid)] = i
    triples: list[tuple[int, np.ndarray, np.ndarray]] = []
    for stem, pid in canonical_ids.items():
        if pid not in first_idx_for_pid:
            continue  # canonical task had no train-split example (shouldn't happen for training subset)
        i = first_idx_for_pid[pid]
        triples.append((pid, inputs[i].copy(), labels[i].copy()))
    return triples


def _make_batch(triples: list[tuple[int, np.ndarray, np.ndarray]]) -> dict[str, torch.Tensor]:
    pids = np.array([p for p, _, _ in triples], dtype=np.int32)
    ins = np.stack([x for _, x, _ in triples])
    lbls = np.stack([y for _, _, y in triples])
    return {
        "inputs": torch.from_numpy(ins).to(torch.int32),
        "labels": torch.from_numpy(lbls).to(torch.int32),
        "puzzle_identifiers": torch.from_numpy(pids).to(torch.int32),
    }


def _exact_match_after_crop(pred_seq: np.ndarray, label_seq: np.ndarray) -> bool:
    pred_grid = _crop_to_grid(pred_seq)
    label_grid = _crop_to_grid(label_seq)
    if pred_grid.shape != label_grid.shape:
        return False
    return bool(np.array_equal(pred_grid, label_grid))


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

@pytest.mark.skipif(
    not DEFAULT_IDENTIFIERS_JSON.is_file(),
    reason=f"dataset not built yet ({DEFAULT_IDENTIFIERS_JSON} missing); "
    "run `python -m dataset.build_arc_dataset ...` from the TRM repo first.",
)
def test_trm_baseline_above_threshold() -> None:
    full = os.environ.get("TRM_BASELINE_FULL", "0") == "1"
    threshold = float(os.environ.get("TRM_BASELINE_THRESH", "0.40"))
    n_tasks = 200 if full else 20

    canonical_ids = get_canonical_200_puzzle_ids(strict=True)
    # Take a deterministic slice ordered by canonical_200 order.
    ordered = [(s, canonical_ids[s]) for s in get_canonical_200() if s in canonical_ids][:n_tasks]
    sliced_ids = dict(ordered)
    triples = _load_canonical_examples(sliced_ids)
    assert len(triples) == n_tasks, f"expected {n_tasks} canonical triples, got {len(triples)}"

    model, meta = load_model(seq_len=SEQ_LEN, batch_size=len(triples))
    batch = _make_batch(triples)

    with torch.no_grad():
        carry = model.initial_carry(batch)
        for step in range(meta.halt_max_steps):
            carry, outputs = model(carry, batch)
            if bool(carry.halted.all()):
                break
        logits = outputs["logits"].to(torch.float32)
        preds = logits.argmax(-1).cpu().numpy()  # (B, seq_len)
    labels = batch["labels"].cpu().numpy()

    hits = sum(_exact_match_after_crop(preds[i], labels[i]) for i in range(len(triples)))
    accuracy = hits / len(triples)
    print(f"[test_trm_baseline] n={n_tasks} hits={hits} accuracy={accuracy:.3f} threshold={threshold:.2f}")
    assert accuracy >= threshold, (
        f"TRM baseline accuracy {accuracy:.3f} < {threshold:.2f} on {n_tasks} canonical tasks. "
        "Check: checkpoint loaded? puzzle_id mapping correct? dataset built with the right subsets?"
    )


if __name__ == "__main__":
    # Allow running outside pytest for manual smoke.
    test_trm_baseline_above_threshold()
