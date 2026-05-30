"""Canonical 200 ARC-AGI-1 task IDs.

Definition (from shared/task_ids/README.md): the first 200 ARC-AGI-1 *training*
tasks when sorted alphabetically by filename stem. Reproducible from the
fchollet/ARC-AGI dataset alone — never check in a JSON.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ARC_TRAIN = Path(
    os.environ.get(
        "ARC_TRAIN_DIR",
        str(Path.home() / "arc-agi-research" / "arc_data" / "ARC-AGI" / "data" / "training"),
    )
)

CANONICAL_N = 200


def get_canonical_200(arc_train_dir: Path | str | None = None) -> list[str]:
    """Return the canonical 200 task IDs (sorted, deterministic)."""
    root = Path(arc_train_dir) if arc_train_dir else DEFAULT_ARC_TRAIN
    if not root.is_dir():
        raise FileNotFoundError(
            f"ARC-AGI training dir not found: {root}. "
            "Set $ARC_TRAIN_DIR or clone fchollet/ARC-AGI to the expected path."
        )
    ids = sorted(p.stem for p in root.glob("*.json"))
    if len(ids) < CANONICAL_N:
        raise RuntimeError(
            f"Only {len(ids)} ARC-AGI training tasks found at {root}; expected ≥{CANONICAL_N}."
        )
    return ids[:CANONICAL_N]


if __name__ == "__main__":
    ids = get_canonical_200()
    print(f"canonical_200: {len(ids)} ids; first={ids[0]} last={ids[-1]}")
