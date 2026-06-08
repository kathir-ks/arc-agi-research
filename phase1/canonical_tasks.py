"""Canonical 200 ARC-AGI-1 task IDs (first 200 alphabetical).

Per ~/arc-agi-research/shared/task_ids/README.md.
"""

from pathlib import Path

ARC_TRAIN = Path.home() / "arc-agi-research/arc_data/ARC-AGI/data/training"


def get_canonical_200() -> list[str]:
    if not ARC_TRAIN.is_dir():
        raise FileNotFoundError(
            f"ARC-AGI training dir not found at {ARC_TRAIN}. "
            "Clone with: git clone https://github.com/fchollet/ARC-AGI.git "
            f"{ARC_TRAIN.parent.parent}"
        )
    ids = sorted(p.stem for p in ARC_TRAIN.glob("*.json"))
    if len(ids) < 200:
        raise RuntimeError(f"Expected ≥200 ARC training tasks at {ARC_TRAIN}, found {len(ids)}")
    return ids[:200]
