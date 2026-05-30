"""Tripwire: assert Phase 2's canonical-200 list matches the shared definition.

Drift here silently breaks Exp E (CKA vs Phase 1) and Phase 4's cross-arch
comparison. Per shared/task_ids/README.md, the canonical set is exactly
``sorted(p.stem for p in ARC_TRAIN.glob('*.json'))[:200]``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `interp` importable when running pytest from the worktree root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interp.canonical_tasks import DEFAULT_ARC_TRAIN, get_canonical_200  # noqa: E402


@pytest.mark.skipif(
    not DEFAULT_ARC_TRAIN.is_dir(),
    reason=f"ARC training dir missing at {DEFAULT_ARC_TRAIN}; set $ARC_TRAIN_DIR.",
)
def test_canonical_200_matches_shared_definition() -> None:
    expected = sorted(p.stem for p in DEFAULT_ARC_TRAIN.glob("*.json"))[:200]
    actual = get_canonical_200()
    assert actual == expected, (
        f"canonical_200 drifted from sorted-first-200: "
        f"symmetric diff = {set(actual) ^ set(expected)}"
    )
    assert len(actual) == 200
    assert len(set(actual)) == 200  # uniqueness
