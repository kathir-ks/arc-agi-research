"""Tripwire #1 — canonical 200 task IDs match the first-200-alphabetical from ARC-AGI-1.

If this drifts, Phase 4's CKA silently compares different tasks across architectures.
"""

from phase1.canonical_tasks import ARC_TRAIN, get_canonical_200


def test_canonical_200_matches_first_alphabetical():
    expected = sorted(p.stem for p in ARC_TRAIN.glob("*.json"))[:200]
    actual = get_canonical_200()
    assert actual == expected, f"drifted from canonical 200: {set(actual) ^ set(expected)}"


def test_canonical_200_length():
    assert len(get_canonical_200()) == 200


def test_canonical_200_is_sorted():
    ids = get_canonical_200()
    assert ids == sorted(ids)
