"""Map canonical_200 task stems → TRM puzzle_emb row IDs.

After `dataset/build_arc_dataset.py` runs, it writes `identifiers.json` — a flat
list where index `i` is the human-readable puzzle name for puzzle_id `i`. Index 0
is "<blank>". For each base task there are up to `num_aug+1` entries:

    "abc123"
    "abc123|||t3|||012345678..."     # augmentation 1
    "abc123|||t7|||987654321..."     # augmentation 2
    ...

For interpretability we want a *single* puzzle_id per canonical task — by
convention, the unaugmented one (no `|||` separator), since that's the puzzle
the original ARC-AGI evaluator sees.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from interp.canonical_tasks import get_canonical_200
from interp.trm_runner import PHASE2_ROOT

PUZZLE_ID_SEP = "|||"
DEFAULT_IDENTIFIERS_JSON = (
    PHASE2_ROOT / "third_party" / "TinyRecursiveModels" / "data" / "arc1concept-aug-1000" / "identifiers.json"
)


def load_identifiers(identifiers_path: Path | str = DEFAULT_IDENTIFIERS_JSON) -> list[str]:
    p = Path(identifiers_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"identifiers.json not found at {p}. "
            "Run `python -m dataset.build_arc_dataset --input-file-prefix kaggle/combined/arc-agi "
            "--output-dir data/arc1concept-aug-1000 --subsets training evaluation concept "
            "--test-set-name evaluation` from the TRM repo first."
        )
    with open(p) as f:
        return json.load(f)


def build_stem_to_ids(identifiers: list[str]) -> dict[str, list[int]]:
    """Group puzzle_ids by their base stem (the part before `|||`)."""
    out: dict[str, list[int]] = defaultdict(list)
    for pid, name in enumerate(identifiers):
        if name == "<blank>":
            continue
        stem = name.split(PUZZLE_ID_SEP, 1)[0]
        out[stem].append(pid)
    return dict(out)


def get_unaugmented_id(stem_to_ids: dict[str, list[int]], identifiers: list[str], stem: str) -> int:
    """Return the puzzle_id for the *unaugmented* version of `stem`."""
    for pid in stem_to_ids[stem]:
        if identifiers[pid] == stem:
            return pid
    raise KeyError(f"no unaugmented puzzle_id found for stem={stem!r} (only augmented variants).")


def get_canonical_200_puzzle_ids(
    identifiers_path: Path | str = DEFAULT_IDENTIFIERS_JSON,
    *,
    arc_train_dir: Path | str | None = None,
    strict: bool = True,
) -> dict[str, int]:
    """Return {canonical_stem: unaugmented_puzzle_id} for all 200 canonical tasks.

    If strict=True (default), raise if any canonical task is missing from the
    identifiers list — that means the dataset build used a different subset mix.
    """
    identifiers = load_identifiers(identifiers_path)
    stem_to_ids = build_stem_to_ids(identifiers)
    canonical = get_canonical_200(arc_train_dir)
    missing: list[str] = []
    mapping: dict[str, int] = {}
    for stem in canonical:
        if stem not in stem_to_ids:
            missing.append(stem)
            continue
        mapping[stem] = get_unaugmented_id(stem_to_ids, identifiers, stem)
    if strict and missing:
        raise RuntimeError(
            f"{len(missing)}/{len(canonical)} canonical tasks missing from identifiers.json "
            f"(first 5: {missing[:5]}). Did the dataset build use --subsets training evaluation concept?"
        )
    return mapping


if __name__ == "__main__":
    import sys

    try:
        mapping = get_canonical_200_puzzle_ids()
    except FileNotFoundError as e:
        print(f"[puzzle_id_map] {e}", file=sys.stderr)
        sys.exit(1)
    print(f"mapped {len(mapping)} canonical stems to puzzle_ids")
    sample = list(mapping.items())[:3]
    for stem, pid in sample:
        print(f"  {stem:>10s} → puzzle_id={pid}")
