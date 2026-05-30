"""TRM checkpoint loader + thin inference adapter.

Loads `Sanjin2024/TinyRecursiveModels-ARC-AGI-1` (or any compatible checkpoint).
Reads architecture dims directly from the state_dict so we don't depend on
`all_config.yaml` having every field. Strips the `_orig_mod.model.` prefix added
by torch.compile + ACTLossHead during training.

Design choices:
- CPU only (Phase 2 spec). bfloat16 forward dtype is fine on torch ≥ 2.1.
- Patches `dataset.common` import via sys.path so we can import the TRM model
  module without installing the third_party tree as a package.
- Does NOT load `puzzle_dataset`; callers must supply pre-tokenized batches.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PHASE2_ROOT = Path(__file__).resolve().parents[1]
TRM_SRC = PHASE2_ROOT / "third_party" / "TinyRecursiveModels"
DEFAULT_CKPT_DIR = PHASE2_ROOT / "third_party" / "checkpoints" / "arc-agi-1"
WRAPPER_PREFIX = "_orig_mod.model."


def _ensure_trm_on_path() -> None:
    """Make `models.*` importable from the TRM clone."""
    p = str(TRM_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclasses.dataclass
class CheckpointMeta:
    vocab_size: int
    hidden_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int
    L_layers: int
    # From all_config.yaml:
    H_cycles: int
    L_cycles: int
    num_heads: int
    expansion: float
    pos_encodings: str
    halt_max_steps: int
    halt_exploration_prob: float
    puzzle_emb_len: int
    no_ACT_continue: bool
    mlp_t: bool
    forward_dtype: str
    # Not in checkpoint — supplied by dataset / fallback default.
    seq_len: int = 900  # 30×30 ARC-AGI grid
    batch_size: int = 1


def _peek_dims(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """Extract dims directly from tensor shapes (more reliable than yaml)."""
    sd = state_dict
    L_layers = max(
        (int(k.split(".layers.")[1].split(".")[0]) for k in sd if ".L_level.layers." in k),
        default=-1,
    ) + 1
    return {
        "vocab_size": sd[f"{WRAPPER_PREFIX}inner.embed_tokens.embedding_weight"].shape[0],
        "hidden_size": sd[f"{WRAPPER_PREFIX}inner.embed_tokens.embedding_weight"].shape[1],
        "num_puzzle_identifiers": sd[f"{WRAPPER_PREFIX}inner.puzzle_emb.weights"].shape[0],
        "puzzle_emb_ndim": sd[f"{WRAPPER_PREFIX}inner.puzzle_emb.weights"].shape[1],
        "L_layers": L_layers,
    }


def load_meta(ckpt_dir: Path | str = DEFAULT_CKPT_DIR, *, ckpt_file: str = "step_155718") -> CheckpointMeta:
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "all_config.yaml"
    state_dict_path = ckpt_dir / ckpt_file
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    arch = cfg["arch"]
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise TypeError(f"expected state_dict mapping, got {type(sd).__name__}")
    dims = _peek_dims(sd)
    return CheckpointMeta(
        vocab_size=dims["vocab_size"],
        hidden_size=dims["hidden_size"],
        num_puzzle_identifiers=dims["num_puzzle_identifiers"],
        puzzle_emb_ndim=dims["puzzle_emb_ndim"],
        L_layers=dims["L_layers"],
        H_cycles=int(arch["H_cycles"]),
        L_cycles=int(arch["L_cycles"]),
        num_heads=int(arch["num_heads"]),
        expansion=float(arch["expansion"]),
        pos_encodings=str(arch["pos_encodings"]),
        halt_max_steps=int(arch["halt_max_steps"]),
        halt_exploration_prob=float(arch["halt_exploration_prob"]),
        puzzle_emb_len=int(arch["puzzle_emb_len"]),
        no_ACT_continue=bool(arch["no_ACT_continue"]),
        mlp_t=bool(arch["mlp_t"]),
        forward_dtype=str(arch["forward_dtype"]),
    )


def _strip_wrapper_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (k[len(WRAPPER_PREFIX) :] if k.startswith(WRAPPER_PREFIX) else k): v
        for k, v in sd.items()
    }


def build_model(meta: CheckpointMeta) -> torch.nn.Module:
    _ensure_trm_on_path()
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    config_dict = dict(
        batch_size=meta.batch_size,
        seq_len=meta.seq_len,
        puzzle_emb_ndim=meta.puzzle_emb_ndim,
        num_puzzle_identifiers=meta.num_puzzle_identifiers,
        vocab_size=meta.vocab_size,
        H_cycles=meta.H_cycles,
        L_cycles=meta.L_cycles,
        H_layers=0,
        L_layers=meta.L_layers,
        hidden_size=meta.hidden_size,
        expansion=meta.expansion,
        num_heads=meta.num_heads,
        pos_encodings=meta.pos_encodings,
        halt_max_steps=meta.halt_max_steps,
        halt_exploration_prob=meta.halt_exploration_prob,
        forward_dtype=meta.forward_dtype,
        mlp_t=meta.mlp_t,
        puzzle_emb_len=meta.puzzle_emb_len,
        no_ACT_continue=meta.no_ACT_continue,
    )
    return TinyRecursiveReasoningModel_ACTV1(config_dict)


def load_model(
    ckpt_dir: Path | str = DEFAULT_CKPT_DIR,
    *,
    ckpt_file: str = "step_155718",
    seq_len: int = 900,
    batch_size: int = 1,
) -> tuple[torch.nn.Module, CheckpointMeta]:
    meta = load_meta(ckpt_dir, ckpt_file=ckpt_file)
    meta.seq_len = seq_len
    meta.batch_size = batch_size

    sd = torch.load(Path(ckpt_dir) / ckpt_file, map_location="cpu", weights_only=False)
    sd = _strip_wrapper_prefix(sd)

    model = build_model(meta)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # We expect the rotary cos_sin buffers (non-persistent) to be missing.
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected}")
    # Filter known-OK missing keys (none yet; tighten later).
    if missing:
        # Only warn for now; the lm_head/embed sharing pattern produces no missing.
        print(f"[trm_runner] note: {len(missing)} missing keys (e.g. {missing[:3]})", file=sys.stderr)
    model.eval()
    return model, meta


__all__ = [
    "CheckpointMeta",
    "DEFAULT_CKPT_DIR",
    "PHASE2_ROOT",
    "TRM_SRC",
    "build_model",
    "load_meta",
    "load_model",
]
