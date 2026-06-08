"""Phase 1 model loader + provenance builder.

Thin wrapper around `models.load_model_for_extraction` that pins the merged
Qwen 2.5-0.5B + LoRA checkpoint and the layers/activation_type we extract
across all four experiments.
"""

from typing import Optional

import jax.numpy as jnp

from models import load_model_for_extraction

PHASE1_MODEL = "KathirKs/qwen-2.5-0.5b"
PHASE1_LAYERS = (6, 12, 18)


def build_provenance(
    config,
    layers: list[int],
    *,
    pipeline: str = "prompt",
    ttt_round: int = 0,
) -> dict:
    """Provenance dict consumed by ActivationStorage and the schema tripwire.

    Matches the schema in ~/arc-agi-research/shared/sae_handoff.md.
    """
    return {
        "model_name": PHASE1_MODEL,
        "model_family": "qwen",
        "hidden_dim": int(config.hidden_size),
        "num_layers": int(config.num_hidden_layers),
        "layers_extracted": list(layers),
        "activation_type": "residual",
        "pipeline": pipeline,
        "num_experts": None,
        "phase": "phase1",
        "extra": {"lora_merged": True, "ttt_round": int(ttt_round)},
    }


def load_phase1_model(
    layers: Optional[list[int]] = None,
    dtype=jnp.bfloat16,
    *,
    pipeline: str = "prompt",
    ttt_round: int = 0,
):
    """Load the Phase 1 model and return everything callers need, including provenance.

    Returns: (jax_model, params, tokenizer, config, spec, provenance_dict)
    """
    layers = list(layers) if layers is not None else list(PHASE1_LAYERS)
    jax_model, params, tokenizer, config, spec = load_model_for_extraction(
        PHASE1_MODEL,
        dtype=dtype,
        layers_to_extract=layers,
        activation_type="residual",
    )
    provenance = build_provenance(config, layers, pipeline=pipeline, ttt_round=ttt_round)
    return jax_model, params, tokenizer, config, spec, provenance
