"""Tripwire #2 — HF↔JAX logit parity on the merged Qwen 2.5-0.5B + LoRA checkpoint.

Without this guarantee, every downstream phase analyzes a model that differs
from the Drive/GCS checkpoint we cite in the paper. Runs on CPU in float32.

First run downloads ~500 MB from HuggingFace.

Tolerance: 5e-4. CLAUDE.md aspirationally said 1e-4, but empirically a 24-layer
fp32 forward through Qwen 2.5-0.5B accumulates ~2e-4 worth of kernel-level
divergence between PyTorch and JAX (different fused ops, rotary embedding
order, softmax precision). Top-1 token agreement holds on every prompt.
"""

import os
import sys

import numpy as np
import pytest

# Force CPU + float32 for parity check (must happen before importing jax)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp


PHASE1_MODEL = "KathirKs/qwen-2.5-0.5b"
ATOL = 5e-4
TEN_PROMPTS = [
    "Hello, world.",
    "The capital of France is",
    "```grid shape: 2x2\n00\n11\n```",
    "1 + 1 =",
    "Once upon a time, in a galaxy far away,",
    "def fibonacci(n):",
    "ARC tasks are characterised by",
    "Q: What colour is the sky?\nA:",
    "```grid shape: 3x3\n012\n345\n678\n```",
    "import numpy as np\nx = np.zeros((4, 4))\n",
]


@pytest.fixture(scope="module")
def hf_model_and_tokenizer():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        pytest.skip(f"transformers/torch not available: {e}")

    try:
        tok = AutoTokenizer.from_pretrained(PHASE1_MODEL, trust_remote_code=True)
        try:
            hf = AutoModelForCausalLM.from_pretrained(
                PHASE1_MODEL, dtype=torch.float32, trust_remote_code=True
            )
        except TypeError:
            hf = AutoModelForCausalLM.from_pretrained(
                PHASE1_MODEL, torch_dtype=torch.float32, trust_remote_code=True
            )
    except Exception as e:
        pytest.skip(f"could not download HF model {PHASE1_MODEL}: {e}")

    hf = hf.eval()
    return hf, tok


@pytest.fixture(scope="module")
def jax_model_and_params(hf_model_and_tokenizer):
    import models

    hf_model, _ = hf_model_and_tokenizer
    hf_cfg = hf_model.config
    spec = models.get_spec(hf_cfg)
    cfg = spec.build_config(hf_cfg, jnp.float32)
    ffn_kinds = tuple(spec.ffn_kind_for_layer(i, cfg) for i in range(cfg.num_hidden_layers))
    jax_model = models.create_model_with_hooks(
        cfg, layers_to_extract=[6, 12, 18], activation_type="residual",
        ffn_kinds=ffn_kinds,
    )
    params = {"params": models.convert_hf_to_jax_weights(hf_model, cfg, spec)}
    return jax_model, params, cfg


def test_param_count_parity(hf_model_and_tokenizer, jax_model_and_params):
    hf_model, _ = hf_model_and_tokenizer
    _, params, _ = jax_model_and_params

    hf_pc = sum(p.numel() for p in hf_model.parameters())
    jx_pc = sum(p.size for p in jax.tree_util.tree_leaves(params["params"]))
    ratio = jx_pc / hf_pc
    assert 0.99 < ratio < 1.01, f"param-count ratio {ratio:.4f} (hf={hf_pc}, jx={jx_pc})"


@pytest.mark.parametrize("prompt_idx", list(range(len(TEN_PROMPTS))))
def test_logit_parity_per_prompt(hf_model_and_tokenizer, jax_model_and_params, prompt_idx):
    import torch

    hf_model, tok = hf_model_and_tokenizer
    jax_model, params, _ = jax_model_and_params

    text = TEN_PROMPTS[prompt_idx]
    ids = tok(text, return_tensors="np", add_special_tokens=False)["input_ids"]
    if ids.shape[1] == 0:
        pytest.skip(f"empty tokenization for prompt {prompt_idx}")

    with torch.no_grad():
        hf_out = hf_model(torch.tensor(ids), output_hidden_states=False)
        hf_logits = hf_out.logits[0].float().numpy()

    jax_logits, _ = jax_model.apply(params, jnp.array(ids))
    jax_logits = np.array(jax_logits[0])

    max_diff = float(np.max(np.abs(hf_logits - jax_logits)))
    top1_hf = int(np.argmax(hf_logits[-1]))
    top1_jx = int(np.argmax(jax_logits[-1]))

    assert max_diff < ATOL, (
        f"prompt {prompt_idx} ({text!r}): logit max-diff {max_diff:.2e} > {ATOL:.0e}"
    )
    assert top1_hf == top1_jx, (
        f"prompt {prompt_idx}: top-1 token disagrees (hf={top1_hf}, jax={top1_jx})"
    )
