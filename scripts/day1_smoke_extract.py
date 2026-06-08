#!/usr/bin/env python3
"""Phase 1 Day 1 smoke extraction.

Load merged Qwen 2.5-0.5B + LoRA, build ARC prompts for the first N canonical
tasks, forward through the JAX model, persist residual activations at layers
6/12/18 via ActivationStorage, and (by default) upload to GCS.

Usage (on TPU main-1 after `source ~/venv-maxtext-py312/bin/activate`):
    python scripts/day1_smoke_extract.py --num_tasks 5
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_ACTIVATION_EXTRACT = Path.home() / "activation-extract"
for p in (_HERE, _ACTIVATION_EXTRACT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax
import jax.numpy as jnp
import numpy as np

from phase1.canonical_tasks import ARC_TRAIN, get_canonical_200
from phase1.model_loader import load_phase1_model

from core.activation_storage import ActivationStorage
from core.dataset_utils import create_prompts_from_dataset
from arc24.encoders import create_grid_encoder

GCS_BUCKET = "arc-mi-research"
GCS_PREFIX = "activations/phase1/_smoke"
DEFAULT_GRID_ENCODER = "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_tasks", type=int, default=5)
    p.add_argument("--predictions_per_task", type=int, default=1,
                   help="Prompt augmentations per task. 1 is enough for a smoke run.")
    p.add_argument("--output_dir", type=str, default=str(_HERE / "artifacts/_smoke"))
    p.add_argument("--no_upload", action="store_true",
                   help="Skip GCS upload; just write shards locally.")
    p.add_argument("--prompt_version", type=str, default="output-from-examples-v0")
    p.add_argument("--grid_encoder", type=str, default=DEFAULT_GRID_ENCODER)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = p.parse_args()

    print("=" * 70)
    print(f"jax.devices(): {jax.devices()}")
    print(f"jax.default_backend(): {jax.default_backend()}")
    print("=" * 70)

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    print(f"\n[1/5] Loading first {args.num_tasks} canonical tasks from {ARC_TRAIN}")
    task_ids = get_canonical_200()[: args.num_tasks]
    tasks = {}
    for tid in task_ids:
        with open(ARC_TRAIN / f"{tid}.json") as f:
            tasks[tid] = json.load(f)
    print(f"  ✓ Loaded: {list(tasks.keys())}")

    print("\n[2/5] Loading merged model + converting HF→JAX weights")
    jax_model, params, tokenizer, config, spec, provenance = load_phase1_model(dtype=dtype)
    print(f"  ✓ {spec.name}: {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"vocab={config.vocab_size}")
    print(f"  ✓ provenance: {provenance}")

    print("\n[3/5] Building ARC prompts")
    grid_encoder = create_grid_encoder(args.grid_encoder)
    prompts_data = create_prompts_from_dataset(
        tasks=tasks,
        grid_encoder=grid_encoder,
        tokenizer=tokenizer,
        prompt_version=args.prompt_version,
        predictions_per_task=args.predictions_per_task,
        random_seed=42,
        verbose=True,
    )
    print(f"  ✓ {len(prompts_data)} prompts")
    print(f"  Sample prompt (first 400 chars):\n{prompts_data[0]['prompt'][:400]}")

    upload = not args.no_upload
    print(f"\n[4/5] Setting up ActivationStorage (upload_to_gcs={upload})")
    storage = ActivationStorage(
        output_dir=args.output_dir,
        upload_to_gcs=upload,
        gcs_bucket=GCS_BUCKET if upload else None,
        gcs_prefix=GCS_PREFIX,
        provenance=provenance,
        shard_size_gb=0.5,
        verbose=True,
    )

    layers_to_extract = provenance["layers_extracted"]
    print(f"\n[5/5] Forward passes — {len(prompts_data)} prompts, layers={layers_to_extract}")
    for sample_idx, pd in enumerate(prompts_data):
        ids = tokenizer(pd["prompt"], return_tensors="np",
                        add_special_tokens=False)["input_ids"]
        input_ids = jnp.array(ids)
        logits, _, acts = jax_model.apply(params, input_ids, return_activations=True)
        top1 = int(jnp.argmax(logits[0, -1]))
        decoded = tokenizer.decode([top1])
        print(f"  task={pd['task_id']} idx={sample_idx} seq_len={ids.shape[1]} "
              f"top1={top1} ({decoded!r})")
        for layer_idx in layers_to_extract:
            layer_act = np.array(acts[f"layer_{layer_idx}"])[0]
            storage.add_activation(
                layer_idx=layer_idx,
                activation=layer_act,
                sample_idx=sample_idx,
                text_preview=f"Task: {pd['task_id']}",
                source_doc_id=pd["task_id"],
            )

    storage.finalize()

    print("\n" + "=" * 70)
    print("✓ Day 1 smoke extraction complete.")
    if upload:
        print(f"  Shards uploaded to: gs://{GCS_BUCKET}/{GCS_PREFIX}/")
    print(f"  Local shards: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
