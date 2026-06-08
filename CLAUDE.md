# CLAUDE.md — Phase 1: Qwen 2.5-0.5B Interpretability

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this worktree is

A `git worktree` of `~/arc-agi-research/` (hub repo) on branch `phase1`. Sibling worktrees (`~/arc-phase{2,3,4}-*/`) hold the other phases. The hub on `main` owns the research plan and the cross-phase contract — **read `~/arc-agi-research/shared/` for invariants** (TPU schedule, SAE handoff format, canonical task IDs, GCS paths). Do not vendor copies of those files here.

The single research-plan reference for this phase is `~/arc-agi-research/Phase 1 — Autoregressive LLM Interpretability (Qwen 0.5B).md` on the hub. Its compute assumptions (Colab + TransformerLens) are obsolete; see "Compute" below for what actually runs.

## What this phase does

Mechanistic interpretability of the merged **Qwen 2.5-0.5B + LoRA** model (the ARC-fine-tuned variant on Google Drive, mirrored to GCS per `shared/gcs_paths.md`). Four experiments, in order:

| Exp | Question | Output |
|---|---|---|
| A | Which attention heads specialize in row / column / color / adjacency? | Head × layer heatmap with ablation-confirmed top heads |
| B | At which layer are spatial concepts linearly decodable? | Per-layer probe accuracy curves for 5+ concepts |
| C | What sparse, monosemantic features does the residual stream learn? | SAE feature dictionary (≥50 labeled features) |
| D | What does TTT actually change in the weights/activations? | Per-layer weight-delta + SAE-feature-delta figures |

## Compute

TPU-only — no Colab, no GPU. Per `shared/tpu_schedule.md`:

- **`main-1` (v4-8, us-central2-b)** for activation collection + hook experiments (Exps A, B, D).
- **`node-v5e-64-europe-west4-b` (v5litepod-64)** on **days 5–6** for SAE training (Exp C). Phase 1 goes first because it has the smallest dataset; do not overrun into Phase 3's slot.

Activate the venv first: `source ~/venv-maxtext-py312/bin/activate`.

## Code reuse — do not start from scratch

`~/activation-extract/` already provides everything needed for Exps A and C's data side:

| Need | Reuse |
|---|---|
| Qwen 2.5 in JAX/Flax with activation hooks | `~/activation-extract/models/qwen.py` + `models/decoder.py` (`GenericModelWithActivations`, `create_model_with_hooks`) |
| Load merged LoRA weights into the JAX model | `models/conversion.py` — HF→Flax template-driven converter; the merged model is just an HF checkpoint with the LoRA already folded in |
| ARC prompt pipeline | `core/dataset_utils.py` + `arc24/` (use `--pipeline prompt`) |
| Grid-only token streams for SAE input | `core/grid_chunking.py` (use `--pipeline grid_chunking`) |
| Multi-host extraction | `multihost_extract.py` (overkill for v4-8 single-host, but available) |
| Shard storage + GCS upload | `core/activation_storage.py` — produces shards in the format `shared/sae_handoff.md` mandates |
| SAE training | `~/sae-worktree/` (branch `feat/sae-training`) — consumes those shards |

**What this worktree adds on top:** attention-pattern recording from the JAX hooks (extend `GenericAttention` or post-process from the residual hooks), linear probing harness, TTT-delta analysis.

## Library pivot vs. the original Phase 1 doc

The original doc lists TransformerLens, `sae-lens`, `circuitsvis`, `nnsight`. **None of these are used.** TransformerLens is CUDA-only and we have no GPU; `sae-lens` is replaced by `~/sae-worktree/`. For circuit visualizations, render statically from probe outputs (matplotlib / plotly) — no interactive `circuitsvis`.

If a sub-task genuinely needs TransformerLens (e.g., for a sanity check against a published result), it must run somewhere else (Colab, Kaggle) — not on this VM.

## Tests

This phase is upstream of two cross-architecture comparisons (Exp E in Phase 2, the CKA in Phase 4), so silent drift is the main risk. Required tests:

- **Canonical-task-IDs tripwire** — assert the task IDs you operate on equal the first-200-alphabetical from ARC-AGI-1 training (see `~/arc-agi-research/shared/task_ids/README.md`).
- **LoRA-merge parity** — assert HF-loaded merged model and JAX-loaded merged model produce identical logits on 10 sample prompts (float32, atol 1e-4). This is the model-equivalence guarantee; without it nothing downstream is meaningful. Model-family parity infra exists at `~/activation-extract/tests/test_models_families.py` — extend that pattern.
- **SAE shard format** — assert `metadata.json` matches the schema in `shared/sae_handoff.md`, with `extra: {lora_merged: true, ttt_round: int}`.

## Cross-phase handoff

Outputs go to GCS at the paths in `shared/gcs_paths.md`:
- Activations → `gs://arc-mi-research/activations/phase1/qwen_layer{6,12,18}/`
- SAE → `gs://arc-mi-research/sae_checkpoints/phase1/`
- Probe results JSON → `gs://arc-mi-research/probing_results/phase1_probes.json`

When done with a TPU, **stop it**: `gcloud compute tpus tpu-vm stop main-1 --zone=us-central2-b`. The hub does not babysit TPU lifecycle.
