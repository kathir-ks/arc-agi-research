# CLAUDE.md — Phase 2: Tiny Recursive Model (TRM) Interpretability

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this worktree is

A `git worktree` of `~/arc-agi-research/` (hub repo) on branch `phase2`. Sibling worktrees (`~/arc-phase{1,3,4}-*/`) hold the other phases. Cross-phase invariants live in `~/arc-agi-research/shared/` — TPU schedule, SAE handoff format, canonical task IDs, GCS paths. Read them; do not vendor copies.

The research-plan reference for this phase is `~/arc-agi-research/Phase 2 — Tiny Recursive Model (TRM) Interpretability.md` on the hub. Its compute/library assumptions are obsolete; see below.

## What this phase does

Mechanistic interpretability of **TRM** (`SamsungSAILMontreal/TinyRecursiveModels`, 7M params, MIT). This is the **critical-path phase** (longest, most novel — no prior MI work on TRM). Five experiments:

| Exp | Question | Output |
|---|---|---|
| A | How does the answer state evolve across recursive iterations? | Convergence curves per task category; phase-transition iterations identified |
| B | What does the latent "scratchpad" encode at each iteration? | "When does TRM know what?" heatmap from per-iteration probes |
| C | How does the `task_id_token` mechanistically control behavior? | Causal-tracing plot showing which layer/iteration consumes the task ID |
| D | What features does the single transformer block compute? | TRM SAE feature library across all iterations |
| E | How similar are TRM's representations to Qwen's? | CKA / RSA matrix between Phase 1 and Phase 2 layer activations |

Exp E is **the** cross-architecture comparison and is downstream of Phase 1's SAE + probe outputs. Run it last.

## Compute

**TRM runs on CPU.** 7M params is comfortably CPU-feasible and torch_xla for a model this small is pure overhead. Reserve `main-4` (v4-8) as a fallback only if iteration-collection becomes a measurable bottleneck — it shouldn't.

SAE training (Exp D) uses the shared **v5litepod-64 on days 9–11**, per `shared/tpu_schedule.md`. Phase 2's SAE is the **largest job** (~23M activation vectors: 200 tasks × ~64 iterations × ~900 token positions × 3 layers within the single block) so it goes last in the staggered slot.

Use the system Python (`/usr/bin/python3`, 3.10) or a fresh CPU venv for TRM; do not use `~/venv-maxtext-py312` here unless you specifically need it for SAE shard inspection — TRM is PyTorch.

## Code reuse

- **TRM model code:** clone `SamsungSAILMontreal/TinyRecursiveModels` into this worktree (gitignore it). Do not fork into the repo. Wrap it with a thin interpretability adapter that exposes hooks at the answer-head, latent-head, and per-iteration residual stream.
- **Storage / SAE handoff:** import `~/activation-extract/core/activation_storage.py` so shards land in the format `shared/sae_handoff.md` mandates. Use `extra: {iteration_index: int, n_iterations: int}` for provenance.
- **SAE training:** `~/sae-worktree/` (branch `feat/sae-training`) consumes the shards. The model-side (TRM) hidden_dim is small (~256–512) so SAE training is fast — the bottleneck is the dataset size, not the model.

## Library pivot vs. the original Phase 2 doc

The original doc proposes `nnsight`. We don't use it. TRM is small enough that a manual PyTorch hook on the single block's forward gives full visibility with less indirection — register `torch.nn.Module.register_forward_hook` on the recursive block and read off `answer_state`, `latent_state`, and the residual each call. No third-party hook library required.

## Tests

Drift here breaks Exp E (the Qwen-vs-TRM comparison) and Phase 4's CKA. Required tests:

- **Canonical-task-IDs tripwire** — same as Phase 1, asserted against `~/arc-agi-research/shared/task_ids/README.md`.
- **TRM baseline ARC score** — assert ≥40% on ARC-AGI-1 eval at n_iterations=64 (paper reports ~45%). This is the model-loaded-correctly smoke test; everything else assumes the model works.
- **Iteration-count invariant** — assert that `n_iterations` recorded in shard `metadata.json` matches what the model actually ran, including for the partial-iterations case (early-exit, if used).
- **SAE shard format** — assert `extra.iteration_index ∈ [0, n_iterations)` for every shard.

## Cross-phase handoff

Outputs go to GCS at the paths in `shared/gcs_paths.md`:
- Activations → `gs://arc-mi-research/activations/phase2/trm_iter{0..63}/`
- SAE → `gs://arc-mi-research/sae_checkpoints/phase2/`
- Probe results → `gs://arc-mi-research/probing_results/phase2_probes.json`
- CKA matrix (Exp E) → `gs://arc-mi-research/cka_matrices/qwen_vs_trm.npy`

## Critical-path note

Phase 4 (paper) blocks on Exp E completing. Phase 1 must publish its probe + SAE outputs *before* Exp E can run. Coordinate with the Phase 1 worktree: read Phase 1's outputs from GCS, not from `~/arc-phase1-qwen/` directly (the worktrees may be at different commits / running different code).
