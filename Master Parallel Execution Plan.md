# ARC-AGI Mechanistic Interpretability — Master Parallel Execution Plan
## All 4 Phases Running Simultaneously

**Researcher:** Kathir
**Execution Mode:** Fully Parallel — All phases start Day 1
**Total Duration:** ~16–18 days (down from 30 days sequential)
**Compute:** 3 × Colab Pro sessions (one per model) + GCP TPU (shared, staggered SAE training)
**Target Output:** arXiv preprint + conference submission
**Last Updated:** May 2026

---

## Why Parallel Execution

Running all 4 phases sequentially takes 30 days. Running them in parallel compresses this to ~16–18 days because:
- The three model analyses (Phases 1, 2, 3) have **no data dependencies on each other** — they study different models on the same ARC task set
- The only real dependency is that **Phase 4 (paper writing) needs results** — but it can start immediately with introduction/background/setup sections, and drop results sections in as each phase finishes
- Google Colab Pro supports multiple simultaneous sessions, and the three models (Qwen 0.5B, TRM 7M, LLaDA 8B) are sufficiently different in compute profile that they don't contend heavily

The bottleneck is Phase 2 (TRM, 11 days of experiments), so the critical path is:
`Day 1 setup → Days 2–9 parallel experiments → Days 10–14 Phase 2 tail + paper drafting → Days 15–18 polish + submission`

---

## Compute Allocation

### Three Colab Sessions (always running in parallel)

| Session | Phase | Model | Primary Use |
|---|---|---|---|
| **Colab 1** | Phase 1 | Qwen 2.5-0.5B | TransformerLens analysis, probing, TTT |
| **Colab 2** | Phase 2 | TRM (7M) | nnsight hooks, iteration tracking, causal tracing |
| **Colab 3** | Phase 3 | LLaDA-8B | nnsight bidirectional hooks, diffusion analysis |

### GCP TPU (Staggered Schedule)

| Days | Phase | Task |
|---|---|---|
| Days 5–6 | Phase 1 | SAE training on Qwen activations |
| Days 7–8 | Phase 3 | Timestep-conditioned SAE training on LLaDA activations |
| Days 9–11 | Phase 2 | SAE training on TRM activations (largest job — ~23M vectors) |

> **TPU scheduling rule:** Never run two SAE training jobs simultaneously. Upload activation `.npy` files to GCS on Colab, then SSH into TPU VM to train. Phase 1 SAE goes first (simplest), then Phase 3, then Phase 2 (largest dataset).

### Google Drive / GCS Layout

```
MyDrive/
├── arc_data/
│   ├── training/        ← ARC-AGI-1 training tasks (shared by all phases)
│   └── evaluation/      ← ARC-AGI-1 eval tasks
├── arc_models/
│   ├── qwen25_05b_arc_merged/   ← Phase 1 model
│   └── llada_8b_architects/     ← Phase 3 model (download on Day 1)
├── arc_results/
│   ├── phase1/          ← Colab 1 outputs
│   ├── phase2/          ← Colab 2 outputs
│   ├── phase3/          ← Colab 3 outputs
│   └── paper/           ← Figures + LaTeX source
└── sae_activations/
    ├── phase1_layer12_acts.npy
    ├── phase3_timestep_acts.h5
    └── phase2_trm_acts.h5
```

---

## Shared Infrastructure (Set Up on Day 1 — All Sessions)

All three Colab sessions share the same ARC dataset pipeline. Run this in each Colab before starting phase-specific work:

```python
# === SHARED ARC DATA PIPELINE — run in all 3 Colab sessions ===
import json, os, numpy as np
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
ARC_PATH = '/content/drive/MyDrive/arc_data/'

def load_arc_tasks(split='training'):
    tasks = {}
    folder = Path(ARC_PATH) / split
    for fname in folder.glob('*.json'):
        with open(fname) as f:
            tasks[fname.stem] = json.load(f)
    return tasks

def grid_to_tokens(grid):
    return '\n'.join(' '.join(str(c) for c in row) for row in grid)

def format_arc_prompt(task, example_idx=0):
    prompt = ""
    for ex in task['train']:
        prompt += f"Input:\n{grid_to_tokens(ex['input'])}\nOutput:\n{grid_to_tokens(ex['output'])}\n\n"
    prompt += f"Input:\n{grid_to_tokens(task['test'][example_idx]['input'])}\nOutput:\n"
    return prompt

# Load shared task set — SAME 200-task subset used by all phases for cross-comparison
arc_tasks = load_arc_tasks('training')
SHARED_TASK_IDS = sorted(arc_tasks.keys())[:200]  # first 200 alphabetically = consistent across sessions
shared_tasks = {tid: arc_tasks[tid] for tid in SHARED_TASK_IDS}
print(f"Shared analysis set: {len(shared_tasks)} tasks")
```

> **Critical:** All three Colab sessions must use the **same `SHARED_TASK_IDS`** for any cross-architecture comparison (Phase 2 Exp E, Phase 4 CKA analysis, paper figures). Use the first 200 alphabetically-sorted task IDs as the canonical shared subset.

---

## Day-by-Day Parallel Timeline

### Day 1 — Universal Setup

| Phase | Work | Deliverable |
|---|---|---|
| **Phase 1** (Colab 1) | Install TransformerLens, load Qwen from Drive, verify model loads, run baseline activation collection on 5 tasks | Confirmed: model loads, tokenization works, activations collected |
| **Phase 2** (Colab 2) | Clone TRM repo, install nnsight, load checkpoint, verify baseline ARC score (~45%), map nnsight hook points | Confirmed: TRM loads, hooks accessible, score verified |
| **Phase 3** (Colab 3) | Install nnsight, download LLaDA-8B-Base + ARChitects weights, test masked forward pass, verify hook points | Confirmed: LLaDA loads, masked inputs work, hooks accessible |
| **Phase 4** (local) | Set up LaTeX skeleton (paper template), write abstract placeholder, draft Introduction (§1) and ARC-AGI Background (§2.1) | 500-word intro draft, paper structure committed to git |

### Days 2–3 — First Experiments

| Phase | Experiment | Deliverable |
|---|---|---|
| **Phase 1** | Exp A: Attention pattern analysis — extract patterns, compute spatial specialization scores, run causal ablations | Head heatmap (all layers × heads); top 5 spatially specialized heads confirmed by ablation |
| **Phase 2** | Exp A: Iteration-by-iteration answer tracking — hook answer state at all 32–64 iterations, compute pixel accuracy curves per task type | Convergence curves (4–6 task categories); phase transition iterations identified |
| **Phase 3** | Exp A: Diffusion step analysis — track unmasking order, collect residual stream activations at 5 representative mask ratios | Unmasking order visualization; mask-ratio activation snapshots saved to GCS |
| **Phase 4** | Write §2.2 (MI Background: SAE, probing, causal tracing), §2.3 (Model architectures overview — Qwen, TRM, LLaDA) | 800-word background section |

### Day 4 — Second Experiments Begin

| Phase | Experiment | Deliverable |
|---|---|---|
| **Phase 1** | Exp B: Residual stream probing — build probe dataset, train row/col/color/boundary probes at all layers | Layer-by-layer probe accuracy curves; peak spatial encoding layers identified |
| **Phase 2** | Exp B: Latent state decoding — collect latent states at all iterations, train probes for 4 semantic concepts | "When does TRM know what?" heatmap |
| **Phase 3** | Exp A: continued — analyze spatial structure of unmasking order (which cells first? object boundaries vs centers?) | Spatial unmasking analysis figure |
| **Phase 4** | Write §3 (Experimental Setup: ARC data pipeline, tokenization, shared task subset, MI toolkit) — incorporates shared infrastructure above | ~600-word methods section |

### Days 5–6 — SAE Preparation + Phase 3 Exp B + Phase 1 Done

| Phase | Experiment | Deliverable |
|---|---|---|
| **Phase 1** | Exp C: Collect SAE activation dataset (500 tasks × 3 layers) → upload to GCS → **hand off to TPU** | `phase1_layer{6,12,18}_acts.npy` on GCS; Colab 1 starts Exp D |
| **Phase 1** (TPU) | SAE training begins on TPU (runs in background) | SAE checkpoint at end of Day 6 |
| **Phase 2** | Exp C: Task ID causal tracing — run clean/corrupted passes, patch-back experiments, per-iteration causal importance | Causal trace bar chart (attn vs MLP by iteration) |
| **Phase 3** | Exp B: Bidirectional attention analysis — compute spatial specialization scores for all LLaDA heads, compare to Qwen | LLaDA head heatmap; Qwen vs LLaDA attention comparison ready |
| **Phase 4** | Write §4 (Phase 1 Results) — attention heatmap figure, probe curves figure, preliminary SAE note | Phase 1 results section draft (update SAE after Day 6) |

### Day 7 — TTT Analysis + LLaDA Exp C + TRM Exp D Prep

| Phase | Experiment | Deliverable |
|---|---|---|
| **Phase 1** | Exp D: TTT before/after — run TTT on 10 tasks, compare activations and SAE feature deltas | TTT delta figures; Phase 1 **COMPLETE** |
| **Phase 1** (TPU) | SAE training finishes; feature interpretation run | SAE checkpoint + feature taxonomy CSV; update §4 with SAE results |
| **Phase 2** | Exp D begins: Collect TRM activation dataset across all iterations → upload to GCS → **hand off to TPU** | `phase2_trm_acts.h5` on GCS; Colab 2 starts Exp E simultaneously |
| **Phase 3** | Exp C: 1D vs 2D RoPE probing — load both LLaDA variants, run linear probes, compute CKA between representations | RoPE comparison probe curves + CKA matrix; Phase 3 **COMPLETE** |
| **Phase 4** | Write §6 (Phase 3 Results — diffusion unmasking, attention comparison, RoPE analysis) | Phase 3 results section draft |

### Days 8–10 — TRM Finishing + Cross-Architecture Synthesis

| Phase | Experiment | Deliverable |
|---|---|---|
| **Phase 1** | **COMPLETE** — finalize figures, write Phase 1 summary notes | All Phase 1 figures in `arc_results/phase1/` |
| **Phase 2** | Exp D (TPU): SAE training on TRM activations | TRM SAE checkpoint; feature interpretation |
| **Phase 2** | Exp E: Qwen vs TRM CKA + feature matching (runs in Colab 2 while TPU trains) | CKA heatmap; universal features list |
| **Phase 3** | **COMPLETE** — finalize all figures | All Phase 3 figures in `arc_results/phase3/` |
| **Phase 4** | Write §5 (Phase 2 Results — convergence, latent decoding, causal tracing); update §4 with final SAE results | Phase 2 draft results (SAE placeholder until Day 10) |

### Days 11–13 — Phase 2 Complete + Paper Integration

| Phase | Experiment | Deliverable |
|---|---|---|
| **Phase 2** | Finalize SAE interpretation, finalize Exp E; Phase 2 **COMPLETE** | All Phase 2 figures in `arc_results/phase2/` |
| **Phase 4** | Complete §5 with SAE and CKA findings; Write §7 (Cross-Architecture Comparison) — the key synthesis section | Cross-architecture comparison section; paper narrative committed |
| **Phase 4** | Produce all 10 paper figures in publication-quality format | Final figures (`.pdf`, 300 DPI, correct fonts) |

### Days 14–16 — Paper Writing Sprint

| Day | Task |
|---|---|
| **Day 14** | Write §8 (Limitations and Future Work); §9 (Conclusion); update Abstract to reflect actual findings |
| **Day 15** | Full paper proofread; fix cross-references; LaTeX formatting pass; bibliography completion |
| **Day 16** | Final figure review; check all captions; author contributions; arXiv metadata |

### Days 17–18 — Submission

| Day | Task |
|---|---|
| **Day 17** | **arXiv v1 submission** (upload PDF + source); post to relevant Slack/Discord communities |
| **Day 18** | Buffer day for revisions; begin identifying target workshop for submission |

---

## Phase 4 "Write as You Go" Schedule

Phase 4 runs throughout the entire project. Here is exactly when each section gets written, keyed to experimental progress:

| Section | Write When | Input Needed |
|---|---|---|
| §1 Introduction | Day 1 | Project framing (available now) |
| §2 Background | Days 2–3 | Literature review (already done) |
| §3 Experimental Setup | Day 4 | Shared infrastructure confirmed working |
| §4 Phase 1 Results (Exps A+B) | Days 5–6 | Attention heatmap + probe curves from Days 2–4 |
| §4 Phase 1 Results (Exps C+D) | Day 7 | SAE results + TTT analysis from Days 5–7 |
| §5 Phase 2 Results (Exps A+B+C) | Days 8–9 | Convergence curves, latent heatmap, causal trace |
| §5 Phase 2 Results (Exps D+E) | Days 11–12 | SAE + CKA from Days 9–11 |
| §6 Phase 3 Results | Day 7–8 | All Phase 3 experiments complete by Day 7 |
| §7 Cross-Architecture Comparison | Days 12–13 | All phases complete |
| §8 Limitations | Day 14 | Full view of what worked and what didn't |
| §9 Conclusion | Day 14 | All results known |
| Abstract (final) | Day 15 | After conclusion is written |

> **Writing rule:** Write each section as a **draft** immediately when its results are available, even if incomplete. Use `[TODO: insert SAE result here]` placeholders. Never wait for a section to be "perfect" before writing the next.

---

## Cross-Phase Coordination Points

There are three moments in the project where information must flow between phases:

### Coordination Point 1 — Day 1 Evening
**Verify shared task IDs are identical across all three Colab sessions.** Print `SHARED_TASK_IDS[:5]` in each Colab and confirm they match. If tokenization differs between models, record the token count per task to handle sequence length alignment in Exp E.

### Coordination Point 2 — Day 7 (after Phase 3 completes)
**Hand off LLaDA attention specialization scores to Phase 4.** The cross-architecture attention comparison (Paper Figure 10) requires attention specialization scores from all three models. Phase 1 produces these on Days 2–3; Phase 2 produces them indirectly through iteration analysis; Phase 3 produces them on Days 5–6. By Day 7, all three should be available to Phase 4 for the comparison figure.

### Coordination Point 3 — Day 11 (after Phase 2 Exp E)
**Merge SAE feature lists from all three phases for the universal features analysis.** The cross-architecture feature matching (Paper §7) requires all three SAE dictionaries. Phase 1 SAE is done by Day 6; Phase 3 SAE by Day 8; Phase 2 SAE by Day 10. Day 11 is the earliest all three are available.

---

## Risk Management

| Risk | Probability | Mitigation |
|---|---|---|
| LLaDA (8B) too slow on Colab T4 | Medium | Use Colab A100; or scope to LLaDA-3B if available; or run fewer diffusion steps |
| TRM nnsight hooks fail inside Python loop | Medium | Inspect TRM `forward()` source manually; use custom hook registration if nnsight context manager fails |
| TPU time runs out before all SAEs train | Low | Train Phase 1 SAE on Colab GPU (slower but feasible for 0.5B d_model=896); Phase 2 SAE is the biggest risk |
| Phase 2 Exp E CKA comparison is inconclusive | Medium | This is a research finding, not a failure — inconclusive CKA is a valid result showing architectural divergence |
| Paper narrative unclear after all results | Low | Pre-commit to one of three narratives (convergence / divergence / mixed) by Day 11; don't wait for perfect clarity |

---

## Updated Timeline at a Glance

```
Day  1  : [P1] Qwen setup  |  [P2] TRM setup  |  [P3] LLaDA setup  |  [P4] Intro + §2
Day  2  : [P1] Exp A attn  |  [P2] Exp A iters |  [P3] Exp A diffuse |  [P4] §2 background
Day  3  : [P1] Exp A abl.  |  [P2] Exp A viz   |  [P3] Exp A spatial |  [P4] §2 architectures
Day  4  : [P1] Exp B probe |  [P2] Exp B latent|  [P3] Exp A contd.  |  [P4] §3 setup
Day  5  : [P1] Exp C coll. |  [P2] Exp C causal|  [P3] Exp B bidir.  |  [P4] §4 P1 attn+probe
Day  6  : [P1] SAE→TPU    |  [P2] Exp C viz   |  [P3] Exp B done    |  [P4] §4 P1 cont.
Day  7  : [P1] Exp D TTT  |  [P2] Exp D coll. |  [P3] Exp C RoPE   |  [P4] §6 P3 results
          [P1] SAE done   |  [P2] Exp E CKA   |  [P3] COMPLETE     |
Day  8  : [P1] COMPLETE   |  [P2] Exp D TPU   |  [P3] fig polish   |  [P4] §4 P1 SAE+TTT
Day  9  : [P4] §5 P2 A+B+C results draft
Day 10  : [P2] SAE done   |  [P2] Exp E feat. |                     |  [P4] §5 P2 update
Day 11  : [P2] COMPLETE   |  [P4] §5 P2 SAE+CKA  |  [P4] Cross-phase SAE merge
Day 12  : [P4] §7 cross-architecture comparison section
Day 13  : [P4] §7 complete; all 10 figures production-quality
Day 14  : [P4] §8 Limitations + §9 Conclusion + abstract final
Day 15  : [P4] Full proofread + LaTeX pass + bibliography
Day 16  : [P4] Final figure review + arXiv metadata
Day 17  : arXiv v1 SUBMISSION
Day 18  : Buffer / begin workshop targeting
```

---

## arXiv Early-Posting Strategy

**After Phase 2 is complete (Day 11), consider posting an arXiv v1 with Phase 1 + Phase 2 results only, then updating to v2 with Phase 3 included.**

Benefits:
- Establishes priority on TRM interpretability findings — the most novel contribution
- Allows community feedback before Phase 3 results are finalized
- Phase 3 (LLaDA) results, if strong, make v2 a significantly upgraded submission

Trigger for early posting: Phase 2 complete + Phase 1 complete + §§1–5 written.

---

## References to Core Papers (Shared Across All Phases)

**Mechanistic Interpretability (LLMs):**
- Mathematical Framework for Transformer Circuits (Anthropic, 2021)
- Towards Monosemanticity (Anthropic, 2023)
- Scaling and Evaluating SAEs (OpenAI, arXiv:2406.04093)
- JumpReLU SAEs (DeepMind, arXiv:2407.14435)
- Gemma Scope (DeepMind, arXiv:2408.05147)
- ROME / Causal Tracing (NeurIPS 2022, arXiv:2202.05262)
- IOI Circuit (arXiv:2211.00593)
- Language Models Represent Space and Time (arXiv:2310.02207)
- Linear Representation Hypothesis (arXiv:2311.03658)
- Open Problems in Mech Interp (arXiv:2501.16496)

**Diffusion Model Interpretability:**
- LLaDA paper (arXiv:2502.09992)
- MDLM (arXiv:2406.07524)
- Emergence in Diffusion Models — SAE on diffusion (arXiv:2504.15473)
- Revelio — k-sparse SAEs on diffusion (arXiv:2411.16725)
- DiffLens — causal interventions in diffusion (arXiv:2503.20483)
- What Does BERT Look At? (arXiv:1906.04341)

**ARC-AGI + Models:**
- TRM paper (arXiv:2510.04871)
- ARChitects 2025 Technical Report (arXiv:2601.10904)
- ARC-AGI-2 paper (arXiv:2505.11831)
- TTT for ARC (arXiv:2411.07279)
- Specialization after Generalization (arXiv:2509.24510)

**Cross-Architecture Comparison:**
- CKA (arXiv:1905.00414)
- Universal Feature Spaces via SAE (arXiv:2410.06981)

---

*This document is the master coordination guide. Keep it open alongside all 4 phase documents during execution. Update the timeline table daily to track actual vs. planned progress.*
