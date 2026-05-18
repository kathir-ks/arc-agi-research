# ARC-AGI Mechanistic Interpretability Research
## Project & Requirements Document

**Researcher:** Kathir
**Compute:** Google Cloud TPUs (primary)
**Goal:** Learn → Publish (arXiv / conference)
**Target Timeline:** 10–30 days
**Last Updated:** May 2026

---

## Executive Summary

This project investigates the internal representations of neural models trained on ARC-AGI tasks using mechanistic interpretability (MI) techniques. The core hypothesis is that different architectures solving the same spatial-reasoning tasks will develop either convergent or divergent internal feature sets — and understanding this could reveal what "visual abstraction" looks like inside a neural network.

Four phases are defined with increasing architectural complexity. The first two phases are the **core research contribution** and are executable within 10–18 days. Phases 3 and 4 extend and synthesize the findings into a publishable paper.

---

## ⚠️ Critical Setup Note: TPUs vs. TransformerLens

TransformerLens (the standard MI library) is **GPU/CUDA-native**. It does not natively run on TPUs. This requires a hybrid compute strategy:

| Task | Recommended Compute |
|---|---|
| Loading & running models for activation collection | Google Colab Pro (A100/T4) or Kaggle GPU |
| Training SAEs on collected activations | Google Cloud TPU (fast matrix ops, cheap) |
| Running TRM (7M params) for interpretability | Any — even CPU works |
| Large model fine-tuning (if needed) | Google Cloud TPU |

**Practical approach:** Use Colab/Kaggle notebooks with GPU for TransformerLens-based analysis. Save activation datasets to Google Drive. Load those activations on TPU to train SAEs at scale. This hybrid workflow keeps costs low while leveraging TPU speed where it matters.

---

## Phase 1: Ground Truth — Autoregressive LLM (Qwen 0.5B)

**Duration:** Days 1–7
**Model:** Your existing Qwen 2.5-0.5B + LoRA merged model (already on Google Drive)
**Compute:** Google Colab Pro (T4/A100) for TransformerLens; GCP TPU for SAE training

### Objective
Establish a baseline understanding of what an ARC-fine-tuned autoregressive LLM internally encodes. Find whether it develops genuine spatial reasoning features or just statistical pattern matching. This phase serves as the interpretability baseline that all subsequent phases compare against.

---

### 1.1 Environment Setup (Day 1)

**Install dependencies:**
```bash
pip install transformer-lens
pip install sae-lens          # for training Sparse Autoencoders
pip install circuitsvis       # for attention visualization
pip install nnsight           # alternative hook-based library
pip install einops
pip install plotly
```

**Load your model into TransformerLens:**
```python
from transformer_lens import HookedTransformer
import torch

# Load your merged Qwen model
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",   # base architecture
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=True,
)
# Then load your fine-tuned weights on top
# (merge your LoRA weights into this loaded model)
```

> **Note:** TransformerLens has native Qwen2 support via TransformerBridge. If loading fails, use `nnsight` which hooks into any HuggingFace model directly.

**ARC input tokenization format:**
```
Input grid (row by row, space-separated color numbers):
"0 0 1 2\n0 1 2 0\n..."
Output: same format
Prompt: "Input:\n{grid}\nOutput:\n"
```

---

### 1.2 Experiment A — Attention Pattern Analysis (Days 2–3)

**Goal:** Find whether attention heads specialize in spatial roles (row attention, column attention, object-boundary detection, color grouping).

**Method:**
```python
# Run the model with hooks to capture attention patterns
tokens = model.to_tokens(arc_prompt)
logits, cache = model.run_with_cache(tokens)

# Extract attention patterns from all layers & heads
attention_patterns = cache.stack_activation("pattern")
# Shape: (n_layers, n_heads, seq_len, seq_len)

# Visualize with circuitsvis
import circuitsvis as cv
cv.attention.attention_patterns(tokens=tokens, attention=attention_patterns[layer])
```

**Analysis to perform:**
- For each layer × head combination, compute average attention between tokens that are in the **same row**, **same column**, **same color**, and **adjacent**.
- Build a 4-metric matrix (n_layers × n_heads × 4 spatial metrics).
- Identify heads that score high on one metric and low on others — these are spatially specialized heads.
- Visualize on 5–10 representative ARC tasks (symmetry, color mapping, object movement, pattern completion).

**Expected output:** A heatmap showing which layer/head specializes in which spatial relationship. This is directly publishable as a figure.

**Success criteria:** At least 3 heads show statistically significant spatial specialization (p < 0.05 compared to shuffled baseline).

---

### 1.3 Experiment B — Residual Stream Probing (Day 4)

**Goal:** Test whether spatial concepts are linearly encoded in the residual stream at different layers.

**Probing concepts to test:**
- Is token X in the top row / bottom row / middle?
- Is token X the same color as token Y?
- Is token X adjacent to token Y?
- What is the color of token X?
- Is this grid a reflection / rotation of the input?

**Method:**
```python
# Collect residual stream activations across many ARC tasks
activations_by_layer = {}
for layer in range(model.cfg.n_layers):
    activations_by_layer[layer] = []

def hook_fn(value, hook, layer_idx):
    activations_by_layer[layer_idx].append(value.detach().cpu())

hooks = [(f"blocks.{i}.hook_resid_post",
          partial(hook_fn, layer_idx=i))
         for i in range(model.cfg.n_layers)]

for prompt, labels in arc_dataset:
    tokens = model.to_tokens(prompt)
    model.run_with_hooks(tokens, fwd_hooks=hooks)

# Train a linear probe at each layer for each concept
from sklearn.linear_model import LogisticRegression
for concept_name, concept_labels in probing_labels.items():
    for layer_idx, activations in activations_by_layer.items():
        probe = LogisticRegression()
        probe.fit(activations, concept_labels)
        accuracy = probe.score(test_activations, test_labels)
        print(f"Layer {layer_idx} | {concept_name}: {accuracy:.3f}")
```

**Expected output:** A layer-by-layer accuracy curve for each spatial concept — showing at which layer the model "knows" each concept. This reveals the model's internal processing pipeline.

**Success criteria:** At least one spatial concept probes above 80% accuracy at some layer (significantly above 50% random baseline).

---

### 1.4 Experiment C — Sparse Autoencoder (SAE) Training (Days 5–6)

**Goal:** Learn a sparse dictionary of features from the model's residual stream that are active during ARC solving. Find interpretable monosemantic features.

**Train SAE on TPU (this is where GCP TPU shines):**
```python
# Collect ~500k activation vectors from residual stream (middle layer)
# Save to Google Drive as a .pt or .npy file

# On TPU: train SAE
import jax
import jax.numpy as jnp

# SAE architecture: encoder maps d_model -> d_sae (overcomplete)
# d_model = 896 for Qwen 0.5B
# d_sae = 4096 to 16384 (4x to 16x expansion)
# L1 sparsity penalty on activations

class SAE:
    def __init__(self, d_model=896, d_sae=8192):
        self.W_enc = jnp.zeros((d_model, d_sae))
        self.W_dec = jnp.zeros((d_sae, d_model))
        self.b_enc = jnp.zeros(d_sae)

# Alternatively use SAELens library which handles this pipeline
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig
```

**After training, interpret features:**
- For each SAE feature, find the top-10 ARC tokens that maximally activate it
- Label features manually: "activates on red tokens in top row", "activates on color boundaries", etc.
- Compute feature frequency: how often is this feature active vs. dead

**Expected output:** A feature dictionary with 50–200 human-interpretable ARC-specific features. This is the core scientific finding of Phase 1.

**Success criteria:** >10% of SAE features are clearly interpretable (can be labeled with a spatial concept).

---

### 1.5 Experiment D — Test-Time Training Effect Analysis (Day 7)

**Goal:** Compare model internals before and after TTT on a specific task to understand what TTT actually changes in the weights/activations.

**Method:**
- Take your fine-tuned model → run TTT on one new ARC task → capture weights/activations before and after
- Compute weight difference norms per layer: which layers change the most?
- Run attention + probing analysis on both snapshots and compare
- Use Logit Lens to see how the predicted output evolves layer-by-layer

```python
# Logit Lens: project residual stream at each layer to vocab space
for layer in range(model.cfg.n_layers):
    resid = cache[f"blocks.{layer}.hook_resid_post"]
    logits = model.unembed(model.ln_final(resid))
    top_token = logits.argmax(dim=-1)
    print(f"Layer {layer}: predicting '{model.to_str_tokens(top_token)}'")
```

**Expected output:** A visual showing how the model "builds up" the correct answer through layers, and what TTT shifts in those layer dynamics.

---

### Phase 1 Deliverables
- [ ] Attention specialization heatmap (figure)
- [ ] Layer-wise probing accuracy curves for 5+ spatial concepts (figure)
- [ ] SAE feature dictionary with 50+ labeled features (dataset)
- [ ] TTT before/after weight and activation comparison (figure)
- [ ] Written summary: "What does Qwen 0.5B learn about ARC-AGI?"

---

## Phase 2: Novel Discovery — Tiny Recursive Model (TRM)

**Duration:** Days 8–18
**Model:** TRM from `SamsungSAILMontreal/TinyRecursiveModels` (7M params, MIT license)
**Compute:** Anywhere — even CPU/free Colab. SAE training on TPU.

### Objective
Apply the same interpretability toolkit from Phase 1 to TRM. This is **virgin territory** — no mechanistic interpretability work has been published on TRM. Given TRM's unique recursive architecture (same weights reused every pass, separate answer and latent states), the questions here are novel and the findings are directly publishable.

---

### 2.1 Setup & Model Loading (Day 8)

**Clone and load TRM:**
```bash
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels
cd TinyRecursiveModels
pip install -r requirements.txt
```

```python
# TRM has a single transformer block applied recursively
# Load with hooks using nnsight (since TransformerLens may not support custom architectures)
from nnsight import NNsight

model = load_trm_checkpoint("path/to/checkpoint.pt")  # from their repo
traced_model = NNsight(model)
```

**Key architectural facts to internalize:**
- TRM input: `[task_id_token | flattened_input_grid | flattened_answer_state | latent_state]`
- Answer state + latent state both start as random noise
- Same single transformer block is applied N times (typically 32–64 iterations)
- Model is supervised at every iteration, not just the final one

**Replicate baseline results:**
```python
# Verify the model achieves reported scores on ARC-AGI-1 eval set
# Before any interpretability, confirm the model works as expected
score = evaluate_on_arc(model, arc_eval_dataset, n_iterations=64)
print(f"TRM score: {score:.1%}")  # should be ~45% on ARC-AGI-1
```

---

### 2.2 Experiment A — Iteration-by-Iteration Answer Tracking (Days 9–10)

**Goal:** Observe how the answer state evolves across recursive iterations. Does it converge smoothly? Are there distinct "phase transitions" where the answer suddenly improves?

**Method:**
```python
answer_states = []
latent_states = []

# Hook into the model to capture states after each iteration
for iteration in range(n_iterations):
    with traced_model.trace(input_tokens):
        # Capture answer portion of the output
        answer_state = traced_model.answer_head.output.save()
        latent_state = traced_model.latent_head.output.save()

    answer_states.append(answer_state.value)
    latent_states.append(latent_state.value)

# Compute per-cell accuracy at each iteration
for t, answer in enumerate(answer_states):
    predicted_grid = decode_grid(answer)
    accuracy = (predicted_grid == target_grid).float().mean()
    print(f"Iteration {t}: {accuracy:.3f}")

# Plot convergence curve
import matplotlib.pyplot as plt
plt.plot(range(n_iterations), per_iteration_accuracy)
plt.xlabel("Iteration"); plt.ylabel("Cell Accuracy"); plt.title("TRM Convergence")
```

**Analysis:**
- Do different task types converge at different speeds?
- Is there a "minimum useful iteration count" per task category?
- Do errors at late iterations cluster in specific grid regions?

**Expected output:** Convergence curves across task categories (color mapping, symmetry, object movement, etc.). Shows whether TRM "reasons step-by-step" or converges chaotically.

---

### 2.3 Experiment B — Latent State Decoding (Days 11–12)

**Goal:** Understand what the latent state (the hidden scratchpad) represents at each iteration.

This is the most novel experiment in the project. The latent state is TRM's internal working memory — but nobody knows what it encodes.

**Method:**
```python
# Collect latent state vectors across many tasks and iterations
# Shape: (n_tasks, n_iterations, latent_dim)

# Train probes to decode spatial concepts from latent state
spatial_concepts = {
    "object_boundaries": ...,   # binary: is this cell on an object edge?
    "color_identity": ...,       # which color is this cell?
    "transformation_type": ...,  # is this a rotation / reflection / color_swap?
    "unchanged_region": ...,     # does this cell stay the same in output?
}

for concept_name, labels in spatial_concepts.items():
    for iteration in range(n_iterations):
        latent_at_t = all_latent_states[:, iteration, :]
        probe = LogisticRegression()
        acc = cross_val_score(probe, latent_at_t, labels, cv=5).mean()
        print(f"Iter {iteration} | {concept_name}: {acc:.3f}")
```

**Analysis:**
- Does the latent state progressively encode higher-level concepts as iterations increase?
- Is there a "discovery moment" where the transformation type becomes decodable?
- Compare latent state trajectories for correctly-solved vs. incorrectly-solved tasks

**Expected output:** A heatmap showing "when" (at which iteration) each concept becomes linearly decodable from the latent state. This directly answers "what is TRM's scratchpad used for?"

---

### 2.4 Experiment C — Task ID Ablation Deep Dive (Days 13–14)

**Goal:** The TRM paper shows replacing the task ID token → zero accuracy. Understand the mechanism behind this.

**Method:**
```python
# Original: correct task ID
output_correct = run_trm(model, task_id=correct_id, input_grid=grid)

# Ablation 1: blank task ID
output_blank = run_trm(model, task_id=BLANK_TOKEN, input_grid=grid)

# Ablation 2: random task ID
output_random = run_trm(model, task_id=random_id, input_grid=grid)

# Ablation 3: wrong task ID (from different task)
output_wrong = run_trm(model, task_id=other_task_id, input_grid=grid)

# Track how task ID information propagates
# Use activation patching: patch task ID activations from correct run into blank run
patched_output = activation_patch(
    model,
    orig_input=blank_tokens,
    patch_source=correct_tokens,
    patch_layer="embedding",  # start at embedding layer
    patch_positions=[task_id_position]
)
```

**Causal tracing:** Patch the task ID embedding from the correct run into the blank run at each layer, measuring how much accuracy recovers. The layer where patching fully restores accuracy is where task ID information is "consumed."

**Expected output:** A causal tracing plot showing which layer uses the task ID. Reveals TRM's task-conditioning mechanism.

---

### 2.5 Experiment D — SAE on TRM's Single Block (Days 15–16)

**Goal:** Find interpretable features in TRM's single transformer block by training an SAE on its residual stream activations across all iterations.

**Advantage over Phase 1:** TRM's single block sees the same representation space at every iteration. This means the SAE is trained on a much richer dataset (n_tasks × n_iterations samples from the same weight space).

```python
# Collect activations: for every task, every iteration, every token position
# n_tasks=400, n_iterations=64, seq_len=~930 (30x30 grid * 2 + metadata)
# Total: ~23M activation vectors from the residual stream

# This is a good workload for TPU SAE training
# d_model for TRM is small (~256 or 512) — very fast to train SAEs on

# After training SAE:
# For each feature, find max-activating inputs across all (task, iteration, position) triples
# Label: "active on color=3 tokens in right column at iteration > 32"
```

**Expected output:** SAE feature library for TRM. Compare to Phase 1's Qwen SAE features — are the same spatial concepts represented? Are there iteration-specific features (features that only activate in early vs. late iterations)?

---

### 2.6 Experiment E — Comparison Against Qwen 0.5B (Days 17–18)

**Goal:** Directly compare the internal representations of TRM vs. Qwen to answer: do different architectures develop similar spatial features?

**Comparison dimensions:**

| Dimension | Qwen 0.5B | TRM |
|---|---|---|
| How many layers encode "color identity"? | Probe across 24 layers | Probe across 64 iterations |
| Which features appear in SAE? | List from Phase 1 | List from Phase 2 |
| Does attention specialize spatially? | Head analysis | Attention in single block per iteration |
| Where do errors cluster? | Last few layers weak? | Last few iterations weak? |

**Method:** Run same probing suite (Phase 1, Exp B) on TRM latent states. Compare probe accuracy curves side by side. Compute representational similarity (RSA / CKA) between layer representations of both models.

**Expected output:** The key comparative finding for the paper — convergence or divergence of spatial representations across architectures.

---

### Phase 2 Deliverables
- [ ] TRM convergence curves per task category (figures)
- [ ] Latent state decoding heatmap — "when does TRM know what?" (figure)
- [ ] Causal tracing plot for task ID mechanism (figure)
- [ ] TRM SAE feature library (dataset)
- [ ] Qwen vs. TRM representational similarity analysis (figure)
- [ ] Written summary: "What does TRM's scratchpad compute?"

---

## Phase 3: Architectural Contrast — Masked Diffusion (LLaDA)

**Duration:** Days 19–25
**Model:** LLaDA-8B (or the ARChitects' fine-tuned version)
**Compute:** GCP TPU for activation collection + SAE training

### Objective
Extend the analysis to a fundamentally different architecture — masked diffusion — to complete the architectural comparison. LLaDA generates all tokens bidirectionally and iteratively, which is structurally similar to TRM's recursion but mechanistically very different. This phase is the "high risk, high reward" component.

> **If compute or time is tight:** This phase can be scoped down to just one key experiment (Exp A) and treated as a preliminary/future-work section in the paper.

---

### 3.1 Setup & Adaptation (Day 19)

**Load LLaDA:**
```bash
# LLaDA is available on HuggingFace
pip install git+https://github.com/ML-GSAI/LLaDA
```

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Base")
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base")
```

**Key challenge:** TransformerLens has limited support for masked diffusion models. Use `nnsight` for hooking into activations — it works with any HuggingFace model.

**ARC input format for LLaDA:** Grid as text tokens, output grid fully masked initially, iteratively unmasked over diffusion steps.

---

### 3.2 Experiment A — Diffusion Step Analysis (Days 20–21)

**Goal:** Track which grid cells get "unmasked" first across diffusion steps, and whether the unmasking order reveals a reasoning strategy.

```python
# Mask all output tokens
masked_output = [MASK_TOKEN] * output_grid_length

# Run 20 diffusion steps
for step in range(n_diffusion_steps):
    logits = model(input + masked_output)
    confidence = logits.softmax(-1).max(-1).values

    # Unmask the highest-confidence positions
    positions_to_unmask = confidence.topk(k=unmask_per_step).indices
    for pos in positions_to_unmask:
        masked_output[pos] = logits[pos].argmax()

    # Record: which positions were unmasked at this step?
    unmasking_order.append(positions_to_unmask)
```

**Analysis:** Are corner cells unmasked first? Same-color-as-input cells? Object centers? This reveals whether the model solves "easy" parts first (high confidence) and defers "hard" parts.

---

### 3.3 Experiment B — Bidirectional Attention vs. Causal Attention (Days 22–23)

**Goal:** Compare attention patterns in LLaDA (bidirectional, no causal mask) against Qwen (causal). Does bidirectional attention enable richer spatial heads?

Run the same attention specialization analysis from Phase 1, Experiment A on LLaDA. Compute spatial specialization scores for all heads and compare distributions.

**Key question:** Does the absence of a causal mask allow LLaDA to develop more symmetric spatial attention (e.g., attending both left-to-right AND right-to-left in the same head)?

---

### 3.4 Experiment C — 2D RoPE Effect (Days 24–25)

**Goal:** The ARChitects replaced LLaDA's 1D RoPE with 2D RoPE. Measure the actual effect on internal representations.

- Load standard LLaDA-8B (1D RoPE)
- Load ARChitects' fine-tuned version (2D RoPE)
- Run probing analysis on both: does 2D RoPE significantly improve the linearity of spatial concept encoding?
- Compute CKA similarity between layer representations of both versions

**Expected output:** Direct evidence for whether 2D positional encoding makes spatial concepts more or less accessible to MI tools.

---

### Phase 3 Deliverables
- [ ] Diffusion step unmasking order visualization (figures)
- [ ] LLaDA vs. Qwen attention specialization comparison (figure)
- [ ] 1D RoPE vs. 2D RoPE probing comparison (figure)
- [ ] Written section: "Does architecture family determine spatial representation?"

---

## Phase 4: Synthesis & Paper Writing

**Duration:** Days 26–30
**Output:** arXiv preprint ready to submit

### Objective
Synthesize findings from all three models into a coherent narrative and write the paper. The central claim should emerge from the data — either "different ARC architectures converge on similar spatial features" or "architectural family determines the nature of spatial representations" (or a nuanced version of both).

---

### 4.1 Paper Structure (Draft)

**Title ideas:**
- "Circuits for Visual Abstraction: Mechanistic Interpretability of ARC-AGI Solvers"
- "What Do ARC-AGI Models Actually Learn? A Mechanistic Comparison Across Architectures"
- "Inside the ARC Solver: Spatial Features, Recursive Scratchpads, and Diffusion Steps"

**Abstract (template):**
*We study the internal representations of three neural architectures trained on ARC-AGI tasks: an autoregressive LLM (Qwen 0.5B), a tiny recursive model (TRM, 7M params), and a masked diffusion LLM (LLaDA-8B). Using attention analysis, linear probing, sparse autoencoders, and causal tracing, we find [FINDING]. Our results suggest [IMPLICATION for spatial reasoning / AGI / interpretability methods].*

**Proposed sections:**
1. Introduction & Motivation
2. Background: ARC-AGI, MI methods, target models
3. Experimental Setup (datasets, tokenization, MI toolkit)
4. Phase 1 results: Autoregressive LLM
5. Phase 2 results: Tiny Recursive Model
6. Phase 3 results: Masked Diffusion
7. Cross-architecture comparison & discussion
8. Limitations & future work
9. Conclusion

---

### 4.2 Figure Plan

| Figure | Content | Source Phase |
|---|---|---|
| Fig 1 | Attention specialization heatmap (Qwen) | Phase 1 |
| Fig 2 | Layer-wise probing curves for 5 spatial concepts (Qwen) | Phase 1 |
| Fig 3 | SAE features — example monosemantic features | Phase 1 |
| Fig 4 | TTT weight change heatmap per layer | Phase 1 |
| Fig 5 | TRM convergence curves per task type | Phase 2 |
| Fig 6 | Latent state decoding — "when does TRM know what?" | Phase 2 |
| Fig 7 | Causal tracing — task ID mechanism | Phase 2 |
| Fig 8 | Qwen vs TRM RSA / CKA similarity | Phase 2 |
| Fig 9 | LLaDA diffusion unmasking order | Phase 3 |
| Fig 10 | Cross-architecture attention specialization comparison | Phases 1-3 |

---

### 4.3 Target Venues

| Venue | Deadline (approx) | Fit |
|---|---|---|
| arXiv (preprint) | Anytime | Immediate publication, good for priority |
| ICLR 2027 Workshop on MI | ~Oct 2026 | High fit |
| NeurIPS 2026 Workshops | ~Sep 2026 | High fit |
| BlackboxNLP at EMNLP | ~May 2026 | Medium fit |
| COLM 2026 | ~Mar 2026 | High fit — specifically about LLMs |

**Recommendation:** Post to arXiv immediately after Phase 2 is complete (even before Phase 3) to establish priority, then submit to a workshop.

---

### 4.4 Writing Schedule

| Day | Task |
|---|---|
| 26 | Write Sections 1–3 (Introduction, Background, Setup) |
| 27 | Write Sections 4–5 (Phase 1 & 2 results) |
| 28 | Write Sections 6–7 (Phase 3 & cross-architecture comparison) |
| 29 | Write Sections 8–9 (Limitations, Conclusion) + polish all figures |
| 30 | Final proofread, format for arXiv, submit |

---

## Full Project Timeline

```
Day  1  : Phase 1 setup — install tools, load Qwen 0.5B, tokenize ARC
Day  2-3: Phase 1A — Attention pattern analysis
Day  4  : Phase 1B — Residual stream probing
Day  5-6: Phase 1C — SAE training (collect activations on Colab, train on TPU)
Day  7  : Phase 1D — TTT before/after comparison + Phase 1 write-up
──────────────────────────────────────────
Day  8  : Phase 2 setup — clone TRM repo, verify scores, set up hooks
Day  9-10: Phase 2A — Iteration-by-iteration answer tracking
Day  11-12: Phase 2B — Latent state decoding probes
Day  13-14: Phase 2C — Task ID causal tracing
Day  15-16: Phase 2D — SAE training on TRM activations (on TPU)
Day  17-18: Phase 2E — Qwen vs TRM comparison + Phase 2 write-up
──────────────────────────────────────────
Day  19  : Phase 3 setup — load LLaDA, adapt hooks
Day  20-21: Phase 3A — Diffusion step unmasking analysis
Day  22-23: Phase 3B — Bidirectional vs. causal attention comparison
Day  24-25: Phase 3C — 2D RoPE vs. 1D RoPE probing
──────────────────────────────────────────
Day  26  : Write Sections 1–3
Day  27  : Write Sections 4–5
Day  28  : Write Sections 6–7
Day  29  : Write Sections 8–9 + figure polish
Day  30  : Final review + arXiv submission
```

---

## ARC Dataset Preparation

All phases share the same dataset pipeline. Set this up on Day 1 and reuse throughout.

```python
import json
import torch

def load_arc_dataset(split="training"):
    """Load ARC-AGI tasks from local files or HuggingFace"""
    # Option 1: from arcprize repo
    # git clone https://github.com/fchollet/ARC-AGI
    tasks = []
    for filepath in Path(f"ARC-AGI/data/{split}").glob("*.json"):
        with open(filepath) as f:
            task = json.load(f)
        tasks.append(task)
    return tasks

def grid_to_tokens(grid):
    """Convert 2D grid to flat token string"""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)

def task_to_prompt(task, n_examples=None):
    """Format ARC task as a text prompt"""
    pairs = task["train"][:n_examples]
    prompt = ""
    for pair in pairs:
        prompt += f"Input:\n{grid_to_tokens(pair['input'])}\nOutput:\n{grid_to_tokens(pair['output'])}\n\n"
    prompt += f"Input:\n{grid_to_tokens(task['test'][0]['input'])}\nOutput:\n"
    return prompt

def create_probing_labels(tasks):
    """Create ground-truth labels for spatial concept probes"""
    labels = {"same_color_as_neighbor": [], "top_row": [], "object_boundary": []}
    for task in tasks:
        grid = task["test"][0]["input"]
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                labels["top_row"].append(1 if i == 0 else 0)
                has_diff_neighbor = any(
                    grid[i+di][j+dj] != cell
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]
                    if 0 <= i+di < len(grid) and 0 <= j+dj < len(row)
                )
                labels["object_boundary"].append(1 if has_diff_neighbor else 0)
    return labels
```

---

## Tools & Libraries Reference

| Tool | Purpose | Install |
|---|---|---|
| `transformer-lens` | Hooks, caching, attention visualization for GPT-style models | `pip install transformer-lens` |
| `nnsight` | Universal model hooking (works on any HuggingFace model incl. LLaDA, TRM) | `pip install nnsight` |
| `sae-lens` | Training Sparse Autoencoders | `pip install sae-lens` |
| `circuitsvis` | Interactive attention pattern visualization | `pip install circuitsvis` |
| `einops` | Tensor reshaping for interpretability | `pip install einops` |
| `sklearn` | Probing classifiers (LogisticRegression, etc.) | `pip install scikit-learn` |
| `plotly` | Interactive figures | `pip install plotly` |
| `wandb` | Experiment tracking | `pip install wandb` |
| `jax` + `flax` | TPU-native computation for SAE training | `pip install jax[tpu] flax` |

---

## Key Papers to Read Before Starting

1. **TRM paper:** "Less is More: Recursive Reasoning with Tiny Networks" — arXiv:2510.04871
2. **ARC Prize 2025 Technical Report** — arXiv:2601.10904
3. **LLaDA paper:** "Large Language Diffusion Models" — arXiv:2502.09992
4. **SAE foundational paper:** "Towards Monosemanticity" — Anthropic (transformer-circuits.pub)
5. **Probing methodology:** "What does BERT look at?" — Clark et al., 2019
6. **Causal tracing:** "Locating and Editing Factual Associations in GPT" — Meng et al., 2022 (ROME)
7. **ARC + MI prior work:** "Mini-ARC" — pfletcherhill.com/mini-arc.pdf

---

*This document is a living project guide. Update findings in each phase before proceeding to the next.*
