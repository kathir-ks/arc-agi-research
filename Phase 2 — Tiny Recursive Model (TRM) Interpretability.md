# Phase 2 — Tiny Recursive Model (TRM) Interpretability
## ARC-AGI Mechanistic Interpretability Study
**Researcher:** Kathir (kathirksw@gmail.com)
**Duration:** Days 1–11 (parallel execution — starts Day 1 alongside Phases 1, 3, 4)
**Compute:** Colab Session 2 / CPU (Experiments A–C, E); GCP TPU Days 9–11 (Experiment D — SAE training)
**Target:** arXiv publication in mechanistic interpretability

---

## ⚡ Parallel Execution Timeline (Phase 2 Only)

> **All 4 phases run simultaneously.** See `Master Parallel Execution Plan.md` for the full coordination schedule, compute allocation (TPU staggering), and cross-phase sync points. This document covers only Phase 2 work.

Phase 2 is the **critical path** — it has the most experiments (5) and the longest SAE training job (~23M activation vectors across all iterations). Everything else finishes by Day 8; Phase 2 wraps Day 11.

| Day | Phase 2 Activity | Colab Session | TPU? |
|---|---|---|---|
| **Day 1** | Clone TRM repo, install nnsight, load checkpoint, verify ARC score (~45%), map hook points | Colab 2 | No |
| **Day 2** | Exp A: Hook answer state at all iterations, compute pixel accuracy curves per task category | Colab 2 | No |
| **Day 3** | Exp A: Visualize convergence curves, detect phase transitions, finalize figure | Colab 2 | No |
| **Day 4** | Exp B: Define probe labels, collect latent states at all iterations for 200 tasks | Colab 2 | No |
| **Day 5** | Exp B: Train probes at each iteration, generate "when does TRM know what?" heatmap | Colab 2 | No |
| **Day 6** | Exp C: Clean/corrupted forward passes, patch-back causal tracing for task ID mechanism | Colab 2 | No |
| **Day 7** | Exp C: Finalize causal trace plots; **begin Exp D**: collect TRM activation dataset, upload to GCS | Colab 2 | Upload |
| **Days 8–9** | Exp D: TPU SAE training on TRM activations; **Exp E** in parallel: Qwen vs TRM CKA on Colab | Colab 2 (Exp E) | **YES — TPU** |
| **Day 10** | Exp D: SAE feature interpretation, task-type clustering; Exp E: feature matching | Colab 2 | No |
| **Day 11** | Finalize all figures, write Phase 2 summary, cross-phase SAE merge with Phases 1 + 3 | Colab 2 | No |

**Phase 2 completes Day 11.** Results handed to Phase 4 for §5 (Phase 2 Results) writing.

**TPU scheduling:** Phase 1 uses TPU Days 5–6; Phase 3 uses TPU Days 7–8. Phase 2 gets TPU Days 9–11 — no overlap. Upload `phase2_trm_acts.h5` to GCS at end of Day 7.

**Shared infrastructure note:** Use the same 200 shared ARC task IDs as Phases 1 and 3 (first 200 alphabetically from ARC-AGI-1 training split). This is mandatory for Experiment E's CKA comparison to be valid. See `Master Parallel Execution Plan.md` for exact IDs and shared pipeline code.

---

## 1. Phase Overview and Objectives

Phase 2 pivots from the Qwen baseline established in Phase 1 to a forensic interpretability study of the **Tiny Recursive Model (TRM)** from Samsung AI (SamsungSAILMontreal/TinyRecursiveModels). TRM achieves approximately 45% on ARC-AGI-1 using only 7 million parameters — a striking efficiency that demands mechanistic explanation. This phase is virgin interpretability territory: no prior mechanistic interpretability (MI) work exists on TRM, making every finding potentially publishable as a standalone contribution.

### Primary Research Questions

1. **Convergence dynamics:** How does TRM's answer state evolve across 32–64 recursive iterations? Do convergence curves show phase transitions, and do different task types exhibit structurally different convergence trajectories?
2. **Latent state semantics:** What information is encoded in the latent state at each iteration? Does TRM first represent spatial structure (e.g., object boundaries) before committing to a transformation rule?
3. **Task conditioning mechanism:** How does the `task_id_token` mechanistically control behavior? Is it a soft key that routes processing, or does it inject representational content directly into the residual stream?
4. **Feature geometry:** What sparse, interpretable features does TRM's single transformer block compute? Are these features re-used across iterations, or does the model specialize across depth?
5. **Cross-architecture universality:** How similar are TRM's learned representations to Qwen's, despite radically different architectures? Do universal ARC-solving features exist?

### Why TRM Is Uniquely Valuable for Interpretability

Standard transformer interpretability faces a confound: layer N's weights differ from layer N+1's, making it unclear whether a feature is specific to that layer or general to the algorithm. TRM eliminates this confound entirely. Because the **same single transformer block is applied N times**, any feature that appears consistently across iterations reflects a genuinely stable algorithmic component — not an artifact of weight differences. This is the interpretability analog of a controlled experiment. The recursive architecture makes TRM a near-ideal substrate for studying how a fixed computational primitive iteratively refines a representation toward a solution.

Additionally, TRM is **supervised at every iteration** (not just the final output), which means the model is explicitly trained to maintain interpretable intermediate states. This training signal biases the model toward solutions with legible convergence trajectories — a significant practical advantage for MI research.

---

## 2. Academic Literature and Adaptation Strategy

Each experiment in Phase 2 adapts a methodological contribution from the MI literature to TRM's unique architecture.

### Exp A — Iteration-by-Iteration Answer Tracking
**Adapts:** "Progress Measures for Grokking via Mechanistic Interpretability" (Neel et al., arXiv:2301.05217)

Neel et al. demonstrate that training loss curves can hide phase transitions visible only in mechanistic measures (e.g., circuit formation precedes generalization). We adapt this insight to the *inference-time* iteration axis: TRM's 32–64 recursive steps are an analog of training steps, with the final answer state playing the role of generalization. We track per-task-type convergence curves to detect inference-time phase transitions — moments where answer quality jumps discontinuously, suggesting a discrete algorithmic step has fired.

### Exp B — Latent State Decoding
**Adapts:** "Language Models Represent Space and Time" (Gurnee & Tegmark, arXiv:2310.02207) and "Linear Representation Hypothesis" (Park et al., arXiv:2311.03658)

Gurnee & Tegmark show that linear probes trained on residual stream activations recover factual world-knowledge features. Park et al. provide the theoretical grounding: if the model uses linear representations, probes are a faithful measurement tool. We apply this methodology to TRM's latent state, probing for spatial concepts (object identity, bounding box, color cluster) as a function of iteration depth.

### Exp C — Task ID Ablation / Causal Tracing
**Adapts:** "ROME: Locating and Editing Factual Associations in GPT" (Meng et al., arXiv:2202.05262) and "In-context Learning and Induction Heads" (Olsson et al., arXiv:2209.11895), with path patching from "Interpretability in the Wild: IOI Circuit" (Wang et al., arXiv:2211.00593)

Meng et al.'s causal tracing framework identifies where factual knowledge is stored by patching intermediate activations and measuring downstream effect. We apply this to the task_id_token: by corrupting the task ID embedding and restoring it at specific iterations, we localize *when* and *where* the task conditioning signal propagates into the answer state.

### Exp D — Sparse Autoencoder on TRM's Single Block
**Adapts:** "Towards Monosemanticity" (Anthropic, 2023) with the TopK SAE architecture from "Scaling and Evaluating Sparse Autoencoders" (Gao et al., arXiv:2406.04093), informed by "Sparse Autoencoders Reveal Universal Feature Spaces" (Tamkin et al., arXiv:2410.06981)

The critical advantage of applying SAEs to TRM vs. standard transformers: TRM has a **single MLP/attention block** whose activations are observed across all iterations. Each iteration provides an independent data point from the same feature basis. This means the SAE trains on a dataset that is N× richer in structure than a single-layer SAE on a standard transformer — with the regularization benefit that features must be consistent across iteration depths.

### Exp E — Qwen vs TRM Comparison
**Adapts:** "Similarity of Neural Network Representations Revisited" (Kornblith et al., arXiv:1905.00414) and "Sparse Autoencoders Reveal Universal Feature Spaces" (Tamkin et al., arXiv:2410.06981)

Kornblith et al.'s Centered Kernel Alignment (CKA) provides a geometry-invariant similarity metric for comparing representations across architectures. Tamkin et al. extend SAE dictionaries to find cross-model universal features. We combine both: first computing CKA between Qwen layer activations and TRM latent states on shared ARC tasks, then matching SAE features between architectures to identify any universal ARC-solving primitives.

---

## 3. Environment Setup (Day 8)

Before experiments begin, establish a clean, reproducible environment.

```bash
# Clone TRM
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels
cd TinyRecursiveModels
pip install -e .

# Core interpretability stack
pip install transformer_lens nnsight einops fancy_einsum
pip install scikit-learn matplotlib seaborn pandas

# SAE training (Experiment D)
pip install torch==2.2.0  # ensure TPU-compatible version on GCP
# For GCP TPU: pip install torch_xla

# ARC task loader
pip install arckit  # loads ARC-AGI-1 tasks as numpy arrays
```

**nnsight vs. TransformerLens note:** TransformerLens requires a pre-defined model architecture and will not work with TRM out of the box. Use `nnsight` throughout Phase 2, which wraps arbitrary PyTorch models and provides activation hooks via a clean context manager API. See "A Practical Review of Mechanistic Interpretability" (Conmy & Heimersheim, arXiv:2407.02646) for guidance on adapting nnsight to custom architectures.

```python
# Minimal TRM + nnsight setup
import torch
from nnsight import NNsight
from trm import TinyRecursiveModel  # adjust import to actual module path

model = TinyRecursiveModel.from_pretrained("SamsungSAILMontreal/TinyRecursiveModels")
model.eval()
nn_model = NNsight(model)

# Verify hook points are accessible
with nn_model.trace(dummy_input):
    latent = nn_model.transformer_block.output.save()  # adapt to actual attribute name
print(latent.shape)  # should be [batch, seq_len, d_model]
```

---

## 4. Experiment A — Iteration-by-Iteration Answer Tracking (Days 9–10)

### Objective
Produce convergence curves for TRM's answer state across all N iterations, stratified by ARC task type (spatial transformation, object counting, pattern completion, symmetry detection).

### Step-by-Step Process

**Step 1: Task categorization.** Manually annotate a 200-task subset of ARC-AGI-1 into 4–6 task type categories. Use arckit to load tasks.

```python
import arckit
import numpy as np

tasks = arckit.load_arc1()  # returns list of (train_pairs, test_pairs)
# Manually label or use heuristic: count distinct colors, check for symmetry axes, etc.

def categorize_task(task):
    grids = [pair[0] for pair in task.train]
    n_colors = len(set(np.unique(g) for g in grids))
    # Simple heuristic — replace with manual labels for final paper
    if n_colors <= 3:
        return "pattern_completion"
    elif check_symmetry(grids):
        return "symmetry"
    else:
        return "spatial_transform"
```

**Step 2: Hook answer state at each iteration.** TRM exposes an answer state that is updated at every recursive step. Use nnsight to capture it at all N steps.

```python
import torch
from collections import defaultdict

def get_iteration_answers(model, nn_model, task_input, n_iterations=32):
    """Returns answer state at each iteration as list of tensors."""
    answer_states = []

    with nn_model.trace(task_input):
        for i in range(n_iterations):
            # Hook the answer state update inside the loop
            # Exact hook point depends on TRM's forward() implementation
            # In TRM, look for self.answer_state or equivalent attribute
            ans = nn_model.answer_state.output[i].save()
            answer_states.append(ans)

    return [a.value for a in answer_states]
```

**Step 3: Compute correctness proxy at each iteration.** Convert the answer state to a predicted grid and compare to ground truth using pixel-wise accuracy.

```python
def answer_state_to_grid(answer_state, output_size):
    """Decode answer state logits to color predictions."""
    logits = answer_state.reshape(output_size[0], output_size[1], -1)
    return logits.argmax(dim=-1).cpu().numpy()

def pixel_accuracy(pred_grid, true_grid):
    return (pred_grid == true_grid).mean()

# Run across task subset
results = defaultdict(list)
for task_id, (task, category) in enumerate(labeled_tasks):
    inp = prepare_input(task)  # flatten + embed
    answers = get_iteration_answers(model, nn_model, inp)
    accs = [pixel_accuracy(answer_state_to_grid(a, task.output_shape), task.test_output)
            for a in answers]
    results[category].append(accs)
```

**Step 4: Visualize convergence curves and detect phase transitions.**

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, len(results), figsize=(16, 4))
for ax, (category, curves) in zip(axes, results.items()):
    curves = np.array(curves)  # shape [n_tasks, n_iterations]
    mean_curve = curves.mean(0)
    std_curve = curves.std(0)
    iters = np.arange(len(mean_curve))

    ax.plot(iters, mean_curve, label=f"{category} (n={len(curves)})")
    ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Pixel Accuracy")
    ax.set_title(category)

    # Mark phase transitions: look for largest second derivative
    second_deriv = np.diff(mean_curve, n=2)
    transition_iter = np.argmax(np.abs(second_deriv)) + 1
    ax.axvline(x=transition_iter, color='red', linestyle='--', label=f"Transition @ iter {transition_iter}")
    ax.legend()

plt.tight_layout()
plt.savefig("exp_a_convergence_curves.pdf", dpi=300)
```

### Expected Outputs
- `exp_a_convergence_curves.pdf` — one panel per task type, showing mean ± std convergence
- `exp_a_phase_transitions.csv` — per-task-type transition iteration
- **Key finding hypothesis:** Spatial transformation tasks converge rapidly (iterations 5–10); symmetry detection tasks show late-stage sharp transitions (iterations 20–28); pattern completion shows smooth monotonic improvement

### Pitfall: TRM's forward loop
TRM's recursion is implemented as a Python for loop, not a native recurrent cell. nnsight hooks must be placed inside the loop body, which requires inspecting the exact source code structure and matching the hook to the iteration counter. Do not assume PyTorch's `nn.LSTM`-style access — read TRM's `forward()` method first.

---

## 5. Experiment B — Latent State Decoding (Days 11–12)

### Objective
Train linear probes on TRM's latent state at each iteration to determine when spatial, semantic, and structural information becomes encoded. Produce a "when does TRM know what?" heatmap.

### Step-by-Step Process

**Step 1: Define probe targets.** For each ARC task, extract ground-truth labels for: (a) dominant output color, (b) number of distinct objects, (c) output grid aspect ratio class, (d) whether a reflection symmetry exists.

```python
def extract_probe_labels(task):
    out = task.test_output
    return {
        "dominant_color": np.bincount(out.flatten()).argmax(),
        "n_objects": count_connected_components(out),
        "aspect_ratio": int(out.shape[0] > out.shape[1]),  # tall vs wide
        "has_symmetry": int(check_horizontal_symmetry(out) or check_vertical_symmetry(out))
    }
```

**Step 2: Collect latent state activations at every iteration.**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

def collect_latent_states(model, nn_model, tasks, n_iterations=32):
    """Returns dict: {iteration: [latent_vec per task]}"""
    all_latents = {i: [] for i in range(n_iterations)}

    for task in tasks:
        inp = prepare_input(task)
        with nn_model.trace(inp):
            for i in range(n_iterations):
                latent = nn_model.latent_state.output[i].save()
            saved = [(i, l) for i, l in enumerate(latent_saves)]

        for i, lat in saved:
            # Pool over sequence dim: mean pool the latent state tokens
            all_latents[i].append(lat.value.mean(dim=1).squeeze().cpu().numpy())

    return all_latents
```

**Step 3: Train probes at each iteration depth.**

```python
probe_results = {}  # {probe_name: {iteration: accuracy}}

for probe_name in ["dominant_color", "n_objects", "aspect_ratio", "has_symmetry"]:
    probe_results[probe_name] = {}
    y = np.array([labels[probe_name] for labels in all_labels])

    for iteration in range(n_iterations):
        X = np.array(all_latents[iteration])
        clf = LogisticRegression(max_iter=500, C=0.1)
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        probe_results[probe_name][iteration] = scores.mean()
```

**Step 4: Generate heatmap.**

```python
import seaborn as sns

probe_matrix = np.array([[probe_results[p][i] for i in range(n_iterations)]
                          for p in probe_results])

plt.figure(figsize=(14, 5))
sns.heatmap(probe_matrix,
            xticklabels=range(n_iterations),
            yticklabels=list(probe_results.keys()),
            cmap="viridis", vmin=0.5, vmax=1.0,
            annot=False)
plt.xlabel("Iteration")
plt.ylabel("Probe Target")
plt.title("Latent State Information Content Across Iterations")
plt.tight_layout()
plt.savefig("exp_b_latent_heatmap.pdf", dpi=300)
```

### Expected Outputs
- `exp_b_latent_heatmap.pdf` — probe accuracy by concept by iteration
- `exp_b_probe_weights.pkl` — saved probe coefficients for visualization
- **Key finding hypothesis:** Structural features (aspect ratio, dominant color) become linearly decodable early (iterations 2–8); compositional features (object count, symmetry) emerge later (iterations 15–25), consistent with a coarse-to-fine computation pattern

---

## 6. Experiment C — Task ID Ablation / Causal Tracing (Days 13–14)

### Objective
Localize the mechanistic role of the `task_id_token` using causal tracing (activation patching). Determine: (1) which iterations are critical for task conditioning; (2) whether task information flows primarily through the attention or MLP pathway.

### Step-by-Step Process

**Step 1: Establish clean vs. corrupted forward passes.**

```python
def run_clean(model, nn_model, task_input):
    """Run normally, save all intermediate activations."""
    saved = {}
    with nn_model.trace(task_input):
        for i in range(N_ITERATIONS):
            saved[f"attn_{i}"] = nn_model.transformer_block.attention.output[i].save()
            saved[f"mlp_{i}"]  = nn_model.transformer_block.mlp.output[i].save()
    return {k: v.value for k, v in saved.items()}

def run_corrupted(model, nn_model, task_input, corrupt_task_id=True):
    """Replace task_id_token embedding with noise."""
    corrupted = task_input.clone()
    if corrupt_task_id:
        corrupted[:, 0, :] = torch.randn_like(corrupted[:, 0, :])  # task ID is first token
    with nn_model.trace(corrupted):
        final_answer = nn_model.answer_state.output[-1].save()
    return final_answer.value
```

**Step 2: Patch-back individual iteration activations and measure recovery.**

```python
def causal_trace(model, nn_model, task_input, clean_cache, n_iterations=32):
    """
    For each iteration i, patch the clean activation back into the corrupted run
    and measure how much of the correct answer is restored.
    """
    results = {"attn": [], "mlp": []}
    base_score = pixel_accuracy(run_corrupted(...), true_answer)

    for pathway in ["attn", "mlp"]:
        for patch_iter in range(n_iterations):
            corrupted = task_input.clone()
            corrupted[:, 0, :] = torch.randn_like(corrupted[:, 0, :])

            with nn_model.trace(corrupted):
                # At iteration patch_iter, restore clean activation
                nn_model.transformer_block.__dict__[pathway].output[patch_iter][:] = \
                    clean_cache[f"{pathway}_{patch_iter}"]
                answer = nn_model.answer_state.output[-1].save()

            recovered_score = pixel_accuracy(answer.value, true_answer)
            effect = recovered_score - base_score
            results[pathway].append(effect)

    return results
```

**Step 3: Visualize causal importance.**

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

iters = np.arange(n_iterations)
ax1.bar(iters, results["attn"], color="steelblue", alpha=0.8, label="Attention")
ax1.set_ylabel("Recovery Effect")
ax1.set_title("Causal Tracing: Task ID Conditioning — Attention Pathway")

ax2.bar(iters, results["mlp"], color="tomato", alpha=0.8, label="MLP")
ax2.set_ylabel("Recovery Effect")
ax2.set_xlabel("Iteration")
ax2.set_title("Causal Tracing: Task ID Conditioning — MLP Pathway")

plt.tight_layout()
plt.savefig("exp_c_causal_trace.pdf", dpi=300)
```

### Expected Outputs
- `exp_c_causal_trace.pdf` — bar chart of causal importance by iteration and pathway
- `exp_c_task_ablation_table.csv` — per-task-type sensitivity to task ID corruption
- **Key finding hypothesis:** Early iterations (1–5) show strong attention-pathway importance (task ID is read via attention); later iterations show stronger MLP importance (task rule is instantiated via MLP computation)

### Pitfall: Task ID token position
TRM's input format is `[task_id_token | flattened_input_grid | flattened_answer_state | latent_state]`. The task_id_token is a learned embedding. Corruption must target exactly token position 0 in the embedded sequence, not the raw integer ID. Verify this by checking the embedding module's output shape before corrupting.

---

## 7. Experiment D — SAE on TRM's Single Block (Days 15–16)

### Objective
Train a sparse autoencoder (SAE) on the residual stream / MLP activations of TRM's single transformer block, collecting activations across all iterations as training data. Identify interpretable monosemantic features and determine whether features cluster by task type or iteration depth.

### Compute: GCP TPU recommended
SAE training requires ~50–100M activation vectors. On CPU/Colab, expect 4–6 hours for a small dictionary (16k features). On GCP TPU v2-8, expect ~45 minutes.

### Step-by-Step Process

**Step 1: Collect activations dataset.**

```python
import h5py, torch
from tqdm import tqdm

# Use nnsight to collect MLP post-activation at every iteration
activation_buffer = []

for task_batch in dataloader:  # iterate over ARC tasks
    with nn_model.trace(task_batch):
        for i in range(N_ITERATIONS):
            act = nn_model.transformer_block.mlp.output[i].save()
        acts_per_iter = [a.value for a in act_saves]

    for iter_acts in acts_per_iter:
        # iter_acts shape: [batch, seq_len, d_mlp]
        # Flatten to [batch * seq_len, d_mlp]
        flat = iter_acts.reshape(-1, iter_acts.shape[-1])
        activation_buffer.append(flat.cpu())

all_activations = torch.cat(activation_buffer, dim=0)  # [N, d_mlp]

# Save to disk
with h5py.File("trm_activations.h5", "w") as f:
    f.create_dataset("activations", data=all_activations.numpy())

print(f"Collected {len(all_activations):,} activation vectors")
```

**Step 2: Define TopK SAE architecture** (following Gao et al., arXiv:2406.04093).

```python
import torch.nn as nn

class TopKSAE(nn.Module):
    def __init__(self, d_in, d_dict, k=32):
        super().__init__()
        self.k = k
        self.d_dict = d_dict
        self.W_enc = nn.Linear(d_in, d_dict, bias=True)
        self.W_dec = nn.Linear(d_dict, d_in, bias=True)
        # Normalize decoder columns to unit norm
        self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.weight.data = nn.functional.normalize(
                self.W_dec.weight.data, dim=0)

    def forward(self, x):
        pre_acts = self.W_enc(x)                    # [batch, d_dict]
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals.clamp(min=0))
        recon = self.W_dec(acts)
        return recon, acts

    def loss(self, x, recon, acts):
        recon_loss = (x - recon).pow(2).mean()
        # TopK SAE: no explicit L1 penalty needed (sparsity is structural)
        return recon_loss
```

**Step 3: Train SAE.**

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(all_activations)
loader = DataLoader(dataset, batch_size=4096, shuffle=True)

sae = TopKSAE(d_in=d_mlp, d_dict=16384, k=32).cuda()
optimizer = torch.optim.Adam(sae.parameters(), lr=2e-4)

for epoch in range(10):
    total_loss = 0
    for (x,) in tqdm(loader):
        x = x.cuda()
        recon, acts = sae(x)
        loss = sae.loss(x, recon, acts)
        optimizer.zero_grad()
        loss.backward()
        # Re-normalize decoder after each step
        sae._normalize_decoder()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}, "
          f"L0={acts.count_nonzero(dim=-1).float().mean():.1f}")

torch.save(sae.state_dict(), "trm_sae_16k.pt")
```

**Step 4: Analyze features — task type and iteration clustering.**

```python
# For each SAE feature (column of W_dec), compute:
# 1. Which task types most activate it
# 2. Whether activation varies systematically across iterations

feature_task_affinities = torch.zeros(sae.d_dict, n_task_types)
feature_iter_profile   = torch.zeros(sae.d_dict, N_ITERATIONS)

# (Collect feature activations per task type and per iteration during eval pass)
# Then visualize via seaborn clustermap

import seaborn as sns
sns.clustermap(feature_task_affinities.numpy().T,
               cmap="YlOrRd", figsize=(20, 6),
               xticklabels=False,
               yticklabels=TASK_TYPE_NAMES)
plt.savefig("exp_d_feature_task_affinities.pdf", dpi=300)
```

### Expected Outputs
- `trm_sae_16k.pt` — trained SAE weights
- `exp_d_feature_task_affinities.pdf` — feature × task type heatmap
- `exp_d_feature_iter_profiles.pdf` — feature × iteration heatmap
- `exp_d_top_features_manual.md` — manual annotation of top 20 interpretable features with activation examples
- **Key finding hypothesis:** ~5–10% of SAE features will be task-type specific; ~20–30% will show systematic iteration profiles (e.g., active early, then declining); a small number will be "planning" features active in middle iterations

---

## 8. Experiment E — Qwen vs TRM Comparison (Days 17–18)

### Objective
Quantify representational similarity between Qwen (from Phase 1) and TRM on shared ARC tasks using Centered Kernel Alignment (CKA), and match SAE dictionaries to find any universal ARC-solving features.

### Step-by-Step Process

**Step 1: Collect Qwen and TRM activations on the same 200-task ARC subset.**

```python
# Qwen activations — use Phase 1 hooks
qwen_acts = {layer: [] for layer in QWEN_LAYERS_OF_INTEREST}
trm_acts  = {iter: [] for iter in range(N_ITERATIONS)}

for task in shared_tasks:
    # ... collect as before
    pass

# Stack into matrices: shape [n_tasks, d_model]
X_qwen = {l: np.stack(qwen_acts[l]) for l in QWEN_LAYERS_OF_INTEREST}
X_trm  = {i: np.stack(trm_acts[i])  for i in range(N_ITERATIONS)}
```

**Step 2: Compute CKA matrix.**

```python
def linear_cka(X, Y):
    """Centered Kernel Alignment — Kornblith et al. 2019 (arXiv:1905.00414)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    numerator   = np.linalg.norm(XtY, "fro") ** 2
    denominator = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")
    return numerator / denominator

# CKA matrix: rows = Qwen layers, cols = TRM iterations
cka_matrix = np.zeros((len(QWEN_LAYERS_OF_INTEREST), N_ITERATIONS))
for i, ql in enumerate(QWEN_LAYERS_OF_INTEREST):
    for j in range(N_ITERATIONS):
        cka_matrix[i, j] = linear_cka(X_qwen[ql], X_trm[j])

# Visualize
plt.figure(figsize=(14, 5))
sns.heatmap(cka_matrix,
            xticklabels=[f"iter_{i}" for i in range(N_ITERATIONS)],
            yticklabels=[f"qwen_l{l}" for l in QWEN_LAYERS_OF_INTEREST],
            cmap="rocket", vmin=0, vmax=1)
plt.title("CKA: Qwen Layers vs TRM Iterations")
plt.tight_layout()
plt.savefig("exp_e_cka_heatmap.pdf", dpi=300)
```

**Step 3: SAE feature matching.** Load TRM SAE from Exp D and Qwen SAE from Phase 1 (if available). Compute cosine similarity between decoder dictionaries.

```python
# Align decoder columns (features) between architectures
W_dec_trm  = sae_trm.W_dec.weight.detach().cpu()   # [d_in_trm, d_dict]
W_dec_qwen = sae_qwen.W_dec.weight.detach().cpu()  # [d_in_qwen, d_dict]

# Project to shared space via PCA if d_in differs
from sklearn.decomposition import PCA

if W_dec_trm.shape[0] != W_dec_qwen.shape[0]:
    pca = PCA(n_components=min(W_dec_trm.shape[0], W_dec_qwen.shape[0]))
    W_trm_proj  = pca.fit_transform(W_dec_trm.numpy().T)
    W_qwen_proj = pca.fit_transform(W_dec_qwen.numpy().T)
else:
    W_trm_proj  = W_dec_trm.numpy().T
    W_qwen_proj = W_dec_qwen.numpy().T

# Cosine similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(W_trm_proj, W_qwen_proj)

# Top matches
top_matches = np.argsort(sim_matrix, axis=1)[:, -5:]
print("Top TRM→Qwen feature matches (cosine sim > 0.5):")
for trm_feat in range(sae_trm.d_dict):
    best_qwen = top_matches[trm_feat, -1]
    if sim_matrix[trm_feat, best_qwen] > 0.5:
        print(f"  TRM feat {trm_feat} ↔ Qwen feat {best_qwen}: "
              f"sim={sim_matrix[trm_feat, best_qwen]:.3f}")
```

### Expected Outputs
- `exp_e_cka_heatmap.pdf` — CKA similarity matrix
- `exp_e_universal_features.csv` — list of high-similarity cross-architecture feature pairs
- `exp_e_qwen_trm_comparison_table.md` — qualitative comparison of what matched features compute
- **Key finding hypothesis:** CKA will show moderate similarity (0.3–0.5) between early Qwen layers and mid-range TRM iterations, suggesting a shared representational strategy for spatial encoding that is architecture-agnostic. A small set of universal features (est. 50–150 out of 16k) will match across architectures.

---

## 9. Common Pitfalls Specific to TRM

### 1. nnsight Hook Placement Inside Python Loops
TRM's recursion is a Python `for` loop, not a compiled RNN cell. nnsight must be configured to intercept each iteration's output. Use a list-based save pattern and verify hook indices match actual loop counters.

### 2. Answer State vs. Latent State Confusion
TRM maintains two separate state tensors — **answer state** (direct output prediction, visible/supervised) and **latent state** (internal computation, opaque). These are concatenated in the input sequence but updated by different head groups. Always verify which state your hook is attached to by checking tensor shapes against TRM's documented architecture.

### 3. Random Initialization Baseline
Both states start as random noise. Any analysis at iteration 0 will reflect the random seed, not meaningful computation. Always start analysis from iteration 1 and report iteration 0 accuracy as the noise baseline.

### 4. Per-Iteration Supervision Artifact
Because TRM is supervised at every iteration, the answer state at iteration 5 may be "trained to look good" even if the internal representation at iteration 5 is not fully formed. Latent state probes (Exp B) are more reliable than answer state accuracy (Exp A) for understanding internal computation timing.

### 5. SAE Training Data Imbalance
Some ARC tasks repeat similar patterns (e.g., simple color swaps) far more than others. Without reweighting, the SAE will over-represent common patterns. Use weighted sampling or task-stratified batching during SAE training.

### 6. CKA Requires Matched Samples
For Exp E's CKA computation, both models must process exactly the same input examples in the same order. ARC inputs are tokenized differently by each model — use the raw numpy grids as the common substrate and apply each model's tokenizer independently.

---

## 10. Success Criteria

| Experiment | Minimum Success | Strong Success |
|------------|----------------|----------------|
| A | Convergence curves differ significantly by task type (ANOVA p < 0.05) | Phase transitions identified and characterized |
| B | At least 2 probe targets decode with >70% accuracy before final iteration | "Coarse-to-fine" ordering confirmed statistically |
| C | Causal tracing identifies >50% of task ID effect concentrated in ≤5 iterations | Attention vs. MLP dissociation confirmed |
| D | SAE trains to <10% reconstruction loss; >50 interpretable features found | Feature clustering by task type shows statistically significant structure |
| E | CKA matrix shows non-trivial block structure | At least 50 universal cross-architecture features identified with consistent interpretations |

---

## 11. Deliverables Checklist

### Figures (for paper)
- [ ] `exp_a_convergence_curves.pdf`
- [ ] `exp_b_latent_heatmap.pdf`
- [ ] `exp_c_causal_trace.pdf`
- [ ] `exp_d_feature_task_affinities.pdf`
- [ ] `exp_d_feature_iter_profiles.pdf`
- [ ] `exp_e_cka_heatmap.pdf`

### Data / Model Artifacts
- [ ] `trm_activations.h5` — full activation dataset (upload to HuggingFace Hub)
- [ ] `trm_sae_16k.pt` — trained SAE weights (release under MIT)
- [ ] `probe_weights.pkl` — all trained linear probes
- [ ] `causal_trace_results.pkl`
- [ ] `exp_e_universal_features.csv`

### Code
- [ ] `phase2_exp_a.py` — self-contained convergence tracking script
- [ ] `phase2_exp_b.py` — latent state probing script
- [ ] `phase2_exp_c.py` — causal tracing script
- [ ] `phase2_exp_d_collect.py` — activation collection for SAE
- [ ] `phase2_exp_d_train.py` — SAE training script
- [ ] `phase2_exp_e.py` — CKA + feature matching script
- [ ] `requirements.txt` — pinned dependency versions

### Writing
- [ ] Methods section draft (~1,200 words) covering TRM architecture and all 5 experiments
- [ ] Results section draft (~1,000 words) with figure references
- [ ] Related work paragraph integrating all 13 cited papers

---

## 12. How Phase 2 Extends Phase 1 and Feeds Phase 3

### Continuity from Phase 1
Phase 1 established Qwen baselines: which layers matter, which attention heads are task-relevant, and what the residual stream geometry looks like on ARC tasks. Phase 2 uses these as comparison anchors. Specifically:
- Exp E's CKA analysis directly compares Phase 1's Qwen layer activations to TRM's iterations
- Phase 1's SAE (if trained) provides the Qwen side of Exp E's feature matching
- Phase 1's task categorization schema carries over to Phase 2's Exp A stratification

### Feeding Phase 3
Phase 3 is the cross-architecture comparison paper. Phase 2 contributes:
1. **TRM SAE dictionary** — the primary artifact for cross-architecture feature comparison in Phase 3
2. **Convergence dynamics characterization** — establishes TRM's "cognitive profile" to contrast with Qwen's layer-by-layer profile
3. **Universal features list** — Phase 3's central claim (if confirmed) that ARC-solving requires a small set of universal representational primitives, regardless of architecture
4. **Causal tracing methodology** — Phase 3 will run the same task ID ablation on Qwen to directly compare task conditioning mechanisms

The core narrative arc: Phase 1 shows *what* Qwen does, Phase 2 shows *what* TRM does, Phase 3 asks *what both do in common* and what that reveals about the nature of abstract reasoning.

---

## 13. Academic Framing for arXiv Submission

The key novelty claim for Phase 2 (whether published standalone or as part of the combined paper):

> "We present the first mechanistic interpretability analysis of a recursively applied transformer, demonstrating that TRM's single transformer block implements a coarse-to-fine spatial reasoning algorithm across its 32–64 iterations. Using linear probes, causal tracing, and sparse autoencoders, we characterize the functional role of TRM's latent state and task conditioning mechanism, and identify a set of universal features shared with a 7B-parameter chain-of-thought model despite a 1000× parameter count difference."

Cite "Open Problems in Mechanistic Interpretability" (Conmy et al., arXiv:2501.16496) to position this as advancing the frontier on interpretability of non-standard architectures — one of their explicitly listed open problems. Cite "A Practical Review of Mechanistic Interpretability" (Conmy & Heimersheim, arXiv:2407.02646) in the Methods section to justify nnsight over TransformerLens for custom architectures.

---

*Document prepared for Phase 2 execution, Days 8–18.*
*Next: Phase 3 — Cross-Architecture Comparison and arXiv Drafting.*
