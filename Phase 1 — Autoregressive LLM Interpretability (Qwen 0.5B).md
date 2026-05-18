# Phase 1 — Autoregressive LLM Interpretability (Qwen 2.5-0.5B)
## Research Execution Document

**Researcher:** Kathir
**Phase Duration:** Days 1–8 (parallel execution — starts Day 1 alongside Phases 2, 3, 4)
**Model:** Qwen 2.5-0.5B + LoRA merged (stored on Google Drive)
**Compute:** Google Colab Pro Session 1 (A100/T4) for TransformerLens; GCP TPU Days 5–6 for SAE training
**Goal:** Establish mechanistic baseline — understand what the fine-tuned LLM actually encodes when solving ARC-AGI tasks
**Target Publication:** arXiv (mechanistic interpretability)

---

## ⚡ Parallel Execution Timeline (Phase 1 Only)

> **All 4 phases run simultaneously.** See `Master Parallel Execution Plan.md` for the full coordination schedule, compute allocation, and cross-phase sync points. This document covers only Phase 1 work.

| Day | Phase 1 Activity | Colab Session | TPU? |
|---|---|---|---|
| **Day 1** | Install deps, load Qwen from Drive, verify tokenization, baseline activation collection | Colab 1 | No |
| **Day 2** | Exp A: Extract attention patterns, compute spatial specialization scores for all heads | Colab 1 | No |
| **Day 3** | Exp A: Causal ablations on top-scoring heads, path patching, finalize head heatmap | Colab 1 | No |
| **Day 4** | Exp B: Build probe dataset, train row/col/color/boundary probes at all layers | Colab 1 | No |
| **Day 5** | Exp C: Collect SAE activation dataset (500 tasks × 3 layers), upload to GCS | Colab 1 | Upload only |
| **Day 6** | Exp C: SAE training on GCP TPU; feature interpretation on Colab in parallel | Colab 1 (interpret) | **YES — TPU** |
| **Day 7** | Exp D: TTT on 10 tasks, before/after activation comparison, SAE feature delta | Colab 1 | No |
| **Day 8** | Finalize all figures, write Phase 1 summary notes, feed results to Phase 4 | Colab 1 | No |

**Phase 1 completes Day 8.** Results handed to Phase 4 for §4 (Phase 1 Results) writing.

**Shared infrastructure note:** Use the same 200 shared ARC task IDs as Phases 2 and 3 (first 200 alphabetically from ARC-AGI-1 training split). See `Master Parallel Execution Plan.md` for the exact task ID list and shared data pipeline code.

---

## 1. Phase Overview

Phase 1 is the interpretability foundation of the entire study. Before comparing architectures (Phases 2–4), we must understand what a standard autoregressive LLM — fine-tuned on ARC-AGI via LoRA — actually learns internally. This is not trivial: ARC-AGI tasks require spatial reasoning, object tracking, pattern abstraction, and analogical thinking, none of which autoregressive LLMs were explicitly designed for. The central question is: **does the model develop genuine spatial representations, or does it rely on shallow statistical pattern matching?**

This phase answers that question using three complementary mechanistic interpretability tools:
1. **Attention pattern analysis** — which heads attend to spatially meaningful positions
2. **Residual stream probing** — whether spatial concepts are linearly encoded
3. **Sparse Autoencoder (SAE) training** — decomposing residual stream activations into monosemantic features
4. **TTT before/after comparison** — whether test-time training changes internal representations

The findings from this phase become the "autoregressive baseline" against which the Transformer Reasoning Model (Phase 2), diffusion model (Phase 3), and hybrid architecture (Phase 4) are compared.

---

## 2. Academic Motivation and Literature Foundation

### 2.1 Why Mechanistic Interpretability for ARC-AGI?

Chollet's ARC-AGI benchmark (2019) is specifically designed to resist statistical pattern matching — each test puzzle requires novel reasoning from first principles. Yet fine-tuned LLMs achieve non-trivial performance. This gap between "shouldn't work" and "does work" makes ARC-AGI a natural testbed for MI. The **Open Problems in Mechanistic Interpretability** paper (arXiv:2501.16496) explicitly identifies spatial reasoning and compositional generalization as unsolved MI challenges, positioning this research at an active frontier.

### 2.2 Theoretical Grounding

The **Mathematical Framework for Transformer Circuits** (Anthropic, 2021) provides the algebraic foundation for this entire phase. It establishes that attention heads can be decomposed into independent QK (query-key, determining "where to attend") and OV (output-value, determining "what to copy") circuits. This decomposition lets us ask: are specific heads computing spatial relationships (e.g., "attend to the cell two rows up") independently of semantic content? We use this framework to interpret attention patterns in Experiment A.

The **Linear Representation Hypothesis** (arXiv:2311.03658) provides theoretical grounding for Experiment B. It proposes that high-level concepts are encoded as linear directions in the residual stream — a strong claim that, if true for spatial concepts in ARC, implies that the model's spatial reasoning is geometrically structured and therefore interpretable via probes.

### 2.3 Literature by Experiment

**Experiment A — Attention Pattern Analysis:**
- *In-context Learning and Induction Heads* (arXiv:2209.11895): Establishes that induction heads (attending to [prev token] → [current token] patterns) are fundamental to in-context learning. We hypothesize ARC-fine-tuned models develop analogous "spatial induction heads" that track grid positions rather than token sequences.
- *Interpretability in the Wild / IOI Circuit* (arXiv:2211.00593): Demonstrates path patching methodology for attributing model behavior to specific circuits. We adapt their attention head classification protocol to the ARC domain.
- *ROME / Causal Tracing* (NeurIPS 2022, arXiv:2202.05262): Provides activation patching methodology. We use this to establish causal attribution — if ablating an attention head disrupts spatial reasoning, that head is causally involved, not just correlated.
- *Practical Review of Mechanistic Interpretability* (arXiv:2407.02646): The primary TransformerLens methodology guide. All hook-based activation collection follows protocols described here.

**Experiment B — Residual Stream Probing:**
- *Language Models Represent Space and Time* (arXiv:2310.02207): Directly adapted methodology. This paper trains linear probes on residual stream activations to test whether spatial/temporal concepts are linearly encoded. We apply the same approach but for ARC-specific spatial concepts: (row, column) position, object color, object boundary presence, and transformation type.
- *Linear Representation Hypothesis* (arXiv:2311.03658): Theoretical motivation. If the hypothesis holds for ARC spatial features, probes should achieve high accuracy with low-dimensional linear classifiers.

**Experiment C — SAE Training:**
- *Towards Monosemanticity* (Anthropic, 2023): Foundational SAE paper. Establishes that dictionary learning with a sparse penalty decomposes superposed polysemantic neurons into monosemantic features. The core loss function `L = ||x - Wx_sparse||^2 + λ||x_sparse||_1` comes from this work.
- *Scaling and Evaluating SAEs* (OpenAI, arXiv:2406.04093): Introduces TopK SAEs, which fix k active features per forward pass rather than using L1 regularization. TopK avoids the "dead feature" problem and scales better. We implement TopK as our primary architecture.
- *JumpReLU SAEs* (Google DeepMind, arXiv:2407.14435): Alternative activation function with a learnable threshold jump at zero, reducing feature shrinkage compared to vanilla ReLU. Evaluated as a secondary baseline.
- *Gemma Scope* (arXiv:2408.05147): Provides the blueprint for comprehensive SAE coverage. Rather than training a single SAE on one layer, Gemma Scope trains SAEs across all layers and residual stream positions. We follow this multi-layer protocol to understand which layers encode ARC-relevant features.
- *SAE Survey* (arXiv:2503.05613): Critical pitfalls guide. We specifically guard against: (1) training SAEs on too few tokens (under-training), (2) evaluating only reconstruction loss without behavioral metrics, (3) ignoring feature geometry (dead features, duplicate features).

**Experiment D — TTT Before/After:**
- *TTT for ARC* (arXiv:2411.07279): Establishes that test-time training (TTT) dramatically improves ARC performance by allowing the model to update weights on the few-shot examples in each puzzle. We use their training protocol as the reference for what "TTT" means operationally.
- *Specialization after Generalization* (arXiv:2509.24510): Provides mechanistic analysis of what TTT actually does to model internals — it argues TTT creates puzzle-specific specialization in early layers while preserving general capabilities in later layers. We test whether this is visible in our SAE features and attention patterns.

---

## 3. Environment Setup — Day 1

### 3.1 Installation

Run in Google Colab Pro (GPU runtime, A100 preferred):

```bash
pip install transformer-lens>=2.0.0
pip install sae-lens>=4.0.0
pip install circuitsvis
pip install einops
pip install plotly
pip install scikit-learn  # for linear probes
pip install datasets      # for ARC data loading
pip install wandb         # experiment tracking
```

### 3.2 Load Model into TransformerLens

```python
from transformer_lens import HookedTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load the merged Qwen model from Google Drive
from google.colab import drive
drive.mount('/content/drive')

MODEL_PATH = '/content/drive/MyDrive/arc_models/qwen25_05b_arc_merged'

# Load as HuggingFace model first
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Step 2: Convert to TransformerLens HookedTransformer
# TransformerLens supports Qwen2 natively as of v2.x
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    hf_model=hf_model,          # pass your fine-tuned weights
    fold_ln=True,               # fold LayerNorm for cleaner analysis
    center_writing_weights=True, # zero-mean weight matrices
    center_unembed=True,
)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model loaded: {model.cfg.n_layers} layers, "
      f"{model.cfg.n_heads} heads, "
      f"d_model={model.cfg.d_model}")
```

> **Fallback if TransformerLens load fails:** Use `nnsight` with direct HuggingFace model:
> ```python
> from nnsight import LanguageModel
> model = LanguageModel(MODEL_PATH, device_map="auto")
> ```

### 3.3 ARC Data Preparation

```python
import json, re, os
import numpy as np

ARC_DATA_PATH = '/content/drive/MyDrive/arc_data/'  # adjust to your path

def load_arc_tasks(split='training'):
    tasks = {}
    folder = os.path.join(ARC_DATA_PATH, split)
    for fname in os.listdir(folder):
        with open(os.path.join(folder, fname)) as f:
            tasks[fname.replace('.json', '')] = json.load(f)
    return tasks

def grid_to_tokens(grid):
    """Convert ARC grid to space-separated string format."""
    return '\n'.join(' '.join(str(c) for c in row) for row in grid)

def format_arc_prompt(task, example_idx=0):
    """Format an ARC task as an LLM prompt."""
    train_examples = task['train']
    prompt = ""
    for ex in train_examples:
        prompt += f"Input:\n{grid_to_tokens(ex['input'])}\nOutput:\n{grid_to_tokens(ex['output'])}\n\n"
    test_input = task['test'][example_idx]['input']
    prompt += f"Input:\n{grid_to_tokens(test_input)}\nOutput:\n"
    return prompt

arc_tasks = load_arc_tasks('training')
print(f"Loaded {len(arc_tasks)} training tasks")

# Tokenize a sample task to check format
sample_task = list(arc_tasks.values())[0]
sample_prompt = format_arc_prompt(sample_task)
tokens = model.to_tokens(sample_prompt)
print(f"Sample prompt tokens: {tokens.shape}")
```

### 3.4 Activation Collection Baseline

```python
def collect_activations(prompts, layer_indices=None, hook_points=None):
    """
    Collect residual stream activations across specified layers.
    Returns dict: {hook_name: tensor of shape [n_prompts, seq_len, d_model]}
    """
    if layer_indices is None:
        layer_indices = list(range(model.cfg.n_layers))
    if hook_points is None:
        hook_points = [f'blocks.{i}.hook_resid_post' for i in layer_indices]

    all_activations = {hp: [] for hp in hook_points}

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_points)
        for hp in hook_points:
            all_activations[hp].append(cache[hp][0].cpu())  # [seq, d_model]

    return {hp: torch.stack(v) for hp, v in all_activations.items()}

# Test on a small batch
sample_prompts = [format_arc_prompt(t) for t in list(arc_tasks.values())[:5]]
activations = collect_activations(sample_prompts, layer_indices=[0, 8, 15, 23])
print("Activation shapes:", {k: v.shape for k, v in activations.items()})
```

---

## 4. Experiment A — Attention Pattern Analysis (Days 2–3)

**Goal:** Identify attention heads that are spatially specialized — heads that consistently attend to structured spatial positions in ARC grids (same row, same column, adjacent cells, symmetric cells).

**Theoretical basis:** Following the QK/OV decomposition from *Mathematical Framework for Transformer Circuits* (Anthropic, 2021), each attention head independently computes attention weights from queries and keys. If a head is spatially specialized, its QK circuit should produce high attention scores between tokens at spatially related grid positions, regardless of their color values.

### 4.1 Attention Weight Extraction

```python
import circuitsvis as cv
from transformer_lens import utils

def get_attention_patterns(prompt, layer=None):
    """Extract attention patterns for all heads across all layers."""
    tokens = model.to_tokens(prompt)
    token_strs = model.to_str_tokens(prompt)
    
    hook_names = [f'blocks.{i}.attn.hook_pattern' for i in range(model.cfg.n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)
    
    # Stack: [n_layers, n_heads, seq_len, seq_len]
    patterns = torch.stack([cache[h][0] for h in hook_names])
    return patterns, token_strs

# Visualize with circuitsvis
patterns, token_strs = get_attention_patterns(sample_prompts[0])

# Display attention for a specific layer and head
cv.attention.attention_patterns(
    tokens=token_strs,
    attention=patterns[8],  # layer 8, all heads
)
```

### 4.2 Spatial Specialization Scoring

For each attention head, compute a **spatial specialization score**: the degree to which attention concentrates on tokens at the same row, same column, or fixed spatial offset in the ARC grid.

```python
def build_grid_position_map(prompt, tokenizer):
    """
    For each token position, record its (row, col) in the ARC grid if applicable.
    Returns: dict mapping token_idx -> (grid_row, grid_col) or None
    """
    pos_map = {}
    lines = prompt.split('\n')
    token_idx = 0
    grid_row = -1
    in_grid = False
    
    for line in lines:
        if line.startswith('Input:') or line.startswith('Output:'):
            grid_row = 0
            in_grid = True
        elif in_grid and line.strip():
            cells = line.split()
            for col, cell in enumerate(cells):
                pos_map[token_idx] = (grid_row, col)
                token_idx += 1
            token_idx += 1  # newline token
            grid_row += 1
        else:
            in_grid = False
            token_idx += len(tokenizer.encode(line)) + 1
    
    return pos_map

def spatial_specialization_score(attn_pattern, pos_map):
    """
    Score = fraction of attention weight going to same-row or same-column tokens,
    averaged over all grid tokens.
    attn_pattern: [seq_len, seq_len] for a single head
    """
    grid_tokens = [t for t, pos in pos_map.items() if pos is not None]
    if len(grid_tokens) < 2:
        return 0.0
    
    scores = []
    for t in grid_tokens:
        row_t, col_t = pos_map[t]
        same_row_or_col = [
            s for s in grid_tokens
            if s != t and (pos_map[s][0] == row_t or pos_map[s][1] == col_t)
        ]
        if not same_row_or_col:
            continue
        attn_to_spatial = attn_pattern[t, same_row_or_col].sum().item()
        attn_total = attn_pattern[t, grid_tokens].sum().item()
        if attn_total > 0:
            scores.append(attn_to_spatial / attn_total)
    
    return np.mean(scores) if scores else 0.0

# Score all heads across all layers
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
scores = np.zeros((n_layers, n_heads))

for prompt in sample_prompts[:20]:
    tokens = model.to_tokens(prompt)
    pos_map = build_grid_position_map(prompt, tokenizer)
    hook_names = [f'blocks.{i}.attn.hook_pattern' for i in range(n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)
    
    for layer in range(n_layers):
        for head in range(n_heads):
            pat = cache[f'blocks.{layer}.attn.hook_pattern'][0, head].cpu().numpy()
            scores[layer, head] += spatial_specialization_score(pat, pos_map)

scores /= 20  # average over prompts
```

### 4.3 Causal Ablation (Activation Patching)

Following *ROME / Causal Tracing* (arXiv:2202.05262), confirm that top-scoring heads are causally necessary, not just correlated:

```python
def ablate_head_and_measure(model, prompt, layer, head, metric_fn):
    """
    Zero-ablate a single attention head and measure performance drop.
    Returns: (baseline_score, ablated_score, delta)
    """
    tokens = model.to_tokens(prompt)
    
    def hook_zero_head(value, hook):
        value[:, head, :, :] = 0.0  # zero out this head's output
        return value
    
    with torch.no_grad():
        # Baseline
        baseline_logits = model(tokens)
        baseline_score = metric_fn(baseline_logits, tokens)
        
        # Ablated
        ablated_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(f'blocks.{layer}.attn.hook_z', hook_zero_head)]
        )
        ablated_score = metric_fn(ablated_logits, tokens)
    
    return baseline_score, ablated_score, baseline_score - ablated_score

def next_token_accuracy(logits, tokens):
    """Measure accuracy of next-token prediction on output grid tokens."""
    pred = logits[0, :-1].argmax(-1)
    return (pred == tokens[0, 1:]).float().mean().item()

# Test top 5 heads
top_heads = np.unravel_index(scores.argsort(axis=None)[-10:], scores.shape)
for l, h in zip(*top_heads):
    base, abl, delta = ablate_head_and_measure(
        model, sample_prompts[0], int(l), int(h), next_token_accuracy
    )
    print(f"Layer {l:2d}, Head {h}: base={base:.3f}, ablated={abl:.3f}, delta={delta:.3f}")
```

### 4.4 Visualization and Output

```python
import plotly.graph_objects as go
import plotly.express as px

# Heatmap of spatial specialization scores
fig = px.imshow(
    scores,
    labels=dict(x="Head", y="Layer", color="Spatial Score"),
    title="Attention Head Spatial Specialization Scores (Qwen 2.5-0.5B, ARC-finetuned)",
    color_continuous_scale="Viridis"
)
fig.write_html('/content/drive/MyDrive/arc_results/phase1/exp_a_head_heatmap.html')
fig.show()
```

**Expected Output:** A heatmap revealing whether spatial specialization is distributed across many heads or concentrated in a small circuit. Based on *Induction Heads* (arXiv:2209.11895), we expect 2–5 heads with significantly above-average spatial scores, likely in middle-to-late layers.

---

## 5. Experiment B — Residual Stream Probing (Day 4)

**Goal:** Test whether spatial concepts are linearly encoded in the residual stream at each layer, following *Language Models Represent Space and Time* (arXiv:2310.02207).

**Concepts to probe:** (1) token row index in grid, (2) token column index in grid, (3) cell color value, (4) whether token is at an object boundary, (5) transformation type (rotation, reflection, recoloring, etc.).

### 5.1 Probe Dataset Construction

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

def build_probe_dataset(tasks, n_tasks=200):
    """
    For each ARC task, collect (activation, label) pairs for each grid token.
    Returns arrays for training linear probes.
    """
    # Labels we'll collect: row, col, color, is_boundary
    X_by_layer = {layer: [] for layer in range(model.cfg.n_layers)}
    Y_row, Y_col, Y_color, Y_boundary = [], [], [], []
    
    for task_id, task in list(tasks.items())[:n_tasks]:
        prompt = format_arc_prompt(task)
        tokens = model.to_tokens(prompt)
        pos_map = build_grid_position_map(prompt, tokenizer)
        
        hook_names = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)
        
        for token_idx, pos in pos_map.items():
            if pos is None:
                continue
            row, col = pos
            grid = task['test'][0]['input']  # target grid
            try:
                color = grid[row][col]
            except IndexError:
                continue
            
            # Collect activations at this token position from each layer
            for layer in range(model.cfg.n_layers):
                act = cache[f'blocks.{layer}.hook_resid_post'][0, token_idx].cpu().numpy()
                X_by_layer[layer].append(act)
            
            Y_row.append(row)
            Y_col.append(col)
            Y_color.append(color)
            # Simple boundary heuristic: token is at grid edge
            max_row = len(grid) - 1
            max_col = len(grid[0]) - 1 if grid else 0
            Y_boundary.append(int(row == 0 or row == max_row or col == 0 or col == max_col))
    
    return (
        {l: np.array(v) for l, v in X_by_layer.items()},
        np.array(Y_row), np.array(Y_col),
        np.array(Y_color), np.array(Y_boundary)
    )

X_layers, Y_row, Y_col, Y_color, Y_boundary = build_probe_dataset(arc_tasks, n_tasks=200)
print(f"Collected {len(Y_row)} token activation samples")
```

### 5.2 Train Linear Probes Per Layer

```python
def train_probes_per_layer(X_layers, Y, task_type='classification'):
    """Train linear probe at each layer and return accuracy/R² by layer."""
    results = {}
    X_keys = sorted(X_layers.keys())
    
    for layer in X_keys:
        X = X_layers[layer]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        if task_type == 'classification':
            probe = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='auto')
            probe.fit(X_train, y_train)
            score = accuracy_score(y_test, probe.predict(X_test))
        else:  # regression
            probe = Ridge(alpha=1.0)
            probe.fit(X_train, y_train)
            score = r2_score(y_test, probe.predict(X_test))
        
        results[layer] = score
        print(f"Layer {layer:2d}: {task_type} score = {score:.3f}")
    
    return results

print("=== Row Position Probe (regression) ===")
row_results = train_probes_per_layer(X_layers, Y_row, 'regression')

print("\n=== Column Position Probe (regression) ===")
col_results = train_probes_per_layer(X_layers, Y_col, 'regression')

print("\n=== Color Probe (classification) ===")
color_results = train_probes_per_layer(X_layers, Y_color, 'classification')

print("\n=== Boundary Probe (classification) ===")
boundary_results = train_probes_per_layer(X_layers, Y_boundary, 'classification')
```

### 5.3 Layer-by-Layer Probe Accuracy Visualization

```python
fig = go.Figure()
fig.add_trace(go.Scatter(y=list(row_results.values()), name='Row (R²)', mode='lines+markers'))
fig.add_trace(go.Scatter(y=list(col_results.values()), name='Col (R²)', mode='lines+markers'))
fig.add_trace(go.Scatter(y=list(color_results.values()), name='Color (Acc)', mode='lines+markers'))
fig.add_trace(go.Scatter(y=list(boundary_results.values()), name='Boundary (Acc)', mode='lines+markers'))
fig.update_layout(
    title='Linear Probe Accuracy by Layer — Qwen 2.5-0.5B ARC-finetuned',
    xaxis_title='Layer', yaxis_title='Score',
    yaxis_range=[0, 1]
)
fig.write_html('/content/drive/MyDrive/arc_results/phase1/exp_b_probe_curves.html')
fig.show()
```

**Expected Output:** Row and column position probes should peak in middle layers (following spatial representation emergence patterns seen in *Language Models Represent Space and Time*). Color probes may peak earlier (simpler feature). If row/col probe accuracy stays near chance (1/max_row), the model does not linearly encode spatial positions — a significant finding.

---

## 6. Experiment C — SAE Training (Days 5–6)

**Goal:** Decompose residual stream activations into sparse, interpretable features using a Sparse Autoencoder. Train on GCP TPU for efficiency.

**Architecture:** TopK SAE following *Scaling and Evaluating SAEs* (OpenAI, arXiv:2406.04093), with k=32 active features per forward pass, trained on residual stream activations collected from Colab and saved to Google Drive.

### 6.1 Collect Activation Dataset (Colab — GPU)

```python
# Collect and save a large activation dataset before switching to TPU

def collect_and_save_activations(tasks, output_path, target_layer=12, n_tasks=500):
    """
    Collect residual stream activations at target_layer for all grid tokens.
    Saves as numpy arrays for TPU training.
    """
    all_acts = []
    all_labels = []
    hook = f'blocks.{target_layer}.hook_resid_post'
    
    for task_id, task in list(tasks.items())[:n_tasks]:
        prompt = format_arc_prompt(task)
        tokens = model.to_tokens(prompt)
        pos_map = build_grid_position_map(prompt, tokenizer)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook])
        
        acts = cache[hook][0].cpu()  # [seq_len, d_model]
        for token_idx, pos in pos_map.items():
            if pos is not None:
                all_acts.append(acts[token_idx].numpy())
                all_labels.append({'task_id': task_id, 'pos': pos})
    
    acts_array = np.array(all_acts, dtype=np.float32)  # [N, d_model]
    np.save(output_path + f'_layer{target_layer}_acts.npy', acts_array)
    
    import pickle
    with open(output_path + f'_layer{target_layer}_labels.pkl', 'wb') as f:
        pickle.dump(all_labels, f)
    
    print(f"Saved {len(all_acts)} activations of shape {acts_array.shape}")

# Collect for key layers (layer 6, 12, 18 for a 24-layer model)
for layer_idx in [6, 12, 18]:
    collect_and_save_activations(
        arc_tasks,
        '/content/drive/MyDrive/arc_results/phase1/sae_data/acts',
        target_layer=layer_idx,
        n_tasks=500
    )
```

### 6.2 SAE Architecture — TopK (GCP TPU)

Switch to GCP TPU environment for this step. Install `jax` and `flax`, or use `sae-lens` with PyTorch on TPU via `torch_xla`.

```python
# On GCP TPU — install torch_xla
# pip install torch torch_xla

import torch
import torch.nn as nn
import numpy as np

class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder from 'Scaling and Evaluating SAEs' (OpenAI, 2406.04093).
    k active features per forward pass; no L1 penalty needed.
    """
    def __init__(self, d_model: int, n_features: int, k: int = 32):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.n_features = n_features
        
        # Encoder
        self.W_enc = nn.Linear(d_model, n_features, bias=True)
        # Decoder (tied norms, free directions)
        self.W_dec = nn.Linear(n_features, d_model, bias=True)
        
        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.weight.data = nn.functional.normalize(
                self.W_dec.weight.data, dim=0
            )
    
    def encode(self, x):
        pre_acts = self.W_enc(x)   # [batch, n_features]
        # TopK: keep only k largest activations
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return acts
    
    def decode(self, acts):
        return self.W_dec(acts)
    
    def forward(self, x):
        acts = self.encode(x)
        x_recon = self.decode(acts)
        return x_recon, acts
    
    def loss(self, x):
        x_recon, acts = self(x)
        recon_loss = (x - x_recon).pow(2).mean()
        # Optional: auxiliary loss to prevent dead features
        n_dead = (acts.abs().sum(0) == 0).float().mean()
        return recon_loss, n_dead

def train_sae(acts_path, d_model=896, n_features=4096, k=32, n_epochs=10, lr=2e-4):
    """Full SAE training loop."""
    acts = torch.from_numpy(np.load(acts_path)).float()
    dataset = torch.utils.data.TensorDataset(acts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    sae = TopKSAE(d_model=d_model, n_features=n_features, k=k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    import wandb
    wandb.init(project='arc_sae_phase1', config={'d_model': d_model, 'n_features': n_features, 'k': k})
    
    for epoch in range(n_epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon_loss, dead_frac = sae.loss(batch)
            recon_loss.backward()
            
            # Normalize decoder columns after each step (per Anthropic protocol)
            with torch.no_grad():
                sae.W_dec.weight.data = nn.functional.normalize(
                    sae.W_dec.weight.data, dim=0
                )
            
            optimizer.step()
            total_loss += recon_loss.item()
        
        avg_loss = total_loss / len(loader)
        wandb.log({'epoch': epoch, 'recon_loss': avg_loss, 'dead_feature_frac': dead_frac.item()})
        print(f"Epoch {epoch+1}/{n_epochs}: recon_loss={avg_loss:.4f}, dead_features={dead_frac:.1%}")
    
    return sae

# Train SAE on layer 12 activations
sae_layer12 = train_sae(
    acts_path='/path/to/gcs/acts_layer12_acts.npy',
    d_model=896,       # Qwen 2.5-0.5B d_model
    n_features=4096,   # 4x expansion ratio
    k=32
)
```

### 6.3 Feature Interpretation

```python
def interpret_sae_features(sae, acts, labels, top_n=20):
    """
    For each SAE feature, find the top activating examples and describe them.
    """
    with torch.no_grad():
        feature_acts = sae.encode(torch.from_numpy(acts))  # [N, n_features]
    
    feature_stats = {}
    for feat_idx in range(sae.n_features):
        feat_col = feature_acts[:, feat_idx]
        activation_density = (feat_col > 0).float().mean().item()
        top_examples = feat_col.topk(top_n).indices.tolist()
        
        top_positions = [labels[i]['pos'] for i in top_examples]
        top_tasks = [labels[i]['task_id'] for i in top_examples]
        
        feature_stats[feat_idx] = {
            'density': activation_density,
            'top_positions': top_positions,
            'top_tasks': top_tasks,
            'mean_activation': feat_col.mean().item(),
        }
    
    # Flag dead features (per SAE Survey, arXiv:2503.05613 pitfall #1)
    dead_features = [f for f, s in feature_stats.items() if s['density'] == 0]
    print(f"Dead features: {len(dead_features)}/{sae.n_features} ({len(dead_features)/sae.n_features:.1%})")
    
    return feature_stats
```

### 6.4 SAE Pitfalls Checklist (from arXiv:2503.05613)

| Pitfall | Mitigation |
|---|---|
| Too few training tokens | Use at least 50M activation vectors; collect from all 500 tasks × all layers |
| Dead features | Monitor density; use auxiliary TopK loss if >10% dead |
| Reconstruction loss only | Also measure: does ablating SAE features change model output? |
| Feature duplication | Check cosine similarity between feature directions; prune near-duplicates |
| Layer selection bias | Train SAEs on multiple layers (6, 12, 18, 24) and compare |
| Polysemantic "junk" features | Manually inspect top-10 activating examples for each feature |

---

## 7. Experiment D — TTT Before/After Analysis (Day 7)

**Goal:** Detect mechanistic changes induced by test-time training (TTT) by comparing SAE feature activations and attention patterns before and after TTT on the same ARC puzzle.

**Theoretical basis:** *Specialization after Generalization* (arXiv:2509.24510) argues that TTT creates early-layer specialization while preserving late-layer generality. *TTT for ARC* (arXiv:2411.07279) establishes the protocol. We test whether this specialization is visible in our SAE feature set.

### 7.1 TTT Protocol

```python
from transformers import Trainer, TrainingArguments

def run_ttt_on_task(base_model_path, task, n_steps=50, lr=1e-4):
    """
    Run test-time training on a single ARC task's few-shot examples.
    Returns: TTT-trained model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    ttt_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Format training data from task's few-shot examples
    train_texts = []
    for example in task['train']:
        inp = grid_to_tokens(example['input'])
        out = grid_to_tokens(example['output'])
        train_texts.append(f"Input:\n{inp}\nOutput:\n{out}")
    
    # Tokenize
    encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
    
    # Quick fine-tune
    optimizer = torch.optim.AdamW(ttt_model.parameters(), lr=lr)
    for step in range(n_steps):
        outputs = ttt_model(**encodings, labels=encodings['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 0:
            print(f"TTT step {step}/{n_steps}: loss={loss.item():.4f}")
    
    return ttt_model

# Select 10 diverse ARC tasks for TTT analysis
ttt_tasks = list(arc_tasks.items())[:10]
```

### 7.2 Compare Activations Before/After TTT

```python
def compare_ttt_activations(task, base_model, ttt_model, layer=12):
    """
    Compare residual stream activations before/after TTT using cosine similarity.
    """
    prompt = format_arc_prompt(task)
    hook = f'blocks.{layer}.hook_resid_post'
    
    # Get base model activations
    base_tl = HookedTransformer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", hf_model=base_model)
    tokens = base_tl.to_tokens(prompt)
    with torch.no_grad():
        _, cache_base = base_tl.run_with_cache(tokens, names_filter=[hook])
    acts_base = cache_base[hook][0].cpu()
    
    # Get TTT model activations
    ttt_tl = HookedTransformer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", hf_model=ttt_model)
    with torch.no_grad():
        _, cache_ttt = ttt_tl.run_with_cache(tokens, names_filter=[hook])
    acts_ttt = cache_ttt[hook][0].cpu()
    
    # Cosine similarity per token
    cos_sim = torch.nn.functional.cosine_similarity(acts_base, acts_ttt, dim=-1)
    delta_norm = (acts_ttt - acts_base).norm(dim=-1)
    
    return cos_sim.numpy(), delta_norm.numpy()

# Run TTT comparison
ttt_results = []
for task_id, task in ttt_tasks:
    ttt_model = run_ttt_on_task(MODEL_PATH, task)
    cos_sims, delta_norms = compare_ttt_activations(task, hf_model, ttt_model)
    ttt_results.append({'task_id': task_id, 'cos_sim': cos_sims, 'delta_norm': delta_norms})
    print(f"Task {task_id}: mean cos_sim={cos_sims.mean():.3f}, mean delta_norm={delta_norms.mean():.3f}")
```

### 7.3 SAE Feature Change Analysis

```python
def ttt_sae_feature_delta(sae, acts_base, acts_ttt):
    """
    For each SAE feature, measure activation change after TTT.
    Returns sorted list of (feature_idx, mean_delta).
    """
    with torch.no_grad():
        feats_base = sae.encode(torch.from_numpy(acts_base))
        feats_ttt = sae.encode(torch.from_numpy(acts_ttt))
    
    delta = (feats_ttt - feats_base).abs().mean(0)  # [n_features]
    ranked = delta.argsort(descending=True)
    return [(int(i), float(delta[i])) for i in ranked[:50]]  # top 50 changed features
```

---

## 8. Common Pitfalls and How to Avoid Them

| Risk | Source | Mitigation |
|---|---|---|
| TransformerLens Qwen2 loading failure | — | Use `nnsight` fallback; verify model config matches |
| Grid tokenization misalignment | — | Validate `pos_map` by decoding tokens and comparing to original grid |
| Probe overfitting | arXiv:2311.03658 §4 | Strict train/test split; use regularized Ridge/LogisticRegression; report cross-validated scores |
| SAE dead features > 15% | arXiv:2503.05613 §3 | Lower learning rate; add auxiliary loss; increase batch size |
| SAE reconstruction too high (collapsed) | arXiv:2406.04093 §5 | Check that k is not too large (k > d_model/4 is suspicious) |
| Induction heads confounding spatial heads | arXiv:2209.11895 §6 | Cross-check: spatial head score should be high even for shuffled-order grids |
| TTT overfitting to training examples | arXiv:2411.07279 §3 | Limit TTT steps (≤100); monitor loss; compare on held-out grids |
| Causal vs. correlational attention patterns | arXiv:2202.05262 §2 | Always follow attention pattern analysis with ablation experiments |

---

## 9. Success Criteria

| Experiment | Minimum Bar | Strong Result |
|---|---|---|
| Exp A | ≥2 heads with spatial score >0.6; ≥1 head causally confirmed by ablation | Full spatial circuit identified: 3–5 heads with clear QK decomposition |
| Exp B | Row/col probe R² > 0.5 in at least one layer | Clear layer-by-layer emergence: low → high → maintained across layers |
| Exp C | SAE reconstructs >85% variance; <10% dead features; ≥5 interpretable ARC features | >20 interpretable spatial features; feature taxonomy by concept |
| Exp D | Measurable activation delta after TTT; early layers change more than late | TTT-specific features identified in SAE; matches *Specialization after Generalization* prediction |

---

## 10. Day-by-Day Schedule

| Day | Task | Key Deliverable |
|---|---|---|
| Day 1 | Environment setup, model load, ARC data pipeline, baseline activation collection | Verified Colab notebook; confirmed model loads correctly |
| Day 2 | Exp A: Attention weight extraction, spatial scoring for all heads | Head heatmap; ranked list of spatially specialized heads |
| Day 3 | Exp A: Ablation experiments; path patching on top heads | Causal confirmation table; circuit diagram sketch |
| Day 4 | Exp B: Probe dataset construction; train probes at all layers | Layer-by-layer probe curves; identification of peak spatial encoding layers |
| Day 5 | Exp C: Collect large activation dataset (500+ tasks, 3 layers); upload to GCS | `acts_layer{6,12,18}_acts.npy` files on GCS |
| Day 6 | Exp C: Train TopK SAEs on GCP TPU; interpret top features | Trained SAE checkpoints; feature interpretation spreadsheet |
| Day 7 | Exp D: Run TTT on 10 tasks; compare activations and SAE features | TTT delta analysis; preliminary evidence for/against specialization hypothesis |

---

## 11. Deliverables Checklist

- [ ] `phase1_setup.ipynb` — Environment, model loading, ARC data pipeline
- [ ] `exp_a_attention.ipynb` — Full Experiment A code and results
- [ ] `exp_b_probing.ipynb` — Full Experiment B code and results
- [ ] `exp_c_sae_training.py` — SAE training script (TPU-compatible)
- [ ] `exp_d_ttt_analysis.ipynb` — Full Experiment D code and results
- [ ] `arc_results/phase1/exp_a_head_heatmap.html` — Interactive head heatmap
- [ ] `arc_results/phase1/exp_b_probe_curves.html` — Layer-by-layer probe accuracy
- [ ] `arc_results/phase1/sae_checkpoints/` — SAE weights for layers 6, 12, 18
- [ ] `arc_results/phase1/feature_taxonomy.csv` — Manually labeled SAE features
- [ ] `arc_results/phase1/ttt_delta_analysis.json` — TTT before/after comparison results
- [ ] Phase 1 summary notes (1–2 pages): key findings, unexpected results, hypotheses for Phase 2

---

## 12. How Phase 1 Findings Feed Into Later Phases

**Phase 2 (TRM — Transformer Reasoning Model):** The spatial heads and SAE features identified here become the comparison baseline. We will run identical Experiments A and B on the TRM and directly compare which heads and features overlap. If ARC-solving mechanisms are architecture-agnostic, we expect convergence; if they are LLM-specific, we expect divergence.

**Phase 3 (Diffusion Model):** The SAE feature taxonomy from Experiment C defines the target concepts to look for in diffusion model activations. For example, if we find a "row position" feature in the LLM's residual stream, we ask whether the diffusion model encodes the same concept. The probe methodology from Experiment B transfers directly.

**Phase 4 (Cross-Architecture Synthesis):** The TTT analysis from Experiment D is critical context for Phase 4's hypothesis: that TTT is mechanistically distinct from fine-tuning and that different architectures implement it differently. The LLM's TTT pattern (early specialization vs. late preservation) becomes the null hypothesis that Phase 4 tests.

**Paper writing (arXiv):** Phase 1 figures — the head heatmap, probe curves, and feature taxonomy — are expected to appear in the paper's main results section. The most important result is whether the LLM develops genuine spatial representations (high probe accuracy + interpretable SAE features) or shallow pattern matching (low probe accuracy + polysemantic SAE features). This finding frames the entire paper's argument.

---

## 13. References

1. Elhage et al. (Anthropic, 2021). *A Mathematical Framework for Transformer Circuits.*
2. Meng et al. (NeurIPS 2022, arXiv:2202.05262). *Locating and Editing Factual Associations in GPT (ROME).*
3. Wang et al. (arXiv:2211.00593). *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2.*
4. Olah et al. (Anthropic, 2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.*
5. Gao et al. (OpenAI, arXiv:2406.04093). *Scaling and Evaluating Sparse Autoencoders.*
6. Rajamanoharan et al. (Google DeepMind, arXiv:2407.14435). *Improving Dictionary Learning with Gated Sparse Autoencoders (JumpReLU SAEs).*
7. Lieberum et al. (arXiv:2408.05147). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.*
8. Ayonrinde et al. (arXiv:2503.05613). *Interpretability of Language Models via Task Spaces (SAE Survey).*
9. Olsson et al. (arXiv:2209.11895). *In-context Learning and Induction Heads.*
10. Gendron et al. (arXiv:2411.07279). *Large Language Models as Agents in the Abstraction and Reasoning Corpus (TTT for ARC).*
11. Anonymous (arXiv:2509.24510). *Specialization after Generalization: What Test-Time Training Does Mechanistically.*
12. Gurnee & Tegmark (arXiv:2310.02207). *Language Models Represent Space and Time.*
13. Park et al. (arXiv:2311.03658). *The Linear Representation Hypothesis and the Geometry of Large Language Models.*
14. Sharkey et al. (arXiv:2501.16496). *Open Problems in Mechanistic Interpretability.*
15. Nanda (arXiv:2407.02646). *A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models.*

---

*Document version: 1.0 | Phase 1 of 4 | Last updated: May 2026*
