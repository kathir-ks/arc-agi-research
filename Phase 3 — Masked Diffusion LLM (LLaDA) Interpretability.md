# Phase 3 — Masked Diffusion LLM (LLaDA) Interpretability
## ARC-AGI Mechanistic Interpretability Research

**Researcher:** Kathir
**Duration:** Days 1–7 (parallel execution — starts Day 1 alongside Phases 1, 2, 4)
**Compute:** Colab Session 3 (A100 strongly recommended for 8B model); GCP TPU Days 7–8 for SAE training
**Model:** ARChitects' ARC-fine-tuned LLaDA-8B with 2D RoPE (arXiv:2601.10904)
**Base Model HuggingFace ID:** `GSAI-ML/LLaDA-8B-Base`
**Risk Level:** High-risk / High-reward

---

## ⚡ Parallel Execution Timeline (Phase 3 Only)

> **All 4 phases run simultaneously.** See `Master Parallel Execution Plan.md` for the full coordination schedule, compute allocation, and cross-phase sync points. This document covers only Phase 3 work.

Phase 3 is the **fastest phase** to complete experimentally — it finishes by Day 7, while Phases 1 and 2 are still running. This means Phase 3 results feed into paper writing early, which is a significant advantage.

| Day | Phase 3 Activity | Colab Session | TPU? |
|---|---|---|---|
| **Day 1** | Install nnsight, download LLaDA-8B-Base + ARChitects weights, test masked forward pass, verify nnsight hooks work on bidirectional model | Colab 3 | No |
| **Day 2** | Exp A: Run T=10–20 denoising steps on 200 shared tasks; track unmasking order per cell | Colab 3 | No |
| **Day 3** | Exp A: Analyze unmasking spatial structure (boundaries/centers/colors first?); collect residual stream at 5 mask ratios; upload to GCS | Colab 3 | Upload |
| **Day 4** | Exp A: Train timestep-conditioned SAE (FiLM conditioning on mask ratio); visualize concept emergence across denoising steps | Colab 3 | No |
| **Day 5** | Exp B: Compute spatial specialization scores for all LLaDA heads at multiple denoising steps; compare to Phase 1 Qwen scores | Colab 3 | No |
| **Day 6** | Exp C: Load both 1D RoPE (base) and 2D RoPE (ARChitects) variants; run row/col/diagonal probing; compute CKA between representations | Colab 3 | No |
| **Day 7** | Finalize all figures, write Phase 3 summary notes; cross-architecture attention scores handed to Phase 4 | Colab 3 | No |

**Phase 3 completes Day 7** — earliest of all experimental phases. Results immediately available for Phase 4 §6.

**TPU scheduling:** Phase 3 SAE training (if needed for LLaDA activations at scale) is scheduled Days 7–8, after Phase 1 TPU job finishes. The timestep-conditioned SAE in Exp A can be trained on Colab A100 if TPU is unavailable (LLaDA's 8B model has larger activations but fewer tasks × fewer timesteps than TRM's 23M vectors).

**A100 strongly recommended:** LLaDA-8B requires ~16GB VRAM for inference. Colab T4 (16GB) is borderline — use A100 (40GB) if available in Colab Pro. Run `torch.cuda.memory_allocated()` after loading to confirm.

**Shared infrastructure note:** Use the same 200 shared ARC task IDs as Phases 1 and 2. The cross-architecture attention comparison (Paper Figure 10) requires that all 3 models run on the same tasks. See `Master Parallel Execution Plan.md` for exact IDs.

---

## 1. Phase Overview

### Why LLaDA Is Architecturally Unique

Phase 1 and Phase 2 studied autoregressive models: Qwen 0.5B and the token-prediction pipeline that generates outputs left-to-right under a causal mask. LLaDA-8B breaks every one of those assumptions.

LLaDA ("Large Language Diffusion with mAsking") is a **masked diffusion language model** (MDLM). Rather than predicting the next token given the past, LLaDA generates all output tokens simultaneously and iteratively refines them over multiple denoising steps. The architecture is a **bidirectional Transformer with no causal mask** — every token attends to every other token at every layer, including future output positions. The forward diffusion process progressively masks output tokens, and the reverse (generative) process unmasks them over T denoising steps (typically T=10..256), producing increasingly confident token assignments.

The ARChitects team (arXiv:2601.10904) fine-tuned LLaDA-8B-Base on ARC tasks and made a further architectural modification: they replaced the standard 1D Rotary Position Embedding (RoPE) with a **"Golden Gate" 2D RoPE** that encodes horizontal (column), vertical (row), and diagonal directions in separate frequency components. This modification is what enabled the model to treat the ARC grid as a true 2D spatial structure rather than a flattened sequence. The system achieved **16.53% on ARC-AGI-2** (2nd place overall in 2025), a striking result for a purely language-model-based approach.

### Mechanistic Interpretability Questions This Opens

This combination of architecture and task raises questions no prior MI work has addressed:

1. **Do diffusion denoising steps function as discrete reasoning stages?** At step T (highly masked), does the model commit to coarse structure (which region changes)? At step 1 (nearly unmasked), does it refine fine-grained color decisions? If so, denoising steps are not mere stochastic sampling — they are mechanistically interpretable reasoning stages.

2. **How does bidirectional attention specialize when there is no causal structure?** Autoregressive models show well-documented attention head roles (induction, previous-token, name-mover — see "What Does BERT Look At?", arXiv:1906.04341). In a bidirectional model generating a structured grid output, do heads specialize by spatial relationship rather than sequential position?

3. **What does 2D RoPE do to the internal geometry of representations?** Replacing 1D with 2D positional encoding should reorganize the representational space. CKA comparisons and probing classifiers can reveal whether row/column structure becomes linearly decodable from residual stream activations in the 2D-RoPE model but not the 1D baseline.

4. **Can Sparse Autoencoders decompose diffusion model activations?** No SAE study exists for masked diffusion LMs. The timestep-dependency of activations (features shift as masking ratio changes) makes this a technically novel adaptation problem.

---

## 2. Why This Is High-Risk / High-Reward

### Technical Challenges

**No TransformerLens support.** TransformerLens does not recognize LLaDA's architecture. Its forward pass deviates from standard HuggingFace Transformers in several ways (custom masking logic, diffusion-specific embeddings). You must use `nnsight` for all hook-based activation collection.

**Timestep-dependent activations.** In autoregressive models, activations for a given input are deterministic. In LLaDA, the same input produces different activations at each denoising step because the masked token positions change. Every downstream analysis (SAE training, attention head analysis) must condition on or marginalize over the masking ratio $\sigma_t$. This invalidates direct application of standard SAE training procedures.

**No causal mask.** All circuit-finding methods (path patching, activation patching) assume directed information flow in an autoregressive graph. Bidirectional attention creates undirected computational graphs with dense cross-token dependencies. Standard IOI circuit analysis does not transfer directly.

**Masked token embedding ambiguity.** The `[MASK]` token has a learned embedding distinct from all vocabulary tokens, but its representation evolves across layers in a way that depends on all surrounding context simultaneously. Interpreting what information this embedding carries at intermediate layers requires careful experimental design.

**Sparse ARC data.** ARC-AGI has only 400 training and 400 evaluation tasks. With LLaDA's stochastic denoising (each forward pass differs), getting statistically robust activation measurements requires multiple forward passes per task and careful variance tracking.

### Why the Reward Justifies the Risk

The gap in the literature is complete. "Emergence and Evolution of Interpretable Concepts in Diffusion Models" (arXiv:2504.15473, NeurIPS 2025 Spotlight) applies SAEs to image diffusion models and finds timestep-semantic structure — but this is entirely in the continuous image domain. "Revelio" (arXiv:2411.16725, ICCV 2025) finds coarse-to-fine semantic organization across timesteps in image diffusion via k-sparse SAEs. Neither paper touches discrete masked diffusion or language-domain tasks. "DiffLens" (arXiv:2503.20483, CVPR 2025) runs causal interventions in diffusion models but again in the image domain. This Phase 3 would produce the **first mechanistic interpretability results for a discrete masked diffusion LM**, and specifically on a frontier spatial reasoning benchmark. That is a clear, defensible novelty claim for arXiv.

---

## 3. Literature Foundation Per Experiment

### Experiment A — Diffusion Step Analysis
- **LLaDA paper** (arXiv:2502.09992): Architecture, masking schedule, denoising procedure.
- **MDLM** (arXiv:2406.07524): Formal mechanics of masked diffusion, forward/reverse process definition.
- **Scaling MDMs** (arXiv:2410.18514): Provides context for LLaDA-8B scale.
- **IterRef / Effective Test-Time Scaling** (arXiv:2511.05562): Frames iterative refinement as a reasoning process — your key theoretical hook for why denoising steps might be reasoning stages.
- **Revelio** (arXiv:2411.16725): Timestep-semantic organization in continuous diffusion — your closest analogous finding in image models.
- **Emergence and Evolution** (arXiv:2504.15473): SAE applied to diffusion activations — most methodologically relevant prior work. Novel gap: discrete masked diffusion, language domain, grid-structured outputs.

### Experiment B — Bidirectional Attention Analysis
- **"What Does BERT Look At?"** (arXiv:1906.04341): The canonical reference for analyzing bidirectional attention head specialization. Defines diagonal/vertical/CLS attention patterns. Direct methodological precursor.
- **Practical Review of MI** (arXiv:2407.02646): Covers `nnsight` for non-AR bidirectional models — your primary tooling reference.
- **ARChitects report** (arXiv:2601.10904): Model specifics and ARC task format.
- **ARC-AGI-2** (arXiv:2505.11831): Task space context, why spatial reasoning is the challenge.
- Novel gap: no prior work analyzes bidirectional attention head roles in a diffusion denoising context or on structured grid tasks.

### Experiment C — 2D RoPE Effect
- **RoPE-ViT** (arXiv:2403.13298): 2D RoPE analysis in vision transformers — direct methodological template for your probing and CKA experiments.
- **CKA** (arXiv:1905.00414): Centered Kernel Alignment for measuring representational similarity between 1D-RoPE and 2D-RoPE models at each layer.
- **ARChitects report** (arXiv:2601.10904): Describes the "Golden Gate" 2D RoPE modification and why it was chosen.
- Novel gap: 2D RoPE interpretability has received zero mechanistic attention in any domain. This experiment would be the first.

---

## 4. Setup Guide (Day 19)

### 4.1 Environment Installation

```bash
pip install nnsight>=0.3.0
pip install transformers>=4.40.0
pip install torch>=2.2.0
pip install einops
pip install scikit-learn        # probing classifiers
pip install plotly kaleido      # visualization
pip install matplotlib seaborn
# SAE library (train from scratch — no LLaDA-compatible pre-trained SAEs exist)
pip install sae-lens            # or implement a lightweight custom SAE
```

### 4.2 Loading LLaDA-8B (ARChitects version)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load ARChitects' fine-tuned version (check their HuggingFace repo from arXiv:2601.10904)
# Base model fallback:
MODEL_ID = "GSAI-ML/LLaDA-8B-Base"  # swap for ARChitects checkpoint when available

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

MASK_TOKEN_ID = tokenizer.mask_token_id
print(f"MASK token ID: {MASK_TOKEN_ID}")
```

### 4.3 Understanding LLaDA's Forward Pass

```python
def create_masked_input(input_ids: torch.Tensor, output_ids: torch.Tensor, 
                         mask_ratio: float) -> torch.Tensor:
    """
    LLaDA's forward pass takes: [prompt tokens | partially masked output tokens]
    mask_ratio = fraction of output tokens currently masked (sigma_t in MDLM notation)
    At t=T (beginning): mask_ratio ~ 1.0 (nearly all output masked)
    At t=1 (end):      mask_ratio ~ 0.0 (nearly all output revealed)
    """
    output_len = output_ids.shape[-1]
    masked_output = output_ids.clone()
    n_to_mask = int(mask_ratio * output_len)
    # Random mask positions (in practice LLaDA uses a fixed schedule)
    mask_positions = torch.randperm(output_len)[:n_to_mask]
    masked_output[0, mask_positions] = MASK_TOKEN_ID
    return torch.cat([input_ids, masked_output], dim=1)

# Example: ARC task forward pass at 50% masking
input_seq = create_masked_input(prompt_ids, output_ids, mask_ratio=0.5)
with torch.no_grad():
    logits = model(input_seq).logits  # shape: [1, seq_len, vocab_size]
    # Output logits at masked positions give token predictions
```

### 4.4 Setting Up nnsight Hooks for Bidirectional Model

```python
from nnsight import LanguageModel

# nnsight wraps any HuggingFace model and provides clean hook API
lm = LanguageModel(MODEL_ID, device_map="auto", torch_dtype=torch.float16)

def collect_residual_stream(input_ids, layer_indices=None):
    """Collect residual stream activations at every layer."""
    if layer_indices is None:
        layer_indices = list(range(lm.config.num_hidden_layers))
    
    layer_activations = {}
    
    with lm.trace(input_ids):
        for layer_idx in layer_indices:
            # Access the residual stream output of each transformer block
            layer_activations[layer_idx] = lm.model.layers[layer_idx].output[0].save()
    
    return {k: v.value for k, v in layer_activations.items()}


def collect_attention_patterns(input_ids, layer_idx: int):
    """Collect all attention head patterns for a given layer."""
    with lm.trace(input_ids):
        # Hook into the attention module's output weights
        attn_weights = lm.model.layers[layer_idx].self_attn.attn_weights.save()
    return attn_weights.value  # shape: [batch, heads, seq, seq]
```

**Critical note on masked inputs:** When collecting activations, always record which token positions are `MASK_TOKEN_ID`. The model's internal representation at masked positions is what you are studying for Experiment A — do not accidentally exclude these positions.

```python
def get_mask_positions(input_ids: torch.Tensor) -> torch.Tensor:
    """Return boolean mask of which positions are [MASK] tokens."""
    return input_ids == MASK_TOKEN_ID
```

---

## 5. Experiment A: Diffusion Step Analysis (Days 20–21)

**Hypothesis:** LLaDA's denoising steps function as mechanistically distinct reasoning stages. High-masking steps (early) commit to coarse spatial structure; low-masking steps (late) refine fine-grained token identity. SAE features active at each timestep should be semantically different.

### 5.1 Unmasking Order Analysis

```python
import torch
import numpy as np
from collections import defaultdict

def run_llada_denoising(model, tokenizer, prompt_ids, output_len, T=20, seed=42):
    """
    Run the full LLaDA denoising chain and record which positions get
    committed (unmasked) at each denoising step.
    Returns: list of (step, newly_unmasked_positions, predicted_tokens)
    """
    torch.manual_seed(seed)
    device = prompt_ids.device
    
    # Initialize fully masked output
    output_ids = torch.full((1, output_len), MASK_TOKEN_ID, dtype=torch.long, device=device)
    full_seq = torch.cat([prompt_ids, output_ids], dim=1)
    prompt_len = prompt_ids.shape[-1]
    
    history = []
    prev_unmasked = set()
    
    for step in range(T, 0, -1):
        mask_ratio = step / T
        
        with torch.no_grad():
            logits = model(full_seq).logits  # [1, seq_len, vocab]
        
        # Get output token predictions
        output_logits = logits[0, prompt_len:]  # [output_len, vocab]
        predicted_tokens = output_logits.argmax(dim=-1)
        
        # Confidence scores at masked positions
        probs = output_logits.softmax(dim=-1)
        confidence = probs.max(dim=-1).values
        
        # Unmask the top-k most confident positions this step
        # (Approximate LLaDA's actual schedule for analysis)
        mask_positions = (full_seq[0, prompt_len:] == MASK_TOKEN_ID).nonzero().squeeze(-1)
        if len(mask_positions) == 0:
            break
        
        n_to_unmask = max(1, int(len(mask_positions) / step))
        top_positions = mask_positions[confidence[mask_positions].topk(n_to_unmask).indices]
        
        newly_unmasked = set(top_positions.tolist()) - prev_unmasked
        history.append({
            'step': step,
            'mask_ratio': mask_ratio,
            'newly_unmasked': list(newly_unmasked),
            'tokens': predicted_tokens[list(newly_unmasked)].tolist(),
            'confidence': confidence[list(newly_unmasked)].tolist(),
        })
        
        # Update sequence
        full_seq[0, prompt_len + top_positions] = predicted_tokens[top_positions]
        prev_unmasked.update(top_positions.tolist())
    
    return history, full_seq


def analyze_unmasking_order(history, grid_width: int, grid_height: int):
    """
    Map unmasked positions back to 2D grid coordinates.
    Tests whether unmasking follows spatial patterns (e.g., corners first, 
    boundary before interior, changed cells before unchanged cells).
    """
    results = []
    for entry in history:
        for pos, tok in zip(entry['newly_unmasked'], entry['tokens']):
            row = pos // (grid_width + 1)   # +1 for newline token
            col = pos % (grid_width + 1)
            results.append({
                'step': entry['step'],
                'mask_ratio': entry['mask_ratio'],
                'row': row, 'col': col,
                'token': tok,
            })
    return results
```

### 5.2 SAE Adaptation for Diffusion Models

Standard SAE training (from "Towards Monosemanticity") trains on residual stream activations from a single model configuration. For LLaDA, activations shift with masking ratio — a feature active at $\sigma=0.9$ may be absent at $\sigma=0.1$. The adaptation strategy is to **condition the SAE on the timestep** or **train separate SAEs per masking-ratio bin**.

```python
import torch
import torch.nn as nn

class TimestepConditionedSAE(nn.Module):
    """
    Sparse Autoencoder conditioned on the diffusion timestep (masking ratio).
    Follows Anthropic's 'Towards Monosemanticity' architecture with 
    timestep conditioning added via FiLM (Feature-wise Linear Modulation).
    
    Reference: arXiv:2504.15473 uses per-timestep SAEs on image diffusion.
    We adapt this to discrete masked diffusion (language domain).
    """
    def __init__(self, d_model: int, d_sae: int, n_timestep_bins: int = 10):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        
        # Standard SAE encoder/decoder
        self.W_enc = nn.Linear(d_model, d_sae, bias=True)
        self.W_dec = nn.Linear(d_sae, d_model, bias=True)
        
        # Timestep conditioning: learn scale/shift per bin
        self.timestep_embed = nn.Embedding(n_timestep_bins, d_model * 2)  # scale + shift
        self.n_bins = n_timestep_bins
        
    def forward(self, x: torch.Tensor, mask_ratio: float):
        """
        x: [batch, seq_len, d_model] — residual stream activations
        mask_ratio: float in [0, 1] — current denoising masking ratio
        """
        # Discretize mask ratio to bin
        bin_idx = min(int(mask_ratio * self.n_bins), self.n_bins - 1)
        t_emb = self.timestep_embed(torch.tensor(bin_idx, device=x.device))
        scale, shift = t_emb.chunk(2, dim=-1)
        
        # FiLM conditioning
        x_conditioned = x * (1 + scale) + shift
        
        # Encode → ReLU → Decode
        pre_acts = self.W_enc(x_conditioned)
        acts = torch.relu(pre_acts)   # sparse activations
        x_recon = self.W_dec(acts)
        
        return x_recon, acts
    
    def get_sparsity_loss(self, acts: torch.Tensor, l1_coeff: float = 1e-3):
        return l1_coeff * acts.abs().sum(dim=-1).mean()


def collect_sae_training_data(lm, arc_tasks, mask_ratios, layer_idx: int):
    """
    Collect residual stream activations at a given layer across multiple
    mask ratios for SAE training. Returns (activations, mask_ratio_labels).
    """
    all_acts = []
    all_ratios = []
    
    for task in arc_tasks:
        prompt_ids = task['prompt_ids']
        output_ids = task['output_ids']
        
        for sigma in mask_ratios:
            masked_input = create_masked_input(prompt_ids, output_ids, sigma)
            acts = collect_residual_stream(lm, masked_input, layer_indices=[layer_idx])
            all_acts.append(acts[layer_idx])
            all_ratios.append(sigma)
    
    return torch.cat(all_acts, dim=0), all_ratios
```

**Training procedure:** Train the `TimestepConditionedSAE` on activations collected at masking ratios $\sigma \in \{0.9, 0.7, 0.5, 0.3, 0.1\}$, using layers 16, 24, 32 (early, mid, late). Compare which SAE features fire at each $\sigma$ — features that shift between high-$\sigma$ and low-$\sigma$ are candidate "reasoning stage" features.

### 5.3 Expected Outputs — Experiment A

- Unmasking order heatmaps: do certain grid positions (corners, changed cells, boundary) get committed earlier?
- SAE feature activation profiles across timesteps: which features are "early-stage" vs "late-stage"?
- Comparison to image diffusion findings from Revelio (arXiv:2411.16725): coarse-to-fine analog in language?

---

## 6. Experiment B: Bidirectional vs. Causal Attention Analysis (Days 22–23)

**Hypothesis:** LLaDA's bidirectional heads develop spatial-relational specializations (row-attender, column-attender, diagonal-attender) rather than the sequential roles seen in autoregressive models. This is enabled by the absence of the causal mask.

### 6.1 Attention Head Classification (adapting "What Does BERT Look At?")

```python
import torch
import numpy as np
from scipy.stats import entropy

def compute_head_specialization(attn_matrix: torch.Tensor, grid_width: int, 
                                 grid_height: int, prompt_len: int):
    """
    Classify attention head behavior on ARC grid outputs.
    attn_matrix: [heads, seq_len, seq_len]
    
    Returns specialization scores per head:
    - row_score: how much within-row attention dominates
    - col_score: how much within-column attention dominates  
    - diagonal_score: diagonal attention pattern strength
    - uniform_score: how close to uniform (entropy-based)
    
    Adapted from "What Does BERT Look At?" (arXiv:1906.04341) to 2D grid domain.
    """
    n_heads = attn_matrix.shape[0]
    output_len = grid_width * grid_height
    
    # Extract output-to-output attention submatrix
    out_attn = attn_matrix[:, prompt_len:prompt_len+output_len, 
                              prompt_len:prompt_len+output_len]
    
    scores = {'row': [], 'col': [], 'diagonal': [], 'entropy': []}
    
    for h in range(n_heads):
        head_attn = out_attn[h].float().cpu().numpy()  # [output_len, output_len]
        
        # Reshape to 2D: [grid_h * grid_w, grid_h * grid_w]
        # Position (i,j) -> (row=i//grid_w, col=i%grid_w)
        
        row_mass = 0.0
        col_mass = 0.0
        diag_mass = 0.0
        
        for i in range(output_len):
            ri, ci = divmod(i, grid_width)
            for j in range(output_len):
                rj, cj = divmod(j, grid_width)
                w = head_attn[i, j]
                if ri == rj and i != j:
                    row_mass += w
                if ci == cj and i != j:
                    col_mass += w
                if abs(ri - rj) == abs(ci - cj) and i != j:
                    diag_mass += w
        
        n_pairs = output_len * (output_len - 1)
        scores['row'].append(row_mass / n_pairs)
        scores['col'].append(col_mass / n_pairs)
        scores['diagonal'].append(diag_mass / n_pairs)
        scores['entropy'].append(entropy(head_attn.mean(axis=0) + 1e-9))
    
    return scores


def compare_head_specialization_across_models(llada_scores, qwen_scores):
    """
    Compare head role distributions between LLaDA (bidirectional, diffusion)
    and Qwen (autoregressive, causal). Tests whether bidirectionality drives
    spatial specialization.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, role in zip(axes, ['row', 'col', 'diagonal']):
        ax.hist(llada_scores[role], alpha=0.6, label='LLaDA (bidirectional)', bins=20)
        ax.hist(qwen_scores[role], alpha=0.6, label='Qwen (causal)', bins=20)
        ax.set_title(f'{role.capitalize()} Attention Specialization')
        ax.set_xlabel('Specialization Score')
        ax.legend()
    plt.tight_layout()
    plt.savefig('attention_head_comparison.png', dpi=150)
```

### 6.2 Attention Head Role Analysis Across Denoising Steps

A unique angle: in autoregressive models, head roles are fixed for a given input. In LLaDA, head roles may shift across denoising steps as the masking ratio changes.

```python
def track_head_roles_across_timesteps(lm, task, mask_ratios, layer_idx: int, 
                                       grid_width: int, grid_height: int):
    """
    For each mask ratio, collect attention patterns and compute head specialization.
    Tests whether head roles are static or shift across denoising steps.
    """
    role_by_step = {}
    
    for sigma in mask_ratios:
        masked_input = create_masked_input(task['prompt_ids'], task['output_ids'], sigma)
        attn = collect_attention_patterns(lm, masked_input, layer_idx)
        roles = compute_head_specialization(
            attn[0], grid_width, grid_height, task['prompt_len']
        )
        role_by_step[sigma] = roles
    
    return role_by_step
```

---

## 7. Experiment C: 2D RoPE Effect Analysis (Days 24–25)

**Hypothesis:** The ARChitects' 2D RoPE modification causes row and column coordinates to become linearly decodable from residual stream activations, whereas the 1D RoPE baseline does not. CKA between the two models should show representation divergence in early-to-mid layers where positional information is primarily encoded.

### 7.1 Probing Row/Column Coordinates

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
import numpy as np

def probe_spatial_coordinates(activations: np.ndarray, 
                               row_labels: np.ndarray,
                               col_labels: np.ndarray,
                               layer_name: str):
    """
    Train linear probes on residual stream activations to predict grid row/col.
    High accuracy = spatial coordinates linearly encoded in residual stream.
    
    Methodology from RoPE-ViT (arXiv:2403.13298) adapted to language domain.
    """
    results = {}
    
    for label_name, labels in [('row', row_labels), ('col', col_labels)]:
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, activations, labels, cv=5, scoring='accuracy')
        results[label_name] = {
            'mean_accuracy': scores.mean(),
            'std': scores.std(),
            'layer': layer_name,
        }
        print(f"  [{layer_name}] {label_name} probe accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    
    return results


def collect_token_activations_with_positions(lm, arc_tasks, layer_idx: int):
    """
    For each output token position, collect its residual stream activation
    and its (row, col) coordinate in the ARC grid.
    Returns: (activations, row_labels, col_labels)
    """
    all_acts = []
    all_rows = []
    all_cols = []
    
    for task in arc_tasks:
        grid_w = task['grid_width']
        full_input = create_masked_input(
            task['prompt_ids'], task['output_ids'], mask_ratio=0.0  # fully revealed
        )
        acts = collect_residual_stream(lm, full_input, layer_indices=[layer_idx])
        layer_acts = acts[layer_idx][0]  # [seq_len, d_model]
        
        prompt_len = task['prompt_len']
        output_len = task['output_ids'].shape[-1]
        
        # Extract output token activations
        for pos in range(output_len):
            row, col = divmod(pos, grid_w + 1)  # +1 for newline
            if col < grid_w:  # skip newline positions
                all_acts.append(layer_acts[prompt_len + pos].float().cpu().numpy())
                all_rows.append(row)
                all_cols.append(col)
    
    return np.array(all_acts), np.array(all_rows), np.array(all_cols)
```

### 7.2 CKA Representational Similarity Between 1D and 2D RoPE Models

```python
def centered_kernel_alignment(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute CKA between representation matrices X and Y.
    CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    Reference: arXiv:1905.00414
    """
    def hsic(K, L):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return np.trace(K @ H @ L @ H) / (n - 1) ** 2
    
    K = X @ X.T
    L = Y @ Y.T
    return hsic(K, L) / np.sqrt(hsic(K, K) * hsic(L, L))


def compare_rope_representations(lm_1d, lm_2d, arc_tasks, layers):
    """
    For each layer, compute CKA between 1D-RoPE and 2D-RoPE model representations
    on the same ARC tasks. Low CKA = representations have diverged due to RoPE change.
    """
    cka_by_layer = {}
    
    for layer_idx in layers:
        acts_1d, rows, cols = collect_token_activations_with_positions(
            lm_1d, arc_tasks, layer_idx
        )
        acts_2d, _, _ = collect_token_activations_with_positions(
            lm_2d, arc_tasks, layer_idx
        )
        
        # Subsample for computational feasibility
        n = min(2000, len(acts_1d))
        idx = np.random.choice(len(acts_1d), n, replace=False)
        
        cka_score = centered_kernel_alignment(acts_1d[idx], acts_2d[idx])
        cka_by_layer[layer_idx] = cka_score
        print(f"  Layer {layer_idx}: CKA(1D, 2D) = {cka_score:.4f}")
    
    return cka_by_layer
```

**Interpretation:** Layers where CKA drops sharply are where 2D RoPE most changes the model's representational geometry. Run probing at those layers to confirm row/col decodability increases.

---

## 8. Common Pitfalls and Mitigations

| Pitfall | Mitigation |
|---|---|
| TransformerLens incompatibility | Use `nnsight` exclusively for LLaDA. Never attempt `HookedTransformer.from_pretrained` with LLaDA's config. |
| Activation shape confusion | LLaDA may output different shapes depending on masking. Always print `tensor.shape` before indexing. |
| MASK token bleeding into analysis | Always record mask positions explicitly; separate analysis of masked vs. revealed token activations. |
| Stochastic masking inflating variance | Use fixed random seeds across all forward passes; run each task with 3+ seeds and average. |
| GPU OOM on 8B model | Use `torch.float16`; offload to CPU when not actively computing; collect activations in batches of 1 task. |
| CKA requires same tasks for both models | Ensure identical tokenization and grid formatting for 1D vs. 2D RoPE comparison. |
| SAE features not interpretable | Fall back to dictionary learning (k-means on activations) as a simpler decomposition baseline. |
| ARChitects checkpoint unavailable | Fall back to GSAI-ML/LLaDA-8B-Instruct; note limitation in paper. |

---

## 9. Day-by-Day Execution Plan

| Day | Activity | Deliverable |
|---|---|---|
| 19 | Environment setup; load LLaDA; validate denoising forward pass; set up nnsight hooks | Working activation collection pipeline |
| 20 | Exp A part 1: Unmasking order analysis across 50 ARC tasks; spatial heatmaps | Unmasking order figures |
| 21 | Exp A part 2: SAE training on multi-timestep activations; feature activation profiles by timestep | SAE trained; timestep-feature matrix |
| 22 | Exp B part 1: Attention head specialization analysis (row/col/diagonal scores) | Head role classification figures |
| 23 | Exp B part 2: Cross-denoising-step head role tracking; comparison to Qwen Phase 1 results | Head role comparison table |
| 24 | Exp C part 1: Probing row/col coordinates in 1D vs 2D RoPE models at each layer | Probing accuracy curves |
| 25 | Exp C part 2: CKA analysis; integration with Phase 1/2 narrative; fallback/cleanup | CKA heatmap; section draft |

---

## 10. Success Criteria and Expected Outputs

### Minimum Success (publishable as a negative result or exploratory finding)
- Unmasking order analysis completed on 50+ ARC tasks with statistical summary
- At least one attention head role analysis comparing LLaDA to Qwen Phase 1 results
- nnsight activation collection pipeline documented and reproducible

### Target Success (strong contribution)
- Diffusion steps show statistically significant spatial-structured unmasking order (p < 0.05, permutation test)
- SAE features show significant activation shifts across timestep bins (timestep-semantic structure)
- LLaDA heads show higher row/column specialization scores than Qwen Phase 1 heads (t-test significant)
- 2D RoPE probing accuracy exceeds 1D RoPE by >10% on row and column prediction

### Deliverables
- `phase3_unmasking_order.ipynb` — Exp A analysis notebook
- `phase3_sae_diffusion.ipynb` — SAE training and feature analysis
- `phase3_attention_heads.ipynb` — Exp B head role comparison
- `phase3_rope_analysis.ipynb` — Exp C probing + CKA
- Figures: unmasking heatmaps, SAE feature timestep profiles, head role distributions, CKA layer curves, probing accuracy curves
- ~1,500-word section draft for arXiv paper

---

## 11. Scoped-Down Fallback Plan

If compute or time is insufficient to complete all three experiments, prioritize in this order:

**Fallback Tier 1 (Days 19–22 only): Unmasking Order + Attention Heads**
- Skip SAE training entirely (most compute-intensive).
- Run Exp A (unmasking order only, no SAE) and Exp B (head role analysis).
- This still produces a novel finding (unmasking order as reasoning strategy) with low compute.
- Estimated cost: ~$20 in GPU hours on Colab.

**Fallback Tier 2 (Days 19–21 only): Unmasking Order Only**
- One core finding: whether diffusion steps are spatially structured reasoning stages.
- Still citable, still novel. Produces 2-3 figures publishable in a larger paper.
- Can be completed on a single A100 session without TPU.

**Fallback Tier 3: Skip Phase 3 entirely**
- Phase 3 is explicitly designated high-risk. Phases 1 and 2 are the core contribution.
- If LLaDA checkpoint is unavailable or setup takes more than Day 20, cut losses.
- Document the attempt, note "future work" in paper, move to Phase 4 synthesis.

---

## 12. Integration With Other Phases

Phase 3 results feed directly into the Phase 4 cross-architecture comparison:

- **Unmasking order** (Exp A) becomes a "reasoning strategy" dimension alongside Qwen's token prediction confidence from Phase 1.
- **Head specialization scores** (Exp B) are compared with Qwen (Phase 1) and any Phase 2 model using the same head role taxonomy.
- **CKA** (Exp C) between LLaDA and Qwen residual streams (using Phase 1 Qwen activations) tests whether bidirectional diffusion and autoregressive AR models converge on similar mid-layer representations — the central thesis of the paper.

The key claim to test is: **"Architecturally different ARC-solving models converge on spatial-relational representations in mid-to-late layers, regardless of whether they use causal or bidirectional attention, and regardless of whether they generate autoregressively or via diffusion."** Phase 3 provides the diffusion half of that comparison.

---

*References:*
- LLaDA: arXiv:2502.09992
- MDLM: arXiv:2406.07524
- Scaling MDMs: arXiv:2410.18514
- ARChitects: arXiv:2601.10904
- ARC-AGI-2: arXiv:2505.11831
- IterRef: arXiv:2511.05562
- Emergence & Evolution (SAE on diffusion): arXiv:2504.15473
- Revelio (k-sparse SAE, diffusion): arXiv:2411.16725
- DiffLens (causal interventions in diffusion): arXiv:2503.20483
- What Does BERT Look At?: arXiv:1906.04341
- RoPE-ViT: arXiv:2403.13298
- CKA: arXiv:1905.00414
- Towards Monosemanticity: Anthropic (2023)
- Practical Review of MI / nnsight: arXiv:2407.02646
