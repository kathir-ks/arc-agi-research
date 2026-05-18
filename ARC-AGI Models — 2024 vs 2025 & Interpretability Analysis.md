# ARC-AGI Models: 2024 vs 2025 Comparison & Interpretability Analysis

---

## Part 1: 2024 Models vs 2025 Models

### Overview Snapshot

| | ARC Prize 2024 | ARC Prize 2025 |
|---|---|---|
| **Dataset** | ARC-AGI-1 (400 tasks) | ARC-AGI-2 (harder, novel tasks) |
| **Top Score** | 55.5% (MindsAI, ineligible) / 53.5% (ARChitects, winner) | 24.03% (NVARC) |
| **Grand Prize** | Unclaimed | Unclaimed |
| **Teams** | ~1,300 | 1,455 |
| **Dominant Theme** | Test-Time Training (TTT) emerges | Refinement Loops + TRM surprise |
| **Papers Submitted** | 47 | 90 |

> The score drop from ~55% (2024) to ~24% (2025) reflects how much harder ARC-AGI-2 is — it was specifically designed to resist the approaches that cracked ARC-AGI-1.

---

### 2024 — Top Models & Solutions

#### 🥇 1st Place: The ARChitects (Daniel Franzen & Jan Disselhoff)
- **Score:** 53.5% (private eval)
- **Base Model:** Mistral-NeMo-Minitron-8B (autoregressive decoder LLM)
- **Approach:** Test-Time Training (TTT) — fine-tune the model on each individual test task at inference time using augmented input-output pairs. Novel augmentations + stability-based selection criterion (solutions that remain stable under augmentation are preferred).
- **Fine-tuning:** Full fine-tuning + LoRA
- **Open source:** Yes — weights on HuggingFace (`da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit`)

#### 🥈 MindsAI (Highest score, ineligible — did not open source)
- **Score:** 55.5% (private eval) — actual top performer
- **Base Model:** Salesforce T5-series (encoder-decoder architecture)
- **Approach:** Pioneered TTT for ARC-AGI starting in 2023. TTT combined with large-scale synthetic data pretraining.
- **Open source:** ❌ No

#### 🥉 Ryan Greenblatt (Notable public submission)
- **Score:** ~42% on ARC-AGI-Pub
- **Base Model:** GPT-4o (API-based, not open)
- **Approach:** LLM-guided program synthesis — GPT-4o generates and evaluates thousands of Python programs per task, selecting the one that correctly maps all input-output pairs.
- **Open source:** Partial (method described publicly)

#### 📄 1st Place Paper: Li et al. — "Combining Induction and Transduction"
- **Score:** ~47.5% (semi-private eval)
- **Base Model:** ~8B parameter LLM with TTT
- **Key insight:** Ensemble of induction (program synthesis) and transduction (direct prediction) — they solve largely distinct task subsets, so combining them significantly boosts scores.

#### 📄 Ice Cuber Style (DSL / Rule-based — legacy but still relevant)
- **Approach:** Brute-force exhaustive search over a hand-coded Domain Specific Language (DSL). No neural network. Won ARC-AGI-1 (2020 Kaggle) outright.
- **Limitation:** Doesn't scale to ARC-AGI-2's novel task types.

---

### 2025 — Top Models & Solutions

#### 🥇 1st Place: NVARC (Ivan Sorokin & Jean-François Puget, NVIDIA)
- **Score:** 24.03%
- **Base Model:** Qwen2-VL-4B (vision-language, autoregressive) + TRM (7M params) — ensemble
- **Approach:** Improved ARChitects-style TTT on Qwen, combined with a TRM component. Generated 100,000+ synthetic ARC-like puzzles. Cost ~$0.20 per task.
- **Open source:** Partial

#### 🥈 2nd Place: The ARChitects (returning team)
- **Score:** 16.53%
- **Base Model:** LLaDA-8B (Large Language Diffusion model — masked diffusion, NOT autoregressive)
- **Approach:** Replaced LLaDA's standard RoPE positional encoding with a **2D variant** suited to grid inputs. Developed recursive latent sampling to iteratively refine predictions. Perspective-based scoring across multiple grid orientations.
- **Key shift from 2024:** Moved from autoregressive (Mistral) → masked diffusion (LLaDA). Fundamentally different generation process.
- **Open source:** Technical report available

#### 🥉 3rd Place: MindsAI
- **Score:** 12.64%
- **Base Model:** Not fully disclosed
- **Approach:** Heavily engineered TTT pipeline — augmentation ensembles, tokenizer dropout, new pretraining tricks.
- **Open source:** ❌ No

#### 📄 1st Place Paper: Tiny Recursive Model / TRM (Jolicoeur-Martineau, Samsung AI)
- **Score:** ~45% on ARC-AGI-1 / ~8% on ARC-AGI-2
- **Parameters:** Only 7 million (!)
- **Architecture:** Single transformer block used recursively. Separate answer state and latent state. Starts with random output + random latent, refines both through many iterations.
- **Key finding:** Recursion can substitute for depth and model size.
- **Open source:** ✅ Fully open, MIT license (`SamsungSAILMontreal/TinyRecursiveModels`)

#### 📄 2nd Place Paper: SOAR (Pourcel, Colas & Oudeyer)
- **Score:** Up to 52% on ARC-AGI-1
- **Architecture:** Evolutionary program synthesis — an LLM fine-tuned on its own successful search traces. Self-improving loop: find solution → add to training set → fine-tune → repeat.
- **No DSL required:** Learns to write Python programs directly.
- **Open source:** Yes

#### 📄 3rd Place Paper: CompressARC
- **Score:** ~20–34% on ARC-AGI-1 / ~4% on ARC-AGI-2
- **Architecture:** MDL (Minimum Description Length) based. Neural code golf — trains a tiny network per puzzle to compress the input-output mapping. No pretraining, no external data.
- **Approach:** Framing ARC as a compression problem.

---

### Side-by-Side Model Comparison

| Model | Year | Params | Architecture Type | Base Model | Score (ARC-2) | Open Source |
|---|---|---|---|---|---|---|
| ARChitects 2024 | 2024 | 8B | Autoregressive LLM + TTT | Mistral-NeMo-Minitron | N/A (ARC-1 era) | ✅ |
| MindsAI 2024 | 2024 | ~? | Encoder-Decoder + TTT | T5-series | N/A | ❌ |
| Greenblatt | 2024 | ~? | Program Synthesis | GPT-4o | N/A | Partial |
| NVARC | 2025 | 4B + 7M | Autoregressive VLM + TRM | Qwen2-VL + TRM | 24.03% | Partial |
| ARChitects 2025 | 2025 | 8B | Masked Diffusion LLM | LLaDA-8B | 16.53% | Partial |
| MindsAI 2025 | 2025 | ~? | TTT Pipeline | Undisclosed | 12.64% | ❌ |
| TRM | 2025 | 7M | Single-block Recursive Transformer | Scratch | ~8% (ARC-2) | ✅ MIT |
| SOAR | 2025 | ~? | Evolutionary Program Synthesis | Open LLM | ~52% (ARC-1) | ✅ |
| CompressARC | 2025 | Tiny | MDL Neural Compression | Scratch | ~4% (ARC-2) | ✅ |

---

## Part 2: Model Architectures & Interpretability Potential

### Architecture Type 1: Autoregressive Decoder LLM (with TTT)
**Examples:** ARChitects 2024 (Mistral-NeMo-Minitron-8B), NVARC's Qwen2-VL-4B, your own Qwen 2.5-0.5B + LoRA

**How it works:**
These are standard transformer decoder models. The ARC grid is tokenized as a flat text sequence (e.g., row by row, color numbers separated by spaces). Test-Time Training (TTT) fine-tunes the model weights on each specific test task before inference — essentially overfitting to the few-shot examples in the task itself.

**Interpretability Potential: ⭐⭐⭐⭐ (High)**

- TransformerLens works natively with these architectures
- Attention patterns can be visualized per layer/head to see which grid tokens attend to which
- SAEs can be trained on residual stream activations to find ARC-specific features (color features, shape features, positional features)
- Causal tracing / activation patching to find which components are responsible for correct answers
- Probing classifiers to test whether layers encode spatial concepts (e.g., "top row", "mirrored", "same color as input")
- **TTT itself is interpretable:** you can compare weights before and after TTT to see what changed — which features the model "learned" per task
- **Your Qwen 0.5B is an ideal starting point** — small enough to run full analysis, already ARC-fine-tuned

**Key Questions to Explore:**
- Do attention heads specialize in spatial relationships (row attention vs. column attention)?
- Does TTT create task-specific "circuits" or just shift existing representations?
- Are color concepts encoded linearly in the residual stream?

---

### Architecture Type 2: Masked Diffusion LLM (LLaDA)
**Examples:** ARChitects 2025 (LLaDA-8B with 2D RoPE)

**How it works:**
Unlike autoregressive models that generate tokens left-to-right, LLaDA predicts all masked tokens simultaneously. The forward process gradually masks the output grid; the reverse process unmasks it iteratively. The ARChitects modified LLaDA with 2D positional encoding so the model understands row/column structure, and used recursive latent sampling to refine answers over multiple passes.

**Interpretability Potential: ⭐⭐⭐ (Medium — novel territory)**

- TransformerLens does NOT natively support diffusion LLMs — would need adaptation (TransformerBridge may help)
- Attention is bidirectional (no causal mask) — richer but more complex to interpret
- SAEs can still be applied to residual stream activations
- The iterative unmasking process offers a unique window: you can track how the model's beliefs about the output evolve across diffusion steps
- 2D positional encoding modification is directly interpretable — does the model learn to attend along rows vs. columns?
- **Largely unexplored** — almost no mechanistic interpretability work on diffusion LLMs exists

**Key Questions to Explore:**
- How does bidirectional attention change the nature of spatial reasoning circuits vs. causal models?
- Do the diffusion steps correspond to interpretable "refinement stages" (e.g., first get colors right, then get positions right)?
- What role does the 2D RoPE play — do specific heads use 2D position information?

---

### Architecture Type 3: Tiny Recursive Model (TRM)
**Examples:** TRM by Jolicoeur-Martineau (7M params), used as a component in NVARC

**How it works:**
A single small transformer block is applied repeatedly (recursively) to the same input. Unlike stacking many layers, the same weights are reused every pass. The model maintains two states: an answer state (the predicted output grid) and a latent state (hidden reasoning scratchpad). Both are initialized randomly and refined over many recursive steps. The model is trained with deep supervised refinement — supervised at every recursive step, not just the final one.

**Interpretability Potential: ⭐⭐⭐⭐⭐ (Highest)**

- Only 7M parameters — the entire model is small enough to exhaustively analyze
- Fully open source (MIT license)
- The recursive computation is explicitly observable: you can intercept the model at any iteration and inspect the answer and latent states
- Because the same weights are reused every pass, any "algorithm" the model learns must be stable across iterations — it cannot rely on positional tricks specific to one layer
- The puzzle-identity ablation (replace task ID → zero accuracy) suggests a specific, traceable mechanism for task conditioning
- SAEs can be trained on the single transformer block's activations across all iterations
- No pretraining noise: the model was trained purely on ARC-like data, so all features are task-relevant
- **Virtually no interpretability work done on it** — completely open research territory

**Key Questions to Explore:**
- What does the latent state represent at each iteration? Is it tracking object boundaries, color assignments, or spatial transforms?
- How does the model use the puzzle ID token — is it stored in the residual stream throughout all iterations?
- Do the iterations correspond to human-like reasoning steps (e.g., "find objects" → "find transformation" → "apply transformation")?
- Why does recursion work? Does each pass incrementally reduce uncertainty in specific regions of the grid?

---

### Architecture Type 4: Program Synthesis / DSL
**Examples:** Greenblatt (GPT-4o + Python search), SOAR, Ice Cuber (pure DSL), CompressARC

**How it works:**
Instead of predicting the output grid directly, the model generates a program (Python code, or expressions in a domain-specific language) that correctly transforms the input into the output. The program is verified by running it on the training examples. If it passes, it is applied to the test input.

**Interpretability Potential: ⭐⭐⭐⭐⭐ (Inherently interpretable — different kind)**

- The programs themselves are the explanation — no need to probe a black box
- For SOAR: the LLM backbone generating programs can be analyzed with standard mechanistic interpretability tools
- The interesting question is not "what does the model compute" but "what types of programs does it prefer to generate" — a behavioral / distributional question
- CompressARC is particularly interesting: the MDL framing means shorter programs = more general solutions. The length of the learned program is a direct measure of task complexity.
- **Not suitable for standard MI (attention analysis, SAEs)** — but ideal for program analysis, DSL grammar analysis, and inductive bias studies

---

### Architecture Type 5: Vision-Language Models (VLM)
**Examples:** NVARC's Qwen2-VL-4B, Qwen-VL experiments in your notes

**How it works:**
Grid inputs are encoded as images (or image-like tokens) by a vision encoder, then passed to a language decoder. The model receives visual spatial information directly rather than relying on the decoder to infer 2D structure from flat token sequences.

**Interpretability Potential: ⭐⭐⭐ (Medium)**

- More complex than pure-text LLMs — two components (vision encoder + language decoder)
- Vision encoder attention maps can be visualized (which image patches the model attends to)
- Cross-attention between vision tokens and language tokens can be studied
- TransformerBridge now supports LLaVA and similar architectures — may support Qwen-VL
- Key question: does the visual encoder actually capture object-level structure, or does it just tokenize the grid similarly to text?
- **Underexplored for ARC** — interesting research gap given that VLMs underperform pure-text models on ARC despite having richer visual input

---

### Summary: Interpretability Recommendation Matrix

| Architecture | MI Tooling | Difficulty | Novelty | Recommended For |
|---|---|---|---|---|
| Autoregressive LLM (Qwen 0.5B) | TransformerLens, SAE | Low | Medium | Starting point — well-tooled |
| Autoregressive LLM (Mistral 8B) | TransformerLens, SAE | Medium | Medium | Richer circuits, harder to run |
| TRM (7M) | Custom + TransformerLens | Low | Very High | Best for novel findings |
| Masked Diffusion (LLaDA) | Needs adaptation | High | Very High | High risk, high reward |
| VLM (Qwen-VL) | Partial tooling | Medium-High | High | Vision-specific questions |
| Program Synthesis (SOAR) | Standard MI on backbone | Medium | Medium | Program generation behavior |
| DSL / CompressARC | No standard MI tools | — | High | Program analysis, not neural MI |

---

## Recommended Research Path for You

1. **Phase 1 — Ground Truth**: Run TransformerLens + SAEs on your **Qwen 0.5B** (already have it). Establish baseline understanding of what ARC-fine-tuned autoregressive LLMs encode.

2. **Phase 2 — Novel Discovery**: Move to **TRM**. Apply the same SAE + probing toolkit. Compare what a recursive model learns vs. a stacked-depth model. This is untouched territory.

3. **Phase 3 — Architectural Contrast**: Pick one of LLaDA (masked diffusion) or Qwen-VL to study a fundamentally different architecture. Compare findings against Phases 1 and 2.

4. **Synthesis**: Write up whether different ARC-solving architectures develop convergent or divergent internal representations — i.e., does spatial reasoning have a universal neural substrate, or is it architecture-dependent?

---

*Compiled: May 2026 | Sources: ARC Prize 2024 & 2025 Technical Reports, ARChitects Technical Reports, TRM papers (arXiv 2510.04871, 2512.11847), LLaDA paper (arXiv 2502.09992)*
