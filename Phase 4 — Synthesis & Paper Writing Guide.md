# Phase 4 — Synthesis & Paper Writing Guide
## ARC-AGI Mechanistic Interpretability: Cross-Architecture Analysis

**Phase Duration:** Days 1–17 (parallel execution — runs throughout the entire project, writing as results arrive)
**Goal:** Synthesize findings from Phases 1–3 into a cohesive, publication-ready arXiv paper
**Mode:** Write-as-you-go — each section is written immediately when its experimental inputs are available

---

## ⚡ Parallel Execution Timeline (Phase 4 Only)

> **Phase 4 runs in parallel with all other phases from Day 1.** It does not wait for experiments to complete — it writes each section as soon as the relevant results are available. See `Master Parallel Execution Plan.md` for the full coordination schedule.

This is the **"write-as-you-go"** model. The key insight: Sections §1–§3 (Introduction, Background, Experimental Setup) can be written from Day 1 with no experimental results. Results sections are written section-by-section as each phase finishes.

| Days | Phase 4 Activity | Waiting For |
|---|---|---|
| **Day 1** | Write §1 Introduction (motivation, contribution summary, ARC-AGI context); set up LaTeX skeleton | Nothing — write now |
| **Days 2–3** | Write §2.1 ARC-AGI Background; §2.2 MI Methods Background (SAE, probing, causal tracing); §2.3 Model Architectures | Nothing — write now |
| **Day 4** | Write §3 Experimental Setup (shared task set, tokenization, MI toolkit, compute setup) | Shared infra confirmed (Day 1) |
| **Days 5–6** | Write §4 Phase 1 Results — Exp A (attention) + Exp B (probing) figures and text | Phase 1 Exps A+B done (Days 2–4) |
| **Day 7** | Write §4 Phase 1 Results — Exp C (SAE) + Exp D (TTT); Write §6 Phase 3 Results (all 3 experiments) | Phase 1 SAE done (Day 6); Phase 3 complete (Day 7) |
| **Days 8–9** | Write §5 Phase 2 Results — Exps A+B+C (convergence, latent, causal tracing) | Phase 2 Exps A–C done (Days 2–6) |
| **Days 10–11** | Update §5 with SAE + CKA results; cross-phase SAE merge for universal features analysis | Phase 2 SAE done (Day 10); all 3 SAEs available |
| **Days 12–13** | Write §7 Cross-Architecture Comparison (the paper's central contribution); produce all 10 figures in final quality | All phases complete (Day 11) |
| **Day 14** | Write §8 Limitations + §9 Conclusion; finalize Abstract | All results known |
| **Day 15** | Full proofread; LaTeX formatting pass; bibliography completion | Full draft exists |
| **Day 16** | Final figure review; check all captions; arXiv metadata preparation | — |
| **Day 17** | **arXiv v1 submission** | — |

**Writing rule:** Never wait for a section to be "perfect." Write a draft immediately with `[TODO: insert X result]` placeholders. The placeholder gets replaced when the experiment finishes — it never delays adjacent sections.

---

## 1. Phase Overview and Synthesis Strategy

Phase 4 is the most intellectually demanding phase of the project. The experimental work is done; the task now is to transform three sets of architecture-specific findings into a single, unified scientific argument. This requires two parallel workstreams: analysis (deciding what the data means) and writing (communicating it clearly).

### 1.1 The Core Synthesis Task

You have mechanistic interpretability findings for three fundamentally different ARC-solving architectures:

- **Qwen 2.5-0.5B** — standard autoregressive LLM with LoRA fine-tuning, analyzed with attention heads, probing classifiers, and SAEs
- **TRM** — a 7M-parameter recursive single-block transformer with test-time training, analyzed with latent state decoding and causal tracing
- **LLaDA-8B** — a masked diffusion LLM with 2D RoPE, analyzed with unmasking order and cross-architecture attention comparison

The synthesis question is not "what does each model do?" but "what does the comparison tell us about spatial reasoning in neural networks?" Frame every result through this lens.

### 1.2 The Two Possible Narratives

Before writing, commit to which narrative the data supports. Both are publishable; only one is true.

**Narrative A — Convergence:** Despite architectural differences, all three models develop similar internal representations for ARC spatial concepts. Row/column structure, color identity, and object adjacency are encoded in geometrically similar ways across LLM, recursive, and diffusion paradigms. This would be a strong universality claim with implications for the Linear Representation Hypothesis and for AI safety (transferable features across architectures).

**Narrative B — Divergence:** Each architecture develops a distinct strategy for spatial reasoning. Qwen uses explicit head specialization; TRM compresses everything into a recursive latent state; LLaDA exploits bidirectional context through iterative unmasking. The finding would argue that spatial cognition is architecture-dependent, not universal, with implications for how we think about alignment and interpretability transfer.

**Narrative C — Mixed (most likely):** Some spatial concepts are universally represented (color identity, basic symmetry detection), while others are architecture-dependent (multi-step relational reasoning, object counting). This nuanced result is often the most scientifically valuable. If you find this, frame it as "partial universality" and identify what determines whether a concept converges.

Document your narrative decision before Day 26 writing begins. Every section should be written to support it.

---

## 2. Cross-Architecture Analysis Framework

### 2.1 CKA/RSA as the Primary Bridge

The CKA (Centered Kernel Alignment) and RSA (Representational Similarity Analysis) scores computed in Phase 3 are the quantitative backbone of the cross-architecture comparison. These provide a principled, model-agnostic measure of whether internal representations are similar.

**Interpreting CKA scores:**
- CKA > 0.7: Strong representational similarity — support Narrative A
- CKA 0.4–0.7: Moderate similarity — support Narrative C (mixed)
- CKA < 0.4: Weak similarity — support Narrative B

Report CKA at three levels: full layer comparison, concept-specific probing subspaces, and attention head functional groups.

### 2.2 Attention Head Taxonomy

Build a shared functional taxonomy across all three architectures. Even if the architectures differ, their attention heads may perform recognizable roles:

| Head Function | Qwen Evidence | TRM Evidence | LLaDA Evidence |
|---|---|---|---|
| Row-tracking | Phase 1 Fig 1 | Phase 2 analysis | Phase 3 analysis |
| Column-tracking | Phase 1 Fig 1 | Phase 2 analysis | Phase 3 analysis |
| Color identity | Phase 1 Fig 1 | Phase 2 analysis | Phase 3 analysis |
| Adjacency detection | Phase 1 Fig 1 | Phase 2 analysis | Phase 3 analysis |

Fill this table from your experimental logs before writing. This table becomes Table 1 in the paper.

### 2.3 Probing-Based Concept Tracking

The layer-wise probing curves (Fig 2 for Qwen) provide a developmental timeline: when does each concept emerge during forward processing? For the synthesis, align these timelines across architectures by normalizing to "fraction of depth" (0.0 = input, 1.0 = output). Ask:

- Do all architectures reach peak probe accuracy for color at the same relative depth?
- Does row/column structure emerge earlier than relational concepts (adjacency, symmetry) in all models?
- Does LLaDA's bidirectional context cause earlier emergence compared to Qwen's causal processing?

These comparisons are the core of the cross-architecture section.

---

## 3. Full Paper Structure and Section-by-Section Writing Guide

**Target length:** 9–10 pages for main body (NeurIPS/ICLR format) + appendix
**Target venue for initial draft:** arXiv preprint, then ICLR 2027 Workshop

### 3.1 Section 1: Introduction (Target: ~600 words)

**Paragraph 1 — The ARC-AGI Challenge:** Open with ARC-AGI as a benchmark that demands genuine abstract reasoning, not pattern-matching. Cite ARC-AGI-2 (arXiv:2505.11831). Establish that recent neural models (fine-tuned LLMs, small recursive models, masked diffusion) have achieved non-trivial ARC performance, but we do not understand *how* — what internal computations they perform.

**Paragraph 2 — The MI Opportunity:** Mechanistic interpretability offers tools to open the black box. Cite "Open Problems in Mechanistic Interpretability" (arXiv:2501.16496) as the program this work contributes to. Briefly name TransformerLens, SAEs, probing, causal tracing as the toolkit.

**Paragraph 3 — The Cross-Architecture Question:** The key novelty: rather than analyzing one model, we ask whether findings generalize. Is spatial reasoning in neural networks universal or architecture-dependent? Cite CKA (arXiv:1905.00414) and "Sparse Autoencoders Reveal Universal Feature Spaces" (arXiv:2410.06981) as prior work on cross-architecture comparison.

**Paragraph 4 — Contributions.** Use a bullet list:
- First mechanistic interpretability study comparing autoregressive LLM, recursive transformer, and masked diffusion on ARC-AGI
- Identification of [N] spatial attention head types across architectures (convergent/divergent finding)
- Layer-wise probing of 5 spatial concepts with cross-architecture timing comparison
- SAE-derived monosemantic ARC features in Qwen 2.5-0.5B
- Causal tracing of task-ID mechanism in TRM
- CKA/RSA cross-architecture similarity scores revealing [convergence/divergence] pattern

### 3.2 Section 2: Background (Target: ~800 words)

**2.1 ARC-AGI Tasks:** Describe the task format (input/output grid pairs, test grid completion). Explain what makes ARC hard: compositional rules, novel generalizations, few-shot regime. Note the five spatial concept categories you probe: row/column position, color identity, object adjacency, local symmetry, global transformation type.

**2.2 Mechanistic Interpretability Methods:** One paragraph each on:
- Attention head analysis (QK/OV framework, cite Anthropic 2021)
- Probing classifiers (cite "Language Models Represent Space and Time," arXiv:2310.02207)
- Sparse Autoencoders (cite "Towards Monosemanticity," "Scaling and Evaluating SAEs" arXiv:2406.04093, Gemma Scope arXiv:2408.05147)
- Causal tracing / activation patching (cite ROME arXiv:2202.05262, IOI Circuit arXiv:2211.00593)
- CKA/RSA (cite arXiv:1905.00414)

**2.3 Target Architectures:** One paragraph each:
- **Qwen 2.5-0.5B:** Standard autoregressive LLM, LoRA fine-tuned on ARC-AGI training set. Cite TTT for ARC (arXiv:2411.07279) for training context. Note tokenization: grid cells as structured tokens.
- **TRM:** Tiny Recursive Model, 7M parameters, single transformer block applied recursively, Samsung AI. Cite "Less is More" (arXiv:2510.04871). Explain test-time training mechanism.
- **LLaDA-8B:** Masked diffusion LLM with 2D RoPE, ARChitects variant. Cite LLaDA (arXiv:2502.09992) and ARChitects Technical Report (arXiv:2601.10904). Explain bidirectional attention and iterative unmasking.

### 3.3 Section 3: Experimental Setup (Target: ~500 words)

This section should be brief but complete enough to ensure reproducibility.

**3.1 Dataset:** ARC-AGI training set (400 tasks), evaluation set (400 tasks). Describe which tasks used for each phase. If you subsampled, state the selection criterion. Mention ARC-AGI-2 (arXiv:2505.11831) for context on benchmark evolution.

**3.2 Toolkit:** List tools with their roles:
- TransformerLens: attention head analysis (Qwen, TRM)
- nnsight: activation patching and causal tracing (all models)
- SAELens: SAE training and feature extraction (Qwen)
- circuitsvis: attention pattern visualization
- sklearn: probing classifier training (logistic regression, 5-fold CV)
- CKA/RSA: implemented via custom code, [link to repo]

**3.3 Probing Protocol:** 5 spatial concepts, logistic regression probes, 5-fold cross-validation, trained on 300 tasks, evaluated on 100 held-out tasks. Report mean accuracy ± std.

**3.4 SAE Training (Qwen only):** TopK SAE, k=[value], hidden dim=[value], trained on [N] ARC grid activations from layer [L]. Cite Gemma Scope (arXiv:2408.05147) for methodology.

### 3.4 Section 4: Phase 1 Results — Qwen 2.5-0.5B (Target: ~700 words)

**4.1 Attention Head Specialization (Fig 1):** Describe the heatmap. Report which heads specialize for which spatial functions. Highlight any surprising findings (heads that handle multiple spatial concepts, heads with no clear spatial role). Connect to "Mathematical Framework for Transformer Circuits" (Anthropic 2021).

**4.2 Layer-wise Probing (Fig 2):** Report when each of the 5 spatial concepts peaks. Is there a depth ordering? (e.g., color identity in early layers, relational concepts in later layers.) Cite "Language Models Represent Space and Time" (arXiv:2310.02207) and "Linear Representation Hypothesis" (arXiv:2311.03658).

**4.3 SAE Features (Fig 3):** Report how many features activate selectively for ARC-relevant concepts. Show 3–5 most interpretable features (Fig 3). Cite "Towards Monosemanticity," "Emergence of Interpretable Concepts in Diffusion Models" (arXiv:2504.15473) for comparison.

**4.4 TTT Weight Changes (Fig 4):** Which layers change most during test-time training? Does the pattern match the probing results (i.e., do the layers with most spatial information show the most TTT modification)? Cite "Specialization after Generalization" (arXiv:2509.24510) for TTT interpretability theory.

### 3.5 Section 5: Phase 2 Results — TRM (Target: ~600 words)

**5.1 Convergence Behavior (Fig 5):** How many recursive steps does TRM need for different task types? Are simpler spatial tasks (color identification) solved in fewer steps than relational tasks (object counting)?

**5.2 Latent State Decoding (Fig 6):** When does TRM "know" the answer? Report probe accuracy by recursive step for each spatial concept. This is the TRM analogue of Qwen's layer-wise probing.

**5.3 Task-ID Mechanism (Fig 7):** Causal tracing results: which activations are causally necessary for correct task identification? Report mean indirect effect scores. Cite ROME (arXiv:2202.05262) and IOI Circuit (arXiv:2211.00593) for methodology.

### 3.6 Section 6: Phase 3 Results — LLaDA-8B (Target: ~600 words)

**6.1 Diffusion Unmasking Order (Fig 9):** Does LLaDA unmask cells in a spatially coherent order? Report whether high-confidence cells (unmasked early) correspond to grid locations that are easier to infer from context. Compare to Qwen's left-to-right causal order.

**6.2 Attention in Bidirectional Context:** Report how LLaDA's attention patterns differ from Qwen's. Does 2D RoPE produce geometrically structured attention (nearby grid cells attending to each other)? Cite "What Does BERT Look At?" (arXiv:1906.04341) for bidirectional attention analysis methodology.

**6.3 Cross-Architecture Comparison Preview:** Briefly note how LLaDA's spatial representations relate to Qwen's (tease the CKA results).

### 3.7 Section 7: Cross-Architecture Comparison (Target: ~900 words) — THE MAIN CONTRIBUTION

This is the most important section. Write it last, after all result sections are drafted.

**7.1 Representational Similarity (Fig 8, 10):** Report CKA and RSA scores for all architecture pairs:
- Qwen vs TRM: [score]
- Qwen vs LLaDA: [score]
- TRM vs LLaDA: [score]

Report scores for full-layer representations and for concept-specific probing subspaces. Discuss what the pattern means (convergence/divergence/mixed).

**7.2 Attention Head Taxonomy:** Present Table 1 (the shared functional taxonomy). Note which spatial attention head types appear in all three models (universal) vs. which are model-specific.

**7.3 Depth Normalization Comparison:** Show the normalized probing curves (fraction of depth) for all 5 concepts across all 3 architectures on a single plot (Fig 10 or an additional figure). The key question: do concepts emerge at the same relative depth?

**7.4 Interpretation:** Argue for your chosen narrative (convergence/divergence/mixed). Connect to the Linear Representation Hypothesis (arXiv:2311.03658), "Sparse Autoencoders Reveal Universal Feature Spaces" (arXiv:2410.06981), and "In-context Learning and Induction Heads" (arXiv:2209.11895).

### 3.8 Section 8: Limitations and Future Work (Target: ~300 words)

Be honest. Common limitations to address:
- Single-task domain (ARC-AGI): findings may not generalize to other spatial reasoning tasks
- Qwen analyzed with LoRA fine-tune, not pre-trained; LoRA may introduce artifacts
- TRM is 7M params; findings may not transfer to larger recursive architectures
- SAE training is noisy; reported monosemantic features may be spurious
- CKA has known limitations for comparing representations of different dimensionalities

Future work: extend to ARC-AGI-2 (cite arXiv:2505.11831), larger LLMs, diffusion models without LoRA, causal intervention at the SAE feature level.

### 3.9 Section 9: Conclusion (Target: ~200 words)

Restate the central question, summarize findings (2–3 sentences), state the main takeaway (convergence/divergence/mixed), and close with the implication for AI safety or interpretability research.

---

## 4. Figure Production Guide

### Figure 1 — Qwen Attention Specialization Heatmap
**What it shows:** For each attention head (x-axis: layer, y-axis: head), a color-coded specialization score for each of 4 spatial categories (row, column, color, adjacency). One heatmap per spatial category, arranged in a 2×2 grid, or a single heatmap with categorical color coding.
**How to generate:** From Phase 1 attention head analysis logs. For each head, compute the probe accuracy for each spatial category on held-out tasks. Normalize per category. Use seaborn `heatmap` with diverging colormap.
**arXiv formatting:** 3.25 inches wide (single column) or 6.75 inches (double column). 300 DPI. PDF preferred. Font size ≥ 8pt for axis labels.

### Figure 2 — Layer-wise Probing Curves (Qwen)
**What it shows:** Line plot with layer depth on x-axis, probe accuracy on y-axis, one line per spatial concept (5 lines). Shaded error bands (±1 std over 5-fold CV).
**How to generate:** From Phase 1 probing logs. Extract accuracy per layer per concept. Plot with matplotlib, use color-blind-safe palette (ColorBrewer "Set1" or "tab10").
**arXiv formatting:** 3.25 inches wide. Legend inside the plot if possible to save space.

### Figure 3 — SAE Feature Visualizations (Qwen)
**What it shows:** For 3–5 selected monosemantic features, show: (a) top activating ARC grid examples, (b) the feature activation heatmap overlaid on the grid, (c) the feature's activation distribution.
**How to generate:** From Phase 1 SAE analysis. Use SAELens feature visualization utilities. For each selected feature, pull top-10 activating examples from the ARC training set.
**arXiv formatting:** Panel figure, 6.75 inches wide. Ensure grid cell colors are distinguishable in grayscale print.

### Figure 4 — TTT Weight Change Heatmap (Qwen)
**What it shows:** Heatmap of parameter change magnitude (L2 norm or cosine similarity to initial) per layer during test-time training, across a sample of ARC tasks (one row per task, one column per layer).
**How to generate:** From Phase 1 TTT analysis. Compute ||W_after - W_before||_F per layer per task. Plot as heatmap. Optionally cluster tasks by type.
**arXiv formatting:** 6.75 inches wide. Use sequential colormap (viridis or plasma).

### Figure 5 — TRM Convergence Curves
**What it shows:** For each task type, plot accuracy vs. number of recursive steps. Show that different task types converge at different rates.
**How to generate:** From Phase 2 TRM analysis. Run TRM with forced early stopping at each step, record accuracy.
**arXiv formatting:** 3.25 inches wide. Separate lines per task type.

### Figure 6 — TRM Latent State Decoding Heatmap
**What it shows:** For each spatial concept (y-axis), probe accuracy at each recursive step (x-axis). Color indicates accuracy. Shows when TRM "knows" each concept.
**How to generate:** From Phase 2 probing logs. Train probes on latent state at each recursive step.
**arXiv formatting:** 3.25 inches wide. Use same color scale as Fig 2 for comparability.

### Figure 7 — TRM Causal Tracing
**What it shows:** Indirect effect score (causal tracing, analogous to ROME) for each attention layer and MLP sublayer of TRM's single block, across recursive steps. Shows which components are causally necessary for task identification.
**How to generate:** From Phase 2 causal tracing logs. Implement activation patching with nnsight. Compute mean indirect effect per component.
**arXiv formatting:** 6.75 inches wide. Follow ROME paper (arXiv:2202.05262) figure style for consistency.

### Figure 8 — Qwen vs TRM RSA/CKA Similarity
**What it shows:** For each pair of layers (Qwen layer i, TRM step j), the CKA similarity score. Shows which layers/steps align representationally.
**How to generate:** From Phase 3 cross-architecture analysis. Compute CKA on matched ARC task activations. Plot as 2D heatmap (Qwen layers × TRM steps).
**arXiv formatting:** 3.25 inches wide.

### Figure 9 — LLaDA Unmasking Order Visualization
**What it shows:** For 2–3 representative ARC tasks, visualize which output cells are unmasked at each diffusion step. Show the order as a sequence of grids with cells colored by the step at which they were first committed.
**How to generate:** From Phase 3 LLaDA analysis. Track mask state at each denoising step. Assign each cell a "commitment step." Render as heatmap overlaid on the output grid.
**arXiv formatting:** Panel figure, 6.75 inches wide. Ensure the colormap communicates temporal order clearly (sequential, e.g., viridis).

### Figure 10 — Cross-Architecture Attention Comparison
**What it shows:** For each of the 4 spatial head categories, a bar chart or heatmap showing the fraction of heads (or the top head's score) for each of the 3 architectures. Enables direct visual comparison of specialization degree.
**How to generate:** Aggregate Phase 1, 2, 3 attention analysis into a shared dataframe. Plot with seaborn `barplot` or `heatmap`.
**arXiv formatting:** 6.75 inches wide. Group bars by architecture or by spatial category (choose based on which comparison is your main point).

---

## 5. Writing Schedule

### Day 26 — Sections 1–3 (Introduction, Background, Setup)
**Morning (3h):** Write Section 1 (Introduction). Do not second-guess the narrative; commit to it. Write the contributions bullet list based on your chosen narrative. Draft the abstract (use the template in Section 6 below).
**Afternoon (3h):** Write Section 2 (Background). Pull from Phase 1–3 planning documents for architecture descriptions. Ensure all 20 references are in your .bib file by end of day.
**Evening (1h):** Write Section 3 (Experimental Setup). This should be mostly factual; use your experimental logs.

**End-of-day checkpoint:** Sections 1–3 drafted, bibliography populated, narrative committed in writing.

### Day 27 — Sections 4–5 (Qwen and TRM Results)
**Morning (3h):** Write Section 4 (Qwen results). Have Figures 1–4 open while writing. Reference each figure at least once in the text.
**Afternoon (3h):** Write Section 5 (TRM results). Have Figures 5–7 open. Ensure claims are supported by specific numbers from your logs.
**Evening (1h):** Generate Figures 1–4 (or polish if already generated). Export to PDF.

**End-of-day checkpoint:** Sections 4–5 drafted, Figures 1–4 finalized.

### Day 28 — Sections 6–7 (LLaDA Results + Cross-Architecture Comparison)
**Morning (2h):** Write Section 6 (LLaDA results). This section may be shorter if Phase 3 data is less complete — that is acceptable.
**Afternoon (4h):** Write Section 7 (Cross-architecture comparison). This is the hardest section. Use the CKA scores and the attention taxonomy table as anchors. Write one subsection at a time. Do not try to write it in one pass.
**Evening (1h):** Generate Figures 8–10. Export to PDF.

**End-of-day checkpoint:** Sections 6–7 drafted, all 10 figures generated.

### Day 29 — Sections 8–9 + Full Figure Polish
**Morning (2h):** Write Section 8 (Limitations) and Section 9 (Conclusion).
**Afternoon (3h):** Read the full draft from start to finish. Fix transitions, ensure narrative coherence, check that every claim is supported by a figure or citation.
**Evening (2h):** Polish all figures to arXiv quality. Check: fonts readable at intended size, no rasterization artifacts, color-blind-safe palette, captions complete.

**End-of-day checkpoint:** Full draft complete, all figures finalized.

### Day 30 — Final Review + arXiv Submission
**Morning (2h):** Final proofread. Check: abstract matches paper, contributions match findings, all figures referenced in text, all references cited in text.
**Late morning (2h):** LaTeX compile check. Fix any compilation errors. Confirm PDF renders correctly. Check arXiv format requirements (no extra packages, no custom fonts).
**Afternoon (2h):** Upload to arXiv. Set to "announced" or "on hold" depending on priority strategy (see Section 9).

---

## 6. Title Options and Abstract Template

### Title Options

**Option A (Convergence narrative):**
"Universal Spatial Representations in ARC-AGI Solvers: A Cross-Architecture Mechanistic Interpretability Study"

**Option B (Divergence narrative):**
"Architecture Shapes Spatial Cognition: Mechanistic Interpretability Across ARC-AGI Solvers"

**Option C (Neutral/mixed, recommended for pre-registration):**
"Inside ARC-AGI Solvers: Mechanistic Interpretability of Spatial Reasoning Across Autoregressive, Recursive, and Diffusion Architectures"

**Option D (Punchy):**
"How Do Neural Networks Solve ARC? A Mechanistic Interpretability Comparison Across Three Architectures"

### Abstract Template

```
Abstract (target: 200–250 words)

Understanding how neural networks solve abstract spatial reasoning tasks is a
central challenge in mechanistic interpretability. We present the first
systematic cross-architecture study of internal representations in ARC-AGI
solvers, analyzing three fundamentally different architectures: Qwen 2.5-0.5B
(autoregressive LLM with LoRA fine-tuning), TRM (7M-parameter recursive
transformer with test-time training), and LLaDA-8B (masked diffusion LLM with
2D RoPE). Using a standardized toolkit — TransformerLens, SAELens, nnsight,
and linear probing — we characterize how five spatial concepts (row/column
position, color identity, object adjacency, local symmetry, and global
transformation type) are represented across these architectures.

[FINDING 1: Attention specialization — e.g., "We find that all three models
develop attention heads specialized for row/column tracking, suggesting
convergent spatial inductive biases."]

[FINDING 2: Probing curves — e.g., "Layer-wise probing reveals that color
identity is encoded at shallower depths than relational concepts across all
three architectures, despite their different processing paradigms."]

[FINDING 3: Cross-architecture CKA — e.g., "CKA analysis reveals moderate
representational similarity (mean CKA = 0.XX) between Qwen and TRM, but
lower similarity between the autoregressive models and LLaDA, suggesting
that bidirectional context produces qualitatively different spatial
representations."]

Our findings [support/challenge] the universality of spatial representations
in neural networks and have implications for interpretability transfer across
architectures. Code and data available at [REPO URL].
```

---

## 7. Submission Guide

### 7.1 LaTeX Setup

Use the NeurIPS 2026 style file for initial drafts (widely accepted as a neutral format). Switch to ICLR style for workshop submission.

**Essential packages:**
```latex
\usepackage{natbib}        % bibliography
\usepackage{graphicx}      % figures
\usepackage{booktabs}      % professional tables
\usepackage{amsmath}       % math
\usepackage{microtype}     % typography
\usepackage{hyperref}      % links (required for arXiv)
\usepackage{cleveref}      % smart cross-references
```

**Figure inclusion best practice:**
```latex
\begin{figure}[t]  % 't' = top of page preferred
  \centering
  \includegraphics[width=\linewidth]{figures/fig1_attention_heatmap.pdf}
  \caption{Attention head specialization in Qwen 2.5-0.5B. Each cell shows
  the probe accuracy for a given spatial concept (rows) in a given attention
  head (columns). Heads are identified by (layer, head) index. [DESCRIPTION
  OF KEY FINDING].}
  \label{fig:qwen_attention}
\end{figure}
```

### 7.2 Appendix Structure

The appendix is where peer reviewers look when they are skeptical. Populate it thoroughly.

**Appendix A — Extended Experimental Details:** Full hyperparameters for SAE training, LoRA configuration, probe training details, TRM training settings.

**Appendix B — Additional Figures:** Extended attention head visualizations, full probing curve matrix (all concepts × all layers), SAE feature gallery (all features, not just top-5).

**Appendix C — Statistical Analysis:** Full significance testing for CKA scores (bootstrap confidence intervals), probe accuracy variance across folds.

**Appendix D — Task Examples:** 5–10 ARC task examples with model predictions and attention maps, to ground the reader in the task domain.

**Appendix E — Failure Analysis:** Cases where the models fail and what the internal representations look like on failure cases. This is often the most interesting content for mechanistic interpretability.

### 7.3 Citation Management

Use a single `references.bib` file. All 20 target references should be in arXiv format (using the `@article` type with `archivePrefix = {arXiv}` and `eprint = {XXXX.XXXXX}` fields).

**Template for arXiv reference:**
```bibtex
@article{elhage2021mathematical,
  title={A Mathematical Framework for Transformer Circuits},
  author={Elhage, Nelson and Nanda, Neel and Olsson, Catherine and others},
  journal={Transformer Circuits Thread},
  year={2021},
  archivePrefix={arXiv},
  note={\url{https://transformer-circuits.pub/2021/framework/index.html}}
}
```

Use Zotero or a plain .bib file. Do not use Mendeley for arXiv submissions (export fidelity is unreliable). Run `bibtex` or `biber` and inspect the output for malformed entries before final compile.

---

## 8. Venue Guide

### 8.1 arXiv Preprint (Immediate)
**Submission deadline:** No deadline — submit when ready (Day 30 target)
**Category:** cs.LG (primary), cs.AI (secondary), cs.NE (optional)
**Fit:** Perfect. arXiv is where mechanistic interpretability work circulates before peer review.
**Checklist:**
- [ ] Paper compiles cleanly from a fresh LaTeX environment
- [ ] All figures included in the .tar.gz upload
- [ ] Abstract under 1920 characters (arXiv limit)
- [ ] No author-identifying information if planning double-blind submission elsewhere
- [ ] Code repository linked (even if private at submission time)

### 8.2 COLM 2026 (Conference on Language Modeling)
**Approximate deadline:** March 2026 (check colmweb.org for exact date)
**Fit:** High. COLM is the premier venue for language model research and has strong mechanistic interpretability representation. LLaDA and Qwen analyses are directly relevant.
**Expected page limit:** 9 pages + references
**Review process:** Double-blind
**Action:** If paper is strong, submit here. This would be the highest-impact venue on the list.

### 8.3 NeurIPS 2026 Workshops
**Approximate deadline:** September 2026 (workshop paper deadlines vary)
**Fit:** High for mechanistic interpretability workshops (e.g., MINT, Attributing Model Behavior). Good for community feedback before ICLR.
**Expected page limit:** 4–6 pages
**Review process:** Light, often single-blind
**Action:** Submit a condensed version focusing on the cross-architecture comparison (Section 7). This is a good intermediate milestone.

### 8.4 ICLR 2027 Workshop on Mechanistic Interpretability
**Approximate deadline:** October 2026
**Fit:** Perfect venue match. The MI workshop at ICLR is the field's most relevant community event.
**Expected page limit:** 4–8 pages
**Action:** This should be the primary peer-reviewed target. Polish the paper based on any NeurIPS workshop feedback before submitting here.

### 8.5 Venue Priority Recommendation

1. **arXiv** — Day 30 (establishes priority)
2. **COLM 2026** — March 2026 (if paper is strong; aggressive but worthwhile)
3. **NeurIPS 2026 Workshop** — September 2026 (community feedback)
4. **ICLR 2027 Workshop** — October 2026 (primary peer-reviewed target)

---

## 9. Early arXiv Strategy: When to Post

**Recommendation: Post to arXiv after Phase 2 is complete (Day ~20), before Phase 3.**

**Rationale:**
- The Qwen + TRM comparison (Phases 1–2) is already a publishable unit. It is the first mechanistic comparison of an autoregressive LLM and a recursive transformer on ARC-AGI.
- Posting early establishes the timestamp of your contribution. If another group publishes similar findings between Day 20 and Day 30, your earlier arXiv preprint protects priority.
- The Phase 2 arXiv draft can be titled something like: "Mechanistic Interpretability of ARC-AGI Solvers: A Comparative Study of Autoregressive and Recursive Architectures" — and updated to v2 when Phase 3 (LLaDA) is complete.

**How to execute the two-stage arXiv strategy:**
1. **v1 (Day ~20):** Sections 1–6 (Introduction, Background, Setup, Qwen results, TRM results, Qwen-vs-TRM comparison). Note in the abstract: "Analysis of a third architecture (masked diffusion) is ongoing and will appear in v2."
2. **v2 (Day 30):** Full paper with all three architectures and the complete cross-architecture comparison.

This strategy is standard in fast-moving fields and is explicitly sanctioned by arXiv's versioning system.

---

## 10. Potential Contribution Claims

These are defensible claims regardless of which narrative (convergence/divergence/mixed) emerges from the data.

**Universal claims (always true regardless of finding):**
1. First mechanistic interpretability study to compare autoregressive, recursive, and masked diffusion architectures on the same task domain (ARC-AGI)
2. First application of SAEs to an ARC-solving LLM
3. First causal tracing analysis of the TRM's task-identification mechanism
4. Novel cross-architecture CKA comparison using ARC spatial concept probing subspaces

**Conditional claims (depend on finding):**
- If convergence: "We establish that certain spatial representations are architecture-invariant, providing evidence for universal neural substrates of abstract spatial reasoning"
- If divergence: "We show that spatial reasoning strategies in neural networks are architecture-dependent, challenging universality assumptions and highlighting the importance of architectural choice in interpretability research"
- If mixed: "We identify a spectrum of concept universality: low-level spatial features (color, position) converge across architectures, while higher-order relational concepts diverge, suggesting that universality is concept-dependent rather than architecture-dependent"

---

## Quick-Reference Checklists

### Pre-Writing Checklist (Complete before Day 26)
- [ ] Narrative committed (convergence/divergence/mixed)
- [ ] All Phase 1–3 experimental logs organized and accessible
- [ ] CKA/RSA scores computed for all architecture pairs
- [ ] All 20 references in .bib file
- [ ] Figure data (numpy arrays, dataframes) exported and labeled
- [ ] Code repository initialized (even if private)
- [ ] LaTeX template compiles cleanly

### Pre-Submission Checklist (Day 30)
- [ ] Abstract ≤ 1920 characters
- [ ] All figures referenced in text
- [ ] All figures have captions with key finding stated
- [ ] All tables have captions and column headers
- [ ] All citations appear in bibliography
- [ ] No TODO or placeholder text remaining
- [ ] LaTeX compiles without warnings (or only ignorable warnings)
- [ ] PDF renders correctly in Adobe Reader and browser PDF viewers
- [ ] Repository URL included in paper
- [ ] Author affiliations and contact information correct
- [ ] arXiv category and metadata filled in

---

*Document version: Phase 4 Planning, prepared Day 25. Update CKA scores and narrative field in Section 2.1 once Phase 3 analysis is complete.*
