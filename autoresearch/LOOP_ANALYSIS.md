# TRM Interpretability Loop — State & Proposal Analysis

_Snapshot of the autonomous autoresearch loop running in `arc-phase2-trm/autoresearch/`._

## 1. What the loop is

A closed propose→run cycle with a single LLM "scientist" (`claude-sonnet-4-6`) at the
center. Every ~10 completed jobs, `exp_propose` fires:

```
every ~10 completed jobs
  └─► exp_propose
        ├─ reads last 30 rows of done.jsonl
        ├─ build_prompt() keeps ONLY the last 12 "ok" results, compressed to
        │    7 fields: {t, stem, fc, kls, klv, halt, slow}
        ├─ shells out: claude -p --model claude-sonnet-4-6  (budget $0.15/call)
        │    → "Propose 2–5 followup experiments on the most interesting tasks above"
        ├─ validates each: type ∈ {a,c,d}?  stem ∈ canonical_200?
        └─ queues survivors → which feed the next window
```

Three experiment kernels are proposable:

| Kernel | Question | Output fields |
|---|---|---|
| `exp_a_convergence` | Does the answer state stabilize across iterations? | per-iter argmax-Hamming, entropy, `first_correct_step` |
| `exp_c_ablate` | How much does the `task_id` token steer the output? | KL(real vs blanked task-id) per step, `peak_kl_step/value` |
| `exp_d_halt` | Does the model know when to stop? | `first_would_halt_step` |

TRM runs on CPU (7M params). The loop has been up continuously for ~7 days.

## 2. Live status

| Metric | Value |
|---|---|
| Loop uptime | ~7d (single CPU core, ~2.5 GB RSS) |
| Jobs done / failed / in-flight | 37,260 / 0 / 0 |
| Per-experiment coverage | exp_a, exp_c, exp_d: **200/200** canonical tasks each |
| Proposer calls | 3,681 |
| Accepted proposals | **14,019** |
| Rejected | 1,675 (10.7%, all "unknown type" — hallucinated experiment names) |
| Proposer API spend | **~$175** |

**Not implemented vs. the written Phase 2 plan:** Exp B (latent linear probes), the
real Exp D (TopK **SAE** training), and Exp E (Qwen↔TRM **CKA**). These are not
proposable because `CORE_TYPES` only contains the three cheap kernels. Phase 4's paper
blocks on Exp E. No activation shards, SAE checkpoints, or CKA matrix exist; no GCS
handoff has occurred.

## 3. The emerging scientific finding

The proposer converged early onto a real, publishable phenomenon: a **dissociation
between task-ID sensitivity (KL), halting (q_halt), and correctness (first_correct)**,
anchored on a stable cast of ~10 hard tasks (`2bcee788`, `2281f1f4`, `6150a2bd`,
`82819916`, `2dc579da`, …). Example: `2bcee788` shows peak KL ≈ 132 at step 0 (strong
task-ID routing) yet **never halts and is never correct** — task conditioning is read
but the halt gate stays flat. This is a clean dissociation worth a figure.

## 4. Were the proposals good?

- **Individually: yes.** Each rationale is locally sound and on-topic; the proposer
  reads the compressed metrics correctly and asks a sensible follow-up. The 10.7%
  rejects are pure schema slips (inventing a 4th experiment type), not bad science.
- **As a portfolio: no.** It found the dissociation early, then spent thousands of
  calls **re-confirming the same hypothesis** instead of advancing. The experiments are
  deterministic, so re-runs return identical numbers and yield ~0 new information.
- **Coverage vs. depth: inverted.** All 200 tasks were touched, but the top-10 tasks
  absorb ~24% of all effort, while the genuinely missing work (Exp B / SAE / CKA) was
  never proposable.

## 5. Redundancy of concepts (the headline)

**Lexical redundancy ≈ 0. Conceptual redundancy ≈ total.**

| Measure | Value |
|---|---|
| Unique `(type, stem)` cells ever proposed | **597** |
| Total accepted proposals | 14,019 |
| **Mean re-proposals per cell** | **23.5×** |
| Most-reproposed single cell | `exp_a_convergence / 2bcee788` — **243×** |
| Verbatim-duplicate rationales | **0.0%** (every rationale is freshly worded) |

Mined for *concepts* rather than wording, the 14k rationales collapse onto a handful of
ideas:

| Concept | Share of rationales |
|---|---|
| KL / task-id ablation | 81% |
| halt / q_halt | 72% |
| convergence / does answer stabilize | 46% |
| "step 0 / early" timing | 40% |
| first_correct / "never correct" | 32% |

Recurring 4-grams expose a hard template: _"…convergence curve will show/reveal
**whether the answer state**…"_ (~4,300 hits). The proposer paraphrases a single
research question thousands of ways — fresh sentence every time, same idea every time.

**Honest characterization:** this is not 14,000 experiments — it is ~3–5 genuine
hypotheses asked ~3,000 times each, in 14,000 distinct sentences.

## 6. Root cause

Redundancy is the guaranteed steady state of three design choices:

1. **12-result goldfish memory.** `build_prompt` does `recent_done[-12:]`. The
   proposer's entire view of a 14k-experiment program is the last ~3 minutes of work —
   which is its own previous output. It cannot know a question was already answered.
2. **Dedup that cannot fire.** `spec.id = f"{type}-{sha1(payload)[:12]}-{uuid4[:4]}"`.
   The random suffix (and an `epoch` that increments every call, baked into the hash)
   means two identical `(type, stem)` proposals always get different IDs. Concept-level
   dedup never collides.
3. **No novelty/"answered" signal.** Nothing rewards a new question over a re-asked
   one, and deterministic experiments never surprise the loop into a new direction.

Result: a self-reinforcing attractor — the window fills with the hard tasks → they are
re-flagged "most interesting" → re-queued → they refill the window. Stable and
cheap-per-call, but globally near-zero-yield.

## 7. Highest-leverage fixes

1. Pass the proposer a **coverage map** (per-cell run counts from `done.jsonl`) and
   instruct it to prefer under-studied cells. Kills the goldfish memory.
2. Add **concept-level dedup**: skip queuing `(type, stem)` already run N times unless
   the result changed.
3. **Widen `CORE_TYPES`** (or add probe/SAE/CKA kernels) so the loop can pursue the
   missing Exp B/D/E the paper actually needs.

---

_Generated from analysis of `autoresearch/state/` (done.jsonl, results/exp_propose/)._
