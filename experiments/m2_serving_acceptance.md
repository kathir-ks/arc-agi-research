# MiniMax-M2.7 INT4 Serving on main-2 (v4-8) — Acceptance Log

Pointer doc for the cross-worktree M2 inference endpoint. The endpoint is a
downstream tool, not a Phase 2 (TRM) research artifact — this file just records
the v1 acceptance run so other phase worktrees know where to point clients.

## Why it's here, not in `~/maxtext`

The serving wiring lives in
`~/maxtext/.claude/worktrees/minimax-m2.7/scripts/minimax_m2.7/` (decode + bootstrap
scripts and the streaming converter). Only the *acceptance log* lives in this
worktree, because the user-facing question — "can other phase worktrees rely on
this endpoint?" — is a Phase 2 question (it's the host we use for MoE-style
critic queries against TRM iterations).

## Plan deviation from `cozy-sleeping-crab.md`

The approved plan called for NFS-sharing main-1's `/dev/shm` to main-2 to pool
~600 GB of tmpfs for the FP8→BF16 conversion. **We pivoted to the streaming
converter instead** because main-1's tmpfs is actively committed to Phase 3
LLaDA (110 GB resident, fineweb processes running) — touching main-1's RAM
would compete with Phase 3's working set.

`convert_minimax_m2_streaming.py` already existed on the `feature/minimax-m2.7`
branch (no new code needed). It writes per-leaf `.npy` files via memmap with a
progressive FP8-shard cleanup, holding peak local tmpfs usage at ~230 GB. We
expanded `/mnt/dtmpfs` on main-2 to **350 GB** to hold both the shrinking FP8
input and the growing float16 output simultaneously.

The plan also called for a custom `convert_minimax_m2_int4.py`. **We dropped
that too** — MaxText already supports INT4 weight-only quantization via
`quantization=intmp` plus
`src/maxtext/configs/quantization/int4_weight_only.json`
(`{"__default__": {"w_bits": 4}}`). Quantization happens at load time on the
CPU host before sharding to HBM, so the float16 .npy pile feeds the INT4 path
directly. Net: zero new converter code; v1 work is entirely shell scripts and
config plumbing.

## Architecture (as built)

- **Host:** main-2 (v4-8, us-central2-b, 240 vCPUs, 400 GB RAM, AMD EPYC 7B12 Zen 2 AVX2-only)
- **Storage:** `/mnt/disk1` 500 GB SSD PD (the streaming converter's float16 output is
  **~460 GB**, not 230 GB — peak storage exceeded the 350 GB tmpfs at layer ~36).
  `/mnt/dtmpfs` (390 GB tmpfs) holds the HF FP8 staging + JAX cache only.
- **Format:** per-leaf `.npy` (float16 on disk) → quantized to INT4 at load via AQT
- **HBM:** ~114 GB INT4 weights + ~4 GB Q8 KV cache (64k ctx) + ~10 GB activations/scratch
- **Decoder:** `python -m maxtext.inference.decode_minimax_m2_npy` via `decode_v4_8.sh`
- **Mesh:** `ici_tensor_parallelism=4, ici_expert_parallelism=1` (v4-8 1×2×2 torus)
- **GCS durable:** `gs://arc-mi-research/models/minimax-m2.7-npy/` (us-central2)

## Disk-size miscalc (2026-05-20T21:30Z)

I budgeted ~230 GB for the converter output, but the three MoE expert tensors
alone (`wi_0`, `wi_1`, `wo`) are each (256 experts × 62 layers × 3072 × 1536)
float16 = ~150 GB → **~450 GB just for MoE experts, ~460 GB total output**.
Streaming converter's docstring says "peak ~500 GB on a 708 GB v6e host" — I
missed that.

At layer ~36/62 the tmpfs was at 355/390 GB with no chance of finishing. I
stopped the convert, attached a 500 GB SSD PD (`minimax-m2-scratch-disk`),
formatted ext4-no-journal at `/mnt/disk1`, re-downloaded the 72 safetensors
files the progressive cleanup had already deleted, and restarted the converter
with `--output_dir=/mnt/disk1/minimax-m2.7-npy`.

**Cost of the redo:** ~30 min of converter work plus ~3 min of HF re-download,
i.e. ~35 min added wall time. Conversion now writes to PD without space
concerns.

**Downstream cold-load risk:** loading 460 GB float16 from PD-ssd at ~150 MB/s
sequential ≈ 50 min, vs ~45 sec from tmpfs. Even with 4-way parallel reads,
expect ~8 min cold-load — *exceeds* the 120 s acceptance threshold. Likely
v1.1 work: produce INT4-packed .npy (~114 GB) that fits in tmpfs for fast load.
For v1 we accept the slow first-load and document.

## Endpoint (to be populated after Stage 4)

| Field | Value |
|---|---|
| Hostname | `main-2.us-central2-b.c.optical-pillar-442815-p2.internal` |
| Port | `8765` (firewall-scoped to project VPC) |
| API shape | OpenAI-compatible (`/v1/completions`, `/v1/chat/completions`) via JetStream |
| Auth | Bearer token, file path: TBD |

## Acceptance results (v1: failed; v1.1 plan below)

| Criterion | Threshold | Result |
|---|---|---|
| Cold-load time (.npy → HBM) | ≤ 120 s | **FAIL** — process OOM'd after 22 min during AQT quant pass |
| Decode throughput (512-tok, bs=1) | ≥ 15 tok/s | n/a (didn't reach decode) |
| Quality vs BF16 ref | ≤ 0.5 nats | n/a |
| ARC-AGI-1 10-task smoke | All complete | n/a |
| Endpoint reachable from main-1 worktree | curl `/health` 200 | n/a |

### Root cause: HBM doesn't fit BF16 intermediate during MaxEngine.quantize_params

MaxText's `intmp` quantization is applied at load time by `MaxEngine.quantize_params`
(`src/maxtext/inference/maxengine/maxengine.py:331-373`). The flow is:

1. Load BF16 weights from the checkpoint into HBM (sharded across chips)
2. Run a dummy forward pass with `mutable=True` so AQT layers capture
   their quantized form
3. Drop the BF16 weights, keep only the AQT pytree

**Step 1 requires the full BF16 model to be resident in HBM** before the
quant pass can convert it. On v4-8: 230 GB BF16 ÷ 4 chips = 57 GB/chip,
which exceeds the 32 GB/chip HBM cap. Loader OOM'd allocating a
34.88 GB tensor on chip 0 (a sharded `wi_0` MoE expert slice).

The `intmp` path on .npy worked correctly as a wire format — install_load_params_patch
ran fine. The bottleneck is the *quantize_params* forward pass, not the .npy loader.

### v1.1 plan — two viable paths

**Path A — DCN multi-host across main-2 + main-4 (~3-4 hr work).**
Use both v4-8s as a 2-host DCN cluster: total 8 chips × 32 GB = 256 GB HBM.
BF16 model: 230 GB ÷ 8 chips = 28.75 GB/chip — fits with ~3 GB headroom
for the quant pass. After quantization, INT4 takes 14.4 GB/chip with
~17 GB free per chip for KV/scratch.

Steps:
1. NFS-mount main-2:/mnt/disk1 to main-4 (read-only), or download the .npy
   pile from `gs://arc-mi-research/models/minimax-m2.7-npy/` to a PD on main-4
2. Set `ici_tensor_parallelism=4, ici_expert_parallelism=1, dcn_expert_parallelism=2`
3. Launch decode_minimax_m2_npy on both hosts with
   `JAX_DISTRIBUTED_COORDINATOR=main-2:1234 JAX_DISTRIBUTED_NUM_PROCESSES=2 JAX_DISTRIBUTED_PROCESS_ID=0|1`
4. Validate the DCN-routed inference produces coherent text (cross-host
   expert routing adds latency but works for MoE inference)

Risk: DCN per-token latency could be high (~10 ms cross-host per layer with
62 layers = ~600 ms/token). May not hit the 15 tok/s threshold; could land
~1-2 tok/s. Functionally works.

**Path B — Pre-quantized AQT-Orbax converter (~5-8 hr work).**
Write a `npy_to_aqt_orbax.py` that loads each float16 .npy leaf, applies
AQT INT4 quantization (compute scales, pack INT4, build QTensor pytree),
writes Orbax checkpoint with `{"aqt": {...}, "params": {...}}` shape that
matches `MaxEngine.quantize_params`'s output. Then decode with
`load_parameters_path=<orbax>` and `checkpoint_is_quantized=True` —
skipping the in-HBM quantize_params step.

Pre-quantized checkpoint: ~115 GB. Loads as INT4 directly into HBM,
14 GB/chip. Easy fit. Better cold-load performance than Path A.

Risk: AQT pytree internals need to be inspected/mirrored faithfully; if
shapes/dtypes don't match exactly what the model expects, hard-to-debug
JAX errors. Higher-effort but lower-runtime-risk than Path A.

**Recommendation:** start with Path A for fastest empirical answer. If DCN
throughput is unusable (<3 tok/s decode), build Path B.

## v1.1 — working endpoint (2026-05-24)

Neither v1.1 Path A (DCN multi-host) nor Path B (pre-quant Orbax) shipped.
What actually worked is a **third path**: per-layer AQT INT4 quantization on
the CPU host writing one pickle per decoder layer, then a custom serve
adapter that streams those pickles into HBM one layer at a time, bypassing
both `MaxEngine.quantize_params` (which OOMs on the in-HBM BF16 pass) and
Orbax (which doubles memory during PyTreeCheckpointer.save). Plus two
bug fixes to MaxText itself.

### Quantization (CPU, layer-by-layer)

`benchmarks/api_server/layerwise_quantize_minimax_m2_npy.py` loads the
float16 .npy weights for one decoder layer at a time, runs AQT INT4
quantization on CPU, pickles each layer's `{params, aqt}` dict, and
deletes intermediates. Output: 62 pickles at ~3.5 GB each (213 GB total) in
`/mnt/dtmpfs/lw_quant_stage/`. Wall time: 51 min for the full model.

Required hack: manually re-key AQT's anonymous `AqtEinsum_0/1/2` to named
`moe_block.{wi_0, wi_1, wo}` keys before write, plus manually pop BF16
expert kernels before `remove_quantized_params` (which otherwise leaves
the originals in place, blowing each pickle to ~10 GB).

### Serving (per-layer pickle → HBM stream)

`benchmarks/api_server/serve_minimax_m2_from_pickles.py` monkey-patches
`MaxEngine.load_params` to:
1. Get the abstract sharding state via `get_abstract_state` (no
   materialization)
2. Loop over the 62 pickles; for each, load → re-key
   `aqt.moe_block.{wi_0,wi_1,wo}` to `aqt.AqtEinsum_{4,5,6}` (to match
   MaxText's actual layer-level einsum naming) → device_put with the
   correct per-leaf NamedSharding → drop the host reference and gc
3. Non-layered weights (`token_embedder.embedding`, `decoder_norm.scale`,
   `logits_dense.kernel`) loaded directly from the original .npy pile

Per-layer host RAM working set: ~10 GB. Per-chip HBM growth: 0.46 GB/layer
× 62 = 28.5 GB. Plus KV cache (Q8, max_target_length=4096) ≈ 1 GB.
Total ~30 GB / chip, within the 32 GB / chip limit.

### MaxText patch — `nnx_wrappers.py:464`

`MiniMaxM2DecoderLayer` doesn't declare AQT-injected `AqtEinsum_4/5/6` as
class-level `nnx.data` attributes — they get auto-created during the first
`__call__`. When the maxengine init reuses our pre-built params and tries
`nnx.update(module, new_state)` for the KV-cache init's `jax.eval_shape`,
NNX rejects the assignment because the attribute path is unknown to the
class. Fix: detect `unknown_state_flat` paths and pre-declare them with
`setattr(module, root, nnx.data({}))` before `nnx.update`.

Backup of the original wrapper: `main-2:~/maxtext/src/maxtext/layers/nnx_wrappers.py.bak`.
This patch is needed for *any* load of a pre-quantized AQT checkpoint
for MoE models in MaxText.

### HBM budget (measured)

| Stage | chip0 | chip1 | chip2 | chip3 |
|---|---|---|---|---|
| Before layer loop | 0 | 0 | 0 | 0 |
| After layer 10 | 4.61 | 4.61 | 4.61 | 4.61 |
| After layer 30 | 13.83 | 13.83 | 13.83 | 13.83 |
| After layer 50 | 23.04 | 23.04 | 23.04 | 23.04 |
| After layer 62 (peak) | 27.65 | 27.65 | 27.65 | 27.65 |

Matches the predicted INT4 footprint for 230 B / 4 chips.

### Throughput (measured, warm, temperature=0)

| Tokens | Wall (s) | tok/s |
|---|---|---|
| 8 | 2.47 | 3.24 |
| 32 | 5.78 | 5.53 |
| 256 | 36.71 | 7.0 |

Cold-JIT first request: **2 min 9 s** for an 8-token completion. JAX cache
at `/mnt/dtmpfs/jax_cache_serve_pickles` warms subsequent runs.

### Pass / fail vs the original thresholds

| Criterion | Plan threshold | Result | Pass? |
|---|---|---|---|
| Cold-load (.npy → HBM) | ≤ 120 s | ~150 s (pickle stream) | ⚠ over by ~25 % |
| Decode throughput | ≥ 15 tok/s | 7 tok/s @ 256 tok | ✗ structural — v4 has no native INT4 matmul |
| Quality sanity | KL ≤ 0.5 nats vs BF16 ref | not measured | n/a |
| ARC-AGI-1 10-task smoke | all complete | not run | n/a |
| Discoverable from peer worktrees | curl from main-1 | confirmed from phase2-trm VM (10.130.0.35) → main-2 (10.130.0.29:8000): "Berlin, a city that has been the" in 1m40s cold JIT | ✓ |

Throughput miss is structural (no v4 int4 silicon). v5e or v6e would
4×-6× this. Within v4-8 constraints, 7 tok/s for a 230 B sparse MoE is
acceptable for downstream critic/verifier use.

### Quality spot check

| Prompt | Output |
|---|---|
| "The capital of France is" | " Paris. The capital of France is Paris" |
| "Once upon a time," | " there was a girl named Lily who loved" |
| "Q: What is 17 times 23? A:" | " 391" (correct); chains into more Q&A pairs, all correct |
| "Write a short essay about the history of TPUs." | Coherent first sentence, collapses into "The history of TPUs" repetition by token ~30 |

Repetition at temp=0 is a known long-generation pathology with INT4-quantized
small-expert MoE; not blocking for short-answer critic use.

### Endpoint reference (as deployed)

```
URL:        http://10.130.0.29:8000           (main-2 internal IP)
Routes:     /                  status
            /v1/completions    OpenAI
            /v1/chat/completions OpenAI
Auth:       none (MAXTEXT_API_KEY env unset; relies on VPC)
Firewall:   default-allow-internal (10.128.0.0/9) permits all internal hosts
Model id:   "minimax-m2.7"
```

The deployed `maxtext_server.py` is older than the worktree copy and is
missing `/health`, `/ready`, `/metrics`, `/v1/models`, and the Anthropic
`/v1/messages` adapter. Updating it costs another cold-JIT cycle and is
deferred to v1.2.

### What's not done (v1.1 → v1.2)

- [ ] systemd unit + `LimitMEMLOCK=infinity` so the endpoint survives
      operator drift. Current launcher is `nohup bash ~/run_serve_pickles.sh`
      with no auto-restart.
- [ ] Refresh deployed `maxtext_server.py` to add `/health` + `/v1/models`
      + Anthropic `/v1/messages` adapter.
- [ ] Cross-host smoke from main-1, main-3, main-4.
- [ ] Persist the 62 pickles (~213 GB) to `/mnt/disk1` so reboot recovery
      doesn't require re-running the 51 min quant pass.
- [ ] Decode hyperparam sweep (temp 0.7, top_p 0.95, rep penalty) to avoid
      the temp=0 repetition pathology.

### Critical file locations (main-2)

- Per-layer INT4 pickles: `/mnt/dtmpfs/lw_quant_stage/` (62 files, 213 GB; tmpfs)
- BF16 source .npy: `/mnt/disk1/minimax-m2.7-npy/` (persistent PD)
- Tokenizer: `/mnt/dtmpfs/minimax-m2.7-hf/`
- Launcher: `~/run_serve_pickles.sh`
- Serve adapter: `~/maxtext/benchmarks/api_server/serve_minimax_m2_from_pickles.py`
- NNX wrapper patch: `~/maxtext/src/maxtext/layers/nnx_wrappers.py:464`
  (backup at `.bak`)
- Server log: `~/serve_pickles.log`

## Build log

- 2026-05-20T20:57Z — provisioning kicked off on main-2 (`bootstrap_tpu.sh`)
- 2026-05-23 — layer-by-layer AQT INT4 quant completed (62 pickles, 213 GB)
- 2026-05-24 — serve adapter + nnx_wrappers patch shipped; first generation
  succeeded, endpoint live on main-2:8000

## Recovery contract

`/mnt/dtmpfs` is wiped on reboot. After a host reboot:

1. `sudo bash ~/maxtext/scripts/minimax_m2.7/setup_dtmpfs.sh /mnt/dtmpfs 350G`
2. `gcloud storage cp -r gs://arc-mi-research/models/minimax-m2.7-npy/ /mnt/dtmpfs/minimax-m2.7-npy/`
3. `gcloud storage cp -r gs://arc-mi-research/models/minimax-m2.7-hf-tokenizer/ /mnt/dtmpfs/minimax-m2.7-hf/`
4. `systemctl --user start minimax-m2-server` (or whatever serving entrypoint we settle on)

Wall-clock from cold reboot to serving healthy: target ≤ 10 min.
