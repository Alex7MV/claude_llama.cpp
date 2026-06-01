# Hybrid Speculative Engine with VRAM KV-Cache

**Date:** 2026-06-01
**Target:** 20+ t/s on RTX 5090 (32 GB VRAM) via hybrid CPU/GPU speculative pipeline

## Target Models

| Model | Layers | Hidden | Heads | kv_lora_rank | Active params |
|---|---|---|---|---|---|
| DeepSeek-V3 | 61 | 7168 | 128 | 512 | ~37B (MoE, top-2) |
| Kimi-k2.6 | 80 | 6144 | 128 | 512 | ~35B (MoE, top-2) |

**KV-cache format:** FP8 compressed latent `c_t^KV` (576 bytes/token/layer), stored in VRAM.

**Draft model:** DeepSeek-V3-Distill-Qwen-7B (IQ4_XS quant, fully in VRAM, ~12 GB).

## VRAM Budget (RTX 5090, 32 GB total)

| Component | Size |
|---|---|
| Draft model (7B, IQ4_XS) | ~12 GB |
| KV-cache 256K context (DSv3) | ~9.0 GB |
| Scratch / CUDA context | ~3 GB |
| **Total used** | **~23.6 GB** |
| Headroom | ~8.4 GB |

## Architecture

### Pipeline: token-to-token

```
t=0               t~3.5ms           t~5.5ms              t~5.7ms
 │                    │                 │                    │
▼                    ▼                 ▼                    ▼
┌────┐  ┌────┐  ┌────────┐  ┌─────────────────┐  ┌────┐  ┌────┐
│RMS │  │ROUT│  │KV COMP │  │    MoE FFN       │  │MERG│  │LOG │
│NORM│─►│ER  │─►│L0RA    │─►│(2 experts, AVX512│─►│E   │─►│ITS │
└───┬┘  └─┬──┘  │7168→512│  │ 3.5ms, prefetched│  └──┬┘  └────┘
    │     │     └───┬─────┘  └────────┬──────────┘    │
    │     │         │                  │              │
    │  [prefetch   [TMA enqueue       │              │
    │   expert      c_t^KV → VRAM]    │              │
    │   weights]                      │              │
    │         │         │             │              │
    │         ▼         ▼             │              │
    │   ┌────────────────────────┐    │              │
    │   │  GPU MLA FLASH ATTN     │◄───┘              │
    │   │  (decompress + attn,    │                   │
    │   │   1.5ms, concurrent)    │                   │
    │   └───────────┬────────────┘                    │
    │               │ h_attn (TMA back, async)        │
    └───────────────┘                                 │
                                                      ▼
                                           ┌──────────────────┐
                                           │ DRAFT MODEL (GPU)│
                                           │ next 4-6 tokens  │
                                           │ (3ms, concurrent)│
                                           └──────────────────┘
```

### Key design decisions

1. **Same-Layer Prefetch (B/TMA):** CPU runs router immediately after RMS Norm, initiates SW prefetch (`_mm_prefetch` T2 hint) for top-2 expert weights while GPU does MLA attention for the same layer's KV.
2. **Seq truncation rollback (B/Truncation):** Speculative rejection rolls back KV-cache via `llama_kv_cache_seq_rm()` — O(1) metadata update on the MLA compressed cache, no PCIe traffic.
3. **SW prefetch only (B/SW-Prefetch):** L3 (384 MB) cannot hold a full expert (~700 MB). Use explicit `_mm_prefetch` immediately after router decision, targeting L3.

### Timing budget (steady state, per token)

| Phase | Time | Resource | Notes |
|---|---|---|---|
| RMS Norm + KV Compress | ~45 µs | CPU | Latent `c_t^KV` produced |
| Router Gate MLP | ~10 µs | CPU | top-2 expert determination |
| TMA enqueue `c_t^KV` | ~3 µs | TMA HW | Async, 576 bytes |
| MoE FFN (top-2) | ~3500 µs | CPU AVX-512 | Prefetched weights hot |
| MLA Flash Attn | ~1500 µs | GPU Tensor Cores | Concurrent with MoE |
| Merge + Logits | ~205 µs | CPU | |
| Draft gen 6 tokens | ~3000 µs | GPU (draft stream) | Overlaps MoE + Attn |

**Total wall-clock per token (pipelined):** ~3.5 ms → ~280 t/s theoretical peak (speculative acceptance ~20-30% → ~55-85 t/s, with `--draft-max-lookahead 6`).

**Target with spec decode overhead:** 20-30+ t/s.

## State Machine: `hybrid_orchestrator`

### Phases

```
IDLE → NORM_DONE → KV_COMPRESSED → TMA_ENQUEUED
       → GPU_ATTN_DONE → MERGE_DONE → (verify) → IDLE
```

### Transitions

1. **IDLE → NORM_DONE:** RMS Norm applied to `h_{t-1}`. Router determines top-2 expert IDs. Initiates `_mm_prefetch` for expert weights.
2. **NORM_DONE → KV_COMPRESSED:** LoRA projection produces `c_t^KV` (7168→512 FP8). Expert prefetch already in flight (overlaps KV compress).
3. **KV_COMPRESSED → TMA_ENQUEUED:** TMA 1D descriptor submitted for H2D transfer of `c_t^KV` to pre-allocated VRAM page pool. Non-blocking.
4. **TMA_ENQUEUED → GPU_ATTN_DONE:** GPU decompresses all `c_t^KV` for current sequence, runs flash attention. CPU concurrently executes MoE FFN.
5. **GPU_ATTN_DONE → MERGE_DONE:** CPU receives `h_attn` via TMA H2D, merges with `h_ffn`, projects logits.
6. **MERGE_DONE:** Token sampled via `sampler_gen_phase` state machine (TEXT / REASON / TOOL_INV).
7. **→ IDLE (next token) OR → verify_and_rollback() (if speculative cycle complete).**

### Key data structures

```
hybrid_orchestrator:
  ├─ stages[61]             : per-layer stage + TMA desc + prefetch state
  ├─ tma_ring[2]            : double-buffered TMA descriptors
  ├─ gpu_compute_stream     : MLA flash attn kernel stream
  ├─ draft:
  │   ├─ draft_stream       : separate stream for draft inference
  │   ├─ lookahead_buffer[] : 4-6 draft tokens
  │   └─ lookahead_kv[]     : c_t^KV per draft token
  └─ verify:
      ├─ accepted           : count of accepted draft tokens
      └─ rejected_at        : position of first mismatch
```

## Speculative Decoding Integration

### Flow

1. **Draft phase:** Draft model (DeepSeek-7B, IQ4_XS, GPU) generates 4-6 lookahead tokens + their `c_t^KV` on `draft_stream`. Target model's KV-cache entries for these positions are pre-populated (placeholder cells).
2. **Verification phase:** Target model runs forward for the draft batch (all tokens in a single expanded batch). `common_sampler_sample_and_accept_n()` cross-references sampled tokens with draft tokens.
3. **Acceptance:** First N matching tokens accepted. Target advances to position `t + N`.
4. **Rollback:** `llama_kv_cache_seq_rm(ctx_tgt, 0, t+N+1, t+draft_len)` — O(1) metadata update. Draft restarts from `t+N`.

### Adaptive lookahead

- Accept rate > 60%: lookahead = 6
- Accept rate 30-60%: lookahead = 4
- Accept rate < 30%: lookahead = 3 (floor)
- Measured as EMA over last 20 cycles.

## Verification Loop

Implemented in `common_sampling_sample_and_accept_n()` (extended):

```cpp
// Added to common_sampler_sample_and_accept_n signature:
//   hybrid_orchestrator *hybrid = nullptr
//
// When hybrid is active:
//   1. Generate logits for each draft position (single batch forward)
//   2. Sample each position through sampler chain (gen_phase first)
//   3. Compare with draft tokens
//   4. Call hybrid->verify_and_rollback()
//   5. Return accepted count
```

The sampler chain operates normally — `sampler_gen_phase` remains first, applying grammar/tool-call masking to each position. This preserves tool-calling isolation.

## Async Staging: c_t^KV transfer

MLA compression (512 elements FP8 per layer per token) is sent to VRAM via TMA descriptors. Implementation approach:

1. **Allocation:** VRAM page pool pre-allocated at context init. Size = `(n_ctx_total + 2 * max_lookahead) * n_layers * kv_lora_rank * sizeof(fp8)`.
2. **Double buffer:** Two TMA ring slots. CPU writes desc to slot N, TMA HW reads and dispatches. GPU processes slot N-1's data while CPU prepares slot N.
3. **TMA fallback:** If Blackwell TMA unavailable, `cudaMemcpyAsync` on default stream (sub-optimal but functional).

## Expert Prefetch

- **Trigger point:** Immediately after `llm_build_deepseek_v3()` router output.
- **Prefetch target:** Top-2 expert weight pointers (gate + up + down matrices).
- **Hint level:** `_MM_HINT_T2` (L3) — expert weights ~700 MB each, L2 too small.
- **Scratchpad:** Prefetch touches ~64 bytes per 4KB cache line. No explicit cache control beyond hint.
- **Verification:** EPYC Turin's hardware prefetcher may extend the stream beyond the hinted lines.

## Error Handling

| Failure mode | Response |
|---|---|---|
| TMA enqueue fails (NACK) | Fall back to `cudaMemcpyAsync` |
| VRAM pool exhausted | Log warning, stall until GPU frees a slot |
| Draft model OOM | Disable speculative mode, fall back to target-only |
| Expert prefetch misses | CPU stalls waiting for DDR5 (penalty ~100 ns, acceptable) |
| `llama_kv_cache_seq_rm` failure (OOM) | Reload from checkpoint (unlikely with O(1) metadata op) |
| Tool-call token in draft lookahead | `sampler_gen_phase` masks it; verification fails → rollback to last non-tool token |
| VRAM pool / `ggml-alloc` conflict | Pre-allocate MLA KV-cache pool on a separate CUDA memory arena, registered as an external backend buffer in `ggml-alloc` via `ggml_backend_alloc_buffer()`. This ensures ggml-alloc tracks the pool and never double-allocates. |

## VRAM Pool Isolation from ggml-alloc

The MLA KV-cache pool must not conflict with ggml-alloc's dynamic device allocations:

1. **Separate arena:** Allocate pool via `cudaMalloc` directly (not through ggml-alloc), then register it as an external `ggml_backend_buffer` via `ggml_backend_cuda_buffer_type()` + `ggml_backend_alloc_buffer()`. This gives ggml-alloc full visibility of the reserved region without surrendering control over its layout.
2. **Pin at context init:** Pool size computed at `llama_init_from_model()` time: `(n_ctx_max + 2 * max_lookahead) * n_layers * kv_lora_rank * sizeof(fp8)`. Allocated once, never resized.
3. **Sub-allocation:** The orchestrator manages its own freelist of `(layer, seq_pos)` slots within the pool. No interaction with ggml-alloc's general allocation path during decode.
4. **Reserved region:** Blackwell TMA descriptors reference GPU virtual addresses within this pool. Register the VA range with CUDA to prevent TLB conflicts with ggml-alloc's allocations.

## Files Changed

| File | Change |
|---|---|
| `common/hybrid_stage.h` | New: `hybrid_orchestrator`, `hybrid_layer_stage`, enums |
| `common/hybrid_stage.cpp` | New: state machine implementation, TMA management |
| `common/CMakeLists.txt` | Add `hybrid_stage.cpp` |
| `llama.cpp` | Hooks in `llm_build_deepseek_v3()` for stage transitions |
| `llama-kv-cache.cpp` | Ensure `seq_rm` is O(1) for MLA compressed cells |
| `common/sampling.cpp` | Extend `sample_and_accept_n` with hybrid verification |
| `common/sampling.h` | Signature update for hybrid param |
| `examples/server/server.cpp` | Init `hybrid_orchestrator` in load_model |
| `examples/server/server-context.cpp` | Wire orchestrator into speculative loop |
| `ggml/src/ggml-cuda/tma-transfer.h` | Add `tma_enqueue_1d` H2D API |
| `common/common.h` | Param flags: `--hybrid-pipeline`, `--kv-vram` |
| `common/common.cpp` | Parse new flags |
| `ggml-alloc` / `ggml-backend` | Register external MLA KV-cache pool buffer; ensure no double-alloc with ggml-alloc's dynamic pool |

## Testing Strategy

1. **Unit:** `hybrid_orchestrator` state machine transitions (mocked GPU)
2. **Integration:** Speculative decode with DSv3-like small model + draft on single GPU
3. **VRAM boundary:** Run with `--kv-vram` at max context, verify no OOM
4. **Regression:** `sampler_gen_phase` masking preserved across speculative cycles
5. **Performance:** `--bench` mode measuring t/s, accept rate, VRAM utilization
