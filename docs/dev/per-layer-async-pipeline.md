# Per-Layer Async Pipeline for DeepSeek V3/R1

## Motivation

DeepSeek V3/R1 uses MLA (Multi-head Latent Attention) + MoE FFN across 61 layers.
The current `llama.cpp` implementation builds ONE monolithic compute graph for all layers
and executes it sequentially. On systems with GPU + CPU hybrid execution, or even on
fully GPU-resident setups, there is significant idle time:

- **GPU** waits while CPU computes RMS norms + attention score math
- **CPU** waits while GPU does matmuls + MoE FFN
- **Flash attention** could overlap with **next-layer QKV projection**
- **TMA K-cache writes** could overlap with **MoE computation**

This document describes a per-layer async pipeline that breaks the monolithic graph into
layer-sized micro-graphs dispatched to multiple CUDA streams, with event-based
synchronization.

## Architecture

### Three-Stage Sliding Window

```
Timeline →

  compute_stream (L-1):  [QKV proj]──[O_proj/resid]──[MoE FFN]──────────────
  compute_stream (L):                                    [QKV proj]──[O_proj/...
  attn_stream (L-1):                        [flash_attn]──[TMA K write]
  attn_stream (L):                                                      [flash_...

  Key:
    [QKV proj]   = rms_norm → q_proj → k_proj → v_proj
    [flash_attn] = Q × K_cache × V_cache → attn_out
    [O_proj]     = attn_out × W_O
    [resid]      = inpL += attn_out
    [MoE FFN]    = rms_norm → gate → top-k → experts
    [TMA K write]= append K,V to page cache
```

Three layers are in flight simultaneously:

| Stage | Stream | Work |
|-------|--------|------|
| L-1 MoE | compute_stream | MoE FFN, residual update |
| L flash_attn | attn_stream | Flash attention, TMA K-cache write |
| L+1 QKV proj | compute_stream | RMS norm, Q/K/V projections |

### Why This Works

The critical insight: **Q/KV projections for layer L+1 depend only on `inpL`
(the residual stream from layer L)**, not on the attention output of layer L+1.

Since `inpL` is available as soon as layer L's MoE FFN + residual update completes,
the QKV proj for L+1 can start immediately — it does NOT need to wait for L+1's
own flash_attn to complete.

This creates a three-stage pipeline where work is always available for both streams.

## Stream Model

### Two Streams (expandable to three)

| Stream | Purpose | CUDA stream |
|--------|---------|-------------|
| `compute_stream` | QKV proj, O_proj, residual, MoE FFN | `main_stream` |
| `attn_stream` | flash_attn, TMA K-cache writes | `attn_global_stream` |

Rationale against a third stream for TMA: TMA operations on Blackwell use
the same stream as the issuing kernel. Adding a third TMA-dedicated stream
adds synchronization complexity for marginal gain (TMA is already async from
the issuing stream's perspective).

### Synchronization

#### CUDA Events (all architectures, fallback path)

```cuda
// Per-layer event pair
cudaEvent_t e_qkv_done[N_LAYERS];  // recorded on compute_stream after QKV proj
cudaEvent_t e_attn_done[N_LAYERS]; // recorded on attn_stream after flash_attn

// Layer L timing:
// compute_stream:
//   QKV proj for layer L
//   cudaEventRecord(e_qkv_done[L], compute_stream)
//   cudaStreamWaitEvent(attn_stream, e_qkv_done[L])
//   (attn_stream now free to start flash_attn for layer L)
//   Wait for attn_stream: cudaStreamWaitEvent(compute_stream, e_attn_done[L])
//   O_proj → residual → MoE FFN for layer L
//   → inpL ready for layer L+1
//
// attn_stream:
//   (waits on e_qkv_done[L] from compute_stream)
//   flash_attn for layer L
//   cudaEventRecord(e_attn_done[L], attn_stream)
//   TMA K-cache write for layer L
//   → attn_stream ready for layer L+1
```

#### CDA Primitives (Blackwell, CUDA 14+, fast path)

Blackwell's CDA (Compute-Dependent Advance) allows direct in-cluster signaling
between streams without host-roundtrip event overhead:

```cuda
// Blackwell CDA barrier pseudocode
__global__ void signal_and_wait(volatile int* barrier, int stage) {
    if (threadIdx.x == 0) {
        if (stage == 0) {
            // compute_stream: QKV done, signal attn_stream
            atomicAdd(barrier, 1);
        } else if (stage == 1) {
            // attn_stream: flash_attn done, signal compute_stream
            atomicAdd(barrier + 1, 1);
        }
    }
}
```

The design detects Blackwell at runtime and selects the sync primitive.
Integration point: `ggml_cuda_concurrent_stream` in `ggml-cuda.cu`.

#### Event Pool

Pre-allocate 122 events (61 layers × 2); recycle via `cudaEventCreate`/`cudaEventDestroy`
at batch boundaries. Events live for the duration of a single `llama_decode`/`llama_encode`.

## Graph Structure

### Per-Layer Micro-Graph Split

Each layer's computation is split into 3 sub-graphs:

#### Sub-graph A: QKV Projection (compute_stream)

```
inpL (residual tensor, persistent GPU allocation)
  → rms_norm → hidden
  → W_Q_A × hidden → q_a
  → W_Q_B × q_a → Q              [n_tokens × n_heads × head_dim]
  → W_KV_A × hidden → kv_a
  → W_K_B × kv_a → KR            [n_tokens × kv_lora_rank]
  → store(Q, kv_a, KR) into scratch
  → record e_qkv_done event
```

Outputs: `Q (n_tokens × 16384)`, `kv_a (n_tokens × 512)`, `KR (n_tokens × 16384)`
stored in pre-allocated scratch GPU buffers.

#### Sub-graph B: Flash Attention (attn_stream)

```
  → wait on e_qkv_done
  → flash_attn(Q, KR, K_cache[L], V_cache[L]) → attn_out
  → record e_attn_done event
  → TMA append: (kv_a, KR) → K_cache[L], V_cache[L]
```

Note: The TMA append uses `kv_a` and `KR` directly (MLA compressed representation),
NOT the up-projected full K,V. DeepSeek's MLA stores compressed KV in the cache.

#### Sub-graph C: O_proj + Residual + MoE FFN (compute_stream)

```
  → wait on e_attn_done
  → W_O × attn_out → attn_proj
  → inpL += attn_proj                 (residual update, in-place)
  → rms_norm(inpL) → ffn_hidden
  → router(ffn_hidden) → top-k indices
  → MoE_FFN(ffn_hidden, expert_weights[top-k]) → ffn_out
  → inpL += ffn_out                   (residual update, in-place)
  → inpL now ready for layer L+1
```

### Residual Stream Management

`inpL` must persist across all micro-graphs for all layers.
Use `ggml_backend_tensor_alloc()` with a persistent allocation:

```c
static struct ggml_tensor * residual_buf;  // allocated once per batch
// Shape: [n_ubatch, n_embd]  (e.g. [1, 7168] for decode)
// Backend: CUDA
```

Scratch tensors (Q, kv_a, KR, attn_out) are recycled per-layer within
a ring-buffer of 3 slots (one per pipeline stage).

## MoE Expert Prefetch (Phase 0/1)

### Problem

With 256 experts × ~14.5B parameters each in Q4, only a subset fit in VRAM.
Each decode step must load top-8 expert weights from CPU RAM → GPU VRAM.
This TMA transfer is on the critical path.

### Solution: Prefetch Overlapped with Attention

During flash_attn for layer L on attn_stream, the compute_stream's
idle time (waiting for flash_attn to complete) can be used to launch
TMA transfers for layer L+1's top-k expert weights.

```cuda
// Between QKV proj done and O_proj start:
// compute_stream has ~0.1-0.5ms gap while flash_attn runs on attn_stream
// → use this gap to issue async expert weight TMA loads
// → weights arrive in VRAM before MoE FFN starts

// Pseudocode:
graph_compute(compute_stream, subgraph_A);  // QKV proj
cudaEventRecord(e_qkv_done, compute_stream);
prefetch_experts(compute_stream, layer_L_plus_1_topk);  // TMA async
cudaStreamWaitEvent(compute_stream, e_attn_done);
graph_compute(compute_stream, subgraph_C);  // O_proj + resid + MoE
```

#### Expert Weight Management

- Maintain a `expert_weight_slots[K]` pinned GPU buffer pool (K = pipeline depth, 3)
- Use `cudaMemcpyAsync` (fallback) or TMA (Blackwell) for H2D transfer
- If all experts fit in VRAM (e.g., 56GB across two GPUs), prefetch is a no-op

## Implementation Plan

### Phase 0: Infrastructure (prerequisite for all)

1. **`ggml-backend`**: Add multi-graph submit API
   - `ggml_backend_sched_submit_micro_graph(ggml_backend_sched, ggml_cgraph*, int stream_id)`
   - `ggml_backend_sched_sync(ggml_backend_sched, cudaEvent_t*)` — record/wait events
   
2. **`ggml-cuda`**: Per-op stream assignment
   - Add `ggml_tensor->extra.stream_id` field (set at graph-build time)
   - Split `ggml_cuda_compute_forward` to dispatch to different streams
   - Implement event record/wait as pseudo-ops (no compute, just sync)

3. **Event pool**: `new_event_pool(N)`, `recycle_pool()`, thread-safe

### Phase 1: Per-Layer Graph Builder

4. **`llama-graph.cpp`**: Refactor `build_deepseek_graph()`
   - `build_deepseek_layer_attn_proj(L)` → subgraph A
   - `build_deepseek_layer_flash_attn(L)` → subgraph B
   - `build_deepseek_layer_ffn(L)` → subgraph C
   - `residual_tensor = ggml_new_tensor_1d(ctx, ...)` — persistent alloc

5. **`llama-decoder.cpp`**: New `llama_build_decode_micro_graphs()` entry point

### Phase 2: Pipeline Scheduler

6. **`pipeline-sched.cpp`** (new): Three-stage pipeline driver
   ```
   for L in range(0, n_layer):
       if L+1 < n_layer:
           submit(subgraph_A(L+1), compute_stream)
           record(e_qkv_done[L+1])
       submit(subgraph_B(L), attn_stream)     // waits on e_qkv_done[L]
       if L > 0:
           prefetch_experts(L, L+1)            // TMA async (Phase 1+)
           wait(e_attn_done[L])
           submit(subgraph_C(L), compute_stream)
           record(e_layer_done[L])
   ```

7. **Event synchronization in `server-context.cpp`**:
   - Replace single `graph_compute` with pipeline dispatch
   - Handle the head (L=0, no L-1 subgraph) and tail (L=N-1, no L+1 subgraph)

### Phase 3: Expert Prefetch

8. **Expert weight TMA**:
   - Detect expert-to-device mapping at load time
   - `prefetch_expert_weights(layer_idx, top_k_indices, stream)` — async H2D
   - Integrate into pipeline scheduler's gap window

### Phase 4: CDA Optimizations (Blackwell)

9. **CDA barriers**:
   - Replace `cudaEventRecord`+`cudaStreamWaitEvent` with CDA atomics
   - Detect `cudaDeviceGetAttribute()` for Blackwell
   - Fall back to events on pre-Blackwell

## Integration Points

| File | Change |
|------|--------|
| `ggml-cuda.h` | `ggml_cuda_set_stream()` API |
| `ggml-cuda.cu` | `ggml_cuda_concurrent_stream` extended for per-op assignment |
| `ggml-backend-impl.h` | `ggml_backend_multi_graph_submit` abstract op |
| `ggml-backend.cpp` | Scheduler multi-graph support |
| `llama-graph.cpp` | Per-layer graph builders |
| `llama-decoder.cpp` | `llama_build_decode()` variant for pipeline |
| `server-context.cpp` | Pipeline driver, event loop |
| `hybrid_stage.cpp` | Attach pipeline to hybrid backend |
| `pipeline-sched.cpp` | NEW: pipeline orchestrator |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Event overhead negates gains | Pre-allocate, batch event ops, CDA fallback |
| TMA K-write races with next flash_attn read | attn_stream serializes all K-cache ops |
| Memory pressure from persistent scratch | Ring buffer of 3 slots, recycle per layer |
| Non-DeepSeek models broken | Pipeline only activates for `LLM_ARCH_DEEPSEEK_V3` / `LLM_ARCH_DEEPSEEK_R1`; fallback to monolithic for all others |
| Debugging complexity | Phase 0 lands with `LLAMA_LOG_INFO` tracing per event; `--no-pipeline` flag to disable |

## Test Strategy

1. **Unit**: Each sub-graph produces bit-exact output vs. monolithic graph
2. **Functional**: `per-layer-async-pipeline == monolithic` for same input (seed, tokens)
3. **Performance**: `llama-bench` with `--no-pipeline` baseline
4. **Stability**: 1000-token decode loop with event tracing, verify no hangs
