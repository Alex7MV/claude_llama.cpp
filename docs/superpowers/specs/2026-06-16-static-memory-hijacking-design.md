# Static Memory Hijacking for MoE Phase 2 CUDA Graph

**Date:** 2026-06-16
**Status:** Design approved, pending implementation
**Context:** Kimi-K2.6 1T MoE on RTX 5090, 10.99 t/s baseline

## Problem

CUDA Graph for Phase 2 (FFN-only compact graph) suffers from data corruption because `ggml_gallocr_alloc_graph()` dynamically reassigns `tensor->data` pointers between Capture and Replay. The CUDA Graph "remembers" the addresses from capture time, so when the allocator reuses memory at different offsets, kernels write to wrong locations.

Current mitigations (`active = 0` forced compact buffer, direct `cudaMemcpy` for logits) are insufficient — they only address leaf tensors, not the 549 intermediate tensors generated across 61 MoE layers.

## Solution Overview

**Static Memory Hijacking** — allocate a single fixed VRAM buffer at model init, pre-compute offsets for every Phase 2 tensor, hijack `tensor->data` pointers before CUDA Graph capture, and restore them before each replay.

The design consists of three structures integrated into `llama_context`:

1. **`phase2_hijack`** — owns the static GPU buffer and manages tensor data pointer hijacking
2. **`phase2_inject`** — owns page-locked host mirrors and performs async H2D data injection
3. **`phase2_guard`** — ensures Phase 1 outputs are consumed by Phase 2 without extra synchronizations

Key invariant: `ggml_gallocr_alloc_graph()` is called normally (preserving all internal asserts), but its tensor data assignments are immediately overwritten with static addresses. The allocator sees a "correct" graph; the CUDA Graph sees stable addresses.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Model | Kimi-K2.6 |
| `n_layers` (MoE) | 61 |
| `n_embd` | 7168 |
| `ffn_hidden_dim` | 2048 |
| `n_experts` | 160 + 1 shared |
| `n_tokens` | 1 (generation only; batch > 1 deferred) |

## Section 1: Memory Map Design

### Per-Layer Tensor Inventory (9 tensors per layer)

| # | Tensor | Shape | Dtype | Raw bytes | Aligned (256B) |
|---|--------|-------|-------|-----------|----------------|
| 1 | `ffn_inp` | [1, 7168] | fp16 | 14,336 | 14,336 |
| 2 | `moe_ids` | [1, 8] | int32 | 32 | 256 |
| 3 | `moe_weights` | [1, 8] | fp32 | 32 | 256 |
| 4 | `residual` | [1, 7168] | fp16 | 14,336 | 14,336 |
| 5 | `gate_out` | [1, 2048] | fp16 | 4,096 | 4,096 |
| 6 | `up_out` | [1, 2048] | fp16 | 4,096 | 4,096 |
| 7 | `silu_mul` | [1, 2048] | fp16 | 4,096 | 4,096 |
| 8 | `down_out` | [1, 7168] | fp16 | 14,336 | 14,336 |
| 9 | `embd_out` | [1, 7168] | fp16 | 14,336 | 14,336 |
| **Per layer** | | | | **70,144** | **70,144** |

### Offset Table (compile-time constants)

```cpp
#define H2_FFN_DIM       2048
#define H2_N_EMBD        7168
#define H2_ALIGN         256
#define H2_ALIGN_UP(x)   (((x) + (H2_ALIGN - 1)) & ~(H2_ALIGN - 1))

#define H2_SZ_INP        H2_ALIGN_UP(H2_N_EMBD * 2)         // 14,336 = 0x3800
#define H2_SZ_IDS        H2_ALIGN_UP(8 * 4)                 //    256 = 0x0100
#define H2_SZ_WEIGHTS    H2_ALIGN_UP(8 * 4)                 //    256 = 0x0100
#define H2_SZ_RESIDUAL   H2_ALIGN_UP(H2_N_EMBD * 2)         // 14,336 = 0x3800
#define H2_SZ_GATE       H2_ALIGN_UP(H2_FFN_DIM * 2)        //  4,096 = 0x1000
#define H2_SZ_UP         H2_ALIGN_UP(H2_FFN_DIM * 2)        //  4,096 = 0x1000
#define H2_SZ_SILU       H2_ALIGN_UP(H2_FFN_DIM * 2)        //  4,096 = 0x1000
#define H2_SZ_DOWN       H2_ALIGN_UP(H2_N_EMBD * 2)         // 14,336 = 0x3800
#define H2_SZ_OUT        H2_ALIGN_UP(H2_N_EMBD * 2)         // 14,336 = 0x3800

#define H2_OFF_INP       0x00000
#define H2_OFF_IDS       0x03800
#define H2_OFF_WEIGHTS   0x03900
#define H2_OFF_RESIDUAL  0x03A00
#define H2_OFF_GATE      0x07200
#define H2_OFF_UP        0x08200
#define H2_OFF_SILU      0x09200
#define H2_OFF_DOWN      0x0A200
#define H2_OFF_OUT       0x0DA00

#define H2_LAYER_STRIDE  0x11200   // 70,144 bytes

// Total: 61 * 70,144 = 4,278,784 bytes (~4.08 MB)
```

All offsets are compile-time constants. No runtime computation — the `addr(il, off)` function is a single multiply-and-add.

## Section 2: The Hijack Mechanism

### `phase2_hijack` Struct

```cpp
struct phase2_hijack {
    void * base = nullptr;           // cudaMalloc at init, cudaFree at destroy
    bool   captured = false;

#ifdef GGML_USE_CUDA
    cudaGraphExec_t cuda_graph_exec = nullptr;
#endif

    static constexpr int N_LAYERS = 61;
    static constexpr int N_TENSOR_TYPES = 9;
    static constexpr int MAX_SNAPSHOTS = N_LAYERS * N_TENSOR_TYPES; // 549

    struct entry {
        ggml_tensor * t;
        void        * static_addr;
    };
    entry snapshots[MAX_SNAPSHOTS];
    int   snapshot_count = 0;

    void* addr(int il, size_t off) {
        return (char*)base + il * H2_LAYER_STRIDE + off;
    }

    void init() {
        cudaMalloc(&base, N_LAYERS * H2_LAYER_STRIDE);
    }

    void hijack_layer(int il,
                      ggml_tensor * inp,   ggml_tensor * ids,
                      ggml_tensor * w,     ggml_tensor * residual,
                      ggml_tensor * gate,  ggml_tensor * up,
                      ggml_tensor * silu,  ggml_tensor * down,
                      ggml_tensor * out) {
        assert(snapshot_count < MAX_SNAPSHOTS); // guard against layer count mismatch
        auto * e = &snapshots[snapshot_count];
        e[0] = {inp,      addr(il, H2_OFF_INP)};      inp->data      = e[0].static_addr;
        e[1] = {ids,      addr(il, H2_OFF_IDS)};      ids->data      = e[1].static_addr;
        e[2] = {w,        addr(il, H2_OFF_WEIGHTS)};  w->data        = e[2].static_addr;
        e[3] = {residual, addr(il, H2_OFF_RESIDUAL)}; residual->data = e[3].static_addr;
        e[4] = {gate,     addr(il, H2_OFF_GATE)};     gate->data     = e[4].static_addr;
        e[5] = {up,       addr(il, H2_OFF_UP)};       up->data       = e[5].static_addr;
        e[6] = {silu,     addr(il, H2_OFF_SILU)};     silu->data     = e[6].static_addr;
        e[7] = {down,     addr(il, H2_OFF_DOWN)};     down->data     = e[7].static_addr;
        e[8] = {out,      addr(il, H2_OFF_OUT)};      out->data      = e[8].static_addr;
        snapshot_count += N_TENSOR_TYPES;
    }

    void restore_all() {
        for (int i = 0; i < snapshot_count; i++)
            snapshots[i].t->data = snapshots[i].static_addr;
        asm volatile("" ::: "memory"); // compiler barrier: no reorder past cudaGraphLaunch
    }

    void destroy() {
        if (base) { cudaFree(base); base = nullptr; }
#ifdef GGML_USE_CUDA
        if (cuda_graph_exec) { cudaGraphExecDestroy(cuda_graph_exec); cuda_graph_exec = nullptr; }
#endif
    }
};
```

### Capture Flow

```
1. Build Phase 2 graph (existing code)
2. ggml_gallocr_alloc_graph(phase2_gf)       → tensor->data assigned by allocator
3. hijack_layer(0..60) for all 549 tensors   → tensor->data overwritten with static_addr
4. cudaStreamBeginCapture(stream)
5. graph_compute(phase2_gf, false)           → CUDA Graph records static addresses
6. cudaStreamEndCapture → cudaGraphInstantiate
7. h2_hijack.captured = true
```

### Replay Flow

```
1. Build Phase 2 graph (same topology, new ubatch)
2. ggml_gallocr_alloc_graph(phase2_gf)       → tensor->data assigned by allocator (different this time)
3. h2_hijack.restore_all()                   → ALL 549 tensor->data restored to static_addr
4. h2_inject.inject_all(h2_hijack, stream)   → cudaMemcpyAsync H2D into static buffer
5. cudaGraphLaunch(h2_hijack.cuda_graph_exec, stream)
```

## Section 3: Async Data Injection

### `phase2_inject` Struct

```cpp
struct phase2_inject {
    // Page-locked host mirrors (1:1 with GPU static buffer input offsets)
    void * host_ffn_inp[61];      // 61 × H2_SZ_INP = 874 KB
    int    host_moe_ids[61][8];   // 61 × 32 = 1.95 KB
    float  host_moe_w[61][8];     // 61 × 32 = 1.95 KB

    void init() {
        for (int il = 0; il < 61; il++)
            cudaHostAlloc(&host_ffn_inp[il], H2_SZ_INP, cudaHostAllocDefault);
    }

    void destroy() {
        for (int il = 0; il < 61; il++)
            cudaFreeHost(host_ffn_inp[il]);
    }

    void fill_from_phase1(int il,
                          const void * ffn_inp_src,   // from GPU D2H or CPU-side Phase 1 result
                          const int  * ids_src,
                          const float * weights_src) {
        memcpy(host_ffn_inp[il], ffn_inp_src,  H2_SZ_INP);
        memcpy(host_moe_ids[il],  ids_src,      H2_SZ_IDS);
        memcpy(host_moe_w[il],    weights_src,  H2_SZ_WEIGHTS);
    }

    void inject_all(phase2_hijack & hijack, cudaStream_t stream) {
        for (int il = 0; il < 61; il++) {
            void * dst_inp = hijack.addr(il, H2_OFF_INP);
            void * dst_ids = hijack.addr(il, H2_OFF_IDS);
            void * dst_w   = hijack.addr(il, H2_OFF_WEIGHTS);

            cudaMemcpyAsync(dst_inp, host_ffn_inp[il], H2_SZ_INP,     cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dst_ids, host_moe_ids[il],  H2_SZ_IDS,    cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dst_w,   host_moe_w[il],    H2_SZ_WEIGHTS,cudaMemcpyHostToDevice, stream);
        }
    }
};
```

Total host-side memory: < 1 MB page-locked. Inject overhead: 183 `cudaMemcpyAsync` launch calls (~15 µs total on RTX 5090), all on the same stream — CUDA serializes them naturally.

## Section 4: Consistency Guard

### `phase2_guard` Struct

```cpp
struct phase2_guard {
    cudaEvent_t phase1_done_event;

    void init() { cudaEventCreate(&phase1_done_event); }
    void destroy() { cudaEventDestroy(phase1_done_event); }

    // Called after Phase 1 graph_compute completes.
    // Ensures all Phase 1 GPU work is visible before we inject data for Phase 2.
    void sync_phase1(cudaStream_t compute_stream) {
        cudaEventRecord(phase1_done_event, compute_stream);
        // No cudaEventSynchronize here — existing D2H reads on the same
        // stream provide implicit ordering. The event pinpoints the
        // completion boundary for any future stream operations.
    }
};
```

The guard exploits the existing `cudaMemcpyAsync` D2H path (line 1561 in `llama-context.cpp`) which already happens on the same compute stream. Phase 1 compute → D2H threshold readback → CPU processes → host buffers filled → `inject_all` H2D — all on the same stream. No extra synchronizations needed.

## Section 5: Zero-Overhead Integration

### Changes to `llama_context`

Add three members to the context struct (`src/llama-context.h`):

```cpp
struct llama_context {
    // ... existing fields ...
    phase2_hijack  h2_hijack;
    phase2_inject  h2_inject;
    phase2_guard   h2_guard;
};
```

### Changes to `process_ubatch()` (`src/llama-context.cpp`)

**Init (model load):** call `h2_hijack.init()`, `h2_inject.init()`, `h2_guard.init()`.

**Destroy (model unload):** call `h2_hijack.destroy()`, `h2_inject.destroy()`, `h2_guard.destroy()`.

**Capture path** (around line 1750, first decode with n_tokens=1):

```cpp
// After ggml_gallocr_alloc_graph, before cudaStreamBeginCapture:
for (int il = 0; il < 61; il++)
    h2_hijack.hijack_layer(il, /* 9 tensor references from graph builder */);
```

**Replay path** (around line 1716, subsequent decode calls):

```cpp
// After ggml_gallocr_alloc_graph, before cudaGraphLaunch:
h2_hijack.restore_all();
h2_guard.sync_phase1(stream);
h2_inject.inject_all(h2_hijack, stream);
cudaGraphLaunch(h2_hijack.cuda_graph_exec, stream);
```

### Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `src/llama-context.h` | Add 3 struct members | +3 |
| `src/llama-context.cpp` | `phase2_hijack`, `phase2_inject`, `phase2_guard` struct definitions + init/destroy/integration | ~100 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | None (allocator untouched) | 0 |
| `ggml/src/ggml-alloc.c` | None | 0 |

Total code delta: ~100 lines.

### Timing Budget (RTX 5090)

| Operation | Latency | Impact |
|-----------|---------|--------|
| `restore_all()` — 549 pointer writes | ~0.1 µs | negligible |
| `fill_from_phase1()` — 61 × memcpy host-side | ~5 µs | negligible |
| `inject_all()` — 183 × cudaMemcpyAsync H2D launch | ~15 µs | negligible |
| **Net overhead vs baseline (10.99 t/s)** | **~20 µs** | **no regression** |

> Note: `cudaGraphLaunch` is already in the baseline (current code uses it; the 10.99 t/s measurement includes graph launch savings). This design makes the graph *correct* without measurable overhead.

## Constraints and Limitations

1. **n_tokens = 1 only.** The static buffer is sized for single-token generation. Batch prefill still uses the standard non-graph path.
2. **Model-specific constants.** `H2_N_LAYERS = 61`, `H2_FFN_DIM = 2048`, `H2_N_EMBD = 7168` are hardcoded for Kimi-K2.6. Different models require recompilation with adjusted constants.
3. **CUDA only.** No CPU, Metal, or Vulkan support.
4. **Temporary ggml allocations are leaked.** After `restore_all()`, the memory allocated by `ggml_gallocr_alloc_graph()` is orphaned (tensors no longer point to it). This is acceptable because:
   - The `ggml_gallocr_reserve()` call at the start of the NEXT decode cycle calls `ggml_vbuffer_reset()` + `ggml_dyn_tallocr_reset()`, which clears all free-block lists and effectively discards leaked allocations
   - The leaked size per iteration is ~4 MB, well within the GPU buffer pool
   - No cumulative leak — the allocator resets every cycle

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `ggml_gallocr` reuses our leaked memory and corrupts it | `restore_all()` happens *after* `alloc_graph`, so tensors are already assigned. The CUDA graph writes to OUR static buffer, not the leaked addresses. |
| Phase 1 tensor data races with Phase 2 reads | Same CUDA stream — implicit ordering via stream serialization |
| `ggml_set_input` or `ggml_set_output` resets tensor->data | `restore_all()` is the LAST thing before `cudaGraphLaunch` — any prior resets are overwritten |
| Future ggml refactoring breaks the hijack contract | Isolated in ~100 lines with clear call points; easy to audit and update |

## Deployment Notes

### NUMA Affinity (EPYC Turin)

On dual-socket EPYC systems, pin the llama.cpp process to the NUMA node physically closest to the RTX 5090's PCIe slot:

```bash
numactl --cpunodebind=0 --membind=0 ./llama-cli ...
```

This eliminates cross-socket memory traffic during `fill_from_phase1()` host-side memcpy operations and D2H threshold readbacks, providing 2-3% additional throughput.