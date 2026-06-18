# Design Spec: MoE Module Decoupling — #huanglawsded Infrastructure

**Date:** 2026-06-18
**Role:** Lead CUDA Systems Architect
**Context:** Decouple monolithic MoE static-hijack / prefetch / cascade infrastructure from `src/llama-context.cpp` into 3 clean, production-grade C++ modules.
**Status:** Design approved, pending implementation plan.

---

## Background

Baseline code achieves stable hybrid inference on Kimi-K2.7 (1T MoE, 600GB) and DeepSeek-V3 using a single RTX 5090 (32GB) and an 80-core AMD EPYC 9V74 (12-channel DDR5). The MoE two-phase pipeline is 100% stable but lives entirely in `src/llama-context.cpp` (lines 91-362) and `src/llama-context.h` (lines 48-220). This spec defines a clean modular decomposition with zero semantic changes.

---

## Preserved Infrastructure (Do Not Lose)

1. **`get_tensor_async` Fallback:** Device ID comparison (`buft_ctx->device != cuda_ctx->device`) returning raw CPU pointers for GGML `cudaMalloc` static buffers.
2. **Direct DMA Execution Path:** `copy_data_to_static` loop issuing `cudaMemcpyAsync` with `cudaMemcpyDefault` for scratch→static D2D transfer.
3. **Cascade GPU Forcing:** Bounded 1-level rule — forces `ffn_moe_*` matched tensors and their direct VIEW (op=37) / ADD (op=2) children to GPU.
4. **Expert IDs Isolation:** Routing indices (ARGSORT, allocation tokens) remain un-forced on CPU to prevent CUDA backend `id >= 0 && id < n_expert` assertion panics.

---

## Module 1: `moe-static-bunker` (Data Layer)

**Files:** `src/moe-static-bunker.h`, `src/moe-static-bunker.cpp`
**Guard:** `#ifdef GGML_USE_CUDA` (no `LLAMA_DEEPSEEK_PIPELINE` req'd)
**Approach:** Free-function extraction — structs are exposed as POD with public fields. The hijacker module operates on them via reference parameters.

### Contents

**Header macros:**
- `H2_ALIGN 256`, `H2_ALIGN_UP(x) (((x)+(H2_ALIGN-1)) & ~(H2_ALIGN-1))`, `H2_N_LAYERS_MAX 128`

**Structs (extracted verbatim from `llama-context.h:116-220`, `llama-context.cpp:91-173`):**

| Struct | What | Lines |
|--------|------|-------|
| `phase2_hijack` | nested `slot`, `TypeId` enum, `TENSOR_NAMES[7]`, CUDA buffer/stream/graph handles | ~80 |
| `phase2_inject` | Host-pinned MoE ID/weight buffers (128 layers × 32 ints + 32 floats) | ~10 |
| `phase2_guard` | CUDA event for Phase 1→2 ordering fence | ~8 |

**Methods retained in struct (data operations only, no graph walking):**
- `phase2_hijack::init(n_moe_layers)` — allocate slots array, create CUDA stream
- `phase2_hijack::destroy()` — free graph exec, buffer, stream, slots
- `phase2_hijack::allocate_slots(max_il)` — `cudaMalloc` static buffer, assign 256-aligned offsets
- `phase2_hijack::slot_idx(il, type_id)` → `il * N_TYPES + type_id` (inline)
- `phase2_inject::init/destroy/fill_layer/inject_all`
- `phase2_guard::init/destroy/record/wait`

**What's REMOVED from the structs:**
- `scan_and_hijack()`, `scan_and_update_snapshots()`, `restore_all()`, `copy_data_to_static()`, `match_name()` — promoted to free functions in moe-hijacker.

**CUDA forward declarations:** Extracted from `llama-context.cpp:65-75` → `moe-static-bunker.cpp`:
`cudaMalloc`, `cudaFree`, `cudaHostAlloc`, `cudaFreeHost`, `cudaEventCreate`, `cudaEventDestroy`, `cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaGraphExecDestroy`, `cudaGraphDestroy`.

**Dependencies:** Only `<cstddef>`, `<cstdio>`, CUDA `extern "C"` declarations. Zero llama-layer dependencies.

---

## Module 2: `moe-hijacker` (Graph Scanning & Cascading)

**Files:** `src/moe-hijacker.h`, `src/moe-hijacker.cpp`
**Guard:** `#ifdef GGML_USE_CUDA`
**Namespace:** `moe`

### Free Functions

```cpp
std::pair<int,int> match_hijack_name(phase2_hijack & h2, const char * name);
int  scan_and_hijack(phase2_hijack & h2, ggml_cgraph * gf);
bool scan_and_update_snapshots(phase2_hijack & h2, ggml_cgraph * gf);
void copy_data_to_static(phase2_hijack & h2, void * cuda_stream);
void cascade_force_moe_consumers(phase2_hijack & h2, ggml_cgraph * gf,
                                  void * sched, void * gpu_backend);
void restore_all(phase2_hijack & h2);
```

### Source mapping

| Old (llama-context.cpp) | New (moe-hijacker.cpp) |
|---|---|
| `phase2_hijack::match_name` (L177-194) | `moe::match_hijack_name(h2, name)` |
| `phase2_hijack::scan_and_hijack` (L227-316) | `moe::scan_and_hijack(h2, gf)` |
| `phase2_hijack::scan_and_update_snapshots` (L318-346) | `moe::scan_and_update_snapshots(h2, gf)` |
| `phase2_hijack::restore_all` (L348-351) | `moe::restore_all(h2)` |
| `phase2_hijack::copy_data_to_static` (L353-360) | `moe::copy_data_to_static(h2, stream)` |
| Cascade loop inline (L2050-2061) | `moe::cascade_force_moe_consumers(h2, gf, sched, gpu)` |

### Preserved Edge Cases

1. **VIEW offset:** `ptrdiff_t off = (char*)t->data - (char*)t->view_src->data` — exact match.
2. **Lazy buffer allocation:** `if (!buffer) allocate_slots(max_il + 1)` — first scan only.
3. **Slot parent marking:** VIEW → parent = -2, non-VIEW → parent = -1. `allocate_slots` skips views.
4. **VIEW hijack address:** `(char*)parent_slot.addr + child_slot.offset` using VIEW's `view_src` slot.
5. **Cascade boundedness:** Only GGML_OP_VIEW (37) and GGML_OP_ADD (2). No transitive fan-out.

**Dependencies:** `moe-static-bunker.h`, `ggml.h`, `<utility>`, `<vector>`.

---

## Module 3: `moe-prefetcher` (Async Background Worker)

**Files:** `src/moe-prefetcher.h`, `src/moe-prefetcher.cpp`
**Guard:** `#ifdef GGML_USE_CUDA`
**Namespace:** `moe`

### Purpose

Extract expert H2D transfers onto a background thread pinned to EPYC 9V74 management cores (indices 64-75), overlapping with main-thread Phase 2 graph build + scheduler allocation. Currently, Phase 1c (`process_ubatch:1947-1983`) runs synchronously — the prefetcher makes it async.

### Architecture

```
Main Thread (cores 0-63)              Prefetcher Thread (cores 64-75)
     |                                      |
  Phase 1a: threshold kernels               |
  Phase 1b: D2H reads → host_mask/remap     |
     |                                      |
  launch_prefetch(work_items) ────────►  receive work
     |                                  pin to cores 64-75
  build Phase 2 graph                   cudaMemcpyAsync H2D
  sched alloc                           (dedicated stream)
  cascade force                         signal done fence
     |                                      |
  wait_prefetch_fence() ◄───────────────┘
     |
  hijack + copy_to_static
  Phase 2 compute
```

### Class API

```cpp
struct prefetch_work_item {
    ggml_tensor * dst_gate, * dst_up, * dst_down;
    ggml_tensor * src_gate, * src_up, * src_down;
    ggml_tensor * expert_mask, * moe_remap;
    void * host_mask, * host_remap;
    size_t slice_bytes[3];
    int max_kept;
    void * prefetch_done; // ggml_backend_event_t
    void * stream;        // dedicated CUDA stream
};

class moe_prefetcher {
public:
    moe_prefetcher();
    ~moe_prefetcher();
    bool start();          // creates thread + CUDA stream, pins to cores 64-75
    void stop();           // signal + join thread
    void launch_prefetch(
        const std::vector<prefetch_work_item> & items,
        void * completion_event);
    void wait_prefetch_fence(void * completion_event);
    void * get_stream() const;

private:
    void worker_loop();
    static bool pin_to_management_cores();  // pthread_setaffinity_np, cores 64-75

    std::thread                worker_;
    std::atomic<bool>          running_{false}, stop_requested_{false};
    std::mutex                 work_mutex_;
    std::condition_variable    work_cv_;
    bool                       work_ready_{false};
    std::vector<prefetch_work_item> work_items_;
    void *                     completion_event_{nullptr};
    void *                     h2d_stream_ = nullptr;
    std::atomic<int>           fence_{0};
};
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single thread, batch work | All layers batched into one `launch_prefetch` call; thread serially issues `cudaMemcpyAsync` per layer on dedicated stream |
| No thread pool | 12 management cores handle DDR5 DMA setup — one thread optimal for serial H2D dispatch |
| Atomic fence for done | `store(1, release)` / `load(acquire)` avoids kernel transition latency |
| `pthread_setaffinity_np` | `#ifdef __linux__` only. `cpu_set_t` with bits 64-75 set |
| Dedicated CUDA stream | `cudaStreamCreate` in `start()`, isolated from compute stream |
| Fallback to sync | If `!moe_prefetch_started` or `ubatch.n_tokens > 1`, uses original synchronous loop |

**Dependencies:** `ggml-cuda.h` (for `pipeline_expert_skip_prefetch`), `<thread>`, `<mutex>`, `<condition_variable>`, `<atomic>`, `<vector>`.

---

## Module 4: Integration Patch

### `llama-context.h` changes

**Remove:** Lines 116-221 (all 3 struct definitions + macros).

**Add includes:**
```cpp
#ifdef GGML_USE_CUDA
#include "moe-static-bunker.h"
#include "moe-hijacker.h"
#include "moe-prefetcher.h"
#endif
```

**Add members:**
```cpp
moe::moe_prefetcher moe_prefetch;
bool                moe_prefetch_started = false;
```

### `llama-context.cpp` changes

**Remove:** Lines 91-362 (all struct method implementations).

**Constructor:** After `h2_init()` → `moe_prefetch_started = moe_prefetch.start();`
**Destructor:** Before `h2_destroy()` → `if (moe_prefetch_started) moe_prefetch.stop();`

### `process_ubatch()` orchestration (single-token path)

```
Phase 1  → routing+attention graph build/alloc/compute      (unchanged)
Phase 1a → threshold kernels (all MoE layers)                (unchanged)
Phase 1b → batched D2H reads                                 (unchanged)
Phase 1c → moe_prefetch.launch_prefetch(items, done_event)   (NEW: async)
         ↓ main thread proceeds in parallel ↓
Phase 2  → build_graph, cascade_force, sched_alloc           (reordered: now between launch + wait)
         → moe_prefetch.wait_prefetch_fence(done_event)      (NEW: block)
         → scan_and_update_snapshots, copy_to_static         (→ moe:: namespace)
         → graph_compute                                     (unchanged)
```

### `src/CMakeLists.txt` patch

Add in alphabetical order:
```
    moe-hijacker.cpp
    moe-prefetcher.cpp
    moe-static-bunker.cpp
```

### Net change footprint

| File | Removed | Added | Net |
|------|---------|-------|-----|
| `llama-context.h` | 105 | 10 | -95 |
| `llama-context.cpp` | 275 | 40 | -235 |
| `moe-static-bunker.h` | — | 90 | new |
| `moe-static-bunker.cpp` | — | 90 | new |
| `moe-hijacker.h` | — | 40 | new |
| `moe-hijacker.cpp` | — | 200 | new |
| `moe-prefetcher.h` | — | 50 | new |
| `moe-prefetcher.cpp` | — | 130 | new |

**Total:** -330 lines in context, +600 lines in 3 new modules. Zero semantic changes to preserved edge cases.

---

## Verification Checklist

1. [ ] `scan_and_hijack` matches same tensors as before (name-based, same `TENSOR_NAMES[7]`)
2. [ ] `scan_and_update_snapshots` saves `orig_data` before overwriting
3. [ ] `copy_data_to_static` issues `cudaMemcpyAsync` with `cudaMemcpyDefault` on same stream
4. [ ] `cascade_force_moe_consumers` only forces VIEW(37) and ADD(2) with matching parent
5. [ ] `force_idxs_to_cpu` NOT called on routing indices (ARGSORT, expert tokens)
6. [ ] Prefetcher launches only for `n_tokens == 1` (generation mode)
7. [ ] Prefetcher wait_fence called before compact buffer access in Phase 2
8. [ ] `moe_prefetch.stop()` before `h2_destroy()` in destructor
9. [ ] Build compiles with `-DGGML_CUDA=ON -DLLAMA_DEEPSEEK_PIPELINE=ON`
10. [ ] Inference correctness verified on Kimi-K2.7 and DeepSeek-V3 (first 100 tokens, seeds 0,1,2)
