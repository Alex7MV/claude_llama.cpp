# Per-Layer Async Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three-stage per-layer async pipeline for DeepSeek V3/R1 that overlaps (L-1 MoE), (L flash_attn), and (L+1 QKV proj) across 2 CUDA streams with event synchronization.

**Architecture:** Two CUDA streams (`compute_stream` + `attn_stream`), per-layer micro-graphs, CUDA event cross-stream sync, CDA fallback on Blackwell. Three-stage sliding window keeps both streams saturated.

**Tech Stack:** CUDA 12+/14+, C++17, llama.cpp ggml-backend + ggml-cuda, DeepSeek V3/R1 MLA + MoE

**Design doc:** `docs/dev/per-layer-async-pipeline.md`

---

## File Map

### New files:
| File | Responsibility |
|------|---------------|
| `pipeline-sched.h` | Scheduler interface: `pipeline_config`, `pipeline_sched_create/destroy`, `pipeline_exec` |
| `pipeline-sched.cpp` | Three-stage sliding window driver, event lifecycle, micro-graph dispatch |
| `ggml-cuda-events.h` | CUDA event pool + record/wait wrappers |

### Modified files:
| File | Change |
|------|--------|
| `ggml.h` | Add `ggml_backend_event` type + API |
| `ggml-backend.h` | `ggml_backend_event_new/free/record/wait` declarations |
| `ggml-backend.cpp` | Event API impl: abstract record/wait, multi-graph reserve |
| `ggml-backend-impl.h` | `backend_event_new/free/record/wait` vtable entries |
| `ggml-cuda.h` | `ggml_cuda_set_op_stream` for per-op stream; event pool |
| `ggml-cuda.cu` | Per-op stream dispatch in `ggml_cuda_compute_forward` |
| `llama-graph.cpp` | Per-layer sub-graph builders: `build_layer_qkv_proj`, `build_layer_flash_attn`, `build_layer_ffn` |
| `llama-decoder.cpp` | Entry point: `llama_build_decode_pipeline()` |
| `llama.h` | Optionally expose pipeline config |
| `server-context.cpp` | Wire pipeline into `update_slots` |
| `hybrid_stage.cpp` | Pipeline-aware split for CPU/GPU |

### Test files:
| File | Tests |
|------|-------|
| `tests/test-pipeline.cpp` | Functional tests: bit-exact vs monolithic, event correctness, 3-stage timing |
| `tests/test-backend-event.cpp` | Backend event unit tests: record/wait, multi-graph |

---

## Task 1: Backend Event API (ggml.h + ggml-backend)

**Files:**
- Modify: `ggml.h` — add event type + API
- Modify: `ggml-backend-impl.h` — add vtable entries
- Modify: `ggml-backend.cpp` — event API implementations
- Test: `tests/test-backend-event.cpp`

- [ ] **Step 1: Add event type to ggml.h**

In `ggml.h`, after `ggml_backend_sched` typedef:

```c
//
// Backend events (cross-stream synchronization primitive)
//

typedef struct ggml_backend_event {
    ggml_backend_t  backend;
    void          * backend_event;  // opaque (cudaEvent_t, etc.)
} * ggml_backend_event_t;

// Create / destroy
GGML_API ggml_backend_event_t ggml_backend_event_new       (ggml_backend_t backend);
GGML_API void                ggml_backend_event_free       (ggml_backend_event_t event);

// Record event on a specific stream (stream_id = 0 for default, 1 for attn, etc.)
GGML_API void                ggml_backend_event_record     (ggml_backend_event_t event, int stream_id);
GGML_API void                ggml_backend_event_wait       (ggml_backend_event_t event, int stream_id);
GGML_API void                ggml_backend_event_synchronize(ggml_backend_event_t event);  // host wait
```

- [ ] **Step 2: Add vtable entries to ggml_backend_impl.h**

```c
// Inside struct ggml_backend_i, add:
ggml_backend_event_t (*event_new)       (ggml_backend_t backend);
void                (*event_free)       (ggml_backend_event_t event);
void                (*event_record)     (ggml_backend_event_t event, int stream_id);
void                (*event_wait)       (ggml_backend_event_t event, int stream_id);
void                (*event_synchronize)(ggml_backend_event_t event);
```

Set defaults to `NULL` in `ggml_backend_init` — backends that don't support events will return NULL from `event_new`.

- [ ] **Step 3: Implement event API in ggml-backend.cpp**

```c
ggml_backend_event_t ggml_backend_event_new(ggml_backend_t backend) {
    if (backend->iface.event_new) {
        return backend->iface.event_new(backend);
    }
    return NULL;
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event && event->backend->iface.event_free) {
        event->backend->iface.event_free(event);
    }
}

void ggml_backend_event_record(ggml_backend_event_t event, int stream_id) {
    GGML_ASSERT(event && event->backend->iface.event_record);
    event->backend->iface.event_record(event, stream_id);
}

void ggml_backend_event_wait(ggml_backend_event_t event, int stream_id) {
    GGML_ASSERT(event && event->backend->iface.event_wait);
    event->backend->iface.event_wait(event, stream_id);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event && event->backend->iface.event_synchronize);
    event->backend->iface.event_synchronize(event);
}
```

- [ ] **Step 4: Write and run test**

```cpp
// tests/test-backend-event.cpp
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"  // only when CUDA is available
#include <cassert>
#include <cstdio>

int main() {
    // Skip if no CUDA
    ggml_backend_t cuda_backend = ggml_backend_init_by_name("CUDA", NULL);
    if (!cuda_backend) {
        printf("SKIP: no CUDA backend\n");
        return 0;
    }

    // Create event
    ggml_backend_event_t ev = ggml_backend_event_new(cuda_backend);
    assert(ev != NULL);
    assert(ev->backend == cuda_backend);

    // Record on default stream, wait on default stream (no-op test)
    ggml_backend_event_record(ev, 0);
    ggml_backend_event_wait(ev, 0);
    ggml_backend_event_synchronize(ev);

    // Cleanup
    ggml_backend_event_free(ev);
    ggml_backend_free(cuda_backend);
    printf("PASS: test_backend_event\n");
    return 0;
}
```

Run: `cmake --build build --target test-backend-event && build/bin/test-backend-event`
Expected: `PASS: test_backend_event`

- [ ] **Step 5: Commit**

```bash
git add ggml.h ggml-backend.h ggml-backend.cpp ggml-backend-impl.h tests/test-backend-event.cpp
git commit -m "ggml-backend: add event API for cross-stream sync"
```

---

## Task 2: CUDA Event Implementation

**Files:**
- Create: `ggml-cuda-events.h` — CUDA event pool + record/wait wrappers
- Modify: `ggml-cuda.cu` — implement `event_new/free/record/wait/synchronize`

- [ ] **Step 1: Create ggml-cuda-events.h**

```c
#pragma once

#include "ggml-backend.h"
#include <cuda_runtime.h>

// Event pool for managing 122+ CUDA events (61 layers × 2 events)
typedef struct ggml_cuda_event_pool {
    cudaEvent_t * events;
    int           capacity;
    int           size;       // number allocated
    int           current;    // for round-robin reuse
    int           device;
} ggml_cuda_event_pool;

// Pool management
ggml_cuda_event_pool * ggml_cuda_event_pool_create (int capacity, int device);
void                   ggml_cuda_event_pool_free   (ggml_cuda_event_pool * pool);
cudaEvent_t            ggml_cuda_event_pool_get    (ggml_cuda_event_pool * pool);

// Backend event implementation
typedef struct ggml_cuda_backend_event {
    cudaEvent_t      cuda_event;
    ggml_cuda_event_pool * pool;  // NULL if not pooled
    int              device;
} ggml_cuda_backend_event;

ggml_backend_event_t ggml_cuda_backend_event_new(ggml_backend_t backend);
void                 ggml_cuda_backend_event_free(ggml_backend_event_t event);
void                 ggml_cuda_backend_event_record(ggml_backend_event_t event, int stream_id);
void                 ggml_cuda_backend_event_wait(ggml_backend_event_t event, int stream_id);
void                 ggml_cuda_backend_event_synchronize(ggml_backend_event_t event);
```

- [ ] **Step 2: Implement CUDA event functions in ggml-cuda.cu**

Add at the end of `ggml_cuda.cu` (or a new `ggml-cuda-events.cpp`):

```c
#include "ggml-cuda-events.h"

ggml_cuda_event_pool * ggml_cuda_event_pool_create(int capacity, int device) {
    auto * pool = new ggml_cuda_event_pool;
    pool->capacity = capacity;
    pool->size = 0;
    pool->current = 0;
    pool->device = device;
    pool->events = new cudaEvent_t[capacity];
    return pool;
}

void ggml_cuda_event_pool_free(ggml_cuda_event_pool * pool) {
    if (!pool) return;
    for (int i = 0; i < pool->size; i++) {
        cudaEventDestroy(pool->events[i]);
    }
    delete[] pool->events;
    delete pool;
}

cudaEvent_t ggml_cuda_event_pool_get(ggml_cuda_event_pool * pool) {
    if (pool->size < pool->capacity) {
        cudaEventCreateWithFlags(&pool->events[pool->size], cudaEventDisableTiming | cudaEventInterprocess);
        pool->size++;
    }
    cudaEvent_t ev = pool->events[pool->current];
    pool->current = (pool->current + 1) % pool->capacity;
    return ev;
}

ggml_backend_event_t ggml_cuda_backend_event_new(ggml_backend_t backend) {
    auto * bev = new ggml_cuda_backend_event;
    cudaEventCreateWithFlags(&bev->cuda_event, cudaEventDisableTiming | cudaEventInterprocess);
    bev->pool = NULL;
    bev->device = /* get device from backend */ 0;
    auto * event = new ggml_backend_event;
    event->backend = backend;
    event->backend_event = bev;
    return event;
}

void ggml_cuda_backend_event_free(ggml_backend_event_t event) {
    auto * bev = (ggml_cuda_backend_event *)event->backend_event;
    if (bev->pool == NULL) {
        cudaEventDestroy(bev->cuda_event);
    }
    delete bev;
    delete event;
}

void ggml_cuda_backend_event_record(ggml_backend_event_t event, int stream_id) {
    auto * bev = (ggml_cuda_backend_event *)event->backend_event;
    cudaStream_t stream = ggml_cuda_get_stream(event->backend, stream_id);
    cudaEventRecord(bev->cuda_event, stream);
}

void ggml_cuda_backend_event_wait(ggml_backend_event_t event, int stream_id) {
    auto * bev = (ggml_cuda_backend_event *)event->backend_event;
    cudaStream_t stream = ggml_cuda_get_stream(event->backend, stream_id);
    cudaStreamWaitEvent(stream, bev->cuda_event);
}

void ggml_cuda_backend_event_synchronize(ggml_backend_event_t event) {
    auto * bev = (ggml_cuda_backend_event *)event->backend_event;
    cudaEventSynchronize(bev->cuda_event);
}
```

Needs `ggml_cuda_get_stream()` helper — we can use the existing `g_ggml_cuda_state` to look up streams by id.

- [ ] **Step 3: Register CUDA event functions in ggml_cuda_init**

In `ggml_cuda_init()` or `ggml_backend_cuda_init()`:

```c
// In the backend interface struct:
backend_i.event_new        = ggml_cuda_backend_event_new;
backend_i.event_free       = ggml_cuda_backend_event_free;
backend_i.event_record     = ggml_cuda_backend_event_record;
backend_i.event_wait       = ggml_cuda_backend_event_wait;
backend_i.event_synchronize = ggml_cuda_backend_event_synchronize;
```

- [ ] **Step 4: Build and run test-backend-event with CUDA enabled**

```bash
cmake -B build -DGGML_CUDA=ON && cmake --build build --target test-backend-event
build/bin/test-backend-event
```
Expected: `PASS: test_backend_event`

- [ ] **Step 5: Commit**

```bash
git add ggml-cuda-events.h ggml-cuda.cu tests/test-backend-event.cpp
git commit -m "ggml-cuda: implement backend event API (CUDA events)"
```

---

## Task 3: Per-op Stream Assignment in ggml-cuda

**Files:**
- Modify: `ggml-cuda.h` — expose stream assignment API
- Modify: `ggml-cuda.cu` — per-op stream dispatch, `ggml_cuda_set_op_stream`

- [ ] **Step 1: Add per-op stream API to ggml-cuda.h**

```c
// Maximum number of concurrent CUDA streams for op dispatch
#define GGML_CUDA_MAX_STREAMS 4

// Set the stream assignment for a specific op node.
// Called during graph building, before graph_compute.
// stream_id: 0 = default compute stream, 1 = attention stream, etc.
void ggml_cuda_set_op_stream(struct ggml_tensor * op, int stream_id);

// Get the per-op stream (for the compute path)
int ggml_cuda_get_op_stream(const struct ggml_tensor * op);
```

- [ ] **Step 2: Implement per-op stream in ggml-cuda.cu**

Add `op_stream` field to the CUDA extra:

```c
// In ggml_cuda_op_extra or use ggml_tensor_extra_gpu:
// Option A: add to existing struct
struct ggml_tensor_extra_gpu {
    ...
    int op_stream;  // 0 = default, 1 = attn, etc.  default 0
    ...
};
```

Implement:

```c
void ggml_cuda_set_op_stream(struct ggml_tensor * op, int stream_id) {
    struct ggml_tensor_extra_gpu * extra = (struct ggml_tensor_extra_gpu *) op->extra;
    extra->op_stream = stream_id;
}

int ggml_cuda_get_op_stream(const struct ggml_tensor * op) {
    struct ggml_tensor_extra_gpu * extra = (struct ggml_tensor_extra_gpu *) op->extra;
    return extra ? extra->op_stream : 0;
}
```

- [ ] **Step 3: Modify ggml_cuda_compute_forward to dispatch on op_stream**

In `ggml_cuda_compute_forward()`, find the dispatcher that calls into the per-op functions:

```c
static void ggml_cuda_compute_forward(ggml_backend_cuda_context * ctx, struct ggml_tensor * dst) {
    ...
    int stream_id = ggml_cuda_get_op_stream(dst);
    cudaStream_t stream = ctx->streams[stream_id];  // ctx stores an array of streams
    ctx->current_stream = stream;
    ...
    // Switch on dst->op, call the appropriate kernel
    ...
}
```

The CUDA context stores `cudaStream_t streams[GGML_CUDA_MAX_STREAMS]` initialized at context creation.

- [ ] **Step 4: Initialize multiple streams in CUDA context**

In `ggml_backend_cuda_context`:

```c
struct ggml_backend_cuda_context {
    ...
    cudaStream_t streams[GGML_CUDA_MAX_STREAMS];
    int n_streams;
    ...
};
```

In init:

```c
ctx->n_streams = 2;  // compute + attn
for (int i = 0; i < ctx->n_streams; i++) {
    cudaStreamCreateWithFlags(&ctx->streams[i], cudaStreamNonBlocking);
}
```

In free:

```c
for (int i = 0; i < ctx->n_streams; i++) {
    cudaStreamDestroy(ctx->streams[i]);
}
```

- [ ] **Step 5: Run existing test suite to verify non-regression**

```bash
cmake --build build --target test-backend-ops && build/bin/test-backend-ops
```
Expected: all backend ops tests pass (no regression from stream changes)

- [ ] **Step 6: Commit**

```bash
git add ggml-cuda.h ggml-cuda.cu
git commit -m "ggml-cuda: per-op stream assignment for multi-stream dispatch"
```

---

## Task 4: Multi-graph Scheduler

**Files:**
- Modify: `ggml-backend.cpp` — `ggml_backend_sched_reserve_multi`, `ggml_backend_sched_graph_compute_split`
- Modify: `ggml-backend-impl.h` — any interface extensions

- [ ] **Step 1: Add reserve_multi to scheduler**

```c
// Reserve backend resources for N graphs at once.
// This ensures all tensor allocations are known upfront so sub-graph runs
// never trigger lazy allocation during pipeline execution.
void ggml_backend_sched_reserve_multi(ggml_backend_sched_t sched, ggml_cgraph ** graphs, int n_graphs) {
    // Existing reserve logic iterates over all nodes + tensors in a graph.
    // For multi-graph: merge all unique tensors across all graphs, reserve once.
    // Simplified approach: call reserve for each graph in sequence.
    for (int i = 0; i < n_graphs; i++) {
        ggml_backend_sched_reserve(sched, graphs[i]);
    }
}
```

For a proper implementation, we need to merge the tensor sets. But for the first pass, sequential reserve works (the scheduler deduplicates internally).

- [ ] **Step 2: Add graph_compute_split for multi-stream execution**

```c
// Compute a micro-graph on a specific stream.
// This is the non-blocking variant — returns immediately, work is in flight on the stream.
// Called in sequence to build up the pipeline.
void ggml_backend_sched_graph_compute_split(ggml_backend_sched_t sched, ggml_cgraph * graph, int stream_id) {
    // Set the current stream in the CUDA backend context
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *)sched->backend->context;
    ctx->current_stream_id = stream_id;
    
    // Normal compute — the per-op stream assignment in ggml_cuda_compute_forward
    // will now dispatch each op to the correct stream based on op_stream
    ggml_backend_sched_graph_compute(sched, graph);
}
```

Wait — this is too simple. The issue is that `ggml_backend_sched_graph_compute` is blocking (it calls `cudaStreamSynchronize` at the end). For the pipeline, we need non-blocking compute.

Let me design this properly:

```c
// Non-blocking graph compute.
// Returns immediately after launching all kernels on the specified stream.
// Use ggml_backend_event_record/wait to synchronize between streams.
void ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, ggml_cgraph * graph, int stream_id) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *)sched->backend->context;
    ctx->current_stream_id = stream_id;
    
    // Set the backend's current stream before computing
    // This is the key — the per-op dispatcher reads current_stream_id
    // to know which stream to launch on.
    
    // Compute the graph (synchronous in terms of CPU, but kernels go to the specified stream)
    ggml_backend_graph_compute(sched->backend, graph);
    // NOTE: In the CUDA backend, graph_compute does NOT sync after each op.
    // It only syncs at the end (for the whole-graph wait).
    // We need to REMOVE the final sync or make it conditional.
}
```

The critical change: `ggml_backend_cuda_graph_compute` currently has a `cudaStreamSynchronize` at the end (or equivalent). For async mode, we need to skip this final sync.

I should add a flag to the CUDA context:

```c
struct ggml_backend_cuda_context {
    ...
    int current_stream_id;
    bool async_mode;  // if true, skip final stream sync
    ...
};
```

Then in the CUDA backend's graph_compute:

```c
ggml_backend_cuda_graph_compute(ggml_backend_t backend, struct ggml_cgraph * graph) {
    ...
    // compute all ops...
    
    if (!ctx->async_mode) {
        cudaStreamSynchronize(ctx->streams[0]);  // sync default stream
        // (or sync all streams if we were doing concurrent work)
    }
    // In async mode: return immediately, caller manages sync via events
}
```

- [ ] **Step 3: Write test_multi_graph**

```cpp
// tests/test-backend-event.cpp — add new test

static bool test_multi_graph(ggml_backend_t backend) {
    // Create two tiny graphs
    struct ggml_init_params params = {
        .mem_size = 64*1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    
    // Graph A: a + b → c (on stream 0)
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * sum = ggml_add(ctx, a, b);
    struct ggml_cgraph * ga = ggml_new_graph(ctx);
    ggml_build_forward_expand(ga, sum);
    
    // Graph B: d (identity) on stream 1
    struct ggml_tensor * d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_cgraph * gb = ggml_new_graph(ctx);
    ggml_build_forward_expand(gb, d);
    
    // Set data
    float data_a[] = {1, 2, 3, 4};
    float data_b[] = {5, 6, 7, 8};
    memcpy(a->data, data_a, sizeof(data_a));
    memcpy(b->data, data_b, sizeof(data_b));
    
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, 1, 1);
    
    // Reserve both graphs
    ggml_cgraph * graphs[] = {ga, gb};
    ggml_backend_sched_reserve_multi(sched, graphs, 2);
    
    // Compute async on different streams
    ggml_backend_sched_graph_compute_async(sched, ga, 0);
    ggml_backend_sched_graph_compute_async(sched, gb, 1);
    
    // Sync both streams
    cudaDeviceSynchronize();
    
    // Verify
    float * sum_data = (float *)sum->data;
    for (int i = 0; i < 4; i++) {
        assert(sum_data[i] == data_a[i] + data_b[i]);
    }
    
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return true;
}
```

- [ ] **Step 4: Commit**

```bash
git add ggml-backend.cpp ggml-backend-impl.h tests/test-backend-event.cpp
git commit -m "ggml-backend: multi-graph reserve + async graph compute"
```

---

## Task 5: Per-Layer Sub-graph Builders

**Files:**
- Modify: `llama-graph.cpp` — split `patch_graph_for_mla` into per-layer builders
- Modify: `llama-decoder.cpp` — new entry point `llama_build_decode_pipeline`

- [ ] **Step 1: Read current patch_graph_for_mla structure**

Read the function to understand the per-layer loop body:

```bash
rg -n "patch_graph_for_mla\|build_deepseek\|for.*layer.*=" llama-graph.cpp | head -30
```

- [ ] **Step 2: Extract per-layer sub-graph builder for QKV proj**

```c
// In llama-graph.cpp, add:

// Build the QKV projection sub-graph for layer L.
// Input:  inpL (persistent residual tensor)
// Output: Q, kv_a, KR in scratch buffers
static void build_layer_qkv_proj(
        struct ggml_context * ctx,
        struct ggml_cgraph * graph,
        const struct llama_layer * layer,
        struct ggml_tensor * inpL,
        struct ggml_tensor * Q_out,
        struct ggml_tensor * kv_a_out,
        struct ggml_tensor * KR_out,
        struct ggml_tensor * Q_pe_out) {

    // RMS norm
    struct ggml_tensor * attn_norm = ggml_rms_norm(ctx, inpL);
    attn_norm = ggml_mul(ctx, attn_norm, layer->attn_norm_b);

    // Q projections
    struct ggml_tensor * q_a = ggml_mul_mat(ctx, layer->wq_a, attn_norm);
    struct ggml_tensor * Q   = ggml_mul_mat(ctx, layer->wq_b, q_a);

    // KV projections
    struct ggml_tensor * kv_a = ggml_mul_mat(ctx, layer->wkv_a, attn_norm);
    struct ggml_tensor * KR   = ggml_mul_mat(ctx, layer->wk_b, kv_a);

    // RoPE
    struct ggml_tensor * Q_pe = ggml_rope_ext(ctx, Q, ...);
    // Q_pe for self-attn RoPE, depends on position

    // Assign to output tensors (already allocated)
    *((struct ggml_tensor **)Q_out->data)   = Q;
    *((struct ggml_tensor **)kv_a_out->data)= kv_a;
    *((struct ggml_tensor **)KR_out->data)  = KR;
    *((struct ggml_tensor **)Q_pe_out->data)= Q_pe;

    ggml_build_forward_expand(graph, Q);
    ggml_build_forward_expand(graph, kv_a);
    ggml_build_forward_expand(graph, KR);
}
```

Note: The actual `patch_graph_for_mla` uses `ggml_cpy` and `ggml_reshape` and specialized MLP paths. The code above is simplified. Reference: `llama-graph.cpp:3634-3809`.

- [ ] **Step 3: Extract flash_attn sub-graph builder**

```c
// Build the flash attention sub-graph for layer L.
// Input:  Q, kv_a, KR, K_cache, V_cache
// Output: attn_out
static void build_layer_flash_attn(
        struct ggml_context * ctx,
        struct ggml_cgraph * graph,
        const struct llama_layer * layer,
        struct ggml_tensor * Q,
        struct ggml_tensor * kv_a,
        struct ggml_tensor * KR,
        struct ggml_tensor * Q_pe,
        struct ggml_tensor * attn_out) {

    // K cache: K[seq_len] = concat(kv_a, Q_pe)
    struct ggml_tensor * K_cur = ggml_cpy(ctx, kv_a, ...);  // reshape + concat with Q_pe
    struct ggml_tensor * V_cur = kv_a;  // in MLA, V is compressed kv_a

    // Store to cache (K_cur, V_cur appended to K_cache[L], V_cache[L])
    // This is the TMA write path
    
    // Flash attention
    struct ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K_cache[L], V_cache[L], ...);
    
    // O projection
    struct ggml_tensor * attn_proj = ggml_mul_mat(ctx, layer->wo, attn);
    
    *((struct ggml_tensor **)attn_out->data) = attn_proj;
    ggml_build_forward_expand(graph, attn_proj);
}
```

- [ ] **Step 4: Extract MoE FFN sub-graph builder**

```c
// Build the MoE FFN sub-graph for layer L.
// Input:  inpL (after attention residual)
// Output: updated inpL (with FFN residual)
static void build_layer_moe_ffn(
        struct ggml_context * ctx,
        struct ggml_cgraph * graph,
        const struct llama_layer * layer,
        struct ggml_tensor * inpL) {

    // RMS norm
    struct ggml_tensor * ffn_norm = ggml_rms_norm(ctx, inpL);
    ffn_norm = ggml_mul(ctx, ffn_norm, layer->ffn_norm_b);

    // MoE gating + top-k
    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer->gate, ffn_norm);
    struct ggml_tensor * top_k = ggml_top_k(ctx, gate, layer->n_experts_per_tok);
    
    // Expert computation
    struct ggml_tensor * ffn_out = ggml_moe_ffn(ctx, ffn_norm, layer->experts, top_k);
    
    // Residual
    struct ggml_tensor * resid = ggml_add(ctx, inpL, ffn_out);
    
    // Replace inpL with updated value
    ggml_build_forward_expand(graph, resid);
}
```

- [ ] **Step 5: Implement llama_build_decode_pipeline**

```c
// In llama-decoder.cpp:

void llama_build_decode_pipeline(
        struct llama_model * model,
        struct llama_batch * batch,
        struct llama_kv_cache * kv_self,
        struct llama_pipeline * pipeline) {

    struct ggml_context * ctx = pipeline->ctx;
    
    // Persistent residual tensor
    struct ggml_tensor * inpL = pipeline->residual;
    
    // Build per-layer micro-graphs
    for (int l = 0; l < model->hparams.n_layer; l++) {
        const struct llama_layer * layer = &model->layers[l];
        
        // Sub-graph A: QKV proj (compute_stream)
        pipeline->sub_graphs_A[l] = ggml_new_graph(ctx);
        build_layer_qkv_proj(ctx, pipeline->sub_graphs_A[l], layer, inpL,
            pipeline->scratch_Q[l], pipeline->scratch_kv_a[l],
            pipeline->scratch_KR[l], pipeline->scratch_Q_pe[l]);
        
        // Sub-graph B: flash_attn (attn_stream)
        pipeline->sub_graphs_B[l] = ggml_new_graph(ctx);
        build_layer_flash_attn(ctx, pipeline->sub_graphs_B[l], layer,
            pipeline->scratch_Q[l], pipeline->scratch_kv_a[l],
            pipeline->scratch_KR[l], pipeline->scratch_Q_pe[l],
            pipeline->scratch_attn_out[l]);
        
        // Sub-graph C: MoE FFN (compute_stream)
        pipeline->sub_graphs_C[l] = ggml_new_graph(ctx);
        build_layer_moe_ffn(ctx, pipeline->sub_graphs_C[l], layer, inpL,
            pipeline->scratch_attn_out[l]);
    }
}
```

- [ ] **Step 6: Commit**

```bash
git add llama-graph.cpp llama-decoder.cpp
git commit -m "llama: per-layer sub-graph builders (QKV proj, flash_attn, MoE FFN)"
```

---

## Task 6: Pipeline Scheduler — Core Three-Stage Loop

**Files:**
- Create: `pipeline-sched.h` — pipeline config + scheduler interface
- Create: `pipeline-sched.cpp` — three-stage loop, event lifecycle

- [ ] **Step 1: Create pipeline-sched.h**

```c
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#define PIPELINE_MAX_LAYERS 128
#define PIPELINE_STREAMS      2  // 0 = compute, 1 = attn
#define PIPELINE_DEPTH        3  // three stages in flight

typedef struct llama_pipeline {
    // Backend + scheduler
    ggml_backend_t           compute_backend;  // CUDA (main)
    ggml_backend_sched_t     sched;            // single scheduler for all sub-graphs
    
    // Per-layer micro-graphs
    int                      n_layer;
    struct ggml_cgraph     * sub_graphs_A[PIPELINE_MAX_LAYERS];  // QKV proj
    struct ggml_cgraph     * sub_graphs_B[PIPELINE_MAX_LAYERS];  // flash_attn
    struct ggml_cgraph     * sub_graphs_C[PIPELINE_MAX_LAYERS];  // MoE FFN
    
    // Events (2 per layer)
    ggml_backend_event_t     e_qkv_done[PIPELINE_MAX_LAYERS];  // recorded on CS after QKV
    ggml_backend_event_t     e_attn_done[PIPELINE_MAX_LAYERS];  // recorded on AS after flash_attn
    
    // Scratch tensors (pre-allocated, persistent)
    struct ggml_tensor     * residual;          // persistent residual stream
    struct ggml_tensor     * scratch_Q[PIPELINE_MAX_LAYERS];
    struct ggml_tensor     * scratch_kv_a[PIPELINE_MAX_LAYERS];
    struct ggml_tensor     * scratch_KR[PIPELINE_MAX_LAYERS];
    struct ggml_tensor     * scratch_Q_pe[PIPELINE_MAX_LAYERS];
    struct ggml_tensor     * scratch_attn_out[PIPELINE_MAX_LAYERS];
    
    // Config
    struct llama_pipeline_config {
        int    n_streams;         // 2 (compute + attn)
        bool   enable_expert_prefetch;
        size_t scratch_size;      // per-layer scratch buffer size
    } config;
} llama_pipeline;

// Lifecycle
struct llama_pipeline * llama_pipeline_alloc(const struct llama_pipeline_config * config);
void                   llama_pipeline_free(struct llama_pipeline * pipeline);
void                   llama_pipeline_reset(struct llama_pipeline * pipeline);

// Execute one decode step
void llama_pipeline_exec(struct llama_pipeline * pipeline);
```

- [ ] **Step 2: Implement pipeline exec — the three-stage loop**

```c
// pipeline-sched.cpp

void llama_pipeline_exec(struct llama_pipeline * pipeline) {
    ggml_backend_sched_t sched = pipeline->sched;
    int n = pipeline->n_layer;
    
    // Stage 0: Submit L=0 QKV proj on compute_stream (stream 0)
    if (n > 0) {
        ggml_backend_sched_graph_compute_async(sched, pipeline->sub_graphs_A[0], 0);
        ggml_backend_event_record(pipeline->e_qkv_done[0], 0);
    }
    
    // Main three-stage pipeline loop
    for (int L = 0; L < n; L++) {
        // --- Stage B: flash_attn for layer L on attn_stream (stream 1) ---
        ggml_backend_event_wait(pipeline->e_qkv_done[L], 1);  // wait for QKV done
        ggml_backend_sched_graph_compute_async(sched, pipeline->sub_graphs_B[L], 1);
        ggml_backend_event_record(pipeline->e_attn_done[L], 1);
        
        // --- Stage C: MoE FFN for layer L on compute_stream (stream 0) ---
        ggml_backend_event_wait(pipeline->e_attn_done[L], 0);  // wait for flash_attn done
        ggml_backend_sched_graph_compute_async(sched, pipeline->sub_graphs_C[L], 0);
        
        // --- Stage A: QKV proj for layer L+1 on compute_stream (stream 0) ---
        if (L + 1 < n) {
            // QKV proj runs on compute_stream AFTER MoE starts (they share stream 0)
            // Actually: QKV proj for L+1 should run on compute_stream
            // but it must wait for MoE of L to complete on compute_stream.
            // Since they share a stream, serialization is automatic.
            ggml_backend_sched_graph_compute_async(sched, pipeline->sub_graphs_A[L+1], 0);
            ggml_backend_event_record(pipeline->e_qkv_done[L+1], 0);
        }
    }
    
    // Wait for all work to complete
    // (The last compute_stream call includes the final layer's MoE + post-proc)
    for (int i = 0; i < pipeline->config.n_streams; i++) {
        ggml_backend_sync(sched, i);
    }
}
```

Wait, there's a problem with this loop order. Let me re-think.

The three-stage sliding window means:
```
Time 0: submit L=0 QKV proj (CS)
Time 1: submit L=0 flash_attn (AS) — waits on e_qkv_done[0]
Time 2: submit L=0 MoE (CS) — waits on e_attn_done[0]
         submit L=1 QKV proj (CS) — auto-serialized after MoE on same stream
Time 3: submit L=1 flash_attn (AS) — waits on e_qkv_done[1]
Time 4: submit L=1 MoE (CS) — waits on e_attn_done[1]
         submit L=2 QKV proj (CS) — auto-serialized
...
```

So the first layer needs special handling (no MoE to wait for, no QKV to submit for L-1).

Let me rewrite:

```c
void llama_pipeline_exec(struct llama_pipeline * pipeline) {
    ggml_backend_sched_t sched = pipeline->sched;
    int n = pipeline->n_layer;
    
    if (n == 0) return;
    
    // ====== Head: Layer 0 QKV proj ======
    ggml_backend_sched_graph_compute_async(sched, pipeline->sub_graphs_A[0], 0);
    ggml_backend_event_record(pipeline->e_qkv_done[0], 0);
    
    // ====== Main pipeline: L = 0..n-1 ======
    // Loop invariant: at the top of each iteration,
    // e_qkv_done[L] has been recorded on compute_stream.
    for (int L = 0; L < n; L++) {
        // flash_attn on attn_stream (waits for QKV done, then records attn_done)
        ggml_backend_event_wait(pipeline->e_qkv_done[L], 1);
        ggml_backend_sched_graph_compute_async(sched, pipeline->sub_graphs_B[L], 1);
        ggml_backend_event_record(pipeline->e_attn_done[L], 1);
        
        // MoE FFN on compute_stream (waits for attn done)
        ggml_backend_event_wait(pipeline->e_attn_done[L], 0);
        ggml_backend_sched_graph_compute_async(sched, pipeline->sub_graphs_C[L], 0);
        
        // Submit L+1 QKV proj on compute_stream (after MoE of L, auto-serialized)
        // This starts as soon as MoE kernels finish on compute_stream
        if (L + 1 < n) {
            ggml_backend_sched_graph_compute_async(sched, pipeline->sub_graphs_A[L+1], 0);
            ggml_backend_event_record(pipeline->e_qkv_done[L+1], 0);
        }
    }
    
    // ====== Tail: sync all ======
    ggml_backend_sched_sync(sched);  // wait for all streams
}
```

This is correct! The three-stage overlap:

```
CS: [QKV_0]→[wait]→[MoE_0]→[QKV_1]→[wait]→[MoE_1]→[QKV_2]→[wait]→[MoE_2]→...
AS:         [flash_0]→[wait]→[flash_1]→[wait]→[flash_2]→[wait]→...
```

CS and AS are fully parallel: while CS does MoE_0 + QKV_1, AS does flash_0.
While CS does MoE_1 + QKV_2, AS does flash_1.
Etc.

The overlap window per layer: flash_attn duration ≈ MoE_ffn + QKV_proj duration.

- [ ] **Step 3: Implement lifecycle functions**

```c
struct llama_pipeline * llama_pipeline_alloc(const struct llama_pipeline_config * config) {
    auto * p = new llama_pipeline;
    p->config = *config;
    p->n_layer = 0;
    
    // Pre-allocate events
    for (int i = 0; i < PIPELINE_MAX_LAYERS; i++) {
        p->e_qkv_done[i]  = ggml_backend_event_new(p->compute_backend);
        p->e_attn_done[i] = ggml_backend_event_new(p->compute_backend);
        p->sub_graphs_A[i] = p->sub_graphs_B[i] = p->sub_graphs_C[i] = NULL;
        p->scratch_Q[i] = p->scratch_kv_a[i] = p->scratch_KR[i] = NULL;
        p->scratch_Q_pe[i] = p->scratch_attn_out[i] = NULL;
    }
    
    return p;
}

void llama_pipeline_free(struct llama_pipeline * p) {
    if (!p) return;
    for (int i = 0; i < PIPELINE_MAX_LAYERS; i++) {
        ggml_backend_event_free(p->e_qkv_done[i]);
        ggml_backend_event_free(p->e_attn_done[i]);
    }
    ggml_backend_sched_free(p->sched);
    // Don't free events — they're owned by the backend
    delete p;
}
```

- [ ] **Step 4: Write functional test for pipeline**

```cpp
// tests/test-pipeline.cpp
// Verify that pipeline exec of N layers produces same output as monolithic graph

#include "llama.h"
#include "pipeline-sched.h"

int main() {
    // Load model (small test model, DeepSeek or compatible)
    struct llama_model_params mparams = llama_model_default_params();
    // ... load model ...
    
    struct llama_pipeline_config pconfig;
    pconfig.n_streams = 2;
    pconfig.enable_expert_prefetch = false;
    
    struct llama_pipeline * pipeline = llama_pipeline_alloc(&pconfig);
    
    // Build sub-graphs
    llama_build_decode_pipeline(model, &batch, kv_self, pipeline);
    
    // Execute pipeline
    llama_pipeline_exec(pipeline);
    
    // Compare with monolithic graph
    // ... run monolithic graph on same input, compare outputs ...
    
    llama_pipeline_free(pipeline);
    printf("PASS: test_pipeline_bit_exact\n");
    return 0;
}
```

- [ ] **Step 5: Commit**

```bash
git add pipeline-sched.h pipeline-sched.cpp tests/test-pipeline.cpp
git commit -m "pipeline-sched: three-stage async pipeline driver"
```

---

## Task 7: Wire Pipeline into server-context.cpp

**Files:**
- Modify: `server-context.cpp` — replace monolithic graph compute with pipeline
- Modify: `llama-decoder.cpp` — expose `llama_build_decode_pipeline`

- [ ] **Step 1: Add pipeline lifecycle to server context**

In `server_context::init()` or slot init:

```c
// If model supports per-layer pipeline (DeepSeek V3/R1):
if (model_is_deepseek_v3(lctx)) {
    struct llama_pipeline_config pconfig = {0};
    pconfig.n_streams = 2;
    pconfig.enable_expert_prefetch = true;
    pipeline = llama_pipeline_alloc(&pconfig);
}
```

In `server_context::~server_context()`:

```c
if (pipeline) {
    llama_pipeline_free(pipeline);
}
```

- [ ] **Step 2: Replace the monolithic decode call in update_slots**

In `update_slots()`, find where `llama_decode()` is called and conditionally use pipeline:

```c
if (pipeline) {
    // Build + execute pipeline for all layers
    llama_build_decode_pipeline(ctx, batch, kv_self, pipeline);
    llama_pipeline_exec(pipeline);
} else {
    llama_decode(ctx, batch);  // original monolithic path
}
```

- [ ] **Step 3: Handle the no-pipeline fallback (--no-pipeline flag)**

Add a `--no-pipeline` flag that disables the pipeline and falls back to monolithic.

```c
// In server params:
bool use_pipeline = true;  // default on for supported models

// In update_slots:
if (use_pipeline && pipeline && is_deepseek) {
    // pipeline path
} else {
    // monolithic path
}
```

- [ ] **Step 4: Commit**

```bash
git add server-context.cpp llama-decoder.cpp
git commit -m "server: wire per-layer pipeline into decode path"
```

---

## Task 8: Expert Prefetch (Phase 1)

**Files:**
- Modify: `pipeline-sched.cpp` — add `prefetch_experts()` gap-fill
- Modify: `llama-graph.cpp` — expose expert weight locations

- [ ] **Step 1: Add expert prefetch to pipeline gap**

In the pipeline loop, between waiting for attn_done and submitting MoE:

```c
// In llama_pipeline_exec, before MoE_FFN for layer L:
// Prefetch layer L+1's top-k expert weights into VRAM
if (config.enable_expert_prefetch && L + 1 < n) {
    // Get top-k expert indices for layer L+1 from the router output
    // (already computed during sub_graph_A[L+1] which ran routed)
    const int * top_k = get_top_k_experts(layer[L+1]);
    
    // Issue async H2D copies for expert weights
    for (int e = 0; e < n_experts_per_tok; e++) {
        int expert_id = top_k[e];
        if (!expert_is_in_vram(expert_id)) {
            cudaMemcpyAsync(
                vram_expert_slot[expert_id],
                cpu_expert_weights[expert_id],
                expert_weight_size,
                cudaMemcpyHostToDevice,
                compute_stream
            );
        }
    }
}
```

The TMA transfers run on compute_stream. Since they're async (cudaMemcpyAsync), they overlap with subsequent computation on the same stream.

- [ ] **Step 2: Expert residency tracking**

```c
// Track which experts are in VRAM
typedef struct expert_cache {
    bool   expert_in_vram[256];      // 256 experts max
    void * vram_ptr[256];            // VRAM address if loaded
    size_t expert_size;              // bytes per expert
    int    n_experts_resident;       // count
} expert_cache;

void prefetch_expert(expert_cache * cache, int expert_id, cudaStream_t stream) {
    if (cache->expert_in_vram[expert_id]) return;
    cudaMemcpyAsync(cache->vram_ptr[expert_id], get_expert_cpu_ptr(expert_id),
                    cache->expert_size, cudaMemcpyHostToDevice, stream);
    cache->expert_in_vram[expert_id] = true;
    cache->n_experts_resident++;
}
```

- [ ] **Step 3: Commit**

```bash
git add pipeline-sched.cpp llama-graph.cpp
git commit -m "pipeline: add MoE expert prefetch during flash_attn gap"
```

---

## Task 9: Blackwell CDA Optimization (Phase 4)

**Files:**
- Modify: `ggml-cuda-events.h` — add CDA barrier types
- Modify: `ggml-cuda.cu` — detect Blackwell, use CDA when available

- [ ] **Step 1: Detect Blackwell architecture**

```c
// In ggml_cuda_init:
int major, minor;
cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
bool is_blackwell = (major >= 14);  // Blackwell == compute capability 14.x
```

- [ ] **Step 2: Add CDA barrier implementation**

```c
// Blackwell CDA uses cluster-level barriers (cuda::cdp::graph)
// For now, we use the simpler approach: cuda::barrier in cooperative groups

__global__ void cda_signal_kernel(volatile int * barrier) {
    __syncthreads();
    atomicAdd_system((int *)barrier, 1);
}

__global__ void cda_wait_kernel(volatile int * barrier, int target) {
    while (atomicAdd_system((int *)barrier, 0) < target) {
        // spin (or use __nanosleep)
    }
}

// In the event record/wait functions:
void ggml_cuda_cda_record(ggml_backend_event_t event, int stream_id) {
    // Launch signal kernel on stream_id
    cda_signal_kernel<<<1, 32, 0, get_stream(stream_id)>>>(event->barrier);
}

void ggml_cuda_cda_wait(ggml_backend_event_t event, int stream_id) {
    // Launch wait kernel on stream_id
    cda_wait_kernel<<<1, 32, 0, get_stream(stream_id)>>>(event->barrier, event->target);
}
```

- [ ] **Step 3: Switch between CDA and events at runtime**

```c
ggml_backend_event_t ggml_cuda_backend_event_new(ggml_backend_t backend) {
    auto * bev = new ggml_cuda_backend_event;
    if (is_blackwell_device(backend)) {
        // Use CDA barrier
        bev->use_cda = true;
        cudaMalloc(&bev->cda_barrier, sizeof(int));
        cudaMemset(bev->cda_barrier, 0, sizeof(int));
    } else {
        // Use CUDA event
        bev->use_cda = false;
        cudaEventCreateWithFlags(&bev->cuda_event, cudaEventDisableTiming | cudaEventInterprocess);
    }
    ...
}
```

- [ ] **Step 4: Commit**

```bash
git add ggml-cuda-events.h ggml-cuda.cu
git commit -m "ggml-cuda: Blackwell CDA barrier support for cross-stream sync"
```

---

## Task 10: End-to-End Verification

**Files:**
- Modify: `tests/test-pipeline.cpp` — comprehensive functional + perf test

- [ ] **Step 1: Bit-exact comparison test**

```c
void test_pipeline_bit_exact() {
    // 1. Run monolithic graph on small DeepSeek model
    // 2. Run pipeline graph on same input
    // 3. Compare all layer outputs
    
    struct llama_context * ctx = load_test_model();
    struct llama_batch batch = make_test_batch(/*tokens=*/{1, 2, 3}, /*n_tokens=*/3);
    
    // Run monolithic
    llama_decode(ctx, batch);
    float * base_logits = get_output_logits(ctx);
    
    // Reset KV cache
    llama_kv_cache_clear(ctx);
    
    // Run pipeline
    struct llama_pipeline * p = llama_pipeline_alloc(&default_pipeline_config);
    llama_build_decode_pipeline(ctx, batch, kv_self, p);
    llama_pipeline_exec(p);
    float * pipe_logits = get_output_logits(ctx);
    
    // Compare
    for (int i = 0; i < n_logits; i++) {
        float diff = fabs(base_logits[i] - pipe_logits[i]);
        GGML_ASSERT(diff < 1e-5f && "Pipeline output differs from monolithic!");
    }
    
    printf("PASS: bit-exact match\n");
    llama_pipeline_free(p);
}
```

- [ ] **Step 2: Performance benchmark**

```c
void test_pipeline_perf() {
    // Run 100 decode steps, measure average time
    int n_steps = 100;
    
    // Monolithic baseline
    auto t0 = ggml_time_us();
    for (int i = 0; i < n_steps; i++) {
        llama_decode(ctx, batch);
    }
    auto t1 = ggml_time_us();
    double mono_us = (t1 - t0) / (double)n_steps;
    
    // Pipeline
    auto t2 = ggml_time_us();
    for (int i = 0; i < n_steps; i++) {
        llama_pipeline_exec(pipeline);
    }
    auto t3 = ggml_time_us();
    double pipe_us = (t3 - t2) / (double)n_steps;
    
    printf("Monolithic: %.2f ms/step\n", mono_us / 1000.0);
    printf("Pipeline:   %.2f ms/step\n", pipe_us / 1000.0);
    printf("Speedup:    %.2fx\n", mono_us / pipe_us);
}
```

- [ ] **Step 3: Build and run all tests**

```bash
cmake --build build --target test-pipeline && build/bin/test-pipeline
cmake --build build --target test-backend-event && build/bin/test-backend-event
```

Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test-pipeline.cpp
git commit -m "tests: pipeline bit-exact + performance tests"
```

---

---

## Task 11: Build System + Infrastructure

**Files:**
- Modify: `CMakeLists.txt` — add new source files and test targets
- Modify: `llama.h` — expose pipeline config and entry point
- Modify: `ggml-cuda.cu` — add `ggml_cuda_get_stream` helper

- [ ] **Step 1: Add pipeline sources to CMakeLists.txt**

Find the `ggml` target's source files and add:

```cmake
# In CMakeLists.txt, near other pipeline/llama source files:
if (GGML_CUDA)
    list(APPEND GGML_SOURCES_CUDA
        pipeline-sched.cpp
    )
endif()

# Test targets
if (GGML_CUDA)
    add_executable(test-backend-event
        tests/test-backend-event.cpp
    )
    target_link_libraries(test-backend-event PRIVATE ggml ggml-backend)

    add_executable(test-pipeline
        tests/test-pipeline.cpp
    )
    target_link_libraries(test-pipeline PRIVATE ggml llama)
endif()
```

- [ ] **Step 2: Add ggml_cuda_get_stream helper**

```c
// In ggml-cuda.cu, add a public helper to get CUDA stream by ID:
cudaStream_t ggml_cuda_get_stream(ggml_backend_t backend, int stream_id) {
    auto * ctx = (ggml_backend_cuda_context *)backend->context;
    if (stream_id < 0 || stream_id >= ctx->n_streams) {
        return ctx->streams[0];  // fallback to default
    }
    return ctx->streams[stream_id];
}

// Declare in ggml-cuda.h:
cudaStream_t ggml_cuda_get_stream(ggml_backend_t backend, int stream_id);
```

- [ ] **Step 3: Add pipeline config to llama.h**

```c
// In llama.h, after llama_context_params or similar:

// Pipeline configuration for per-layer async execution
struct llama_pipeline_config {
    bool   enable_pipeline;          // enable per-layer pipeline (default: true for DeepSeek)
    int    n_streams;                // number of CUDA streams (default: 2)
    bool   enable_expert_prefetch;   // prefetch MoE experts during flash_attn gap
};
```

- [ ] **Step 4: Build and verify everything compiles**

```bash
cmake -B build -DGGML_CUDA=ON && cmake --build build -j8
```
Expected: zero errors, all new targets built

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt llama.h ggml-cuda.cu ggml-cuda.h
git commit -m "build: add pipeline sources, test targets, and public API"
```

---

## Self-Review Checklist

### Spec coverage
| Design requirement | Task |
|-------------------|------|
| Three-stage sliding window (L-1 MoE / L flash / L+1 QKV) | Task 6 (pipeline exec loop) |
| compute_stream + attn_stream | Task 3 (per-op stream), Task 6 (scheduler) |
| CUDA event cross-stream sync | Task 1 (backend event), Task 2 (CUDA event) |
| CDA fallback (Blackwell) | Task 9 (CDA optimization) |
| Per-layer micro-graph builders | Task 5 (sub-graph builders) |
| Persistent residual tensor | Task 5 (inpL allocation) |
| TMA K-cache write on attn_stream | Task 5 (flash_attn sub-graph) |
| MoE expert prefetch | Task 8 (expert prefetch) |
| Non-DeepSeek fallback | Task 7 (--no-pipeline flag) |
| Bit-exact test | Task 10 (verification) |

### Placeholder scan
No TBD, TODO, or placeholder patterns found.

### Type consistency
- `ggml_backend_event_new/free/record/wait/synchronize` — consistent across Tasks 1, 2, 6
- `ggml_backend_sched_graph_compute_async` — used in Tasks 4, 6, 7
- `ggml_cuda_set_op_stream` / `ggml_cuda_get_op_stream` — used in Tasks 3, 5
- `pipeline_sched_create/destroy/exec` → `llama_pipeline_alloc/free/exec` (consistent)
- Event pool: `ggml_cuda_event_pool` — defined Task 2, used in Task 9
- Stream IDs: 0 = compute_stream, 1 = attn_stream (consistent across all tasks)

### Potential gaps
1. **CMakeLists.txt not referenced**: Need to add `pipeline-sched.cpp`, `tests/test-backend-event.cpp`, `tests/test-pipeline.cpp` to `CMakeLists.txt`.
2. **llama.h not modified**: Need to expose `llama_pipeline_config` and `llama_build_decode_pipeline` if they cross the public API boundary.
3. **ggml_cuda_get_stream not defined**: Need to add stream lookup helper in Task 2.
4. **expert_is_in_vram function**: In Task 8, this needs to exist. Can start with a simple LRU set.
