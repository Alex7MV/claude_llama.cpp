#pragma once

#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pipelined backend scheduler for hybrid CPU+GPU prefill.
// Wraps an existing ggml_backend_sched and executes its splits
// in a depth-3 pipeline: CPU compute -> TMA transfer -> GPU compute.

typedef struct ggml_backend_sched_pipelined * ggml_backend_sched_pipelined_t;

// C-compatible slot descriptor for GPU dispatch callbacks.
struct ggml_pipeline_slot_info {
    int32_t  split_idx;
    int32_t  first_layer;
    int32_t  num_layers;
    void *   cpu_buf;      // pinned host buffer (source for TMA)
    void *   gpu_buf;      // VRAM destination
    size_t   buf_bytes;
    int32_t  tma_desc_idx;
    void *   completion_event;  // opaque event handle for async completion tracking
    void *   gpu_async_done;    // pointer to std::atomic<bool> flag; callback sets to true when done
};

// Callback invoked by the orchestrator when a GPU-ready slot is available.
// Implementations (e.g. in ggml-cuda-epyc-pipeline.cu) launch the
// Blackwell tile kernel or cudaMemcpyAsync for this slot.
typedef void (*ggml_pipeline_gpu_dispatch_fn)(
    void * user_data,
    const struct ggml_pipeline_slot_info * slot);

ggml_backend_sched_pipelined_t ggml_backend_sched_pipelined_init(
    ggml_backend_sched_t base,
    int depth,
    int split_size,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll,
    ggml_backend_t gpu_backend);

// Register GPU dispatch callback and pinned buffer (set by CUDA glue).
void ggml_backend_sched_pipelined_set_gpu_dispatch(
    ggml_backend_sched_pipelined_t sched,
    ggml_pipeline_gpu_dispatch_fn fn,
    void * user_data);

// Set an opaque pinned buffer handle for slot allocations.
// The handle is passed through to the gpu_dispatch callback as slot->cpu_buf.
void ggml_backend_sched_pipelined_set_pinned_buf(
    ggml_backend_sched_pipelined_t sched,
    void * pinned_buf);

// Legacy compute path: single graph, threadpool rotation only.
// Does NOT use the 3-stage queue; kept for backward compatibility.
enum ggml_status ggml_backend_sched_pipelined_compute(
    ggml_backend_sched_pipelined_t sched,
    struct ggml_cgraph * gf);

// 3-stage pipelined compute for one split graph.
//
// The caller builds partial graphs (layer ranges) and submits them
// sequentially. The orchestrator maintains the lock-free queue across
// calls and dispatches stages:
//   - CPU-backend split  -> compute on rotated threadpool, enqueue slot
//   - GPU-backend split  -> dispatch GPU callback for queued slots,
//                           then compute GPU graph
//
// Depth limits in-flight slots. When depth is reached, this function
// blocks until a slot drains from the GPU stage.
//
// @param gf           Partial graph for this split
// @param first_layer  First layer in this split's range
// @param num_layers   Number of layers in this split
// @param is_cpu       true if this split runs on CPU backend
enum ggml_status ggml_backend_sched_pipelined_compute_split(
    ggml_backend_sched_pipelined_t sched,
    struct ggml_cgraph * gf,
    int32_t first_layer,
    int32_t num_layers,
    bool is_cpu);

// Finalize the pipeline after all splits are submitted.
// Waits for all in-flight slots to complete and recycles them.
enum ggml_status ggml_backend_sched_pipelined_drain(
    ggml_backend_sched_pipelined_t sched);

void ggml_backend_sched_pipelined_synchronize(ggml_backend_sched_pipelined_t sched);

void ggml_backend_sched_pipelined_free(ggml_backend_sched_pipelined_t sched);

#ifdef __cplusplus
}
#endif
