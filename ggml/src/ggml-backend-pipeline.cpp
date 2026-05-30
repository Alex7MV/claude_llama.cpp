#include "ggml-backend-pipeline.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-backend-pipeline-queue.h"
#include <cstring>
#include <algorithm>
#include <atomic>
#include <mutex>

// NOTE: This file is part of ggml-base and must NOT depend on CUDA.
// TMA / Blackwell kernel integration is done at the llama-context level
// or via the gpu_dispatch callback registered with
// ggml_backend_sched_pipelined_set_gpu_dispatch.

#define GGML_PIPELINE_MAX_EVENTS 2

struct ggml_backend_sched_pipelined {
    ggml_backend_sched_t base;
    int depth;
    int split_size;

    // CPU threadpools (dual CCD pairs)
    ggml_threadpool_params tp_params[2];
    ggml_threadpool_t cpu_tp[2];
    int num_tp;
    int active_pool;
    std::atomic<ggml_backend_t> cpu_backend{nullptr};
    std::once_flag cpu_backend_init_flag;

    // GPU
    ggml_backend_t gpu_backend;
    ggml_backend_event_t stage_events[GGML_PIPELINE_MAX_EVENTS];  // ping-pong
    ggml_backend_event_t * slot_events;  // per-slot completion events
    ggml_backend_dev_t gpu_device;

    // 3-stage lock-free queue
    ggml_pipeline_queue * queue;
    int next_split_idx;   // Monotonic counter for slot assignment

    // GPU dispatch callback (set by CUDA glue, e.g. llama-context.cpp)
    ggml_pipeline_gpu_dispatch_fn gpu_dispatch;
    void * gpu_dispatch_user_data;

    // Pipeline state across split submissions
    bool graph_allocated;
    std::mutex alloc_mutex;   // protects graph allocation (call_once pattern)

    // Opaque pinned buffer handle (set by CUDA glue, e.g. llama-context.cpp)
    void * pinned_buf;

    bool initialized;
};

ggml_backend_sched_pipelined_t ggml_backend_sched_pipelined_init(
    ggml_backend_sched_t base,
    int depth,
    int split_size,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll,
    ggml_backend_t gpu_backend) {

    auto * sched = new ggml_backend_sched_pipelined();
    sched->base = base;
    sched->depth = depth;
    sched->split_size = split_size;
    sched->cpu_backend = nullptr;
    sched->gpu_backend = gpu_backend;
    sched->num_tp = 0;
    sched->active_pool = 0;
    sched->gpu_device = nullptr;
    sched->stage_events[0] = nullptr;
    sched->stage_events[1] = nullptr;
    sched->slot_events = nullptr;
    sched->queue = nullptr;
    sched->next_split_idx = 0;
    sched->gpu_dispatch = nullptr;
    sched->gpu_dispatch_user_data = nullptr;
    sched->graph_allocated = false;
    sched->pinned_buf = nullptr;
    sched->initialized = false;

    // Initialize dual threadpools from CCD pairs
    int threads_per_pair = n_threads / 2;
    sched->num_tp = ggml_cpu_init_dual_threadpool(sched->tp_params, 2,
                                                   threads_per_pair, prio, poll);

    for (int i = 0; i < sched->num_tp; i++) {
        sched->cpu_tp[i] = ggml_threadpool_new(&sched->tp_params[i]);
        if (!sched->cpu_tp[i]) {
            for (int j = 0; j < i; j++) ggml_threadpool_free(sched->cpu_tp[j]);
            delete sched;
            return nullptr;
        }
    }

    // Fallback: if dual pool fails, use single pool
    if (sched->num_tp < 2) {
        for (int j = 0; j < sched->num_tp; j++) {
            if (sched->cpu_tp[j]) ggml_threadpool_free(sched->cpu_tp[j]);
        }
        sched->num_tp = 1;
        sched->cpu_tp[0] = nullptr;
        memset(&sched->tp_params[0], 0, sizeof(ggml_threadpool_params));
        sched->tp_params[0].n_threads = n_threads;
        sched->tp_params[0].prio = prio;
        sched->tp_params[0].poll = poll;
        sched->cpu_tp[0] = ggml_threadpool_new(&sched->tp_params[0]);
        if (!sched->cpu_tp[0]) {
            delete sched;
            return nullptr;
        }
    }

   // Create lock-free queue if depth > 0
    if (depth > 0) {
        // Round depth up to next power of two for fast modulo
        int queue_depth = 1;
        while (queue_depth < depth) queue_depth <<= 1;
        sched->queue = ggml_pipeline_queue_init(queue_depth);
        if (!sched->queue) {
            for (int i = 0; i < sched->num_tp; i++) {
                if (sched->cpu_tp[i]) ggml_threadpool_free(sched->cpu_tp[i]);
            }
            delete sched;
            return nullptr;
        }
    }

    // Initialize backend events for pipeline synchronization.
    if (gpu_backend) {
        sched->gpu_device = ggml_backend_get_device(gpu_backend);
        if (sched->gpu_device) {
            for (int i = 0; i < GGML_PIPELINE_MAX_EVENTS; i++) {
                sched->stage_events[i] = ggml_backend_event_new(sched->gpu_device);
            }
            // Per-slot completion events for async transfer tracking.
            if (sched->queue) {
                sched->slot_events = (ggml_backend_event_t *)calloc(
                    sched->queue->depth, sizeof(ggml_backend_event_t));
                for (int i = 0; i < sched->queue->depth; i++) {
                    sched->slot_events[i] = ggml_backend_event_new(sched->gpu_device);
                    sched->queue->slots[i].completion_event = sched->slot_events[i];
                }
            }
        }
    }

    sched->initialized = true;
    return sched;
}

void ggml_backend_sched_pipelined_set_gpu_dispatch(
    ggml_backend_sched_pipelined_t sched,
    ggml_pipeline_gpu_dispatch_fn fn,
    void * user_data) {

    if (!sched) return;
    sched->gpu_dispatch = fn;
    sched->gpu_dispatch_user_data = user_data;
}

void ggml_backend_sched_pipelined_set_pinned_buf(
    ggml_backend_sched_pipelined_t sched,
    void * pinned_buf) {

    if (!sched) return;
    sched->pinned_buf = pinned_buf;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void ggml_pipeline_ensure_cpu_backend(ggml_backend_sched_pipelined_t sched) {
    if (sched->cpu_backend.load(std::memory_order_acquire)) return;
    std::call_once(sched->cpu_backend_init_flag, [sched]() {
        int n_be = ggml_backend_sched_get_n_backends(sched->base);
        for (int i = 0; i < n_be; i++) {
            ggml_backend_t be = ggml_backend_sched_get_backend(sched->base, i);
            if (ggml_backend_is_cpu(be)) {
                sched->cpu_backend.store(be, std::memory_order_release);
                break;
            }
        }
    });
}

// Drain exactly one GPU-done slot (called when queue is full or at final drain).
static bool ggml_pipeline_drain_one_slot(ggml_backend_sched_pipelined_t sched) {
    if (!sched->queue) return false;
    int idx = ggml_pipeline_dequeue_gpu_done(sched->queue);
    if (idx < 0) return false;
    (void)idx;
    return true;
}

// Block until at least one slot is empty (for backpressure).
static void ggml_pipeline_wait_for_empty_slot(ggml_backend_sched_pipelined_t sched) {
    if (!sched->queue) return;
    while (ggml_pipeline_count_empty(sched->queue) == 0) {
        bool made_progress = false;

        // Try to force-complete any TMA_READY slots whose async work is done.
        for (int i = 0; i < sched->queue->depth; i++) {
            uint32_t state = sched->queue->slots[i].state.load(std::memory_order_acquire);
            if (state == GGML_SLOT_TMA_READY &&
                sched->queue->slots[i].gpu_async_done.load(std::memory_order_acquire)) {
                if (ggml_pipeline_force_complete_slot(sched->queue, i)) {
                    made_progress = true;
                }
            }
        }

        // Try to drain completed GPU slots to make room
        if (ggml_pipeline_drain_one_slot(sched)) {
            made_progress = true;
        }

        if (!made_progress) {
            // No progress yet; brief spin
            #if defined(__x86_64__)
            __asm__ volatile ("pause" ::: "memory");
            #endif
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy compute path (backward compatibility)
// ---------------------------------------------------------------------------

enum ggml_status ggml_backend_sched_pipelined_compute(
    ggml_backend_sched_pipelined_t sched,
    struct ggml_cgraph * gf) {

    if (!sched || !sched->initialized) {
        return ggml_backend_sched_graph_compute_async(sched->base, gf);
    }

  ggml_pipeline_ensure_cpu_backend(sched);

    ggml_backend_t cpu_be = sched->cpu_backend.load(std::memory_order_acquire);
    if (cpu_be) {
        ggml_threadpool_t tp = sched->cpu_tp[sched->active_pool];
        ggml_backend_cpu_set_threadpool(cpu_be, tp);
        ggml_threadpool_resume(tp);
    }

    enum ggml_status status = ggml_backend_sched_graph_compute_async(sched->base, gf);
    if (sched->num_tp >= 2) {
        sched->active_pool = 1 - sched->active_pool;
    }

    return status;
}

// ---------------------------------------------------------------------------
// 3-stage pipelined compute_split
// ---------------------------------------------------------------------------
// The caller submits partial graphs sequentially (one per layer range).
// The orchestrator:
//   1. Ensures the base graph is allocated (first call only)
//   2. For CPU splits: compute on rotated threadpool, enqueue slot
//   3. For GPU splits: dispatch GPU callback for ready slots, compute graph
//   4. Respects depth limit via backpressure
// ---------------------------------------------------------------------------

enum ggml_status ggml_backend_sched_pipelined_compute_split(
    ggml_backend_sched_pipelined_t sched,
    struct ggml_cgraph * gf,
    int32_t first_layer,
    int32_t num_layers,
    bool is_cpu) {

    if (!sched || !sched->initialized) {
        return ggml_backend_sched_graph_compute_async(sched->base, gf);
    }

    // Lazy graph allocation (thread-safe)
    {
        std::lock_guard<std::mutex> lock(sched->alloc_mutex);
        if (!sched->graph_allocated) {
            if (!ggml_backend_sched_alloc_graph(sched->base, gf)) {
                return GGML_STATUS_ALLOC_FAILED;
            }
            sched->graph_allocated = true;
        }
    }

    int split_idx = sched->next_split_idx++;
    int event_idx = split_idx % GGML_PIPELINE_MAX_EVENTS;
    ggml_backend_t cpu_be = nullptr;

    if (is_cpu) {
        // ---- Stage 1: CPU Compute ----
        ggml_pipeline_ensure_cpu_backend(sched);

        cpu_be = sched->cpu_backend.load(std::memory_order_acquire);
        if (cpu_be) {
            int pool_idx = (sched->num_tp > 0) ? (split_idx % sched->num_tp) : 0;
            ggml_threadpool_t tp = sched->cpu_tp[pool_idx];
            ggml_backend_cpu_set_threadpool(cpu_be, tp);
            ggml_threadpool_resume(tp);
        }

        // Compute this CPU split
        enum ggml_status status = ggml_backend_sched_graph_compute_async(sched->base, gf);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }

        // ---- Enqueue slot for Stage 2 (TMA) / Stage 3 (GPU) ----
        // The CPU split produced KV outputs in pinned host memory.
        // We enqueue a slot describing this output for the GPU dispatch.
        if (sched->queue) {
            // Backpressure: if queue is full, drain GPU-done slots until room
            ggml_pipeline_wait_for_empty_slot(sched);

            // TODO: get actual pinned buffer pointer from gf outputs
            // For now, use nullptr placeholders; the caller (llama-context)
            // will populate real buffers via the dispatch callback.
            int idx = ggml_pipeline_enqueue_cpu_ready(
                sched->queue,
                split_idx,
                first_layer,
                num_layers,
                nullptr,   // cpu_buf: set by caller before GPU dispatch
                nullptr,   // gpu_buf: set by caller before GPU dispatch
                0,         // buf_bytes: set by caller
                -1);       // tma_desc_idx

            if (idx < 0) {
                // Should not happen after wait_for_empty_slot, but handle
                return GGML_STATUS_FAILED;
            }
        }

        // Record event so GPU stage can wait on this CPU split
        if (sched->stage_events[event_idx] && cpu_be) {
            ggml_backend_event_record(sched->stage_events[event_idx], cpu_be);
        }

    } else {
        // ---- Stage 3: GPU Compute ----
        // Wait for CPU stage to finish producing inputs
        if (sched->stage_events[event_idx]) {
            ggml_backend_event_wait(sched->gpu_backend, sched->stage_events[event_idx]);
        }

        // Dispatch GPU callback for any queued slots BEFORE computing the
        // next GPU split graph. This allows TMA/WGMMA to overlap with
        // scheduler input copies.
        if (sched->queue && sched->gpu_dispatch) {
            int ready_count = ggml_pipeline_count_cpu_ready(sched->queue);
            for (int r = 0; r < ready_count; r++) {
                int idx = ggml_pipeline_dequeue_cpu_ready(sched->queue);
                if (idx < 0) break;

                ggml_pipeline_slot & slot = sched->queue->slots[idx];
                struct ggml_pipeline_slot_info info = {};
                info.split_idx        = slot.split_idx;
                info.first_layer      = slot.first_layer;
                info.num_layers       = slot.num_layers;
                info.cpu_buf          = slot.cpu_buf;
                info.gpu_buf          = slot.gpu_buf;
                info.buf_bytes        = slot.buf_bytes;
                info.tma_desc_idx     = slot.tma_desc_idx;
                info.completion_event = slot.completion_event;
                info.gpu_async_done   = &slot.gpu_async_done;

                sched->gpu_dispatch(sched->gpu_dispatch_user_data, &info);

                // Slot stays in TMA_READY state.  The GPU dispatch callback
                // records the completion_event and sets gpu_async_done when
                // async work finishes.  drain() polls gpu_async_done and
                // transitions to GPU_DONE.
            }
        }

        // Compute GPU split graph
        enum ggml_status status = ggml_backend_sched_graph_compute_async(sched->base, gf);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }

    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Drain all in-flight slots
// ---------------------------------------------------------------------------

enum ggml_status ggml_backend_sched_pipelined_drain(
    ggml_backend_sched_pipelined_t sched) {

    if (!sched || !sched->queue) {
        return GGML_STATUS_SUCCESS;
    }

    // Poll until all slots are EMPTY (recycled)
    int iterations = 0;
    const int max_iterations = 1000000; // safety bound
    while (iterations < max_iterations) {
        bool any_in_flight = false;
        for (int i = 0; i < sched->queue->depth; i++) {
            uint32_t state = sched->queue->slots[i].state.load(std::memory_order_acquire);
            if (state != GGML_SLOT_EMPTY) {
                any_in_flight = true;
                break;
            }
        }
        if (!any_in_flight) break;

        // Complete TMA_READY slots whose async work is finished.
        for (int i = 0; i < sched->queue->depth; i++) {
            uint32_t state = sched->queue->slots[i].state.load(std::memory_order_acquire);
            if (state == GGML_SLOT_TMA_READY &&
                sched->queue->slots[i].gpu_async_done.load(std::memory_order_acquire)) {
                ggml_pipeline_force_complete_slot(sched->queue, i);
            }
        }

        // Drain completed GPU slots
        while (ggml_pipeline_drain_one_slot(sched)) {}

        // Brief spin
        #if defined(__x86_64__)
        __asm__ volatile ("pause" ::: "memory");
        #endif

        iterations++;
    }

    if (iterations >= max_iterations) {
        return GGML_STATUS_FAILED;
    }

    // Reset for next sequence
    sched->next_split_idx = 0;
    sched->graph_allocated = false;

    return GGML_STATUS_SUCCESS;
}

void ggml_backend_sched_pipelined_synchronize(ggml_backend_sched_pipelined_t sched) {
    if (!sched) return;
    ggml_backend_sched_synchronize(sched->base);
}

void ggml_backend_sched_pipelined_free(ggml_backend_sched_pipelined_t sched) {
    if (!sched) return;

    if (sched->gpu_device) {
        for (int i = 0; i < GGML_PIPELINE_MAX_EVENTS; i++) {
            if (sched->stage_events[i]) {
                ggml_backend_event_free(sched->stage_events[i]);
            }
        }
        if (sched->slot_events) {
            for (int i = 0; i < sched->queue->depth; i++) {
                if (sched->slot_events[i]) {
                    ggml_backend_event_free(sched->slot_events[i]);
                }
            }
            free(sched->slot_events);
        }
    }

    if (sched->queue) {
        ggml_pipeline_queue_free(sched->queue);
    }

    for (int i = 0; i < sched->num_tp; i++) {
        if (sched->cpu_tp[i]) {
            ggml_threadpool_free(sched->cpu_tp[i]);
        }
    }

    delete sched;
}
