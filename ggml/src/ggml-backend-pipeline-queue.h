#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <new>

// ---------------------------------------------------------------------------
// Lock-Free 3-Stage Pipeline Queue
// ---------------------------------------------------------------------------
// Stages: CPU_COMPUTE -> TMA_TRANSFER -> GPU_WGMMA -> [recycle]
//
// Each slot is a ping-pong buffer cell that cycles through atomic states.
// Depth limits in-flight splits (design doc: depth=3).
//
// Threading model (per design doc):
//   - CPU threads enqueue splits after CPU compute (producing CPU_READY)
//   - GPU TMA warp dequeues CPU_READY, initiates TMA, produces TMA_READY
//   - GPU WGMMA warp group dequeues TMA_READY, computes, produces GPU_DONE
//   - CPU recycler dequeues GPU_DONE, resets to EMPTY
//
// All cross-stage handoffs are single-producer / single-consumer on
// disjoint threads, mapped to independent atomic sequences.
// ---------------------------------------------------------------------------

enum ggml_pipeline_slot_state : uint32_t {
    GGML_SLOT_EMPTY     = 0,  // Available for new split assignment
    GGML_SLOT_CPU_READY = 1,  // CPU compute done, buffer valid for TMA
    GGML_SLOT_TMA_READY = 2,  // TMA transfer complete, SMEM valid for WGMMA
    GGML_SLOT_GPU_DONE  = 3,  // WGMMA done, outputs committed, safe to recycle
};

struct ggml_pipeline_slot {
    // Atomic state drives the state machine. Only forward transitions
    // EMPTY->CPU_READY->TMA_READY->GPU_DONE->EMPTY are legal.
    alignas(64) std::atomic<uint32_t> state{ GGML_SLOT_EMPTY };

    // Split metadata (written once by producer before state advance)
    int32_t  split_idx;       // Which graph split this slot carries
    int32_t  first_layer;     // Layer range [first, first+n)
    int32_t  num_layers;

    // Buffer handles (pinned CPU -> GPU SMEM/VRAM)
    void *   cpu_buf;         // Pinned host buffer (128-byte aligned)
    void *   gpu_buf;         // VRAM destination (or nullptr if direct SMEM)
    size_t   buf_bytes;       // Bytes to transfer / compute

    // TMA descriptor index for this slot (-1 if not using TMA)
    int32_t  tma_desc_idx;

    // mbarrier shared-memory handle for TMA->WGMMA sync on this slot
    // Populated by GPU-side init; opaque to CPU queue logic.
    uint64_t mbar_handle;

    // Opaque event handle for async completion tracking (ggml_backend_event_t).
    // Set by the orchestrator, recorded by the GPU dispatch callback.
    void * completion_event;

    // Flag set by GPU dispatch callback when async work is complete.
    // Used for non-blocking poll in drain() and wait_for_empty_slot().
    alignas(64) std::atomic<bool> gpu_async_done{ false };
};

struct ggml_pipeline_queue {
    int32_t  depth;           // Capacity (e.g. 3)
    int32_t  mask;            // depth-1 for power-of-two fast modulo
    ggml_pipeline_slot * slots;

    // Per-stage cursor pairs. Each stage has a "tail" it produces to
    // and a "head" it consumes from. Sequences are monotonic.
    //
    // Sequence mapping:
    //   cpu_seq  : tail = enqueue_cpu_ready,  head = dequeue_gpu_done (recycle)
    //   tma_seq  : tail = enqueue_tma_ready,  head = dequeue_cpu_ready
    //   gpu_seq  : tail = enqueue_gpu_done,   head = dequeue_tma_ready
    //
    // A slot is "owned" by stage N when stage N's head cursor points to it
    // and the slot state matches stage N's expected input state.
    alignas(64) std::atomic<int64_t> cpu_tail{ 0 };
    alignas(64) std::atomic<int64_t> tma_tail{ 0 };
    alignas(64) std::atomic<int64_t> gpu_tail{ 0 };

    alignas(64) std::atomic<int64_t> cpu_head{ 0 }; // recycler
    alignas(64) std::atomic<int64_t> tma_head{ 0 }; // TMA consumer
    alignas(64) std::atomic<int64_t> gpu_head{ 0 }; // WGMMA consumer
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

static inline ggml_pipeline_queue * ggml_pipeline_queue_init(int32_t depth) {
    assert(depth > 0 && (depth & (depth - 1)) == 0); // power of two
    auto * q = new ggml_pipeline_queue();
    q->depth  = depth;
    q->mask   = depth - 1;
    q->slots  = new ggml_pipeline_slot[depth]();
    // All atomic counters already zero-initialized.
    return q;
}

static inline void ggml_pipeline_queue_free(ggml_pipeline_queue * q) {
    if (!q) return;
    delete[] q->slots;
    delete q;
}

// ---------------------------------------------------------------------------
// Producer/Consumer primitives
// ---------------------------------------------------------------------------
// Each function returns the slot index on success, or -1 if the stage
// would block (queue full / no ready work). Callers spin or yield.
// ---------------------------------------------------------------------------

// Called by CPU compute thread after finishing a split.
// Finds an EMPTY slot, writes metadata, advances state to CPU_READY.
static inline int32_t ggml_pipeline_enqueue_cpu_ready(
    ggml_pipeline_queue * q,
    int32_t  split_idx,
    int32_t  first_layer,
    int32_t  num_layers,
    void *   cpu_buf,
    void *   gpu_buf,
    size_t   buf_bytes,
    int32_t  tma_desc_idx) {

    const int64_t tail = q->cpu_tail.load(std::memory_order_relaxed);
    const int64_t head = q->cpu_head.load(std::memory_order_acquire);
    if (tail - head >= q->depth) {
        return -1; // No EMPTY slot available (all slots in flight)
    }

    const int32_t idx = static_cast<int32_t>(tail & q->mask);
    ggml_pipeline_slot & slot = q->slots[idx];

    uint32_t expected = GGML_SLOT_EMPTY;
    if (!slot.state.compare_exchange_strong(
            expected, GGML_SLOT_CPU_READY,
            std::memory_order_release,
            std::memory_order_relaxed)) {
        return -1; // Slot not empty (should not happen with correct head/tail)
    }

    // Write metadata *after* CAS succeeds but before tail increment visible.
    // Release ordering on CAS ensures metadata is visible to TMA consumer.
    slot.split_idx    = split_idx;
    slot.first_layer  = first_layer;
    slot.num_layers   = num_layers;
    slot.cpu_buf      = cpu_buf;
    slot.gpu_buf      = gpu_buf;
    slot.buf_bytes    = buf_bytes;
    slot.tma_desc_idx     = tma_desc_idx;
    slot.mbar_handle      = 0;
    slot.completion_event = nullptr;
    slot.gpu_async_done.store(false, std::memory_order_relaxed);

    q->cpu_tail.store(tail + 1, std::memory_order_release);
    return idx;
}

// Called by GPU TMA warp. Dequeues a CPU_READY slot and flips to TMA_IN_PROGRESS.
// The caller is responsible for initiating TMA and later calling enqueue_tma_ready.
static inline int32_t ggml_pipeline_dequeue_cpu_ready(ggml_pipeline_queue * q) {
    const int64_t head = q->tma_head.load(std::memory_order_relaxed);
    const int64_t tail = q->cpu_tail.load(std::memory_order_acquire);
    if (head >= tail) {
        return -1; // No CPU_READY work
    }

    const int32_t idx = static_cast<int32_t>(head & q->mask);
    ggml_pipeline_slot & slot = q->slots[idx];

    uint32_t expected = GGML_SLOT_CPU_READY;
    if (!slot.state.compare_exchange_strong(
            expected, GGML_SLOT_TMA_READY,
            std::memory_order_acquire,
            std::memory_order_relaxed)) {
        // Another consumer got here first, or state not yet CPU_READY.
        // Advance head anyway to avoid stalling on a slow producer.
        // The caller should retry.
        return -1;
    }

    q->tma_head.store(head + 1, std::memory_order_release);
    return idx;
}

// Called by GPU TMA warp after TMA descriptor is committed.
// Slot is already marked TMA_READY by dequeue_cpu_ready (early-start model).
// This function is a no-op in the simple model; provided for explicit phase
// separation if TMA_WAIT_GROUP is done by a different warp.
static inline int32_t ggml_pipeline_enqueue_tma_ready(ggml_pipeline_queue * q, int32_t idx) {
    // In the "early start" design, the TMA commit and WGMMA start are
    // mbarrier-synchronized within the same slot. The queue state is
    // already TMA_READY. We only advance the gpu_head cursor gate here.
    (void)q; (void)idx;
    return idx;
}

// Called by GPU WGMMA warp group. Dequeues a TMA_READY slot.
static inline int32_t ggml_pipeline_dequeue_tma_ready(ggml_pipeline_queue * q) {
    const int64_t head = q->gpu_head.load(std::memory_order_relaxed);
    const int64_t tail = q->tma_tail.load(std::memory_order_acquire);
    if (head >= tail) {
        return -1;
    }

    const int32_t idx = static_cast<int32_t>(head & q->mask);
    ggml_pipeline_slot & slot = q->slots[idx];

    uint32_t expected = GGML_SLOT_TMA_READY;
    if (!slot.state.compare_exchange_strong(
            expected, GGML_SLOT_GPU_DONE,
            std::memory_order_acquire,
            std::memory_order_relaxed)) {
        return -1;
    }

    q->gpu_head.store(head + 1, std::memory_order_release);
    return idx;
}

// Called by GPU WGMMA warp group after committing accumulator to VRAM.
static inline void ggml_pipeline_enqueue_gpu_done(
    ggml_pipeline_queue * q,
    int32_t idx) {
    // State is already GPU_DONE from dequeue_tma_ready.
    // Advance gpu_tail so recycler can reclaim.
    q->gpu_tail.fetch_add(1, std::memory_order_release);
    (void)idx;
}

// Called by CPU recycler thread. Reclaims a GPU_DONE slot, resets to EMPTY.
static inline int32_t ggml_pipeline_dequeue_gpu_done(ggml_pipeline_queue * q) {
    const int64_t head = q->cpu_head.load(std::memory_order_relaxed);
    const int64_t tail = q->gpu_tail.load(std::memory_order_acquire);
    if (head >= tail) {
        return -1;
    }

    const int32_t idx = static_cast<int32_t>(head & q->mask);
    ggml_pipeline_slot & slot = q->slots[idx];

    uint32_t expected = GGML_SLOT_GPU_DONE;
    if (!slot.state.compare_exchange_strong(
            expected, GGML_SLOT_EMPTY,
            std::memory_order_release,
            std::memory_order_relaxed)) {
        return -1;
    }

    q->cpu_head.store(head + 1, std::memory_order_release);
    return idx;
}

// ---------------------------------------------------------------------------
// Forced completion: used when async work completes out-of-band
// (e.g., via event synchronization in drain()).
// Advances all intermediate cursors so the slot can be recycled.
// ---------------------------------------------------------------------------
static inline bool ggml_pipeline_force_complete_slot(
        ggml_pipeline_queue * q, int32_t idx) {
    ggml_pipeline_slot & slot = q->slots[idx];

    uint32_t expected = GGML_SLOT_TMA_READY;
    if (!slot.state.compare_exchange_strong(
            expected, GGML_SLOT_GPU_DONE,
            std::memory_order_release,
            std::memory_order_relaxed)) {
        return false;
    }

    // Advance the cursors that were skipped in the async path.
    q->tma_tail.fetch_add(1, std::memory_order_release);
    q->gpu_head.fetch_add(1, std::memory_order_release);
    q->gpu_tail.fetch_add(1, std::memory_order_release);
    return true;
}

// ---------------------------------------------------------------------------
// Non-blocking peek helpers (for polling loops)
// ---------------------------------------------------------------------------

static inline int32_t ggml_pipeline_count_cpu_ready(ggml_pipeline_queue * q) {
    int64_t tail = q->cpu_tail.load(std::memory_order_acquire);
    int64_t head = q->tma_head.load(std::memory_order_acquire);
    return static_cast<int32_t>(tail - head);
}

static inline int32_t ggml_pipeline_count_empty(ggml_pipeline_queue * q) {
    int64_t tail = q->cpu_tail.load(std::memory_order_acquire);
    int64_t head = q->cpu_head.load(std::memory_order_acquire);
    return q->depth - static_cast<int32_t>(tail - head);
}
