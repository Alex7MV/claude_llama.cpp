#pragma once

#include "ggml-backend-pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// GPU health-check for pipeline init
// ---------------------------------------------------------------------------
// Verifies:
//  1. PCIe NUMA affinity (reads /sys/class/drm/card*/device/numa_node)
//  2. Pinned buffer 128-byte alignment
//  3. VRAM buffer 16-byte alignment
//  4. 48-bit VA assertion
//
// @return true if all checks pass, false on alignment failure
bool ggml_cuda_epyc_pipeline_health_check(int device_id, void * pinned_base, void * vram_base);

// ---------------------------------------------------------------------------
// KV cache merge: linear prefill layout -> ring buffer
// ---------------------------------------------------------------------------
// Runs on a dedicated non-blocking CUDA stream, ordered after
// the last GPU split via `last_split_event`.
void ggml_cuda_epyc_pipeline_merge_kv(
    void * dst_ring,
    const void * src_linear,
    size_t bytes_per_layer,
    int n_layers,
    int ring_stride,
    void * last_split_event);   // cudaEvent_t cast to void* for C compat

// ---------------------------------------------------------------------------
// GPU dispatch callback for pipeline slots
// ---------------------------------------------------------------------------
// Called by the orchestrator for each CPU_READY slot.
// Currently uses cudaMemcpyAsync; will use TMA kernel when enabled.
void ggml_cuda_epyc_pipeline_dispatch(
    void * user_data,           // cudaStream_t
    const struct ggml_pipeline_slot_info * slot);

#ifdef __cplusplus
}
#endif
