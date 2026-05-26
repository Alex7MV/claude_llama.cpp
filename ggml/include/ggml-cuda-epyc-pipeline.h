#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// TMA transfer between pinned system RAM and GPU VRAM.
// Falls back to cudaMemcpyAsync when TMA is unavailable.

typedef struct ggml_tma_transfer * ggml_tma_transfer_t;

// Check if TMA engine is available on the current device.
// Requires GGML_CUDA_TMA=1 env var, CUDA >= 12.4, SM >= 90.
int ggml_tma_supported(void);

// Initialize a TMA transfer descriptor.
// src_pinned: pinned RAM address (from ggml_backend_cpu_pinned_buffer_type)
// dst_vram:   GPU VRAM address (from cudaMalloc)
// num_elements: element count (e.g. float16/bf16 elements to transfer)
// elem_size:    element size in bytes (2 for float16/bf16, 4 for float32)
// stream:       CUDA stream (pass NULL to auto-use pipeline compute stream)
int ggml_tma_init_transfer(ggml_tma_transfer_t * out,
    void * src_pinned,
    void * dst_vram,
    size_t num_elements,
    size_t elem_size,
    void * stream);

// Launch the transfer asynchronously on the configured stream.
void ggml_tma_launch_transfer(ggml_tma_transfer_t transfer);

// Free TMA transfer resources.
void ggml_tma_free_transfer(ggml_tma_transfer_t transfer);

// EPYC pipeline GPU-side state (streams + events).

// Get the pipeline's compute stream (NULL if not initialized).
void * ggml_epyc_pipeline_get_compute_stream(void);

// Get the pipeline's merge stream (NULL if not initialized).
void * ggml_epyc_pipeline_get_merge_stream(void);

// Dispatch async linear-to-ring KV cache merge on the merge stream.
// Copies n_rows from a linear buffer into the ring buffer with dst_stride.
// The merge stream waits on the last compute stage before starting.
void ggml_epyc_pipeline_dispatch_kv_merge(
    void * src,
    void * dst,
    size_t row_size,
    int32_t n_rows,
    size_t dst_stride);

// Record a stage completion event on the given stream.
void ggml_epyc_pipeline_record_stage(int stage, void * stream);

// Make a stream wait for a stage event.
void ggml_epyc_pipeline_wait_stage(void * stream, int stage);

#ifdef __cplusplus
}
#endif
