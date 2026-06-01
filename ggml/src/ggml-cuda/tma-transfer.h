#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// TMA capability probe.
// Returns true if TMA is available and GGML_CUDA_TMA=1 is set.
// Requires: CUDA runtime >= 12.4, device compute capability >= sm_90.
bool ggml_cuda_tma_supported(void);

// TMA transfer between pinned system RAM and GPU VRAM.
// Falls back to cudaMemcpyAsync when TMA is unavailable.

typedef struct ggml_tma_transfer * ggml_tma_transfer_t;

// Initialize a TMA transfer descriptor.
// src_pinned: pinned RAM address (from ggml_backend_cpu_pinned_buffer_type)
// dst_vram:   GPU VRAM address (from cudaMalloc)
// num_elements: element count (e.g. float16/bf16 elements to transfer)
// elem_size:    element size in bytes (2 for float16/bf16, 4 for float32)
// stream:       CUDA stream for the transfer
// Returns true on success, false if TMA unavailable (will use memcpy fallback).
bool ggml_tma_init_transfer(ggml_tma_transfer_t * out,
    void * src_pinned,
    void * dst_vram,
    size_t num_elements,
    size_t elem_size,
    void * stream);  // cudaStream_t passed as void* to keep C-compatible

// Launch the transfer asynchronously on the configured stream.
void ggml_tma_launch_transfer(ggml_tma_transfer_t transfer);

// Free TMA transfer resources (descriptor device memory, etc).
void ggml_tma_free_transfer(ggml_tma_transfer_t transfer);

#ifdef __cplusplus
}

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>

// C++ convenience wrapper: enqueue a 1D H2D transfer from pinned CPU to VRAM.
// Returns true if TMA was launched; false => caller should use cudaMemcpyAsync.
inline bool ggml_tma_enqueue_h2d_1d(
    void *       dst_vram,
    const void * src_pinned,
    size_t       num_bytes,
    void *       stream,
    void *       cuda_event_out = nullptr)
{
    if (!ggml_cuda_tma_supported()) {
        cudaError_t err = cudaMemcpyAsync(
            dst_vram, src_pinned, num_bytes,
            cudaMemcpyHostToDevice, (cudaStream_t)stream);
        if (err == cudaSuccess && cuda_event_out) {
            cudaEventRecord((cudaEvent_t)cuda_event_out, (cudaStream_t)stream);
        }
        return err == cudaSuccess;
    }

    ggml_tma_transfer_t tma_desc = nullptr;
    if (!ggml_tma_init_transfer(
            &tma_desc,
            const_cast<void *>(src_pinned),
            dst_vram,
            num_bytes,
            1,
            stream))
    {
        return false;
    }

    ggml_tma_launch_transfer(tma_desc);
    if (cuda_event_out) {
        cudaEventRecord((cudaEvent_t)cuda_event_out, (cudaStream_t)stream);
    }
    ggml_tma_free_transfer(tma_desc);
    return true;
}
#endif // GGML_USE_CUDA

#endif // __cplusplus
