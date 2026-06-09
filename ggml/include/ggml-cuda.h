#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

GGML_BACKEND_API void ggml_backend_cuda_print_timing(ggml_backend_t backend);

// Set the active stream index for the CUDA backend (0 = compute, 1 = attn, etc.)
GGML_BACKEND_API void ggml_backend_cuda_set_stream(ggml_backend_t backend, int stream_id);

// Get the active stream index for the CUDA backend
GGML_BACKEND_API int  ggml_backend_cuda_get_stream(ggml_backend_t backend);

// Get the raw CUDA stream pointer for a given stream index.
// Returns a cudaStream_t cast to void* to avoid exposing CUDA runtime types.
// Used by pipeline prefetch to access the transfer stream for async H2D copies.
GGML_BACKEND_API void * ggml_backend_cuda_get_stream_ptr(ggml_backend_t backend, int stream_id);

// CDA barrier — GPU-side flag for sub-microsecond stream synchronization.
// Replaces cudaEventRecord/cudaStreamWaitEvent with a device memory write + spin
// to avoid CPU involvement in the inter-stream ordering path.
typedef void * ggml_backend_cda_t;

// Create a CDA barrier (allocates device-side flag).
// Returns NULL on failure.
GGML_BACKEND_API ggml_backend_cda_t ggml_backend_cuda_cda_create(ggml_backend_t backend);

// Destroy a CDA barrier.
GGML_BACKEND_API void ggml_backend_cuda_cda_free(ggml_backend_t backend, ggml_backend_cda_t cda);

// Signal the barrier on the backend's current stream (writes flag to 1).
// All preceding work on the current stream is completed before the signal.
GGML_BACKEND_API void ggml_backend_cuda_cda_signal(ggml_backend_t backend, ggml_backend_cda_t cda);

// Wait for the barrier on the backend's current stream (spins until flag is 1).
// The wait is a GPU-side spin loop — no CPU involvement after launch.
GGML_BACKEND_API void ggml_backend_cuda_cda_wait(ggml_backend_t backend, ggml_backend_cda_t cda);

// Asynchronous expert prefetch for DeepSeek pipeline (all CUDA operations on
// a single transfer stream managed internally via ggml_backend_cuda_set_stream).
// - Switches backend to transfer stream (id=2).
// - Waits for qkv_done event on that stream.
// - Reads top_k indices GPU→host via cudaMemcpyAsync @ transfer stream.
// - Deduplicates expert IDs.
// - Launches cudaMemcpyAsync H2D copies for each unique expert's gate/up/down weights.
// - Records completion_event.
// - Restores backend to compute stream.
// dst[s] are GPU prefetch buffers, src[s] are CPU model weight tensors.
GGML_BACKEND_API void ggml_backend_cuda_pipeline_expert_prefetch(
    struct ggml_tensor  * dst[3],
    struct ggml_tensor  * src[3],
    size_t                slice_bytes[3],
    struct ggml_tensor  * top_k,
    ggml_backend_event_t  qkv_done,
    ggml_backend_event_t  completion_event,
    ggml_backend_t        backend);

#ifdef  __cplusplus
}
#endif
