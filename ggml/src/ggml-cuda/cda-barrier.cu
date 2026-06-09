#include "ggml-cuda.h"
#include "vendors/cuda.h"

// ---- CDA Barrier — GPU-side flag for sub-microsecond inter-stream sync ----
//
// Signal (producer stream):
//   __threadfence() + atomicExch(flag, 1)
//   - __threadfence() ensures preceding producer work (on same stream) is visible
//   - atomicExch provides release semantics, publishing the flag device-wide
//
// Wait (consumer stream):
//   atomicAdd(flag, 0) spin loop with __nanosleep
//   - atomicAdd provides acquire semantics, pairing with signal's release
//   - After seeing flag == 1, all producer writes are visible to consumer
//
// This avoids CPU involvement in the inter-stream ordering path.
// CUDA events (cudaEventRecord/cudaStreamWaitEvent) are NOT needed — the
// atomic release/acquire pair + same-stream producer ordering guarantees
// correct visibility.

struct cda_device_flag {
    uint64_t value;
};

// Signal kernel: release fence + atomic store
__global__ void cda_signal_kernel(cda_device_flag * flag) {
    __threadfence();
    atomicExch((unsigned long long int *)&flag->value, 1ULL);
}

// Wait kernel: acquire atomic + spin with nanosleep
__global__ void cda_wait_kernel(const cda_device_flag * flag) {
    while (atomicAdd((unsigned long long int *)&flag->value, 0ULL) == 0ULL) {
        __nanosleep(100);
    }
}

ggml_backend_cda_t ggml_backend_cuda_cda_create(ggml_backend_t backend) {
    (void)backend;
    cda_device_flag * d_flag = nullptr;
    cudaError_t err = cudaMalloc(&d_flag, sizeof(cda_device_flag));
    if (err != cudaSuccess || !d_flag) {
        return nullptr;
    }
    // Initialize to 0
    err = cudaMemset(d_flag, 0, sizeof(cda_device_flag));
    if (err != cudaSuccess) {
        cudaFree(d_flag);
        return nullptr;
    }
    return (ggml_backend_cda_t)d_flag;
}

void ggml_backend_cuda_cda_free(ggml_backend_t backend, ggml_backend_cda_t cda) {
    (void)backend;
    if (cda) {
        cudaFree((cda_device_flag *)cda);
    }
}

void ggml_backend_cuda_cda_signal(ggml_backend_t backend, ggml_backend_cda_t cda) {
    if (!backend || !cda) return;
    cda_device_flag * d_flag = (cda_device_flag *)cda;
    int sid = ggml_backend_cuda_get_stream(backend);
    cudaStream_t stream = (cudaStream_t)ggml_backend_cuda_get_stream_ptr(backend, sid);
    if (!stream) return;
    cda_signal_kernel<<<1, 1, 0, stream>>>(d_flag);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_backend_cuda_cda_wait(ggml_backend_t backend, ggml_backend_cda_t cda) {
    if (!backend || !cda) return;
    cda_device_flag * d_flag = (cda_device_flag *)cda;
    int sid = ggml_backend_cuda_get_stream(backend);
    cudaStream_t stream = (cudaStream_t)ggml_backend_cuda_get_stream_ptr(backend, sid);
    if (!stream) return;
    cda_wait_kernel<<<1, 1, 0, stream>>>(d_flag);
    CUDA_CHECK(cudaGetLastError());
}
