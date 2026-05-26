#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// TMA descriptors (128 bytes, packed) — available in host and device code
#pragma pack(push, 1)
struct tma_store_desc {
    uint64_t d[16];
};
#pragma pack(pop)
static_assert(sizeof(tma_store_desc) == 128, "TMA store desc must be 128 bytes");

#pragma pack(push, 1)
struct tma_load_desc {
    uint64_t d[16];
};
#pragma pack(pop)
static_assert(sizeof(tma_load_desc) == 128, "TMA load desc must be 128 bytes");

#ifndef __CUDA_ARCH__
// Host-side declarations so kernels are visible for launching
__global__ void tma_copy_store_kernel(const tma_store_desc *);
__global__ void tma_copy_load_kernel(const tma_load_desc *);
#endif

// Device code only — stub implementations; the actual cp.async.bulk PTX
// requires the correct compiler intrinsic for the target CUDA toolkit.
// The runtime probe (ggml_tma_runtime_supported) guards actual TMA usage.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__global__ void tma_copy_store_kernel(const tma_store_desc * desc) {
    (void)desc;
    // TODO: insert cp.async.bulk PTX intrinsic once toolkit support is verified
    __threadfence_system();
}

__global__ void tma_copy_load_kernel(const tma_load_desc * desc) {
    (void)desc;
    // TODO: insert cp.async.bulk PTX intrinsic once toolkit support is verified
    __threadfence_system();
}
#endif
