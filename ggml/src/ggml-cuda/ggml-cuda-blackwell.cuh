#pragma once
#if __CUDA_ARCH__ >= 1000 || !defined(__CUDA_ARCH__)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// TMA store descriptor (128 bytes, packed)
// Matches hwTmaStoreDesc format from PTX ISA
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

/**
 * TMA copy kernel for store operations.
 * Launch with 1 block, 1 thread. Uses cp.async.bulk intrinsic.
 * The descriptor must be in device memory and 128-byte aligned.
 */
__global__ void tma_copy_store_kernel(const tma_store_desc * desc) {
#if __CUDA_ARCH__ >= 1000
    __cp_async_bulk_uniform_raw_tma_crypto_zero_copy_mem_first_pass(
        *(const unsigned long long*)desc, 0);
    __threadfence_system();
#endif
}

/**
 * TMA load kernel for prefetch operations.
 */
__global__ void tma_copy_load_kernel(const tma_load_desc * desc) {
#if __CUDA_ARCH__ >= 1000
    __cp_async_bulk_uniform_raw_tma_crypto_zero_copy_mem_first_pass(
        *(const unsigned long long*)desc, 0);
    __threadfence_system();
#endif
}

// mbarrier wrappers for TMA completion sync
__device__ inline void mbarrier_arrive_expect_tx(void * barrier, unsigned int expected_tx) {
#if __CUDA_ARCH__ >= 900
    __mbarrier_arrive_expect_tx(__cvta_generic_to_shared(barrier), expected_tx);
#endif
}

__device__ inline void mbarrier_wait(void * barrier, unsigned int phase) {
#if __CUDA_ARCH__ >= 900
    __mbarrier_wait(__cvta_generic_to_shared(barrier), phase);
#endif
}

#endif // __CUDA_ARCH__ >= 1000
