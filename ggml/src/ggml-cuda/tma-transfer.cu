#include "common.cuh"

#include "tma-transfer.h"
#include "ggml-cuda-blackwell.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>

// Forward declare EPYC pipeline accessor (defined in ggml-cuda-epyc-pipeline.cu)
extern "C" cudaStream_t ggml_epyc_pipeline_get_compute_stream(void);

// TMA-based transfer layer for moving KV cache data between pinned
// system RAM and GPU VRAM.  On SM 100+ the kernel can use the TMA
// engine; otherwise it falls back to cudaMemcpyAsync.
//
// We define our own copy of the 16-byte TMA descriptor type here to
// avoid pulling in tma.cuh host-side helpers (which are gated behind
// #ifndef __CUDA_ARCH__ and cause visibility issues in .cu translation
// units).  The layout matches ggml_cuda_tma_desc from tma.cuh.

#pragma pack(push, 1)
struct tma_transfer_desc {
    uint64_t d[2];
};
#pragma pack(pop)
static_assert(sizeof(tma_transfer_desc) == 16);

struct ggml_tma_transfer {
    tma_transfer_desc * desc;      // device-side TMA descriptor (16 bytes)
    cudaStream_t stream;
    size_t num_elements;
    size_t elem_size;
    void * src_pinned;
    void * dst_vram;
    bool use_tma;                  // true if SM >= 100 and kernel is verified
};

// Runtime probe: check if TMA is supported on current device
static bool ggml_tma_runtime_supported(int * device_out) {
    const char * env = getenv("GGML_CUDA_TMA");
    if (!env || atoi(env) != 1) return false;

    int cuda_version = 0;
    if (cudaRuntimeGetVersion(&cuda_version) != cudaSuccess) return false;
    if (cuda_version < 12040) return false;

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;
    if (prop.major < 9) return false;

    if (device_out) *device_out = device;
    return true;
}

int ggml_tma_supported(void) {
    return ggml_tma_runtime_supported(nullptr) ? 1 : 0;
}

bool ggml_tma_init_transfer(ggml_tma_transfer_t * out,
    void * src_pinned,
    void * dst_vram,
    size_t num_elements,
    size_t elem_size,
    void * stream) {

    if (!out) return false;
    if (!src_pinned || !dst_vram || num_elements == 0) {
        *out = nullptr;
        return false;
    }

    // Auto-resolve stream: use EPYC pipeline compute stream if available
    if (!stream) {
        cudaStream_t ps = ggml_epyc_pipeline_get_compute_stream();
        if (ps) {
            stream = (void*)ps;
        }
    }

    ggml_tma_transfer * transfer = new ggml_tma_transfer();
    transfer->src_pinned  = src_pinned;
    transfer->dst_vram    = dst_vram;
    transfer->num_elements = num_elements;
    transfer->elem_size   = elem_size > 0 ? elem_size : 2;  // default float16/bf16
    transfer->stream      = (cudaStream_t)stream;
    transfer->desc        = nullptr;
    transfer->use_tma     = false;

    // Check if TMA is supported and enabled
    {
        int device = 0;
        bool supported = ggml_tma_runtime_supported(&device);
        if (supported) {
            // Validate alignment requirements
            if ((uintptr_t)src_pinned % 128 != 0) {
                GGML_LOG_WARN("TMA disabled: pinned buffer not 128-byte aligned\n");
                supported = false;
            } else if ((uintptr_t)dst_vram % 16 != 0) {
                GGML_LOG_WARN("TMA disabled: VRAM target not 16-byte aligned\n");
                supported = false;
            } else if ((uintptr_t)src_pinned >= (1ULL << 48)) {
                GGML_LOG_ERROR("TMA requires 48-bit VA, got %lx\n", (uintptr_t)src_pinned);
                supported = false;
            }
        }
        transfer->use_tma = supported;
    }

    // Build a 1D TMA descriptor pointing at the pinned system RAM source.
    // Encoding matches ggml_cuda_tma_make_load_desc_1d from tma.cuh:
    //   d[0] = lower 48 bits of address
    //   d[1] = lower 48 bits of byte count
    {
        size_t num_bytes = num_elements * transfer->elem_size;
        tma_transfer_desc host_desc;
        uint64_t addr = (uint64_t)src_pinned & 0xFFFFFFFFFFFFUL;
        host_desc.d[0] = addr;
        host_desc.d[1] = num_bytes & 0xFFFFFFFFFFFFUL;

        CUDA_CHECK(cudaMalloc(&transfer->desc, sizeof(tma_transfer_desc)));
        cudaMemcpy(transfer->desc, &host_desc, sizeof(tma_transfer_desc),
                        cudaMemcpyHostToDevice);
    }

    *out = transfer;
    return true;
}

void ggml_tma_launch_transfer(ggml_tma_transfer_t transfer) {
    if (!transfer) return;

    size_t bytes = transfer->num_elements * transfer->elem_size;

    if (!transfer->use_tma) {
        cudaMemcpyAsync(transfer->dst_vram, transfer->src_pinned, bytes,
                        cudaMemcpyHostToDevice, transfer->stream);
        return;
    }
    // Launch TMA store kernel (1 block, 1 thread) using descriptor in device memory.
    // The kernel is defined in ggml-cuda-blackwell.cuh and guarded by __CUDA_ARCH__ >= 1000.
    // Fallback to async memcpy if kernel fails to launch.
    cudaFuncAttributes attr;
    cudaError_t lerr = cudaFuncGetAttributes(&attr, tma_copy_store_kernel);
    if (lerr == cudaSuccess) {
        tma_copy_store_kernel<<<1, 1, 0, transfer->stream>>>(
            reinterpret_cast<const tma_store_desc*>(transfer->desc));
    } else {
        GGML_LOG_WARN("TMA kernel not available, falling back to memcpy\n");
        cudaMemcpyAsync(transfer->dst_vram, transfer->src_pinned, bytes,
                        cudaMemcpyHostToDevice, transfer->stream);
    }
}

void ggml_tma_free_transfer(ggml_tma_transfer_t transfer) {
    if (!transfer) return;
    if (transfer->desc) {
        cudaFree(transfer->desc);
    }
    delete transfer;
}
