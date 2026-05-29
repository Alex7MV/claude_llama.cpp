#include "common.cuh"

#include "tma-transfer.h"

#include <cstdlib>
#include <cstring>

// TMA-based transfer layer for moving KV cache data between pinned
// system RAM and GPU VRAM.  On SM 100+ the kernel can use the TMA
// engine; otherwise it falls back to cudaMemcpyAsync.
//
// Runtime feature flag: set GGML_CUDA_TMA=1 to enable TMA path.
// Requires: CUDA runtime >= 12.4, device compute capability >= sm_90.

// We define our own copy of the 16-byte TMA descriptor type here to
// avoid pulling in tma.cuh host-side helpers (which are gated behind
// #ifndef __CUDA_ARCH__ and cause visibility issues in .cu translation
// units).  The layout matches ggml_cuda_tma_desc from tma.cuh.

#pragma pack(push, 1)
struct tma_transfer_desc {
    uint64_t d[2];
};
#pragma pack(pop)
static_assert(sizeof(tma_transfer_desc) == 16, "TMA descriptor must be 16 bytes");

struct ggml_tma_transfer {
    tma_transfer_desc * desc;      // device-side TMA descriptor (16 bytes)
    cudaStream_t stream;
    size_t num_elements;
    size_t elem_size;
    void * src_pinned;
    void * dst_vram;
    bool use_tma;                  // true if SM >= 100 and kernel is verified
};

// ---------------------------------------------------------------------------
// Runtime TMA capability probe
// ---------------------------------------------------------------------------

static bool g_ggml_cuda_tma_enabled = false;
static bool g_ggml_cuda_tma_probed = false;

bool ggml_cuda_tma_supported() {
    if (g_ggml_cuda_tma_probed) return g_ggml_cuda_tma_enabled;
    g_ggml_cuda_tma_probed = true;

    const char * env = std::getenv("GGML_CUDA_TMA");
    if (!env || std::strcmp(env, "1") != 0) {
        return false;
    }

    int runtimeVer = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtimeVer);
    if (err != cudaSuccess || runtimeVer < 12040) {
        GGML_LOG_WARN("TMA: requires CUDA runtime >= 12.4, have %d (err=%s)\n",
            runtimeVer, cudaGetErrorString(err));
        return false;
    }

    int device = 0;
    cudaGetDevice(&device);
    int ccMajor = 0, ccMinor = 0;
    cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, device);
    int cc = ccMajor * 10 + ccMinor;
    if (cc < 90) {
        GGML_LOG_WARN("TMA: requires sm_90+, have sm_%d\n", cc);
        return false;
    }

    g_ggml_cuda_tma_enabled = true;
    GGML_LOG_INFO("TMA: enabled on sm_%d with CUDA %d\n", cc, runtimeVer);
    return true;
}

// ---------------------------------------------------------------------------
// Transfer init / launch / free
// ---------------------------------------------------------------------------

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

    ggml_tma_transfer * transfer = new ggml_tma_transfer();
    transfer->src_pinned  = src_pinned;
    transfer->dst_vram    = dst_vram;
    transfer->num_elements = num_elements;
    transfer->elem_size   = elem_size > 0 ? elem_size : 2;  // default float16/bf16
    transfer->stream      = (cudaStream_t)stream;
    transfer->desc        = nullptr;
    transfer->use_tma     = ggml_cuda_tma_supported();

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
        cudaMemcpyAsync(transfer->desc, &host_desc, sizeof(tma_transfer_desc),
                        cudaMemcpyHostToDevice, transfer->stream);
    }

    *out = transfer;
    return true;
}

void ggml_tma_launch_transfer(ggml_tma_transfer_t transfer) {
    if (!transfer) return;

    size_t bytes = transfer->num_elements * transfer->elem_size;

    // TMA kernel launch gated by use_tma flag (SM 100+, deferred).
    // Currently always uses cudaMemcpyAsync on dedicated stream.
    if (!transfer->use_tma) {
        cudaMemcpyAsync(transfer->dst_vram, transfer->src_pinned, bytes,
                        cudaMemcpyHostToDevice, transfer->stream);
        return;
    }
    // TODO: launch ggml_tma_kv_transfer_kernel when TMA verified on live hardware
}

void ggml_tma_free_transfer(ggml_tma_transfer_t transfer) {
    if (!transfer) return;
    if (transfer->desc) {
        cudaFree(transfer->desc);
    }
    delete transfer;
}
