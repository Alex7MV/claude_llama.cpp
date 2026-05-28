#include "ggml-cuda-blackwell-pipeline.h"
#include "ggml-cuda-blackwell.cuh"
#include "common.cuh"
#include "ggml-backend-pipeline.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <atomic>

// ---------------------------------------------------------------------------
// Host function for cudaLaunchHostFunc: sets the gpu_async_done flag.
// ---------------------------------------------------------------------------
static void CUDART_CB ggml_bw_host_complete(void * user_data) {
    auto * flag = reinterpret_cast<std::atomic<bool> *>(user_data);
    if (flag) {
        flag->store(true, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// GPU dispatch callback for async pipeline fallback.
// ---------------------------------------------------------------------------
void ggml_bw_pipeline_gpu_dispatch(
    void * user_data,
    const struct ggml_pipeline_slot_info * slot) {

    if (!slot || !slot->cpu_buf || !slot->gpu_buf || slot->buf_bytes == 0) {
        // Nothing to transfer; mark done immediately.
        if (slot && slot->gpu_async_done) {
            reinterpret_cast<std::atomic<bool> *>(slot->gpu_async_done)
                ->store(true, std::memory_order_release);
        }
        return;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(user_data);

    cudaError_t err = cudaMemcpyAsync(
        slot->gpu_buf,
        slot->cpu_buf,
        slot->buf_bytes,
        cudaMemcpyHostToDevice,
        stream);

    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemcpyAsync failed: %s\n", __func__, cudaGetErrorString(err));
        // Mark done so the pipeline doesn't hang.
        if (slot->gpu_async_done) {
            reinterpret_cast<std::atomic<bool> *>(slot->gpu_async_done)
                ->store(true, std::memory_order_release);
        }
        return;
    }

    // Record completion event if provided (for cross-stream ordering).
    if (slot->completion_event) {
        cudaEventRecord(static_cast<cudaEvent_t>(slot->completion_event), stream);
    }

    // Schedule host callback to set the done flag when memcpy completes.
    if (slot->gpu_async_done) {
        cudaLaunchHostFunc(stream, ggml_bw_host_complete, slot->gpu_async_done);
    }
}

// ---------------------------------------------------------------------------
// Blackwell Tile-Pipeline Kernel
// ---------------------------------------------------------------------------
// Each block processes tiles from one pipeline slot.
// Block dims: 128 threads (4 warps). Warp 0 = TMA, Warps 1-3 = WGMMA.
//
// Shared memory layout per block:
//   [0..7]     mbarrier (uint64_t)
//   [16..K)    K tile buffer  (tile_m * tile_n * type_size bytes)
//   [K..K+V)   V tile buffer  (same size)
//   (padded to 16-byte alignment)
//
// The kernel loops over tiles within the slot's byte range.
// For each tile:
//   1. TMA warp: mbarrier.init, arrive_expect_tx(tile_bytes*2)
//   2. TMA warp: cp.async.bulk for K and V tiles
//   3. TMA warp: cp.async.bulk.commit_group
//   4. WGMMA warps: mbarrier.wait (unblocks when BOTH K and V are resident)
//   5. WGMMA warps: compute (stub — real WGMMA call lives in mmq.cuh)
//   6. Advance to next tile
//
// Zero-bubble: step 4 unblocks as soon as the tile is in SMEM, not when
// the whole slot is copied. The next tile's TMA can overlap with the
// current tile's WGMMA because each tile uses the same SMEM slot
// sequentially within the block.
// ---------------------------------------------------------------------------

#define BW_PIPELINE_THREADS_PER_BLOCK 128
#define BW_PIPELINE_WARPS_PER_BLOCK   4
#define BW_PIPELINE_TMA_WARP_ID       0

// Per-block dynamic shared memory is declared extern; size set at launch.
extern __shared__ char bw_pipeline_smem[];

struct bw_pipeline_smem_layout {
    uint64_t mbar;
    // K and V buffers follow at runtime-computed offsets
};

__device__ inline uint64_t * bw_smem_mbar(char * smem) {
    return reinterpret_cast<uint64_t *>(smem);
}

__device__ inline void * bw_smem_k_buf(char * smem, size_t mbar_size) {
    return smem + mbar_size;
}

__device__ inline void * bw_smem_v_buf(char * smem, size_t mbar_size, size_t tile_bytes) {
    return smem + mbar_size + tile_bytes;
}

// ---------------------------------------------------------------------------
// TMA descriptor device type (matches tma.cuh layout)
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct bw_tma_desc {
    uint64_t d[2];
};
#pragma pack(pop)

// Build a 1D TMA load descriptor on the device (used when host passes
// raw pointers instead of pre-built descriptors).
__device__ inline bw_tma_desc bw_make_tma_desc_1d(const void * base_ptr, size_t num_bytes) {
    bw_tma_desc desc;
    uint64_t addr = (uint64_t)base_ptr & 0xFFFFFFFFFFFFUL;
    desc.d[0] = addr;
    desc.d[1] = num_bytes & 0xFFFFFFFFFFFFUL;
    return desc;
}

// ---------------------------------------------------------------------------
// Core tile orchestration (device)
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(GGML_CUDA_HAS_WGMMA)

__device__ inline void bw_tile_tma_wgmma(
    char * smem,
    size_t smem_mbar_offs,
    size_t smem_k_offs,
    size_t smem_v_offs,
    const void * src_k,      // pinned host K pointer for this tile
    const void * src_v,      // pinned host V pointer for this tile
    uint32_t tile_bytes,
    int lane_id,
    int warp_id) {

    uint64_t * mbar = reinterpret_cast<uint64_t *>(smem + smem_mbar_offs);
    void * smem_k = smem + smem_k_offs;
    void * smem_v = smem + smem_v_offs;

    bool is_tma_warp  = (warp_id == BW_PIPELINE_TMA_WARP_ID);
    bool is_wgmma_warp = (warp_id > BW_PIPELINE_TMA_WARP_ID);

    // ---- Phase 1: init mbarrier (leader thread only) ----
    if (lane_id == 0) {
        ggml_blackwell_mbar_init(mbar, 1);
    }
    __syncthreads();

    // ---- Phase 2: TMA warp arrives with expected tx ----
    uint64_t mbar_state = 0;
    if (is_tma_warp && lane_id == 0) {
        mbar_state = ggml_blackwell_mbar_arrive_expect_tx(mbar, tile_bytes * 2);
    }
    __syncthreads();

    // ---- Phase 3: Issue TMA bulk copies (TMA warp) ----
    if (is_tma_warp) {
        // Build descriptors for this tile's K and V sources
        bw_tma_desc desc_k = bw_make_tma_desc_1d(src_k, tile_bytes);
        bw_tma_desc desc_v = bw_make_tma_desc_1d(src_v, tile_bytes);

        // Prefetch into TMA descriptor cache
        ggml_blackwell_tma_desc_cache_prefetch(&desc_k);
        ggml_blackwell_tma_desc_cache_prefetch(&desc_v);

        // Issue cp.async.bulk for K tile
        asm volatile (
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], [%2], %3;"
            :: "r"((uint32_t)__cvta_generic_to_shared(smem_k)),
               "l"(&desc_k),
               "r"((uint32_t)__cvta_generic_to_shared(mbar)),
               "r"(tile_bytes)
            : "memory"
        );

        // Issue cp.async.bulk for V tile
        asm volatile (
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], [%2], %3;"
            :: "r"((uint32_t)__cvta_generic_to_shared(smem_v)),
               "l"(&desc_v),
               "r"((uint32_t)__cvta_generic_to_shared(mbar)),
               "r"(tile_bytes)
            : "memory"
        );

        // Commit TMA group
        asm volatile ("cp.async.bulk.commit_group;" ::: "memory");
    }

    // ---- Phase 4: WGMMA warps wait on mbarrier (EARLY START) ----
    if (is_wgmma_warp) {
        // Wait until both K and V tiles are resident in SMEM
        ggml_blackwell_mbar_wait(mbar, mbar_state);

        // At this point WGMMA can begin using smem_k / smem_v.
        // For this test-harness, we perform a minimal SMEM touch to ensure
        // the compiler does not optimize away the wait.
        // Real WGMMA: call ggml_cuda_wgmma::mma_sync<...>() here.
        volatile char * vk = (volatile char *)smem_k;
        volatile char * vv = (volatile char *)smem_v;
        #pragma unroll
        for (int i = lane_id; i < (int)tile_bytes; i += 32) {
            vk[i] = vk[i];
            vv[i] = vv[i];
        }
    }
    __syncthreads();
}

#endif // __CUDA_ARCH__ >= 1000 && defined(GGML_CUDA_HAS_WGMMA)

// Fallback path for pre-Blackwell or when TMA is disabled:
// Simple cudaMemcpyAsync-equivalent via global loads + dummy compute.
__device__ inline void bw_tile_memcpy_fallback(
    char * dst_vram,
    const char * src_pinned,
    size_t tile_bytes,
    int lane_id,
    int warp_id) {

    (void)warp_id;
    size_t global_offs = blockIdx.x * tile_bytes;
    const char * src = src_pinned + global_offs;
    char * dst = dst_vram + global_offs;

    // Coalesced copy by all threads in the block
    for (int i = threadIdx.x * 16; i < (int)tile_bytes; i += blockDim.x * 16) {
        if (i + 16 <= (int)tile_bytes) {
            int4 val = *((const int4 *)(src + i));
            *((int4 *)(dst + i)) = val;
        } else {
            for (int j = i + lane_id; j < (int)tile_bytes && j < i + 16; j += 32) {
                dst[j] = src[j];
            }
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Global kernel
// ---------------------------------------------------------------------------
__global__ void ggml_bw_pipeline_kernel(
    const char * src_pinned,
    char * dst_vram,
    size_t total_bytes,
    int32_t first_layer,
    int32_t num_layers,
    ggml_blackwell_tile_shape tile,
    bool use_tma) {

    (void)first_layer;
    (void)num_layers;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    uint32_t tile_bytes = ggml_blackwell_mbar_tx_bytes_kv(tile);
    if (tile_bytes == 0) return;

    size_t smem_mbar_size = sizeof(uint64_t);
    size_t smem_k_offs = (smem_mbar_size + 15) & ~15;
    size_t smem_v_offs = smem_k_offs + tile_bytes;
    size_t smem_total  = smem_v_offs + tile_bytes;
    smem_total = (smem_total + 15) & ~15;

    // Each block processes tiles sequentially.
    // Block b handles tiles: b, b+gridDim.x, b+2*gridDim.x, ...
    int num_tiles = (int)((total_bytes + tile_bytes - 1) / tile_bytes);

    for (int t = blockIdx.x; t < num_tiles; t += gridDim.x) {
        size_t tile_offs = (size_t)t * tile_bytes;
        size_t this_tile_bytes = (tile_offs + tile_bytes <= total_bytes)
                                 ? tile_bytes
                                 : (total_bytes - tile_offs);

        if (use_tma) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(GGML_CUDA_HAS_WGMMA)
            bw_tile_tma_wgmma(
                bw_pipeline_smem,
                0,  // mbarrier at start of dynamic SMEM
                smem_k_offs,
                smem_v_offs,
                src_pinned + tile_offs,
                src_pinned + tile_offs + this_tile_bytes, // V follows K in buffer
                (uint32_t)this_tile_bytes,
                lane_id,
                warp_id);
            #else
            // Should not reach here — host should not launch TMA on non-BW
            bw_tile_memcpy_fallback(dst_vram, src_pinned, this_tile_bytes, lane_id, warp_id);
            #endif
        } else {
            bw_tile_memcpy_fallback(dst_vram, src_pinned, this_tile_bytes, lane_id, warp_id);
        }

        // Write tile output from SMEM to VRAM (WGMMA result).
        // In the TMA path, the "compute" above is a stub; for a real kernel
        // the WGMMA accumulators would be stored here. For the memcpy fallback,
        // the copy already happened in-place.
        if (!use_tma) {
            // Already copied to dst_vram in fallback path
        } else {
            // TODO: store WGMMA accumulators to dst_vram after real matmul
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher implementation
// ---------------------------------------------------------------------------

bool ggml_bw_pipeline_launch(
    void * stream_ptr,
    void * src_pinned,
    void * dst_vram,
    size_t bytes,
    int32_t first_layer,
    int32_t num_layers,
    const struct ggml_bw_tile_shape * tile,
    bool use_tma) {

    if (!stream_ptr || !src_pinned || !dst_vram || bytes == 0 || !tile) {
        return false;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // Convert C tile shape to internal C++ type
    ggml_blackwell_tile_shape bw_tile = {
        tile->tile_m,
        tile->tile_n,
        tile->tile_k,
        tile->type_size
    };

    uint32_t tile_bytes = ggml_blackwell_mbar_tx_bytes_kv(bw_tile);
    if (tile_bytes == 0) {
        return false;
    }

    int num_tiles = (int)((bytes + tile_bytes - 1) / tile_bytes);
    if (num_tiles == 0) num_tiles = 1;

    // Launch up to 'num_tiles' blocks, but cap at a reasonable concurrency
    // limit to avoid SMEM pressure. For the test harness, use min(num_tiles, 16).
    int num_blocks = num_tiles;
    if (num_blocks > 16) num_blocks = 16;
    if (num_blocks < 1) num_blocks = 1;

    size_t smem_per_block = ggml_blackwell_smem_bytes_per_tile_slot(bw_tile);

    // If using TMA on Blackwell, set dynamic shared memory size.
    // Fallback path does not need extra SMEM.
    if (use_tma) {
        cudaError_t err = cudaFuncSetAttribute(
            ggml_bw_pipeline_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_per_block);
        if (err != cudaSuccess) {
            // SMEM too large for this config — fall back to memcpy
            use_tma = false;
        }
    }

    dim3 grid(num_blocks);
    dim3 block(BW_PIPELINE_THREADS_PER_BLOCK);

    size_t dynamic_smem = use_tma ? smem_per_block : 0;

    ggml_bw_pipeline_kernel<<<grid, block, dynamic_smem, stream>>>(
        static_cast<const char *>(src_pinned),
        static_cast<char *>(dst_vram),
        bytes,
        first_layer,
        num_layers,
        bw_tile,
        use_tma);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        GGML_LOG_ERROR("%s: kernel launch failed: %s\n", __func__, cudaGetErrorString(launch_err));
        return false;
    }

    return true;
}

void ggml_bw_pipeline_sync(void * stream_ptr) {
    if (!stream_ptr) return;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

size_t ggml_bw_pipeline_smem_per_block(const struct ggml_bw_tile_shape * tile) {
    if (!tile) return 0;
    ggml_blackwell_tile_shape bw_tile = {
        tile->tile_m, tile->tile_n, tile->tile_k, tile->type_size
    };
    return ggml_blackwell_smem_bytes_per_tile_slot(bw_tile);
}

bool ggml_bw_pipeline_validate_device(
    int device_id,
    const struct ggml_bw_tile_shape * tile,
    int num_inflight_blocks) {

    if (!tile || num_inflight_blocks <= 0) return false;

    ggml_blackwell_tile_shape bw_tile = {
        tile->tile_m, tile->tile_n, tile->tile_k, tile->type_size
    };

    ggml_blackwell_pipeline_caps caps = ggml_blackwell_probe_caps(device_id);
    if (!caps.wgmma_supported) return false;

    size_t smem_needed = ggml_blackwell_smem_total(bw_tile, num_inflight_blocks);
    return smem_needed <= static_cast<size_t>(caps.smem_per_block);
}

// ---------------------------------------------------------------------------
// Non-blocking CUDA stream for pipeline dispatch
// ---------------------------------------------------------------------------

void * ggml_bw_pipeline_get_stream(void) {
    static cudaStream_t g_bw_pipeline_stream = nullptr;
    static bool g_bw_stream_init_attempted = false;

    if (!g_bw_stream_init_attempted) {
        g_bw_stream_init_attempted = true;
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            err = cudaStreamCreateWithFlags(&g_bw_pipeline_stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                g_bw_pipeline_stream = nullptr;
            }
        }
    }
    return g_bw_pipeline_stream;
}
