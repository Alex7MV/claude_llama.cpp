#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Forward declaration to avoid including the full pipeline header here.
struct ggml_pipeline_slot_info;

// ---------------------------------------------------------------------------
// GPU dispatch callback for the async pipeline fallback path.
//
// Launches cudaMemcpyAsync for the slot's cpu_buf -> gpu_buf transfer
// and sets the gpu_async_done flag via cudaLaunchHostFunc when complete.
//
// @param user_data  cudaStream_t cast to void*
// @param slot       Slot descriptor from the pipeline queue
// ---------------------------------------------------------------------------
void ggml_bw_pipeline_gpu_dispatch(
    void * user_data,
    const struct ggml_pipeline_slot_info * slot);

// Return a non-blocking CUDA stream for the pipeline dispatch callback.
// The stream is created lazily and persists for the process lifetime.
// Returns nullptr if CUDA is not available.
void * ggml_bw_pipeline_get_stream(void);

// Host-side launcher for the Blackwell tile-pipeline kernel.
// This file is C-compatible so it can be called from ggml-backend-pipeline.cpp
// via function pointer or direct linkage when both CUDA and base are linked.

// Tile geometry for KV-cache tiles (matches ggml_blackwell_tile_shape)
struct ggml_bw_tile_shape {
    int32_t tile_m;
    int32_t tile_n;
    int32_t tile_k;
    int32_t type_size;
};

// Opaque handle to a launched pipeline kernel instance
typedef struct ggml_bw_pipeline_kernel * ggml_bw_pipeline_kernel_t;

// ---------------------------------------------------------------------------
// Launch the Blackwell tile-pipeline kernel for one pipeline slot.
//
// The kernel runs a loop over tiles within the assigned layer range.
// For each tile:
//   - TMA warp issues cp.async.bulk to move K/V from pinned host -> SMEM
//   - mbarrier signals when the tile is resident
//   - WGMMA warps start computing immediately (early-start / zero-bubble)
//   - Output is written to dst_vram
//
// @param stream      CUDA stream for async execution
// @param src_pinned  Pinned host buffer (128-byte aligned) containing K/V
// @param dst_vram    GPU VRAM output buffer (16-byte aligned)
// @param bytes       Total bytes in src_pinned for this slot
// @param first_layer First layer index in this slot's range
// @param num_layers  Number of layers in this slot's range
// @param tile        Tile geometry (usually 64x128x64 BF16)
// @param use_tma     If false, falls back to cudaMemcpyAsync + dummy compute
//
// @return true if kernel was launched, false on error (logs to ggml log)
// ---------------------------------------------------------------------------
bool ggml_bw_pipeline_launch(
    void * stream,  // cudaStream_t passed as void* for C compat
    void * src_pinned,
    void * dst_vram,
    size_t bytes,
    int32_t first_layer,
    int32_t num_layers,
    const struct ggml_bw_tile_shape * tile,
    bool use_tma);

// Synchronize a launched kernel (wraps cudaStreamSynchronize)
void ggml_bw_pipeline_sync(void * stream);

// Host-side SMEM size calculation (for cudaFuncSetAttribute)
size_t ggml_bw_pipeline_smem_per_block(const struct ggml_bw_tile_shape * tile);

// Validate that the tile shape + inflight count fits device limits
bool ggml_bw_pipeline_validate_device(
    int device_id,
    const struct ggml_bw_tile_shape * tile,
    int num_inflight_blocks);

#ifdef __cplusplus
}
#endif
