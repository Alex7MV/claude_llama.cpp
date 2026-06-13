#include "ggml-cuda.h"
#include "tma-transfer.h"
#include "vendors/cuda.h"

#include <cstdlib>
#include <cstring>

// Read skip_mask from GPU → host, iterate non-skipped experts, copy each to compact slot.
// Uses TMA for H2D when available (sm_90+ / RTX 5090), falls back to cudaMemcpyAsync.
//
// dst[3]: compact GPU prefetch buffer (gate/up/down) [..., PREFETCH_MAX_EXPERTS]
// src[3]: CPU model weight tensors [..., n_expert]
// slice_bytes[3]: bytes per single expert in each weight type
// expert_mask: GPU tensor [ceil(n_expert/64)] u64, bit=1 → skip
// moe_remap:   GPU tensor [n_expert] i32, value = compact_slot for kept, -1 for skipped
//
// For each non-skipped expert e (expert_mask bit = 0):
//   slot = moe_remap[e]  (guaranteed >= 0 and < LLAMA_PIPELINE_PREFETCH_MAX_EXPERTS)
//   for each weight type s:
//     TMA/cudaMemcpyAsync(dst[s] + slot * slice_bytes[s],
//                         src[s] + e   * slice_bytes[s],
//                         slice_bytes[s], H2D, transfer_stream)
//
void ggml_backend_cuda_pipeline_expert_skip_prefetch(
    struct ggml_tensor  * dst[3],
    struct ggml_tensor  * src[3],
    size_t                slice_bytes[3],
    struct ggml_tensor  * expert_mask,
    struct ggml_tensor  * moe_remap,
    const uint64_t      * host_mask_ptr,
    const int32_t       * host_remap_ptr,
    ggml_backend_event_t  moe_ready,
    ggml_backend_event_t  completion_event,
    ggml_backend_t        backend) {

    const int transfer_stream_id = 2;

    // Switch backend to transfer stream
    ggml_backend_cuda_set_stream(backend, transfer_stream_id);

    // Wait for MoE routing + threshold to complete before reading skip_mask
    if (moe_ready) {
        ggml_backend_event_wait(backend, moe_ready);
    }

    cudaStream_t stream = (cudaStream_t)ggml_backend_cuda_get_stream_ptr(backend, transfer_stream_id);

    // Read n_expert from moe_remap shape: [n_expert] i32
    int n_expert = (int)(ggml_nbytes(moe_remap) / (int)sizeof(int32_t));
    size_t remap_nbytes = ggml_nbytes(moe_remap);
    bool remap_alloc = false;

    // ---- Step 1: Read expert_mask from GPU → host (D2H async, tiny) ----
    // expert_mask is [ceil(n_expert/64)] u64 → max 9 uint64 = 72 bytes
    size_t mask_nbytes = ggml_nbytes(expert_mask);
    bool mask_alloc = false;
    uint64_t * host_mask;
    if (host_mask_ptr) {
        host_mask = (uint64_t *)host_mask_ptr;
    } else {
        host_mask = (uint64_t *)malloc(mask_nbytes);
        if (!host_mask) {
            goto skip_prefetch_fail;
        }
        mask_alloc = true;
        cudaMemcpyAsync(host_mask, expert_mask->data, mask_nbytes,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // ---- Step 2: Read moe_remap from GPU → host (also tiny) ----
    int32_t * host_remap;
    if (host_remap_ptr) {
        host_remap = (int32_t *)host_remap_ptr;
    } else {
        host_remap = (int32_t *)malloc(remap_nbytes);
        if (!host_remap) {
            if (mask_alloc) free(host_mask);
            goto skip_prefetch_fail;
        }
        remap_alloc = true;
        cudaMemcpyAsync(host_remap, moe_remap->data, remap_nbytes,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // ---- Step 3: Fast Skip — iterate experts, H2D only non-skipped ----
    for (int e = 0; e < n_expert; e++) {
        // Check skip bit in mask
        int word = e / 64;
        int bit  = e % 64;
        bool skip = (host_mask[word] >> bit) & 1ULL;
        if (skip) continue;  // ← Fast Skip: no memcpy at all

        // Compact slot from remap table
        int slot = host_remap[e];
        if (slot < 0) continue; // safety: should never happen for non-skipped

        // Use TMA for H2D when available, fallback to cudaMemcpyAsync
        for (int s = 0; s < 3; s++) {
            if (!dst[s] || !src[s] || slice_bytes[s] == 0) continue;

            void * gpu_dst = (char *)dst[s]->data + (size_t)slot * slice_bytes[s];
            void * cpu_src = (char *)src[s]->data + (size_t)e     * slice_bytes[s];

            // Try TMA first (sm_90+ Blackwell / RTX 5090 feature)
            if (!ggml_tma_enqueue_h2d_1d(gpu_dst, cpu_src, slice_bytes[s], stream)) {
                // TMA unavailable/fallback → use cudaMemcpyAsync
                cudaMemcpyAsync(gpu_dst, cpu_src, slice_bytes[s],
                                cudaMemcpyHostToDevice, stream);
            }
        }
    }

    if (mask_alloc) free(host_mask);
    if (remap_alloc) free(host_remap);

    // Record completion event on transfer stream
    if (completion_event) {
        ggml_backend_event_record(completion_event, backend);
    }

    // Restore backend to compute stream
    ggml_backend_cuda_set_stream(backend, 0);
    return;

skip_prefetch_fail:
    if (completion_event) {
        ggml_backend_event_record(completion_event, backend);
    }
    ggml_backend_cuda_set_stream(backend, 0);
}
