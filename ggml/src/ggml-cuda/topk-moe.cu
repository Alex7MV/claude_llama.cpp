#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool with_norm;
    bool delayed_softmax;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_cuda(const float *         logits,
                                                                  float *               weights,
                                                                  int32_t *             ids,
                                                                  float *               bias,
                                                                  const int             n_rows,
                                                                  const int             n_expert_used,
                                                                  const float           clamp_val,
                                                                  const float           scale_val,
                                                                  const topk_moe_config config) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = 4;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Hyper-sparse MoE Phase 2: Cumulative Threshold Kernel
//
// 1 block × 32 threads. Reads per-expert contribution totals after Phase A
// routing (moe_ids + moe_weights), computes cumulative weight threshold at
// 0.95 with floor(3), builds skip_mask (uint64_t bitmask) and remap table.
//
// Inputs:
//   moe_ids:       [n_expert_used, n_tokens] i32 — expert IDs per token per slot
//   moe_weights_in: [n_expert_used, n_tokens] f32 — pre-renormalization weights
//
// Outputs (GPU global memory):
//   moe_weights_out: [n_expert_used, n_tokens] f32 — renormalized (skipped → 0.0)
//   skip_mask:       [ceil(n_expert/64)] uint64_t — bit=1 → skip this expert
//   remap:           [n_expert] int32_t — remap[expert_id] = compact_slot or -1
//   kept_count:      [1] int32_t — number of kept experts
// ============================================================================
__global__ void cumulative_threshold_kernel(
    const int32_t * moe_ids,
    const float   * moe_weights_in,
    float         * moe_weights_out,
    uint64_t      * skip_mask,
    int32_t       * remap,
    int32_t       * kept_count,
    int             n_tokens,
    int             n_expert_used,
    int             n_expert,
    int             n_padded,       // next power of 2 >= n_expert (bitonic sort requires POW2)
    float           threshold,
    int             floor_experts) {

    // Shared memory for per-expert contributions and sort indices.
    // Padded to n_padded for bitonic sort (max 1024 → 12 KB < 48 KB).
    extern __shared__ float shmem[];
    float * contrib   = shmem;                     // [n_padded]
    float * s_vals    = shmem + n_padded;          // [n_padded] sorted copy
    int   * s_idx     = (int *)(shmem + 2 * n_padded);  // [n_padded]

    const int tid = threadIdx.x;  // 0..31
    const int total_slots = n_tokens * n_expert_used;

    // ---- Step 1: Clear contribution buffer ----
    for (int i = tid; i < n_padded; i += 32) {
        contrib[i] = 0.0f;
    }
    __syncthreads();

    // ---- Step 2: Scatter weights → per-expert contributions ----
    for (int idx = tid; idx < total_slots; idx += 32) {
        int e = moe_ids[idx];
        float w = moe_weights_in[idx];
        if (e >= 0 && e < n_expert) {
            atomicAdd(&contrib[e], w);
        }
    }
    __syncthreads();

    // ---- Step 3: Compute total contribution sum ----
    float total = 0.0f;
    for (int i = tid; i < n_expert; i += 32) {
        total += contrib[i];
    }
    total = warp_reduce_sum(total);

    // Copy contrib → s_vals, init s_idx; pad extras with -FLT_MAX
    for (int i = tid; i < n_padded; i += 32) {
        if (i < n_expert) {
            s_vals[i] = contrib[i];
            s_idx[i]  = i;
        } else {
            s_vals[i] = -FLT_MAX;
            s_idx[i]  = i;
        }
    }
    __syncthreads();

    // ---- Step 4: Bitonic sort descending by contribution ----
    // Operates on n_padded elements; sentinel -FLT_MAX sink to bottom.
    for (int k = 2; k <= n_padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj;
            for (int base = tid; base < n_padded; base += 32) {
                int i = base;
                ixj = i ^ j;
                if (ixj > i) {
                    bool descending = ((i & k) == 0);
                    bool swap = descending ? (s_vals[i] < s_vals[ixj]) : (s_vals[i] > s_vals[ixj]);
                    if (swap) {
                        float tmp_f = s_vals[i];
                        s_vals[i]   = s_vals[ixj];
                        s_vals[ixj] = tmp_f;
                        int tmp_i   = s_idx[i];
                        s_idx[i]    = s_idx[ixj];
                        s_idx[ixj]  = tmp_i;
                    }
                }
            }
            __syncthreads();
        }
    }

    // ---- Step 5: Cumulative threshold (iterate only real n_expert, not padded) ----
    float cum = 0.0f;
    float target = total * threshold;
    int split = n_expert;

    if (tid == 0) {
        for (int i = 0; i < n_expert; i++) {
            cum += s_vals[i];
            if ((cum >= target || i >= n_expert - floor_experts) && i >= floor_experts - 1) {
                split = i + 1;
                if (cum < target) {
                    split = i + 1;
                }
                break;
            }
        }
        if (split < floor_experts) {
            split = floor_experts;
        }
        *kept_count = split;
    }
    __syncthreads();

    // ---- Step 6: Build skip_mask (bit=1 → skip) ----
    const int n_mask_words = (n_expert + 63) / 64;

    // Reuse contrib area as byte "is_kept" flag (contrib dead after sort)
    bool * is_kept = (bool *)shmem;

    for (int e = tid; e < n_expert; e += 32) {
        is_kept[e] = false;
    }
    __syncthreads();

    for (int i = tid; i < split; i += 32) {
        is_kept[s_idx[i]] = true;
    }
    __syncthreads();

    for (int w = tid; w < n_mask_words; w += 32) {
        uint64_t mask = 0ULL;
        int base = w * 64;
        int limit = min(64, n_expert - base);
        for (int b = 0; b < limit; b++) {
            if (!is_kept[base + b]) {
                mask |= (1ULL << b);
            }
        }
        skip_mask[w] = mask;
    }
    __syncthreads();

    // ---- Step 7: Build remap table (original expert ID → compact slot) ----
    // S4: skipped experts get remap[e]=0 (safe sentinel; zero weight kills output).
    //     DO NOT write -1 — ggml_get_rows treats it as OOB → GPU crash.
    for (int e = tid; e < n_expert; e += 32) {
        remap[e] = 0;
    }
    __syncthreads();

    for (int i = tid; i < split; i += 32) {
        remap[s_idx[i]] = i;
    }
    __syncthreads();

    // ---- Step 8: Renormalize weights (skipped → 0.0, kept → w / Σ_kept w) ----
    for (int t = tid; t < n_tokens; t += 32) {
        const float * w_in  = moe_weights_in + t * n_expert_used;
        float *       w_out = moe_weights_out + t * n_expert_used;

        float kept_sum = 0.0f;
        for (int k = 0; k < n_expert_used; k++) {
            int e = moe_ids[t * n_expert_used + k];
            int word = e / 64;
            int bit  = e % 64;
            bool skipped = (skip_mask[word] >> bit) & 1ULL;
            if (!skipped) {
                kept_sum += w_in[k];
            }
        }

        float inv_sum = kept_sum > 1e-10f ? 1.0f / kept_sum : 1.0f;
        for (int k = 0; k < n_expert_used; k++) {
            int e = moe_ids[t * n_expert_used + k];
            int word = e / 64;
            int bit  = e % 64;
            bool skipped = (skip_mask[word] >> bit) & 1ULL;
            w_out[k] = skipped ? 0.0f : w_in[k] * inv_sum;
        }
    }
}

// ============================================================================
// Host wrapper: launches cumulative_threshold_kernel with proper block config.
// Called from pipeline-sched.cpp after Phase A compute completes.
// ============================================================================
void ggml_backend_cuda_pipeline_moe_threshold(
    struct ggml_tensor  * moe_ids,
    struct ggml_tensor  * moe_weights_in,
    struct ggml_tensor  * moe_weights_out,
    struct ggml_tensor  * skip_mask,
    struct ggml_tensor  * remap,
    struct ggml_tensor  * kept_counts,
    int                   layer_idx,
    float                 threshold,
    int                   floor_experts,
    ggml_backend_t        backend) {

    auto * ctx = (ggml_backend_cuda_context *)backend->context;

    int n_expert_used = moe_ids->ne[0];
    int n_tokens      = moe_ids->ne[1];
    int n_expert      = remap->ne[0];

    // S1: pad to next power of 2 for bitonic sort
    int n_padded = 1;
    while (n_padded < n_expert) n_padded <<= 1;

    cudaStream_t stream = ctx->stream();

    // shared memory sized for padded arrays
    size_t shmem_bytes = (2 * sizeof(float) + sizeof(int)) * n_padded;

    dim3 grid_dims(1, 1, 1);
    dim3 block_dims(32, 1, 1);

    // kept_counts[layer_idx] = split (the number of kept experts)
    int32_t * kept_ptr = (int32_t *)kept_counts->data + layer_idx;

    cumulative_threshold_kernel<<<grid_dims, block_dims, shmem_bytes, stream>>>(
        (const int32_t *)moe_ids->data,
        (const float   *)moe_weights_in->data,
        (float         *)moe_weights_out->data,
        (uint64_t      *)skip_mask->data,
        (int32_t       *)remap->data,
        kept_ptr,
        n_tokens,
        n_expert_used,
        n_expert,
        n_padded,
        threshold,
        floor_experts);
}
