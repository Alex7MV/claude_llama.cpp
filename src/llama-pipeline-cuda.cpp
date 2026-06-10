#include "pipeline-sched.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"

// Bridge callback matching the new llama_pipeline_expert_prefetch_fn signature.
// Delegates all CUDA work to ggml_backend_cuda_pipeline_expert_skip_prefetch.
static void deepseek_skip_prefetch_callback(
    struct ggml_tensor  * dst_tensors[3],
    struct ggml_tensor  * src_tensors[3],
    size_t                slice_bytes[3],
    struct ggml_tensor  * expert_mask,
    struct ggml_tensor  * moe_remap,
    int                   layer,
    ggml_backend_event_t  moe_ready,
    ggml_backend_event_t  completion_event,
    void                * user_data) {

    (void)layer;
    ggml_backend_t backend = (ggml_backend_t)user_data;
    ggml_backend_cuda_pipeline_expert_skip_prefetch(
        dst_tensors, src_tensors, slice_bytes,
        expert_mask, moe_remap,
        moe_ready, completion_event, backend);
}

void llama_pipeline_cuda_init_prefetch(
    struct llama_pipeline_sched * p,
    ggml_backend_t               backend) {
    llama_pipeline_sched_set_prefetch_fn(p, deepseek_skip_prefetch_callback, (void *)backend);
}

#else

void llama_pipeline_cuda_init_prefetch(
    struct llama_pipeline_sched * p,
    ggml_backend_t               backend) {
    // No-op: CUDA not available
    (void)p;
    (void)backend;
}

#endif // GGML_USE_CUDA
