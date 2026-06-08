#include "pipeline-sched.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"

// Bridge callback matching llama_pipeline_expert_prefetch_fn.
// Delegates all CUDA work to ggml_backend_cuda_pipeline_expert_prefetch.
static void deepseek_prefetch_callback(
    struct ggml_tensor  * dst_tensors[3],
    struct ggml_tensor  * src_tensors[3],
    size_t                slice_bytes[3],
    struct ggml_tensor  * top_k,
    int                   layer,
    ggml_backend_event_t  qkv_done,
    ggml_backend_event_t  completion_event,
    void                * user_data) {

    ggml_backend_t backend = (ggml_backend_t)user_data;
    ggml_backend_cuda_pipeline_expert_prefetch(
        dst_tensors, src_tensors, slice_bytes, top_k,
        qkv_done, completion_event, backend);
}

void llama_pipeline_cuda_init_prefetch(
    struct llama_pipeline_sched * p,
    ggml_backend_t               backend) {
    llama_pipeline_sched_set_prefetch_fn(p, deepseek_prefetch_callback, (void *)backend);
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
