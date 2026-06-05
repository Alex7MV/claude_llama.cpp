#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

// Hybrid orchestrator sets these to redirect flash_attn to a separate stream.
// attn_global_event is the fence event for GPU→CPU coherence.
// Defined in fattn.cu.
extern cudaStream_t attn_global_stream;
extern cudaEvent_t  attn_global_event;

static inline void ggml_cuda_attn_set_async_ctx(cudaStream_t stream, cudaEvent_t event) {
    attn_global_stream   = stream;
    attn_global_event    = event;
}
