#include "ggml-cuda.h"
#include "vendors/cuda.h"

void ggml_backend_cuda_pipeline_expert_prefetch(
    struct ggml_tensor  * dst[3],
    struct ggml_tensor  * src[3],
    size_t                slice_bytes[3],
    struct ggml_tensor  * top_k,
    ggml_backend_event_t  qkv_done,
    ggml_backend_event_t  completion_event,
    ggml_backend_t        backend) {

    const int transfer_stream_id = 2;

    // Switch backend to transfer stream
    ggml_backend_cuda_set_stream(backend, transfer_stream_id);

    // Wait for QKV to complete on the transfer stream
    ggml_backend_event_wait(backend, qkv_done);

    // Get raw CUDA stream pointer for cudaMemcpyAsync
    cudaStream_t stream = (cudaStream_t)ggml_backend_cuda_get_stream_ptr(backend, transfer_stream_id);

    // Read top_k indices from GPU to host (async D2H on transfer stream)
    size_t topk_nbytes = ggml_nbytes(top_k);
    int * host_topk = (int *)malloc(topk_nbytes);
    if (!host_topk) {
        ggml_backend_event_record(completion_event, backend);
        ggml_backend_cuda_set_stream(backend, 0);
        return;
    }

    cudaMemcpyAsync(host_topk, top_k->data, topk_nbytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Deduplicate expert IDs (small array — O(n^2) is fine)
    int n_total = (int)(topk_nbytes / (int)sizeof(int));
    int unique[256];
    int n_unique = 0;
    for (int i = 0; i < n_total && n_unique < 256; i++) {
        int e = host_topk[i];
        int j;
        for (j = 0; j < n_unique; j++) {
            if (unique[j] == e) break;
        }
        if (j == n_unique) unique[n_unique++] = e;
    }
    free(host_topk);

    // Launch async H2D copies for each unique expert's three weight types
    for (int i = 0; i < n_unique; i++) {
        int e = unique[i];
        for (int s = 0; s < 3; s++) {
            if (!dst[s] || !src[s] || slice_bytes[s] == 0) continue;
            void * gpu_dst = (char *)dst[s]->data + (size_t)i * slice_bytes[s];
            void * cpu_src = (char *)src[s]->data + (size_t)e * slice_bytes[s];
            cudaMemcpyAsync(gpu_dst, cpu_src, slice_bytes[s], cudaMemcpyHostToDevice, stream);
        }
    }

    // Record completion event on transfer stream
    ggml_backend_event_record(completion_event, backend);

    // Restore backend to compute stream
    ggml_backend_cuda_set_stream(backend, 0);
}
