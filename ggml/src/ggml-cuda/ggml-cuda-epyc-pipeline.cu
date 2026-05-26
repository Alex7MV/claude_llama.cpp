#include "common.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

struct ggml_epyc_pipeline_gpu {
    cudaStream_t    compute_stream;
    cudaStream_t    merge_stream;
    cudaEvent_t     stage_events[4];
    int             depth;
    bool            health_ok;
    bool            numa_affinity_ok;
    bool            alignment_ok;
};

static ggml_epyc_pipeline_gpu * g_epyc_pipeline = nullptr;

extern "C" {

bool ggml_epyc_pipeline_health_check(int gpu_device_id) {
    bool numa_ok = true;

    // Check 1: GPU NUMA affinity via sysfs
    char path[512];
    int gpu_numa_node = -1;
    int n = snprintf(path, sizeof(path),
        "/sys/class/drm/card%d/device/numa_node", gpu_device_id);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "WARN: sysfs path too long for health check\n");
        return true;
    }
    FILE * f = fopen(path, "r");
    if (f) {
        fscanf(f, "%d", &gpu_numa_node);
        fclose(f);
    }
    if (gpu_numa_node < 0) {
        numa_ok = false;
        fprintf(stderr, "WARN: could not determine GPU NUMA node for card%d\n",
            gpu_device_id);
    }

    // Check 2: Verify async engine count for TMA (non-fatal, no CUDA_CHECK)
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "WARN: health check skipped, CUDA not initialized (%s)\n",
            cudaGetErrorString(err));
        return true;
    }
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "WARN: could not read device properties (%s)\n",
            cudaGetErrorString(err));
        return true;
    }

    if (prop.asyncEngineCount < 2) {
        fprintf(stderr, "WARN: GPU has only %d async engines, TMA may be limited\n",
            prop.asyncEngineCount);
    }

    if (g_epyc_pipeline) {
        g_epyc_pipeline->numa_affinity_ok = numa_ok;
        g_epyc_pipeline->alignment_ok = true;
        g_epyc_pipeline->health_ok = true;
    }
    return true; // Non-fatal warnings only
}

ggml_epyc_pipeline_gpu * ggml_epyc_pipeline_gpu_init(int depth, cudaStream_t main_stream) {
    if (depth < 1 || depth > 4) return nullptr;
    if (g_epyc_pipeline) return g_epyc_pipeline; // Prevent double-init leak

    g_epyc_pipeline = new ggml_epyc_pipeline_gpu();
    memset(g_epyc_pipeline, 0, sizeof(*g_epyc_pipeline));
    g_epyc_pipeline->depth = depth;
    g_epyc_pipeline->compute_stream = main_stream;
    g_epyc_pipeline->health_ok = false;

    // Create merge stream for KV cache linear-to-ring transfer (non-blocking)
    CUDA_CHECK(cudaStreamCreateWithFlags(&g_epyc_pipeline->merge_stream,
        cudaStreamNonBlocking));

    // Create stage events for pipeline sync
    for (int i = 0; i < depth; i++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&g_epyc_pipeline->stage_events[i],
            cudaEventDisableTiming));
    }

    return g_epyc_pipeline;
}

void ggml_epyc_pipeline_gpu_free(ggml_epyc_pipeline_gpu * pipe) {
    if (!pipe) return;
    if (pipe->merge_stream) CUDA_CHECK(cudaStreamDestroy(pipe->merge_stream));
    for (int i = 0; i < pipe->depth && i < 4; i++) {
        if (pipe->stage_events[i]) CUDA_CHECK(cudaEventDestroy(pipe->stage_events[i]));
    }
    delete pipe;
    if (g_epyc_pipeline == pipe) g_epyc_pipeline = nullptr;
}

cudaStream_t ggml_epyc_pipeline_get_compute_stream() {
    return g_epyc_pipeline ? g_epyc_pipeline->compute_stream : nullptr;
}

cudaStream_t ggml_epyc_pipeline_get_merge_stream() {
    return g_epyc_pipeline ? g_epyc_pipeline->merge_stream : nullptr;
}

void ggml_epyc_pipeline_record_stage(int stage, cudaStream_t stream) {
    if (!g_epyc_pipeline || stage < 0 || stage >= g_epyc_pipeline->depth) return;
    CUDA_CHECK(cudaEventRecord(g_epyc_pipeline->stage_events[stage], stream));
}

void ggml_epyc_pipeline_wait_stage(cudaStream_t stream, int stage) {
    if (!g_epyc_pipeline || stage < 0 || stage >= g_epyc_pipeline->depth) return;
    CUDA_CHECK(cudaStreamWaitEvent(stream, g_epyc_pipeline->stage_events[stage]));
}

// Dispatch async linear-to-ring KV cache merge on the merge stream.
// Copies n_rows from a linear buffer into the ring buffer with dst_stride.
// The merge stream implicitly waits on the compute stream's last event.
void ggml_epyc_pipeline_dispatch_kv_merge(
    void * src,
    void * dst,
    size_t row_size,
    int32_t n_rows,
    size_t dst_stride) {
    if (!g_epyc_pipeline || !src || !dst || n_rows <= 0 || row_size == 0) return;
    if (!dst_stride) dst_stride = row_size;  // contiguous fallback

    // Make merge stream wait on compute stream to ensure data is ready
    CUDA_CHECK(cudaStreamWaitEvent(g_epyc_pipeline->merge_stream,
        g_epyc_pipeline->stage_events[g_epyc_pipeline->depth - 1], 0));

    // Dispatch row-by-row async copies (each row is one KV cell)
    for (int32_t i = 0; i < n_rows; i++) {
        CUDA_CHECK(cudaMemcpyAsync(
            (char*)dst + i * (ssize_t)dst_stride,
            (char*)src + i * (ssize_t)row_size,
            row_size,
            cudaMemcpyDeviceToDevice,
            g_epyc_pipeline->merge_stream));
    }

    // Record merge completion event for generation to wait on if needed
    CUDA_CHECK(cudaEventRecord(g_epyc_pipeline->stage_events[0],
        g_epyc_pipeline->merge_stream));
}

} // extern "C"
