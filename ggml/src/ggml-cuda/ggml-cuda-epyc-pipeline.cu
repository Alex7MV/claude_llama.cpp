#include "ggml-cuda-epyc-pipeline.h"
#include "ggml-cuda-blackwell.cuh"
#include "common.cuh"
#include "ggml-backend-pipeline.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Health-check: verify PCIe bus NUMA affinity and alignment
// ---------------------------------------------------------------------------

bool ggml_cuda_epyc_pipeline_health_check(int device_id, void * pinned_base, void * vram_base) {
    // 1. Check PCIe bus NUMA affinity
    int numa_node = -1;
    char path[256];
    int ret = snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/numa_node", device_id);
    if (ret > 0 && (size_t)ret < sizeof(path)) {
        int fd = open(path, O_RDONLY);
        if (fd >= 0) {
            char buf[16];
            ssize_t n = read(fd, buf, sizeof(buf) - 1);
            if (n > 0) {
                buf[n] = '\0';
                numa_node = atoi(buf);
            }
            close(fd);
        }
    }
    if (numa_node >= 0) {
        GGML_LOG_INFO("EPYC-GPU: device %d on NUMA node %d\n", device_id, numa_node);
    }

    // 2. Alignment checks
    uintptr_t pinned_addr = (uintptr_t)pinned_base;
    uintptr_t vram_addr = (uintptr_t)vram_base;

    if (pinned_base && (pinned_addr & 127) != 0) {
        GGML_LOG_ERROR("EPYC-GPU: pinned buffer not 128-byte aligned (%p)\n", pinned_base);
        return false;
    }
    if (vram_base && (vram_addr & 15) != 0) {
        GGML_LOG_ERROR("EPYC-GPU: VRAM buffer not 16-byte aligned (%p)\n", vram_base);
        return false;
    }
    if (pinned_addr >= (1ULL << 48)) {
        GGML_LOG_ERROR("EPYC-GPU: pinned buffer VA exceeds 48 bits (%p)\n", pinned_base);
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Merge stream for KV cache linear-to-ring transfer
// ---------------------------------------------------------------------------

static cudaStream_t g_merge_stream = nullptr;

static cudaStream_t ggml_cuda_epyc_get_merge_stream() {
    if (!g_merge_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_merge_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            GGML_LOG_ERROR("EPYC-GPU: failed to create merge stream: %s\n", cudaGetErrorString(err));
            g_merge_stream = nullptr;
        }
    }
    return g_merge_stream;
}

void ggml_cuda_epyc_pipeline_merge_kv(
    void * dst_ring,
    const void * src_linear,
    size_t bytes_per_layer,
    int n_layers,
    int ring_stride,
    void * last_split_event) {

    if (!dst_ring || !src_linear || bytes_per_layer == 0 || n_layers <= 0) {
        return;
    }

    cudaStream_t stream = ggml_cuda_epyc_get_merge_stream();
    if (!stream) return;

    // Wait for last GPU split to finish before merging
    if (last_split_event) {
        cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(last_split_event), 0);
    }

    // Each layer's KV chunk is copied from linear prefill layout to ring buffer
    char * dst = static_cast<char *>(dst_ring);
    const char * src = static_cast<const char *>(src_linear);

    for (int il = 0; il < n_layers; il++) {
        cudaError_t err = cudaMemcpyAsync(
            dst + il * ring_stride,
            src + il * bytes_per_layer,
            bytes_per_layer,
            cudaMemcpyDeviceToDevice,
            stream);
        if (err != cudaSuccess) {
            GGML_LOG_ERROR("EPYC-GPU: merge memcpy layer %d failed: %s\n", il, cudaGetErrorString(err));
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// GPU dispatch callback for pipeline slots
// ---------------------------------------------------------------------------

static void CUDART_CB ggml_epyc_host_complete(void * user_data) {
    auto * flag = reinterpret_cast<std::atomic<bool> *>(user_data);
    if (flag) {
        flag->store(true, std::memory_order_release);
    }
}

void ggml_cuda_epyc_pipeline_dispatch(
    void * user_data,
    const struct ggml_pipeline_slot_info * slot) {

    if (!slot || !slot->cpu_buf || !slot->gpu_buf || slot->buf_bytes == 0) {
        if (slot && slot->gpu_async_done) {
            reinterpret_cast<std::atomic<bool> *>(slot->gpu_async_done)
                ->store(true, std::memory_order_release);
        }
        return;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(user_data);

    // TODO: When TMA is enabled and this buffer is pinned+aligned,
    // launch ggml_bw_pipeline_kernel here instead of cudaMemcpyAsync.
    // For now, use cudaMemcpyAsync as the proven fallback.
    cudaError_t err = cudaMemcpyAsync(
        slot->gpu_buf,
        slot->cpu_buf,
        slot->buf_bytes,
        cudaMemcpyHostToDevice,
        stream);

    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemcpyAsync failed: %s\n", __func__, cudaGetErrorString(err));
        if (slot->gpu_async_done) {
            reinterpret_cast<std::atomic<bool> *>(slot->gpu_async_done)
                ->store(true, std::memory_order_release);
        }
        return;
    }

    if (slot->completion_event) {
        cudaEventRecord(static_cast<cudaEvent_t>(slot->completion_event), stream);
    }

    if (slot->gpu_async_done) {
        cudaLaunchHostFunc(stream, ggml_epyc_host_complete, slot->gpu_async_done);
    }
}
