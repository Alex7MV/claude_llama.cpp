#ifdef GGML_USE_CUDA

#include "moe-prefetcher.h"

#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdio>

#ifdef __linux__
#include <pthread.h>
#endif

extern "C" {
    int cudaStreamCreate(void ** stream);
    int cudaStreamDestroy(void * stream);
    int cudaStreamSynchronize(void * stream);
}

namespace moe {

moe_prefetcher::moe_prefetcher() {
}

moe_prefetcher::~moe_prefetcher() {
    if (running_.load(std::memory_order_acquire)) {
        stop();
    }
}

bool moe_prefetcher::start() {
    if (running_.load(std::memory_order_acquire)) return true;

    int e = cudaStreamCreate(&h2d_stream_);
    if (e != 0) {
        fprintf(stderr, "moe_prefetcher: cudaStreamCreate failed: %d\n", e);
        return false;
    }

    running_.store(true, std::memory_order_release);
    stop_requested_.store(false, std::memory_order_release);

    try {
        worker_ = std::thread(&moe_prefetcher::worker_loop, this);
    } catch (const std::exception & ex) {
        fprintf(stderr, "moe_prefetcher: thread creation failed: %s\n", ex.what());
        running_.store(false, std::memory_order_release);
        return false;
    }

    fprintf(stderr, "moe_prefetcher: started background worker\n");
    return true;
}

void moe_prefetcher::stop() {
    if (!running_.load(std::memory_order_acquire)) return;

    {
        std::lock_guard<std::mutex> lk(work_mutex_);
        stop_requested_.store(true, std::memory_order_release);
    }
    work_cv_.notify_one();

    if (worker_.joinable()) {
        worker_.join();
    }

    if (h2d_stream_) {
        cudaStreamSynchronize(h2d_stream_);
        cudaStreamDestroy(h2d_stream_);
        h2d_stream_ = nullptr;
    }

    running_.store(false, std::memory_order_release);
    fprintf(stderr, "moe_prefetcher: stopped\n");
}

void moe_prefetcher::launch_prefetch(
    const std::vector<prefetch_work_item> & items,
    void * completion_event)
{
    if (!running_.load(std::memory_order_acquire)) {
        for (auto & item : items) {
            const void * host_mask_ptr = item.host_mask;
            const void * host_remap_ptr = item.host_remap;
            if (host_mask_ptr && host_remap_ptr) {
                ggml_tensor * dst_arr[] = { item.dst_gate, item.dst_up, item.dst_down };
                ggml_tensor * src_arr[] = { item.src_gate, item.src_up, item.src_down };
                size_t sb[3] = { item.slice_bytes[0], item.slice_bytes[1], item.slice_bytes[2] };
                ggml_backend_cuda_pipeline_expert_skip_prefetch(
                    dst_arr, src_arr, sb,
                    item.expert_mask, item.moe_remap,
                    (const uint64_t *)host_mask_ptr,
                    (const int32_t *)host_remap_ptr,
                    nullptr, (ggml_backend_event_t)completion_event, (ggml_backend_t)item.gpu_backend);
            }
        }
        return;
    }

    {
        std::lock_guard<std::mutex> lk(work_mutex_);
        work_items_ = items;
        completion_event_ = completion_event;
        work_ready_ = true;
    }
    work_cv_.notify_one();
}

void moe_prefetcher::wait_prefetch_fence(void * /*completion_event*/) {
    while (fence_.load(std::memory_order_acquire) == 0) {
    }
    fence_.store(0, std::memory_order_release);
}

void moe_prefetcher::worker_loop() {
    pin_to_management_cores();

    while (true) {
        {
            std::unique_lock<std::mutex> lk(work_mutex_);
            work_cv_.wait(lk, [this] {
                return work_ready_ || stop_requested_.load(std::memory_order_acquire);
            });
            if (stop_requested_.load(std::memory_order_acquire)) break;
        }

        for (auto & item : work_items_) {
            if (!item.host_mask || !item.host_remap) continue;

            ggml_tensor * dst_arr[] = { item.dst_gate, item.dst_up, item.dst_down };
            ggml_tensor * src_arr[] = { item.src_gate, item.src_up, item.src_down };
            size_t sb[3] = { item.slice_bytes[0], item.slice_bytes[1], item.slice_bytes[2] };

            ggml_backend_t be = (ggml_backend_t)item.gpu_backend;
            int saved_stream = ggml_backend_cuda_get_stream(be);
            ggml_backend_cuda_set_stream(be, 2);

            ggml_backend_cuda_pipeline_expert_skip_prefetch(
                dst_arr, src_arr, sb,
                item.expert_mask, item.moe_remap,
                (const uint64_t *)item.host_mask,
                (const int32_t *)item.host_remap,
                nullptr,
                (ggml_backend_event_t)(item.prefetch_done ? item.prefetch_done : completion_event_),
                be);

            ggml_backend_cuda_set_stream(be, saved_stream);
        }

        if (completion_event_) {
            ggml_backend_event_synchronize((ggml_backend_event_t)completion_event_);
        }
        fence_.store(1, std::memory_order_release);

        {
            std::lock_guard<std::mutex> lk(work_mutex_);
            work_ready_ = false;
            work_items_.clear();
        }
    }
}

bool moe_prefetcher::pin_to_management_cores() {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int core = 64; core <= 75; core++) {
        CPU_SET(core, &cpuset);
    }
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        fprintf(stderr, "moe_prefetcher: pthread_setaffinity_np failed: %d\n", rc);
        return false;
    }
    fprintf(stderr, "moe_prefetcher: pinned to cores 64-75\n");
    return true;
#else
    fprintf(stderr, "moe_prefetcher: core affinity not supported on this platform\n");
    return false;
#endif
}

} // namespace moe

#endif // GGML_USE_CUDA