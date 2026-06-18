#pragma once

// Async background worker for expert H2D prefetch on EPYC 9V74.
// Pins to management cores 64-75, uses dedicated CUDA stream.
// Overlaps with main-thread Phase 2 graph build / sched alloc.

#ifdef GGML_USE_CUDA

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

struct ggml_tensor;

namespace moe {

struct prefetch_work_item {
    ggml_tensor * dst_gate     = nullptr;
    ggml_tensor * dst_up       = nullptr;
    ggml_tensor * dst_down     = nullptr;
    ggml_tensor * src_gate     = nullptr;
    ggml_tensor * src_up       = nullptr;
    ggml_tensor * src_down     = nullptr;
    ggml_tensor * expert_mask  = nullptr;
    ggml_tensor * moe_remap    = nullptr;
    void *        host_mask    = nullptr;
    void *        host_remap   = nullptr;
    size_t        slice_bytes[3] = {0, 0, 0};
    int           max_kept     = 0;
    void *        prefetch_done = nullptr;
    void *        stream       = nullptr;
    void *        gpu_backend  = nullptr;
};

class moe_prefetcher {
public:
    moe_prefetcher();
    ~moe_prefetcher();

    moe_prefetcher(const moe_prefetcher &) = delete;
    moe_prefetcher & operator=(const moe_prefetcher &) = delete;

    bool start();
    void stop();

    void launch_prefetch(
        const std::vector<prefetch_work_item> & items,
        void * completion_event);

    void wait_prefetch_fence(void * completion_event);

    void * get_stream() const { return h2d_stream_; }

private:
    void worker_loop();
    static bool pin_to_management_cores();

    std::thread                worker_;
    std::atomic<bool>          running_{false};
    std::atomic<bool>          stop_requested_{false};

    std::mutex                 work_mutex_;
    std::condition_variable    work_cv_;
    bool                       work_ready_{false};
    std::vector<prefetch_work_item> work_items_;
    void *                     completion_event_{nullptr};

    void * h2d_stream_ = nullptr;

    std::atomic<int> fence_{0};
};

} // namespace moe

#endif // GGML_USE_CUDA