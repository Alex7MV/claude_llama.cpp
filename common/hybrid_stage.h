#pragma once
#include <cstdint>
#include <atomic>
#include <vector>
#include <functional>
#include "llama.h"
#include "common.h"
#include "hybrid_vram_pool.h"

enum class hybrid_phase : uint8_t {
    IDLE,
    NORM_DONE,
    KV_COMPRESSED,
    TMA_ENQUEUED,
    GPU_ATTN_DONE,
    MERGE_DONE,
};

struct hybrid_layer_stage {
    std::atomic<hybrid_phase> phase = hybrid_phase::IDLE;

    struct {
        const float * cpu_src   = nullptr;
        uint64_t      bytes     = 0;
        bool          enqueued  = false;
    } tma;

    struct {
        int32_t  expert_ids[2] = {-1, -1};
        bool     prefetched    = false;
    } prefetch;
};

struct hybrid_orchestrator {
    hybrid_vram_pool pool;

    std::vector<hybrid_layer_stage> stages;
    uint32_t n_layers       = 0;
    uint32_t kv_lora_rank   = 0;

    uint32_t current_layer  = 0;
    uint32_t current_token  = 0;

    void * gpu_compute_stream = nullptr;
    void * draft_stream       = nullptr;

    void * tma_events[2] = {};
    int    tma_head      = 0;

    struct {
        std::vector<llama_token> lookahead_buffer;
        bool in_flight = false;
    } draft;

    struct {
        uint32_t accepted    = 0;
        uint32_t rejected_at = UINT32_MAX;
        uint32_t n_draft     = 0;
    } verify;

    bool init(
        uint32_t n_layers,
        uint32_t n_ctx_max,
        uint32_t kv_lora_rank,
        uint32_t max_lookahead,
        ggml_backend_t backend,
        void * compute_stream,
        void * draft_stream);

    void free_all();

    void on_norm_done(uint32_t layer, const int32_t expert_ids[2]);
    void on_kv_compressed(uint32_t layer, const float * c_tkv_cpu);
    void on_tma_enqueued(uint32_t layer);
    void on_gpu_attn_done(uint32_t layer);
    void on_merge_done(uint32_t layer);

    void advance_token();

    uint32_t verify_and_rollback(
        llama_context * ctx_tgt,
        const llama_tokens & target_tokens);

    void start_draft_batch(std::function<void()> draft_fn);
};
