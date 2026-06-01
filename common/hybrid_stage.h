#pragma once
#include <cstdint>
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
    hybrid_phase phase = hybrid_phase::IDLE;

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
        uint32_t accepted         = 0;
        uint32_t rejected_at      = UINT32_MAX;
        uint32_t n_draft          = 0;
        uint32_t total_attempts   = 0;
        uint32_t total_accepted   = 0;
    } verify;

    uint32_t max_draft        = 6;
    uint32_t current_draft    = 6;
    float    accept_rate_ema  = 0.0f;

    bool init(
        uint32_t n_layers,
        uint32_t n_ctx_max,
        uint32_t kv_lora_rank,
        uint32_t max_lookahead,
        ggml_backend_dev_t device,
        void * compute_stream,
        void * draft_stream,
        uint32_t max_draft = 6,
        uint32_t key_dim = 0);

    void free_all();

    void on_norm_done(uint32_t layer, const int32_t expert_ids[2]);
    void on_kv_compressed(uint32_t layer, const float * c_tkv_cpu);
    void on_tma_enqueued(uint32_t layer);
    bool wait_tma_event();
    void on_gpu_attn_done(uint32_t layer);
    void on_merge_done(uint32_t layer);
    void trace_tma_verify(uint32_t layer);

    void advance_token();

    // Walk the full cgraph and reassign GGML_OP_FLASH_ATTN_EXT to GPU backend.
    // The scheduler will auto-copy Q/K/V/mask from CPU to GPU before compute.
    static void patch_graph_for_mla(
        struct ggml_cgraph * gf,
        ggml_backend_sched_t sched,
        ggml_backend_t gpu_backend);

    uint32_t verify_and_rollback(
        llama_context * ctx_tgt,
        const llama_tokens & target_tokens);

    void start_draft_batch(std::function<void()> draft_fn);
};
