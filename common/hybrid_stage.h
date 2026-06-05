#pragma once
#include <cstdint>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <memory>
#include "llama.h"
#include <mutex>
#include "common.h"
#include "hybrid_vram_pool.h"

// Shared mutex for draft context (ctx_dft) access serialization.
// Declared in hybrid_stage.cpp.
extern std::mutex g_dft_mutex;

struct common_sampler;

// Thread-local pointer set by process_ubatch before graph compute.
// Read by the CUDA flash_attn_ext handler to detect async mode.
// Defined in hybrid_stage.cpp.
#ifndef HYBRID_STAGE_H_
#define HYBRID_STAGE_H_
struct hybrid_orchestrator;
extern thread_local hybrid_orchestrator * g_hybrid_ctx;
#endif

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

    // --- Async flash_attn (separate stream, pinned output, event fence) ---
    void * attn_stream = nullptr;   // separate cudaStream_t for async flash_attn
    void * attn_event  = nullptr;   // cudaEvent_t recorded after attn completes

    // Per-layer pinned CPU buffers for flash_attn output (F32, kq_head_dim elements)
    std::vector<void *> attn_pinned;

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

    // --- Async pre-generation (GPU/CPU overlap for speculative decoding) ---
    //
    // Pre-generates the next draft batch on GPU DURING the CPU MoE FFN of the
    // current verification step.  Uses greedy (temperature=0) sampling so the
    // pre-generated tokens are deterministically identical to what a fresh
    // generation from the same state would produce.  After verification accepts
    // k tokens, the suffix [k..N] of the pre-generated batch is a valid next
    // draft (same tokens, correct prefix).
    struct {
        std::vector<uint8_t> state_buf;     // captured target state (the pre-verification position)
        llama_seq_id         seq_id   = 0;
        std::thread          thread;         // background worker
        std::atomic<bool>    done{false};    // set by worker when tokens ready
        std::vector<llama_token> tokens;     // the generated draft
        uint32_t             n_draft  = 0;
    } pregen;

    // Capture the target state buffer (call BEFORE verification, e.g. inside
    // the draft-generation lambda).  The buffer is used later by start_pregen().
    void capture_pregen_state(std::vector<uint8_t> state_buf, llama_seq_id seq_id);

    // Launch the background pre-generation thread using the captured state.
    // The thread copies target state -> draft model, then decodes n_draft
    // tokens with deterministic (greedy) sampling.
    // Call this just before llama_decode(ctx_tgt, ...) so the GPU draft-gen
    // overlaps with the CPU MoE FFN of the target verification.
    bool start_pregen(
        llama_context * ctx_dft,
        struct common_sampler * smpl,
        llama_seq_id seq_id,
        uint32_t n_draft);

    // Wait for the background thread and return the suffix starting at index
    // n_accepted (i.e. the tokens that continue from the last verified
    // position).  If n_accepted >= pre-gen length, returns empty (all rejected).
    // Also generates extra tokens on ctx_dft if the suffix is shorter than
    // n_draft_target, extending the draft to the desired length.
    // trim_pos: if >= 0, truncates ctx_dft to this position (safe because
    // the background thread has been joined).
    std::vector<llama_token> finish_pregen(
        llama_context * ctx_dft,
        struct common_sampler * smpl,
        uint32_t n_accepted,
        uint32_t n_draft_target,
        llama_pos trim_pos = -1);

    ~hybrid_orchestrator() { free_all(); }

    bool init(
        uint32_t n_layers,
        uint32_t n_ctx_max,
        uint32_t kv_lora_rank,
        uint32_t max_lookahead,
        ggml_backend_dev_t device,
        void * compute_stream,
        void * draft_stream,
        uint32_t max_draft = 6,
        ggml_type  cache_type_k = GGML_TYPE_F16,
        uint32_t   key_dim      = 0);

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

    // CPU fence: blocks until the orchestrator's attn_event signals.
    // Call before any CPU operation that reads the pinned attn output buffer.
    void hybrid_gpu_fence();

    uint32_t verify_and_rollback(
        llama_context * ctx_tgt,
        const llama_tokens & target_tokens);

    void start_draft_batch(std::function<void()> draft_fn);
};
