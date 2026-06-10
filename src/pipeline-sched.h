#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

struct llm_graph_context;

#define LLAMA_PIPELINE_MAX_LAYERS 128
#define LLAMA_PIPELINE_DEPTH 2  // sliding window: 2 layers in flight

// Per-layer scratch tensors (all in a persistent GPU buffer)
// Bridges data between phases across reset boundaries.
struct llama_layer_scratch {
    struct ggml_tensor * qcur;    // [n_embd_head_k, n_head, n_tokens]
    struct ggml_tensor * kcur;    // [kv_lora_rank+n_rope, 1, n_tokens]
    struct ggml_tensor * vcur;    // [kv_lora_rank, 1, n_tokens]
    struct ggml_tensor * top_k;       // [n_top_k, n_tokens] i32 (DSA indexer)
    struct ggml_tensor * ffn_inp;     // [n_embd, n_tokens]

    // ---- Hyper-sparse MoE scratch (populated by Phase A, consumed by Phase C) ----
    struct ggml_tensor * moe_ids;       // [n_expert_used, n_tokens] i32 — cached expert IDs
    struct ggml_tensor * moe_weights;   // [n_expert_used, n_tokens] f32 — renormalized weights
    struct ggml_tensor * expert_mask;   // [ceil(n_expert/64), 1] u64 — skip bitmask
    struct ggml_tensor * moe_remap;     // [n_expert] i32 — remap[e]=compact_slot or -1
    struct ggml_tensor * moe_kept_count; // [1] i32 — how many experts kept after thresholding
};

// Maximum number of MoE experts that can be prefetched per layer (compact buffer)
#define LLAMA_PIPELINE_PREFETCH_MAX_EXPERTS 128

// Callback for expert prefetch (Host→Device async copy via TMA/cudaMemcpyAsync).
// Called after Phase A's MoE routing completes (between MoE routing and FFN).
// The callback receives pointers to:
//   - dst_tensors[0..2]: GPU prefetch buffers (compact, slot per kept expert)
//   - src_tensors[0..2]: CPU model weight tensors for gate/up/down
//   - slice_bytes[0..2]: bytes per single expert slice in each weight type
//   - expert_mask: GPU tensor with skip bitmask [ceil(n_expert/64)] u64
//                  (bit=1 → skip this expert, no H2D copy needed)
//   - moe_remap:   GPU tensor [n_expert] i32 — remap[expert_id]=compact_slot
//   - layer: current layer index
//   - moe_ready: event recorded on compute stream when MoE routing + thresholding
//                completes (transfer stream must wait on this before reading mask)
// The callback must record completion_event on its transfer stream after all
// H2D copies are submitted (pipeline waits on this event before FFN phase).
// If NULL, prefetch is disabled (CPU direct path used).
//
// Implementation steps for CUDA callback:
//   1. Set backend stream to transfer stream (e.g., stream 2)
//   2. Wait for moe_ready event
//   3. Read expert_mask: D2H async (max 9 uint64 = 72 bytes → trivial)
//   4. Synchronize transfer stream (tiny read, fast)
//   5. For each NON-skipped expert e: cudaMemcpyAsync(
//        dst + moe_remap[e]*sz, src + e*sz, sz, H2D, transfer_stream)
//   6. Record completion_event on transfer stream
typedef void (*llama_pipeline_expert_prefetch_fn)(
    struct ggml_tensor  * dst_tensors[3],  // [gate, up, down] GPU compact prefetch buf
    struct ggml_tensor  * src_tensors[3],  // [gate, up, down] CPU model weight tensors
    size_t                slice_bytes[3],   // bytes per expert slice for each type
    struct ggml_tensor  * expert_mask,      // GPU [ceil(n_expert/64)] u64 skip bitmask
    struct ggml_tensor  * moe_remap,        // GPU [n_expert] i32 remap → compact slot
    int                   layer,            // current layer index
    ggml_backend_event_t  moe_ready,        // MoE routing done event
    ggml_backend_event_t  completion_event, // record on transfer stream after all copies
    void                * user_data);       // user context (e.g., CUDA stream handle)

struct llama_pipeline_sched {
    // Backend
    ggml_backend_t         backend;
    ggml_backend_dev_t     device;

    // Three separate schedulers with independent gallocr instances:
    //   sched[0] = QKV phase (stream 0)
    //   sched[1] = flash_attn phase (stream 1)
    //   sched[2] = FFN phase (stream 0)
    ggml_backend_sched_t   sched_pipe[3];
    // Original caller scheduler (for output head computation)
    ggml_backend_sched_t   sched_orig;

    // Stream control function pointers (set by CUDA-capable caller)
    // Allows switching between compute_stream (0) and attn_stream (1) etc.
    void (*set_stream)(ggml_backend_t, int);
    int  (*get_stream)(ggml_backend_t);

    // Per-layer phase graphs (built once during init)
    struct ggml_cgraph * gf_qkv  [LLAMA_PIPELINE_MAX_LAYERS];
    struct ggml_cgraph * gf_attn [LLAMA_PIPELINE_MAX_LAYERS];
    struct ggml_cgraph * gf_ffn  [LLAMA_PIPELINE_MAX_LAYERS];
    struct ggml_context * ctx_layers[LLAMA_PIPELINE_MAX_LAYERS]; // per-layer metadata
    int n_layer;

    // Events — used both as fallback and for prefetch (which always needs real cudaEvent_t)
    ggml_backend_event_t e_qkv_done [LLAMA_PIPELINE_MAX_LAYERS]; // QKV→attn sync
    ggml_backend_event_t e_attn_done[LLAMA_PIPELINE_MAX_LAYERS]; // attn→FFN sync
    ggml_backend_event_t e_moe_done [LLAMA_PIPELINE_MAX_LAYERS]; // MoE routing → prefetch
    ggml_backend_event_t e_prefetch_done[LLAMA_PIPELINE_MAX_LAYERS]; // prefetch done → FFN

    // CDA barriers (GPU-side flag sync, replaces events for main pipeline ordering)
    bool     use_cda;  // true when CDA barriers are active
    void   * cda_qkv [LLAMA_PIPELINE_MAX_LAYERS]; // QKV→attn sync
    void   * cda_attn[LLAMA_PIPELINE_MAX_LAYERS]; // attn→FFN sync

    // Persistent scratch buffer + tensors (survives sched reset)
    ggml_backend_buffer_t scratch_buf;
    struct ggml_context  * scratch_ctx;
    struct llama_layer_scratch scratch[LLAMA_PIPELINE_DEPTH];

    // Scratch copies of all input tensors needed across phases/resets
    struct ggml_tensor * inpL_embd;       // token embedding copy (scratch)
    struct ggml_tensor * scratch_inp_pos; // position tensor copy (scratch)
    struct ggml_tensor * inpL_next[LLAMA_PIPELINE_DEPTH]; // per-slot residual output

    // Scratch DSA input tensors (survives sched reset — copied before pipeline compute)
    struct ggml_context  * ctx_persist;   // persistent ggml_context for DSA/metadata
    struct ggml_tensor * scratch_k_rot_lid;         // Hadamard rotation [nrot,nrot]
    struct ggml_tensor * scratch_k_idxs_lid;        // indexer KV indices [n_tokens]
    struct ggml_tensor * scratch_kq_mask_lid;       // indexer attention mask [n_ctx, n_tokens, 1, 1]
    struct ggml_tensor * scratch_k_idxs_mla;        // MLA KV indices [n_tokens]
    struct ggml_tensor * scratch_kq_mask_mla;       // MLA attention mask [n_ctx, n_tokens, 1, 1]

    // Persistent DSA input object (references scratch tensors above)
    // Created during init; passed to phase builders so graphs reference scratch data.
    class llm_graph_input_attn_k_dsa * inp_attn_persist;

    // Pointers to the ORIGINAL main-graph tensors (from the most recent build_graph).
    // These are used by copy_inputs to locate the source data.
    struct ggml_tensor                  * saved_inp_pos;        // original inp_pos from build_graph
    class llm_graph_input_attn_k_dsa   * saved_inp_attn_dsa;  // original DSA input from build_graph

    // Output head graph (built once during init)
    struct ggml_cgraph * gf_output;       // output head: last_hidden → norm → lm_head
    struct ggml_context * ctx_output;     // ggml_context for output head tensors

    // Model output references (for building output head)
    struct ggml_tensor * output_norm_weight;
    float                norm_rms_eps;
    struct ggml_tensor * lm_head_weight;

    // ---- Expert Prefetch (Host→Device async copy) ----
    // Per-layer MoE indicator (true for layers with MoE experts)
    bool                 has_moe[LLAMA_PIPELINE_MAX_LAYERS];

    // Backing GPU buffer for prefetch destination tensors (compact)
    ggml_backend_buffer_t  prefetch_buf;

    // Compact prefetch destination tensors (sized for PREFETCH_MAX_EXPERTS × 3 weight types)
    struct ggml_tensor * prefetch_gate;  // [n_embd, n_ff_exp, PREFETCH_MAX_EXPERTS]
    struct ggml_tensor * prefetch_up;    // [n_embd, n_ff_exp, PREFETCH_MAX_EXPERTS]
    struct ggml_tensor * prefetch_down;  // [n_ff_exp, n_embd, PREFETCH_MAX_EXPERTS]

    // Original (CPU) model weight tensor references for each MoE layer
    struct ggml_tensor * model_gate_exps[LLAMA_PIPELINE_MAX_LAYERS];
    struct ggml_tensor * model_up_exps  [LLAMA_PIPELINE_MAX_LAYERS];
    struct ggml_tensor * model_down_exps[LLAMA_PIPELINE_MAX_LAYERS];

    // Expert remap buffer (GPU): maps original expert ID → compact prefetch slot
    // ex: remap_buf->data[e] = compact_slot for kept, -1 for skipped
    // expert_mask_buf->data = skip bitmask [ceil(n_expert/64)] u64
    // kept_count_buf->data = [1] i32, how many kept after thresholding
    ggml_backend_buffer_t  remap_buf;
    struct ggml_tensor   * remap_tensor;     // [n_expert] i32, GPU
    struct ggml_tensor   * expert_mask_perm; // [ceil(n_expert/64)] u64, GPU persistent

    // Prefetch function + user data (provided by CUDA-capable caller)
    llama_pipeline_expert_prefetch_fn prefetch_fn;
    void                            * prefetch_user_data;

    // ---- Hyper-sparse MoE params ----
    int64_t n_expert_stats;       // n_expert from model hparams
    float   moe_threshold;        // cumulative weight threshold (e.g. 0.95)
    int     moe_floor;            // minimum kept experts (e.g. 3)

    // ---- Hyper-sparse MoE stats ----
    struct {
        int64_t n_total_experts;       // Σ(n_expert) across all layers processed
        int64_t n_skipped_experts;     // Σ(skipped) across all layers
        double  cumulative_sparsity;   // running avg for /metrics
    } stats;
    int32_t stats_counter;             // log throttling counter
    int32_t stats_log_interval;        // log every N layers (0 = no logging)
};

// Initialize pipeline: build per-layer phase graphs, allocate scratch, create events.
// ctx must already have inpL set to the token embedding input.
// set_stream/get_stream are CUDA backend stream control (can be NULL on non-CUDA).
// prefetch_fn is an optional callback for async Host→Device expert weight prefetch.
// When non-NULL, the pipeline allocates GPU prefetch buffers and calls prefetch_fn
// after each QKV phase with the selected expert indices.
struct llama_pipeline_sched * llama_pipeline_sched_init(
    ggml_backend_t                      backend,
    ggml_backend_sched_t                sched,
    const struct llama_model          & model,
    struct llm_graph_context          & ctx,
    struct ggml_tensor                * inpL,
    struct ggml_tensor                * inp_pos,
    class llm_graph_input_attn_k_dsa * inp_attn_dsa,
    struct ggml_tensor                * inp_out_ids,
    float                               kq_scale,
    void                              (*set_stream)(ggml_backend_t, int),
    int                               (*get_stream)(ggml_backend_t),
    llama_pipeline_expert_prefetch_fn   prefetch_fn,
    void                              * prefetch_user_data);

// Copy caller's input data to persistent scratch (must be called before compute,
// after set_inputs on the main graph). The scratch copies survive sched resets
// and are safely readable by all pipeline phases.
void llama_pipeline_sched_copy_inputs(
    struct llama_pipeline_sched  * p,
    ggml_backend_t                 backend,
    struct ggml_tensor           * src_inpL,
    struct ggml_tensor           * src_inp_pos,
    class llm_graph_input_attn_k_dsa * src_dsa);

// Execute the three-stage sliding window across compute/attn streams.
// After return, the last layer's hidden state is in p->inpL_next[(n_layer-1) % LLAMA_PIPELINE_DEPTH].
void llama_pipeline_sched_compute(struct llama_pipeline_sched * p, int n_layer);

// Build and compute the output head using the pipeline's last-layer output.
// This allocates and computes a small graph: last_hidden → norm → lm_head.
// Output tensors (logits, embd) are placed into the caller's res object.
// Must be called after llama_pipeline_sched_compute.
void llama_pipeline_sched_compute_output_head(
    struct llama_pipeline_sched  * p,
    class llm_graph_result      * res,
    struct ggml_tensor           * output_norm_weight,
    float                          norm_rms_eps,
    struct ggml_tensor           * lm_head_weight);

// Set or update the expert prefetch callback after initialization.
// This allows the CUDA-capable caller to provide the async H2D copy implementation
// after the pipeline has been created (e.g. from process_ubatch where CUDA context is available).
void llama_pipeline_sched_set_prefetch_fn(
    struct llama_pipeline_sched  * p,
    llama_pipeline_expert_prefetch_fn fn,
    void * user_data);

// Convenience: register the CUDA-based prefetch callback on pipeline p.
// The callback uses ggml_backend_cuda_pipeline_expert_prefetch internally.
// backend must be a valid CUDA ggml_backend_t.
void llama_pipeline_cuda_init_prefetch(
    struct llama_pipeline_sched * p,
    ggml_backend_t               backend);

void llama_pipeline_sched_free(struct llama_pipeline_sched * p);
