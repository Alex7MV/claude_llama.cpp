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
    struct ggml_tensor * top_k;   // [n_top_k, n_tokens] i32
    struct ggml_tensor * ffn_inp; // [n_embd, n_tokens]
};

// Maximum number of MoE experts that can be prefetched per layer
#define LLAMA_PIPELINE_PREFETCH_MAX_EXPERTS 16

// Callback for expert prefetch (Host→Device async copy via TMA/cudaMemcpyAsync).
// Called after QKV phase (between QKV and attn on the dispatch timeline).
// The callback receives pointers to:
//   - dst_tensors[0..2]: GPU prefetch buffers for gate/up/down expert weights
//   - src_tensors[0..2]: CPU model weight tensors for gate/up/down
//   - slice_bytes[0..2]: bytes per single expert slice in each weight type
//   - top_k: GPU tensor with DSA top-k indices (can be read via cudaMemcpyDeviceToHost)
//   - layer: current layer index
//   - qkv_done: event recorded on compute stream when QKV completes
//              (transfer stream must wait on this before reading top_k)
// The callback must record completion_event on its transfer stream after all
// H2D copies are submitted (pipeline waits on this event before FFN phase).
// If NULL, prefetch is disabled (scheduler's built-in copy used in phase C).
//
// Implementation steps for CUDA callback:
//   1. Set backend stream to transfer stream (e.g., stream 2)
//   2. Wait for qkv_done event: cudaStreamWaitEvent(transfer_stream, qkv_done_event, 0)
//   3. Read top_k: cudaMemcpyDeviceToAsync(host_buf, top_k->data, size, transfer_stream)
//   4. Synchronize transfer stream (small top_k, fast)
//   5. For each unique expert ID: cudaMemcpyAsync(dst + i*sz, src + id*sz, sz,
//      cudaMemcpyHostToDevice, transfer_stream)
//   6. Record completion_event on transfer stream
typedef void (*llama_pipeline_expert_prefetch_fn)(
    struct ggml_tensor  * dst_tensors[3],  // [gate, up, down] GPU prefetch buffers
    struct ggml_tensor  * src_tensors[3],  // [gate, up, down] CPU model weight tensors
    size_t                slice_bytes[3],   // bytes per expert slice for each type
    struct ggml_tensor  * top_k,            // GPU tensor with DSA top-k indices
    int                   layer,            // current layer index
    ggml_backend_event_t  qkv_done,         // QKV completion event (wait before reading top_k)
    ggml_backend_event_t  completion_event, // record on transfer stream after all copies
    void                * user_data);       // user context (e.g., CUDA stream handle)

struct llama_pipeline_sched {
    // Backend
    ggml_backend_t         backend;
    ggml_backend_sched_t   sched;
    ggml_backend_dev_t     device;

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

    // Events
    ggml_backend_event_t e_qkv_done [LLAMA_PIPELINE_MAX_LAYERS];
    ggml_backend_event_t e_attn_done[LLAMA_PIPELINE_MAX_LAYERS];

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

    // Backing GPU buffer for prefetch destination tensors
    ggml_backend_buffer_t  prefetch_buf;

    // Prefetch destination tensors (GPU-resident, sized for n_expert_used × 3 weight types)
    struct ggml_tensor * prefetch_gate;  // [n_embd, n_ff_exp, n_expert_used]
    struct ggml_tensor * prefetch_up;    // [n_embd, n_ff_exp, n_expert_used]
    struct ggml_tensor * prefetch_down;  // [n_ff_exp, n_embd, n_expert_used]

    // Original (CPU) model weight tensor references for each MoE layer
    struct ggml_tensor * model_gate_exps[LLAMA_PIPELINE_MAX_LAYERS];
    struct ggml_tensor * model_up_exps  [LLAMA_PIPELINE_MAX_LAYERS];
    struct ggml_tensor * model_down_exps[LLAMA_PIPELINE_MAX_LAYERS];

    // Prefetch completion event (recorded on transfer stream after copies submitted)
    ggml_backend_event_t e_prefetch_done;

    // Prefetch function + user data (provided by CUDA-capable caller)
    llama_pipeline_expert_prefetch_fn prefetch_fn;
    void                            * prefetch_user_data;
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
