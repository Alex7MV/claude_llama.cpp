#include "pipeline-sched.h"
#include "models/models.h"
#include "llama-graph.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"

#include <cmath>
#include <cstring>
#include <cassert>

// ============================================================================
// Phase A:  norm -> QKV proj -> DSA indexer -> Qcur/Kcur/Vcur/top_k
//
// Reads:  inpL  (persistent scratch -> inpL_next[(il-1) % 2] or original inpL_embd)
// Writes: scratch->{qcur,kcur,vcur,top_k}  (persistent buffer)
// ============================================================================
static void build_phase_a(
        const struct llama_model            & model,
        struct llm_graph_context            & ctx,
        struct ggml_context                 * ctx0_layer,
        struct ggml_cgraph                  * gf_layer,
        struct ggml_tensor                  * inpL,
        struct ggml_tensor                  * inp_pos,
        struct llm_graph_input_attn_k_dsa   * inp_attn_dsa,
        float                                 kq_scale,
        uint32_t                              il,
        struct llama_layer_scratch          * scratch) {

    struct ggml_context * ctx0_saved = ctx.ctx0;
    struct ggml_cgraph  * gf_saved   = ctx.gf;
    ctx.ctx0 = ctx0_layer;
    ctx.gf   = gf_layer;

    const int64_t n_embd_head_k  = ctx.hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v  = ctx.hparams.n_embd_head_v_mla();
    GGML_UNUSED(n_embd_head_v);

    const int64_t n_embd_head_qk_rope = ctx.hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t n_indexer_head      = ctx.hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = ctx.hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = ctx.hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const uint32_t n_indexer_top_k    = ctx.hparams.indexer_top_k;

    const uint32_t kv_lora_rank = ctx.hparams.n_lora_kv;
    const int64_t  n_tokens     = ctx.n_tokens;
    const int64_t  n_head       = ctx.n_head;

    // --- norm ---
    ggml_tensor * cur;
    cur = ctx.build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
    ctx.cb(cur, "attn_norm", il);

    // --- QKV proj + DSA ---
    {
        ggml_tensor * qr = ggml_mul_mat(ctx0_layer, model.layers[il].wq_a, cur);
        ctx.cb(qr, "qr", il);

        qr = ctx.build_norm(qr, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        ctx.cb(qr, "qr", il);

        // ---- DSA indexer ----
        ggml_tensor * top_k = nullptr;
        {
            ggml_tensor * indexer_q = ggml_mul_mat(ctx0_layer, model.layers[il].indexer_attn_q_b, qr);
            ctx.cb(indexer_q, "indexer_q", il);

            ggml_tensor * idx_q_pe =
                ggml_view_3d(ctx0_layer, indexer_q, n_embd_indexer_head_rope, n_indexer_head, n_tokens,
                             ggml_row_size(indexer_q->type, n_embd_indexer_head),
                             ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head, 0);
            ctx.cb(idx_q_pe, "indexer_q_pe", il);

            ggml_tensor * idx_q_nope =
                ggml_view_3d(ctx0_layer, indexer_q, n_embd_indexer_head_nope, n_indexer_head, n_tokens,
                             ggml_row_size(indexer_q->type, n_embd_indexer_head),
                             ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
                             ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));
            ctx.cb(idx_q_nope, "indexer_q_nope", il);

            idx_q_pe = ggml_rope_ext(ctx0_layer, idx_q_pe, inp_pos, nullptr, ctx.n_rot,
                             LLAMA_ROPE_TYPE_NEOX, ctx.n_ctx_orig, ctx.freq_base, ctx.freq_scale,
                             ctx.ext_factor, ctx.attn_factor, ctx.beta_fast, ctx.beta_slow);
            ctx.cb(idx_q_pe, "indexer_q_pe", il);

            indexer_q = ggml_concat(ctx0_layer, idx_q_pe, idx_q_nope, 0);
            ctx.cb(indexer_q, "indexer_q", il);

            ggml_tensor * indexer_k = ggml_mul_mat(ctx0_layer, model.layers[il].indexer_attn_k, cur);
            ctx.cb(indexer_k, "indexer_k", il);

            indexer_k = ctx.build_norm(indexer_k, model.layers[il].indexer_k_norm,
                                       model.layers[il].indexer_k_norm_b, LLM_NORM, il);
            ctx.cb(indexer_k, "indexer_k", il);

            ggml_tensor * idx_k_pe =
                ggml_view_3d(ctx0_layer, indexer_k, n_embd_indexer_head_rope, 1, n_tokens,
                             ggml_row_size(indexer_k->type, n_embd_indexer_head),
                             ggml_row_size(indexer_k->type, n_embd_indexer_head) * 1, 0);
            ctx.cb(idx_k_pe, "indexer_k_pe", il);

            ggml_tensor * idx_k_nope =
                ggml_view_3d(ctx0_layer, indexer_k, n_embd_indexer_head_nope, 1, n_tokens,
                             ggml_row_size(indexer_k->type, n_embd_indexer_head),
                             ggml_row_size(indexer_k->type, n_embd_indexer_head) * 1,
                             ggml_row_size(indexer_k->type, n_embd_indexer_head_nope));
            ctx.cb(idx_k_nope, "indexer_k_nope", il);

            idx_k_pe = ggml_rope_ext(ctx0_layer, idx_k_pe, inp_pos, nullptr, ctx.n_rot,
                             LLAMA_ROPE_TYPE_NEOX, ctx.n_ctx_orig, ctx.freq_base, ctx.freq_scale,
                             ctx.ext_factor, ctx.attn_factor, ctx.beta_fast, ctx.beta_slow);
            ctx.cb(idx_k_pe, "indexer_k_pe", il);

            indexer_k = ggml_concat(ctx0_layer, idx_k_pe, idx_k_nope, 0);
            ctx.cb(indexer_k, "indexer_k", il);

            // Access scratch k_rot_lid (persistent): use from the persistent DSA object
            indexer_q = ggml_mul_mat(ctx0_layer, inp_attn_dsa->self_k_rot_lid, indexer_q);
            ctx.cb(indexer_q, "indexer_q", il);
            indexer_k = ggml_mul_mat(ctx0_layer, inp_attn_dsa->self_k_rot_lid, indexer_k);
            ctx.cb(indexer_k, "indexer_k", il);

            const auto * mctx_lid = inp_attn_dsa->mctx->get_lid();
            const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
            ggml_build_forward_expand(gf_layer, mctx_lid->cpy_k(ctx0_layer, indexer_k, k_idxs_lid, il));

            ggml_tensor * indexer_weights = ggml_mul_mat(ctx0_layer, model.layers[il].indexer_proj, cur);
            ctx.cb(indexer_weights, "indexer_weights", il);

            indexer_k = mctx_lid->get_k(ctx0_layer, il);

            const auto n_stream = indexer_k->ne[3];
            indexer_q = ggml_view_4d(ctx0_layer, indexer_q, indexer_q->ne[0], indexer_q->ne[1],
                                     indexer_q->ne[2]/n_stream, n_stream,
                                     indexer_q->nb[1], indexer_q->nb[2],
                                     indexer_q->nb[3]/n_stream, 0);
            indexer_weights = ggml_view_4d(ctx0_layer, indexer_weights, indexer_weights->ne[0],
                                     indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
                                     indexer_weights->nb[1], indexer_weights->nb[2]/n_stream,
                                     indexer_weights->nb[3]/n_stream, 0);

            indexer_q = ggml_permute(ctx0_layer, indexer_q, 0, 2, 1, 3);
            ctx.cb(indexer_q, "indexer_q", il);
            indexer_k = ggml_permute(ctx0_layer, indexer_k, 0, 2, 1, 3);
            ctx.cb(indexer_k, "indexer_k", il);

            ggml_tensor * indexer_kq = ggml_mul_mat(ctx0_layer, indexer_k, indexer_q);
            ctx.cb(indexer_kq, "indexer_kq", il);

            indexer_kq = ggml_cont(ctx0_layer, ggml_permute(ctx0_layer, indexer_kq, 2, 1, 0, 3));
            ctx.cb(indexer_kq, "indexer_kq", il);

            ggml_tensor * indexer_score = ggml_relu(ctx0_layer, indexer_kq);
            ctx.cb(indexer_score, "indexer_score", il);

            indexer_weights = ggml_scale(ctx0_layer, indexer_weights,
                                         1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head)));
            ctx.cb(indexer_weights, "indexer_weights", il);

            indexer_score = ggml_mul(ctx0_layer, indexer_score, indexer_weights);
            ctx.cb(indexer_score, "indexer_score", il);

            indexer_score = ggml_sum_rows(ctx0_layer, indexer_score);
            ctx.cb(indexer_score, "indexer_score", il);

            indexer_score = ggml_cont(ctx0_layer, ggml_permute(ctx0_layer, indexer_score, 2, 1, 0, 3));
            ctx.cb(indexer_score, "indexer_score", il);

            // Use persistent scratch mask (survives sched reset)
            ggml_tensor * indexer_kq_mask = inp_attn_dsa->get_kq_mask_lid();
            indexer_score = ggml_add(ctx0_layer, indexer_score, indexer_kq_mask);
            ctx.cb(indexer_score, "indexer_score", il);

            uint32_t n_top_k = indexer_score->ne[0] < n_indexer_top_k
                             ? indexer_score->ne[0] : n_indexer_top_k;
            top_k = ggml_cont(ctx0_layer, ggml_top_k(ctx0_layer, indexer_score, n_top_k));
            ctx.cb(top_k, "top_k", il);
        }

        // Q projection
        ggml_tensor * q = ggml_mul_mat(ctx0_layer, model.layers[il].wq_b, qr);
        ctx.cb(q, "q", il);

        ggml_tensor * q_nope =
            ggml_view_3d(ctx0_layer, q, n_embd_head_qk_nope, n_head, n_tokens,
                         ggml_row_size(q->type, n_embd_head_k),
                         ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
        ctx.cb(q_nope, "q_nope", il);

        ggml_tensor * q_pe = ggml_view_3d(
            ctx0_layer, q, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));
        ctx.cb(q_pe, "q_pe", il);

        ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0_layer, model.layers[il].wkv_a_mqa, cur);
        ctx.cb(kv_cmpr_pe, "kv_cmpr_pe", il);

        ggml_tensor * kv_cmpr =
            ggml_view_2d(ctx0_layer, kv_cmpr_pe, kv_lora_rank, n_tokens,
                         ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
        ctx.cb(kv_cmpr, "kv_cmpr", il);

        ggml_tensor * k_pe = ggml_view_3d(ctx0_layer, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
        ctx.cb(k_pe, "k_pe", il);

        q_pe = ggml_rope_ext(ctx0_layer, q_pe, inp_pos, nullptr, ctx.n_rot, ctx.rope_type,
                             ctx.n_ctx_orig, ctx.freq_base, ctx.freq_scale,
                             ctx.ext_factor, ctx.attn_factor, ctx.beta_fast, ctx.beta_slow);
        ctx.cb(q_pe, "q_pe", il);

        k_pe = ggml_rope_ext(ctx0_layer, k_pe, inp_pos, nullptr, ctx.n_rot, ctx.rope_type,
                             ctx.n_ctx_orig, ctx.freq_base, ctx.freq_scale,
                             ctx.ext_factor, ctx.attn_factor, ctx.beta_fast, ctx.beta_slow);
        ctx.cb(k_pe, "k_pe", il);

        kv_cmpr = ctx.build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        ctx.cb(kv_cmpr, "kv_cmpr", il);

        // ---- Build Qcur, Kcur, Vcur for flash_attn ----
        q_nope = ggml_permute(ctx0_layer, q_nope, 0, 2, 1, 3);
        ctx.cb(q_nope, "q_nope_perm", il);

        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0_layer, model.layers[il].wk_b, q_nope);
        ctx.cb(q_nope_absorbed, "q_nope_absorbed", il);

        q_nope_absorbed = ggml_permute(ctx0_layer, q_nope_absorbed, 0, 2, 1, 3);
        ctx.cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

        ggml_tensor * Qcur = ggml_concat(ctx0_layer, q_nope_absorbed, q_pe, 0);
        ctx.cb(Qcur, "Qcur", il);

        kv_cmpr = ggml_reshape_3d(ctx0_layer, kv_cmpr, kv_lora_rank, 1, n_tokens);
        ctx.cb(kv_cmpr, "kv_cmpr_reshape", il);

        ggml_tensor * Kcur = ggml_concat(ctx0_layer, kv_cmpr, k_pe, 0);
        ctx.cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = kv_cmpr;
        ctx.cb(Vcur, "Vcur", il);

        // Copy Qcur/Kcur/Vcur/top_k to persistent scratch (survives sched reset)
        ggml_build_forward_expand(gf_layer, ggml_cpy(ctx0_layer, Qcur, scratch->qcur));
        ggml_build_forward_expand(gf_layer, ggml_cpy(ctx0_layer, Kcur, scratch->kcur));
        ggml_build_forward_expand(gf_layer, ggml_cpy(ctx0_layer, Vcur, scratch->vcur));
        if (scratch->top_k && top_k) {
            ggml_build_forward_expand(gf_layer, ggml_cpy(ctx0_layer, top_k, scratch->top_k));
        }
    }

    ctx.ctx0 = ctx0_saved;
    ctx.gf   = gf_saved;
}

// ============================================================================
// Phase B:  flash_attn + O_proj + residual_add -> ffn_inp
//
// Reads:  scratch->{qcur,kcur,vcur,top_k}  (persistent buffer)
//         inpL (inpSA for residual -- the same as Phase A's input)
// Writes: scratch->ffn_inp  (persistent buffer)
// ============================================================================
static void build_phase_b(
        const struct llama_model            & model,
        struct llm_graph_context            & ctx,
        struct ggml_context                 * ctx0_layer,
        struct ggml_cgraph                  * gf_layer,
        struct ggml_tensor                  * inpL,
        struct llama_layer_scratch          * scratch,
        struct llm_graph_input_attn_k_dsa   * inp_attn_dsa,
        struct ggml_tensor                  * inp_out_ids,
        float                                 kq_scale,
        uint32_t                              il) {

    struct ggml_context * ctx0_saved = ctx.ctx0;
    struct ggml_cgraph  * gf_saved   = ctx.gf;
    ctx.ctx0 = ctx0_layer;
    ctx.gf   = gf_layer;

    const uint32_t effective_n_layers = ctx.hparams.n_layer - ctx.hparams.nextn_predict_layers;

    ggml_tensor * cur;
    ggml_tensor * inpSA = inpL;

    // ---- flash_attn using scratch Qcur/Kcur/Vcur/top_k ----
    // inp_attn_dsa references persistent scratch copies of masks/indices
    cur = ctx.build_attn(inp_attn_dsa,
            model.layers[il].wo, NULL, model.layers[il].wo_s,
            scratch->qcur, scratch->kcur, scratch->vcur,
            nullptr, nullptr, model.layers[il].wv_b,
            scratch->top_k, kq_scale, il);

    // ---- residual ----
    if (il == effective_n_layers - 1 && inp_out_ids) {
        cur   = ggml_get_rows(ctx0_layer, cur, inp_out_ids);
        inpSA = ggml_get_rows(ctx0_layer, inpSA, inp_out_ids);
    }
    ggml_tensor * ffn_inp = ggml_add(ctx0_layer, cur, inpSA);
    ctx.cb(ffn_inp, "ffn_inp", il);

    // Copy to persistent scratch for Phase C
    ggml_build_forward_expand(gf_layer, ggml_cpy(ctx0_layer, ffn_inp, scratch->ffn_inp));

    ctx.ctx0 = ctx0_saved;
    ctx.gf   = gf_saved;
}

// ============================================================================
// Phase C:  FFN (dense or MoE) + residual -> new inpL
//
// Reads:  scratch->ffn_inp  (persistent buffer)
// Writes: inpL_next  (persistent buffer -- bridges to next layer)
//
// When prefetch_gate/up/down are not NULL, they are used INSTEAD of the
// model's original MoE weight tensors. These are GPU-resident prefetch
// buffers that receive async H2D copies during phase B.
// ============================================================================
static void build_phase_c(
        const struct llama_model            & model,
        struct llm_graph_context            & ctx,
        struct ggml_context                 * ctx0_layer,
        struct ggml_cgraph                  * gf_layer,
        struct llama_layer_scratch          * scratch,
        struct ggml_tensor                  * inpL_next_out, // persistent dest
        struct ggml_tensor                  * inp_out_ids,
        uint32_t                              il,
        struct ggml_tensor                  * prefetch_gate  = nullptr,
        struct ggml_tensor                  * prefetch_up    = nullptr,
        struct ggml_tensor                  * prefetch_down  = nullptr) {

    struct ggml_context * ctx0_saved = ctx.ctx0;
    struct ggml_cgraph  * gf_saved   = ctx.gf;
    ctx.ctx0 = ctx0_layer;
    ctx.gf   = gf_layer;

    const uint32_t n_layer_dense_lead = ctx.hparams.n_layer_dense_lead;

    ggml_tensor * cur;

    // ---- FFN ----
    cur = ctx.build_norm(scratch->ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
    ctx.cb(cur, "ffn_norm", il);

    if ((uint32_t) il < n_layer_dense_lead) {
        cur = ctx.build_ffn(cur,
            model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
            model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
            model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        ctx.cb(cur, "ffn_out", il);
    } else {
        // Use prefetch buffer tensors (GPU-resident) when available, else model weights (may be CPU)
        struct ggml_tensor * ffn_up_exps   = prefetch_up   ? prefetch_up   : model.layers[il].ffn_up_exps;
        struct ggml_tensor * ffn_gate_exps = prefetch_gate ? prefetch_gate : model.layers[il].ffn_gate_exps;
        struct ggml_tensor * ffn_down_exps = prefetch_down ? prefetch_down : model.layers[il].ffn_down_exps;

        ggml_tensor * moe_out = ctx.build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            ffn_up_exps,
            ffn_gate_exps,
            ffn_down_exps,
            model.layers[il].ffn_exp_probs_b,
            ctx.n_expert, ctx.n_expert_used,
            LLM_FFN_SILU, ctx.hparams.expert_weights_norm,
            ctx.hparams.expert_weights_scale,
            (llama_expert_gating_func_type) ctx.hparams.expert_gating_func,
            il,
            nullptr,
            model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
        ctx.cb(moe_out, "ffn_moe_out", il);

        {
            ggml_tensor * ffn_shexp =
                ctx.build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                    model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                    model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            ctx.cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0_layer, moe_out, ffn_shexp);
            ctx.cb(cur, "ffn_out", il);
        }
    }

    cur = ggml_add(ctx0_layer, cur, scratch->ffn_inp);
    cur = ctx.build_cvec(cur, il);
    ctx.cb(cur, "l_out", il);

    // Copy to persistent inpL_next for the next layer
    ggml_build_forward_expand(gf_layer, ggml_cpy(ctx0_layer, cur, inpL_next_out));

    ctx.ctx0 = ctx0_saved;
    ctx.gf   = gf_saved;
}

// ============================================================================
// Pipeline Scheduler -- init
// ============================================================================

struct llama_pipeline_sched * llama_pipeline_sched_init(
    ggml_backend_t                      backend,
    ggml_backend_sched_t                sched,
    const struct llama_model          & model,
    struct llm_graph_context          & ctx,
    struct ggml_tensor                * inpL,
    struct ggml_tensor                * inp_pos,
    struct llm_graph_input_attn_k_dsa * inp_attn_dsa,
    struct ggml_tensor                * inp_out_ids,
    float                               kq_scale,
    void                              (*set_stream)(ggml_backend_t, int),
    int                               (*get_stream)(ggml_backend_t),
    llama_pipeline_expert_prefetch_fn   prefetch_fn,
    void                              * prefetch_user_data) {

    auto * p = new llama_pipeline_sched();
    memset(p, 0, sizeof(*p));
    p->backend     = backend;
    p->sched       = sched;
    p->device      = backend ? ggml_backend_get_device(backend) : nullptr;
    p->set_stream  = set_stream;
    p->get_stream  = get_stream;
    p->n_layer     = 0;
    p->inpL_embd   = nullptr;
    p->scratch_buf = nullptr;
    p->scratch_ctx = nullptr;

    p->gf_output       = nullptr;
    p->ctx_output      = nullptr;
    p->inp_attn_persist = nullptr;

    p->prefetch_fn        = prefetch_fn;
    p->prefetch_user_data = prefetch_user_data;

    const uint32_t n_layers = ctx.hparams.n_layer - ctx.hparams.nextn_predict_layers;
    if (n_layers > LLAMA_PIPELINE_MAX_LAYERS) { delete p; return nullptr; }
    p->n_layer = (int)n_layers;
    const int64_t n_ctx_max = ctx.cparams.n_ctx;

    // ---- Allocate persistent scratch buffer ----
    {
        const int64_t n_embd_head_k  = ctx.hparams.n_embd_head_k_mla();
        const int64_t n_embd_head_qk_rope = ctx.hparams.n_rot();
        const uint32_t kv_lora_rank   = ctx.hparams.n_lora_kv;
        const uint32_t n_indexer_top_k = ctx.hparams.indexer_top_k;
        const int64_t  n_embd         = ctx.n_embd;
        const int64_t  n_head         = ctx.n_head;
        const int64_t  n_tokens       = ctx.n_tokens;

        // DSA k_rot size: smallest power-of-2 divisor of indexer_head_size
        int nrot = 64;
        {
            const int32_t n_embd_head_lid = ctx.hparams.indexer_head_size;
            int tmp = 64;
            do { tmp *= 2; } while (n_embd_head_lid % tmp == 0);
            nrot = tmp / 2;
        }

        // Scratch sizes
        const size_t s_inpL_embd     = ggml_row_size(GGML_TYPE_F32, n_embd * n_tokens);
        const size_t s_Qcur          = ggml_row_size(GGML_TYPE_F32, n_embd_head_k * n_head * n_tokens);
        const size_t s_Kcur          = ggml_row_size(GGML_TYPE_F32, (kv_lora_rank + n_embd_head_qk_rope) * 1 * n_tokens);
        const size_t s_Vcur          = ggml_row_size(GGML_TYPE_F32, kv_lora_rank * 1 * n_tokens);
        const size_t s_top_k         = ggml_row_size(GGML_TYPE_I32, n_indexer_top_k * n_tokens);
        const size_t s_ffn_inp       = ggml_row_size(GGML_TYPE_F32, n_embd * n_tokens);
        const size_t s_inpL_next     = ggml_row_size(GGML_TYPE_F32, n_embd * n_tokens);
        const size_t s_inp_pos       = ggml_row_size(GGML_TYPE_I32, n_tokens);

        // DSA scratch sizes (persistent copies -- one set shared across all layers)
        const size_t s_k_rot_lid     = ggml_row_size(GGML_TYPE_F32, nrot * nrot);
        const size_t s_k_idxs_lid    = ggml_row_size(GGML_TYPE_I64, n_tokens);
        const size_t s_k_idxs_mla    = ggml_row_size(GGML_TYPE_I64, n_tokens);
        const size_t s_mask_lid      = ggml_row_size(GGML_TYPE_F32, (size_t)n_ctx_max * n_tokens);
        const size_t s_mask_mla      = ggml_row_size(GGML_TYPE_F16, (size_t)n_ctx_max * n_tokens);

        const size_t slot_size = s_inpL_embd + s_Qcur + s_Kcur + s_Vcur + s_top_k + s_ffn_inp + s_inpL_next + s_inp_pos;
        const size_t dsa_scratch = s_k_rot_lid + s_k_idxs_lid + s_k_idxs_mla + s_mask_lid + s_mask_mla;
        const size_t total_scratch = LLAMA_PIPELINE_DEPTH * slot_size + dsa_scratch;

        // ggml_context for scratch tensor metadata
        const int n_tensors_per_slot = 7; // qcur, kcur, vcur, top_k, ffn_inp, inpL_next, inp_pos
        const int n_dsa_tensors = 5; // k_rot_lid, k_idxs_lid, k_idxs_mla, mask_lid, mask_mla
        const int n_output_tensors = 4; // logits, embd, norm_output, etc.
        const size_t ctx_meta = LLAMA_PIPELINE_DEPTH * (n_tensors_per_slot * ggml_tensor_overhead()) +
                                (n_dsa_tensors + n_output_tensors) * ggml_tensor_overhead() + 8192;
        uint8_t * ctx_buf = new uint8_t[ctx_meta];
        p->scratch_ctx = ggml_init({ctx_meta, ctx_buf, false});
        if (!p->scratch_ctx) { delete[] ctx_buf; delete p; return nullptr; }

        // Persistent GPU buffer
        p->scratch_buf = ggml_backend_alloc_buffer(backend, total_scratch);
        if (!p->scratch_buf) { ggml_free(p->scratch_ctx); delete p; return nullptr; }

        size_t offset = 0;
        auto make_scratch = [&](ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
                -> struct ggml_tensor * {
            struct ggml_tensor * t = ggml_new_tensor_4d(p->scratch_ctx, type, ne0, ne1, ne2, ne3);
            if (!t) return nullptr;
            size_t sz = ggml_row_size(type, ne0 * ne1 * ne2 * ne3);
            t->data   = (uint8_t *)ggml_backend_buffer_get_base(p->scratch_buf) + offset;
            t->buffer = p->scratch_buf;
            t->flags |= GGML_TENSOR_FLAG_INPUT;  // gallocr skips allocation
            offset += sz;
            return t;
        };

        // Per-slot scratch tensors for intermediate data
        for (int s = 0; s < LLAMA_PIPELINE_DEPTH; s++) {
            auto & sc = p->scratch[s];
            sc.qcur     = make_scratch(GGML_TYPE_F32, n_embd_head_k, n_head, n_tokens, 1);
            sc.kcur     = make_scratch(GGML_TYPE_F32, kv_lora_rank + n_embd_head_qk_rope, 1, n_tokens, 1);
            sc.vcur     = make_scratch(GGML_TYPE_F32, kv_lora_rank, 1, n_tokens, 1);
            sc.top_k    = make_scratch(GGML_TYPE_I32, n_indexer_top_k, n_tokens, 1, 1);
            sc.ffn_inp  = make_scratch(GGML_TYPE_F32, n_embd, n_tokens, 1, 1);
            p->inpL_next[s] = make_scratch(GGML_TYPE_F32, n_embd, n_tokens, 1, 1);
        }
        // inpL_embd: embedding input copy (only needed by layer 0)
        p->inpL_embd = make_scratch(GGML_TYPE_F32, n_embd, n_tokens, 1, 1);

        // inp_pos scratch copy
        const int64_t n_pos_per_embd = ctx.hparams.n_pos_per_embd();
        p->scratch_inp_pos = make_scratch(GGML_TYPE_I32, n_tokens * n_pos_per_embd, 1, 1, 1);

        // ---- Persistent DSA scratch tensors ----
        // These are shared by all layers (one copy), referenced via p->inp_attn_persist.
        p->scratch_k_rot_lid   = make_scratch(GGML_TYPE_F32, nrot, nrot, 1, 1);
        p->scratch_k_idxs_lid  = make_scratch(GGML_TYPE_I64, n_tokens, 1, 1, 1);
        p->scratch_kq_mask_lid = make_scratch(GGML_TYPE_F32, n_ctx_max, n_tokens, 1, 1);
        p->scratch_k_idxs_mla  = make_scratch(GGML_TYPE_I64, n_tokens, 1, 1, 1);
        p->scratch_kq_mask_mla = make_scratch(GGML_TYPE_F16, n_ctx_max, n_tokens, 1, 1);

        // Report scratch size
        (void)offset;
    }

    // ---- Create persistent DSA input object referencing scratch tensors ----
    {
        // k_rot_lid is stable (filled once below from the original DSA context)
        p->inp_attn_persist = new llm_graph_input_attn_k_dsa(
            inp_attn_dsa->hparams, inp_attn_dsa->cparams, inp_attn_dsa->mctx);
        p->inp_attn_persist->self_k_rot_lid   = p->scratch_k_rot_lid;
        p->inp_attn_persist->self_k_idxs_lid  = p->scratch_k_idxs_lid;
        p->inp_attn_persist->self_kq_mask_lid = p->scratch_kq_mask_lid;
        p->inp_attn_persist->self_kq_mask_lid_cnv = p->scratch_kq_mask_lid;
        p->inp_attn_persist->self_k_idxs_mla  = p->scratch_k_idxs_mla;
        p->inp_attn_persist->self_kq_mask_mla = p->scratch_kq_mask_mla;
        p->inp_attn_persist->self_kq_mask_mla_cnv = p->scratch_kq_mask_mla;

        // Fill k_rot_lid once (it is stable -- doesn't change between decode steps).
        // The set_input_k_rot function needs a host buffer; use the original DSA
        // tensor's buffer type for the scratch k_rot_lid as well.
        // We just copy the data from the original using backend tensor copy.
        if (inp_attn_dsa->self_k_rot_lid) {
            ggml_backend_tensor_copy(inp_attn_dsa->self_k_rot_lid, p->scratch_k_rot_lid);
        }
    }

    // ---- Build per-layer phase graphs ----
    struct ggml_tensor * layer_inpL = p->inpL_embd;
    for (int il = 0; il < p->n_layer; il++) {
        const size_t ctx_size = 2 * 1024 * 1024;
        uint8_t * ctx_buf = new uint8_t[ctx_size];
        p->ctx_layers[il] = ggml_init({ctx_size, ctx_buf, false});
        if (!p->ctx_layers[il]) {
            delete[] ctx_buf;
            llama_pipeline_sched_free(p);
            return nullptr;
        }

        p->gf_qkv[il]  = ggml_new_graph(p->ctx_layers[il]);
        p->gf_attn[il] = ggml_new_graph(p->ctx_layers[il]);
        p->gf_ffn[il]  = ggml_new_graph(p->ctx_layers[il]);

        int slot = il % LLAMA_PIPELINE_DEPTH;

        // Use persistent DSA input (scratch tensors) for all phases
        build_phase_a(model, ctx, p->ctx_layers[il], p->gf_qkv[il],
                      layer_inpL, p->scratch_inp_pos, p->inp_attn_persist,
                      kq_scale, il, &p->scratch[slot]);

        build_phase_b(model, ctx, p->ctx_layers[il], p->gf_attn[il],
                      layer_inpL, &p->scratch[slot],
                      p->inp_attn_persist, inp_out_ids, kq_scale, il);

        build_phase_c(model, ctx, p->ctx_layers[il], p->gf_ffn[il],
                      &p->scratch[slot], p->inpL_next[slot],
                      inp_out_ids, il);

        layer_inpL = p->inpL_next[slot];
    }

    // ---- Determine which layers have MoE experts ----
    // Dense: il < n_layer_dense_lead; MoE: il >= n_layer_dense_lead
    const uint32_t n_layer_dense_lead = ctx.hparams.n_layer_dense_lead;
    for (int il = 0; il < p->n_layer; il++) {
        p->has_moe[il] = ((uint32_t)il >= n_layer_dense_lead);
    }

    // ---- Save model weight tensor references for MoE layers ----
    for (int il = 0; il < p->n_layer; il++) {
        if (p->has_moe[il]) {
            p->model_gate_exps[il] = model.layers[il].ffn_gate_exps;
            p->model_up_exps[il]   = model.layers[il].ffn_up_exps;
            p->model_down_exps[il] = model.layers[il].ffn_down_exps;
        }
    }

    // ---- Allocate expert prefetch buffers (GPU-resident) ----
    // We allocate for n_expert_used experts × 3 weight types.
    // The prefetch buffer has ne[2] = prefetch_max_experts (smaller than n_expert).
    // At runtime, selected expert slices are copied from CPU model weights
    // into the prefetch buffer slots, with indices remapped.
    {
        // Determine the shape from the first MoE layer's weight tensor
        int moe_il = -1;
        for (int il = 0; il < p->n_layer && moe_il < 0; il++) {
            if (p->has_moe[il] && p->model_up_exps[il]) {
                moe_il = il;
            }
        }

        if (p->prefetch_fn && moe_il >= 0) {
            // Allocate prefetch buffers for n_expert_used experts per weight type
            const int n_expert_used = (int)ctx.hparams.n_expert_used;
            const int ne2 = (n_expert_used > 0 && n_expert_used <= LLAMA_PIPELINE_PREFETCH_MAX_EXPERTS)
                          ? n_expert_used : LLAMA_PIPELINE_PREFETCH_MAX_EXPERTS;

            auto * ref_gate = p->model_gate_exps[moe_il];
            auto * ref_up   = p->model_up_exps[moe_il];
            auto * ref_down = p->model_down_exps[moe_il];
            if (ref_gate && ref_up && ref_down) {
                const size_t s_gate = ggml_row_size(ref_gate->type,
                    ref_gate->ne[0] * ref_gate->ne[1] * ne2);
                const size_t s_up   = ggml_row_size(ref_up->type,
                    ref_up->ne[0] * ref_up->ne[1] * ne2);
                const size_t s_down = ggml_row_size(ref_down->type,
                    ref_down->ne[0] * ref_down->ne[1] * ne2);
                const size_t s_total = s_gate + s_up + s_down;

                p->prefetch_gate = ggml_new_tensor_3d(p->scratch_ctx,
                    ref_gate->type, ref_gate->ne[0], ref_gate->ne[1], ne2);
                p->prefetch_up   = ggml_new_tensor_3d(p->scratch_ctx,
                    ref_up->type,   ref_up->ne[0],   ref_up->ne[1],   ne2);
                p->prefetch_down = ggml_new_tensor_3d(p->scratch_ctx,
                    ref_down->type, ref_down->ne[0], ref_down->ne[1], ne2);

                if (p->prefetch_gate && p->prefetch_up && p->prefetch_down) {
                    p->prefetch_buf = ggml_backend_alloc_buffer(backend, s_total);
                    if (p->prefetch_buf) {
                        size_t off = 0;
                        auto assign = [&](struct ggml_tensor * t, size_t sz) {
                            if (!t) return;
                            t->data   = (uint8_t *)ggml_backend_buffer_get_base(p->prefetch_buf) + off;
                            t->buffer = p->prefetch_buf;
                            t->flags |= GGML_TENSOR_FLAG_INPUT;
                            off += sz;
                        };
                        assign(p->prefetch_gate, s_gate);
                        assign(p->prefetch_up,   s_up);
                        assign(p->prefetch_down, s_down);
                    }
                }

                if (p->device) {
                    p->e_prefetch_done = ggml_backend_event_new(p->device);
                }
            }
        }
    }

    // ---- Now rebuild phase C for MoE layers to use prefetch buffers ----
    // We rebuild the FFN graphs since the weight tensor references changed
    for (int il = 0; il < p->n_layer; il++) {
        if (p->has_moe[il] && p->prefetch_gate) {
            // Free the existing FFN graph and rebuild with prefetch tensors
            if (p->ctx_layers[il]) {
                // Can't partially free a ggml_context, but we can rebuild
                // by clearing the graph context and rebuilding phase C.
                // For the phase C sub-graph, we need a fresh ggml_context.

                // We can't easily modify the existing graph, so we create
                // a reinterpretation: the graph stores tensor pointers.
                // Since the graph is already built, we need to rebuild it.
                // For simplicity, we modify the weight tensor references
                // by patching the graph's source tensors.
                // Actually, we rebuild the phase by clearing and re-running
                // build_phase_c with prefetch overrides.

                // Get a new context for the rebuilt graph
                const size_t ctx_size = 2 * 1024 * 1024;
                uint8_t * ctx_buf = new uint8_t[ctx_size];
                // Free old context
                ggml_free(p->ctx_layers[il]);
                p->ctx_layers[il] = ggml_init({ctx_size, ctx_buf, false});
                p->gf_ffn[il] = ggml_new_graph(p->ctx_layers[il]);

                // Rebuild with prefetch tensors
                int slot = il % LLAMA_PIPELINE_DEPTH;
                struct ggml_tensor * layer_inpL = (il == 0) ? p->inpL_embd : p->inpL_next[(il-1) % LLAMA_PIPELINE_DEPTH];
                // Re-run only phase C with prefetch overrides
                // build_phase_c needs scratch data and the layer's inpL_next output
                build_phase_c(model, ctx, p->ctx_layers[il], p->gf_ffn[il],
                              &p->scratch[slot], p->inpL_next[slot],
                              inp_out_ids, il,
                              p->prefetch_gate, p->prefetch_up, p->prefetch_down);
            }
        }
    }

    // ---- Save original main-graph tensor pointers (used by copy_inputs) ----
    p->saved_inp_pos = inp_pos;
    p->saved_inp_attn_dsa = inp_attn_dsa;

    // ---- Create events ----
    if (p->device) {
        for (int i = 0; i < p->n_layer; i++) {
            p->e_qkv_done[i]  = ggml_backend_event_new(p->device);
            p->e_attn_done[i] = ggml_backend_event_new(p->device);
        }
    }

    return p;
}

// ============================================================================
// copy_inputs -- copy caller's GPU-buffer data to persistent scratch
//
// After set_inputs fills the original DSA tensors (in CPU/host buffer), this
// copies inpL_embd, inp_pos, and all DSA mask/index data to our persistent
// scratch for safe access across sched resets.
// ============================================================================

void llama_pipeline_sched_copy_inputs(
    struct llama_pipeline_sched  * p,
    ggml_backend_t                 backend,
    struct ggml_tensor           * src_inpL,
    struct ggml_tensor           * src_inp_pos,
    struct llm_graph_input_attn_k_dsa * src_dsa) {

    if (!p) return;

    // Copy embedding and position to persistent scratch
    if (src_inpL && p->inpL_embd) {
        ggml_backend_tensor_copy(src_inpL, p->inpL_embd);
    }
    if (src_inp_pos && p->scratch_inp_pos) {
        ggml_backend_tensor_copy(src_inp_pos, p->scratch_inp_pos);
    }

    // Copy DSA mask/index data from the source DSA input (CPU buffer, filled
    // by set_inputs) to our persistent scratch.
    // k_rot_lid is already copied at init (stable); only masks and indices change.
    if (src_dsa && p->inp_attn_persist) {
        if (src_dsa->self_k_idxs_mla && p->scratch_k_idxs_mla) {
            ggml_backend_tensor_copy(src_dsa->self_k_idxs_mla, p->scratch_k_idxs_mla);
        }
        if (src_dsa->get_kq_mask_mla() && p->scratch_kq_mask_mla) {
            ggml_backend_tensor_copy(src_dsa->get_kq_mask_mla(), p->scratch_kq_mask_mla);
        }
        if (src_dsa->self_k_idxs_lid && p->scratch_k_idxs_lid) {
            ggml_backend_tensor_copy(src_dsa->self_k_idxs_lid, p->scratch_k_idxs_lid);
        }
        if (src_dsa->get_kq_mask_lid() && p->scratch_kq_mask_lid) {
            ggml_backend_tensor_copy(src_dsa->get_kq_mask_lid(), p->scratch_kq_mask_lid);
        }
    }
}

// ============================================================================
// Three-stage sliding window dispatch
//
//   compute_stream (0): [QKV_0] -> [wait] -> [prefetch_launch] -> [FFN_0]
//   attn_stream   (1):            [attn_0]
//   transfer_stream(2): [prefetch_H2D overlaps with attn on s1]
//
// Events:
//   e_qkv_done[L]  -- QKV done on CS -> AS waits -> starts attn
//   e_attn_done[L] -- attn done on AS -> CS waits -> starts FFN
//   e_prefetch_done -- H2D copies done on TS -> FFN ready
// ============================================================================

void llama_pipeline_sched_compute(struct llama_pipeline_sched * p, int n_layer) {
    if (!p || !p->backend || n_layer <= 0) return;
    if (n_layer > p->n_layer) n_layer = p->n_layer;

    // Whether prefetch is enabled for this batch
    const bool use_prefetch = (p->prefetch_fn != nullptr) &&
                              (p->prefetch_gate != nullptr) &&
                              (p->device != nullptr);

    for (int L = 0; L < n_layer; L++) {
        // ---- Stage 1: QKV proj on compute stream (0) ----
        ggml_backend_sched_synchronize(p->sched);
        ggml_backend_sched_reset(p->sched);
        if (p->set_stream) p->set_stream(p->backend, 0);
        ggml_backend_sched_graph_compute_async(p->sched, p->gf_qkv[L]);
        if (p->e_qkv_done[L]) ggml_backend_event_record(p->e_qkv_done[L], p->backend);

        // ---- Expert Prefetch Launch (after QKV, before attn) ----
        // Call prefetch_fn with GPU top_k tensor, model weight tensors, and
        // prefetch destination buffers. The callback reads top_k on GPU,
        // determines which experts to prefetch, and launches async H2D copies
        // on a dedicated transfer stream. The completion event syncs with FFN.
        if (use_prefetch && p->has_moe[L]) {
            struct ggml_tensor * top_k_t = p->scratch[L % LLAMA_PIPELINE_DEPTH].top_k;
            if (top_k_t && p->model_gate_exps[L]) {
                struct ggml_tensor * dst[3]  = {p->prefetch_gate, p->prefetch_up, p->prefetch_down};
                struct ggml_tensor * src[3]  = {p->model_gate_exps[L], p->model_up_exps[L], p->model_down_exps[L]};
                size_t sb[3];
                for (int i = 0; i < 3; i++) {
                    sb[i] = src[i] ? ggml_row_size(src[i]->type, src[i]->ne[0] * src[i]->ne[1]) : 0;
                }
                if (dst[0] && dst[1] && dst[2] && src[0] && src[1] && src[2]) {
                    p->prefetch_fn(dst, src, sb, top_k_t, L,
                                   p->e_qkv_done[L], p->e_prefetch_done,
                                   p->prefetch_user_data);
                }
            }
        }

        // ---- Stage 2: flash_attn on attn stream (1) ----
        ggml_backend_sched_synchronize(p->sched);
        ggml_backend_sched_reset(p->sched);
        if (p->set_stream) p->set_stream(p->backend, 1);
        if (p->e_qkv_done[L]) ggml_backend_event_wait(p->backend, p->e_qkv_done[L]);
        ggml_backend_sched_graph_compute_async(p->sched, p->gf_attn[L]);
        if (p->e_attn_done[L]) ggml_backend_event_record(p->e_attn_done[L], p->backend);

        // ---- Stage 3: FFN on compute stream (0) ----
        ggml_backend_sched_synchronize(p->sched);
        ggml_backend_sched_reset(p->sched);
        if (p->set_stream) p->set_stream(p->backend, 0);
        if (p->e_attn_done[L]) ggml_backend_event_wait(p->backend, p->e_attn_done[L]);
        ggml_backend_sched_graph_compute_async(p->sched, p->gf_ffn[L]);
    }

    ggml_backend_synchronize(p->backend);
}

// ============================================================================
// compute_output_head -- build + compute final norm + lm_head
//
// After pipeline layers are done, inpL_next[last_slot] holds the hidden state.
// This builds a tiny graph: last_hidden -> rms_norm -> lm_head.
// The output tensor is placed in res->t_logits / res->t_embd.
// ============================================================================

void llama_pipeline_sched_compute_output_head(
    struct llama_pipeline_sched  * p,
    struct llm_graph_result      * res,
    struct ggml_tensor           * output_norm_weight,
    float                          norm_rms_eps,
    struct ggml_tensor           * lm_head_weight) {

    if (!p || !res) return;

    // Find the last layer's output slot
    int last_slot = (p->n_layer - 1) % LLAMA_PIPELINE_DEPTH;
    struct ggml_tensor * last_hidden = p->inpL_next[last_slot];

    // If output head graph was already created, check if dimensions still match
    // and reuse; otherwise rebuild.
    if (p->gf_output && p->ctx_output) {
        // Check if the hidden state shape matches (n_embd, n_tokens)
        // If the graph shape is stale, recreate it.
        // For simplicity, we rebuild each time if the hidden state changed.
        // This is fine since the output head graph is tiny.
    }
    if (p->ctx_output) {
        ggml_free(p->ctx_output);
        p->ctx_output = nullptr;
        p->gf_output = nullptr;
    }

    // Build a tiny output head context
    const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false) + 1024;
    uint8_t * ctx_buf = new uint8_t[ctx_size];
    p->ctx_output = ggml_init({ctx_size, ctx_buf, false});
    if (!p->ctx_output) { delete[] ctx_buf; return; }

    p->gf_output = ggml_new_graph(p->ctx_output);

    // Build: last_hidden -> rms_norm -> lm_head
    ggml_tensor * cur = ggml_rms_norm(p->ctx_output, last_hidden, norm_rms_eps);
    ggml_set_name(cur, "result_norm");

    cur = ggml_mul_mat(p->ctx_output, output_norm_weight, cur);
    ggml_set_name(cur, "result_norm_w");

    cur = ggml_mul_mat(p->ctx_output, lm_head_weight, cur);
    ggml_set_name(cur, "result_output");

    ggml_build_forward_expand(p->gf_output, cur);
    ggml_set_output(cur);

    // Also produce embedding if the model has one
    ggml_tensor * embd = nullptr;
    if (res->get_embd() != nullptr) {
        // Normalized hidden state as embedding
        embd = ggml_rms_norm(p->ctx_output, last_hidden, norm_rms_eps);
        ggml_set_output(embd);
        ggml_build_forward_expand(p->gf_output, embd);
    }

    // Allocate and compute using the pipeline's scheduler
    // (same sched used for layers)
    if (!p->sched) return;

    ggml_backend_sched_synchronize(p->sched);
    ggml_backend_sched_reset(p->sched);

    if (!ggml_backend_sched_alloc_graph(p->sched, p->gf_output)) {
        LLAMA_LOG_ERROR("%s: failed to allocate output head graph\n", __func__);
        return;
    }

    ggml_backend_sched_graph_compute(p->sched, p->gf_output);

    // Copy results to the result object for the caller
    res->t_logits = cur;
    if (embd) {
        res->t_embd = embd;
    }
}

void llama_pipeline_sched_set_prefetch_fn(
    struct llama_pipeline_sched  * p,
    llama_pipeline_expert_prefetch_fn fn,
    void * user_data) {
    if (p) {
        p->prefetch_fn        = fn;
        p->prefetch_user_data = user_data;
    }
}

void llama_pipeline_sched_free(struct llama_pipeline_sched * p) {
    if (!p) return;

    if (p->device) {
        for (int i = 0; i < p->n_layer; i++) {
            if (p->e_qkv_done[i])  ggml_backend_event_free(p->e_qkv_done[i]);
            if (p->e_attn_done[i]) ggml_backend_event_free(p->e_attn_done[i]);
        }
        if (p->e_prefetch_done) ggml_backend_event_free(p->e_prefetch_done);
    }

    if (p->scratch_buf) ggml_backend_buffer_free(p->scratch_buf);
    if (p->scratch_ctx) ggml_free(p->scratch_ctx);

    if (p->prefetch_buf) ggml_backend_buffer_free(p->prefetch_buf);

    if (p->ctx_output) {
        ggml_free(p->ctx_output);
        p->ctx_output = nullptr;
        p->gf_output = nullptr;
    }

    delete p->inp_attn_persist;
    p->inp_attn_persist = nullptr;

    for (int i = 0; i < p->n_layer; i++) {
        if (p->ctx_layers[i]) {
            ggml_free(p->ctx_layers[i]);
            p->ctx_layers[i] = nullptr;
        }
    }

    delete p;
}
