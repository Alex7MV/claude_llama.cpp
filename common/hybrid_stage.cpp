#include "hybrid_stage.h"
#include <cstdio>

// Thread-local pointer read by the CUDA flash_attn_ext handler
thread_local hybrid_orchestrator * g_hybrid_ctx = nullptr;

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#include "ggml-cuda/tma-transfer.h"
#include "ggml-cuda/fattn.cuh"

bool hybrid_orchestrator::init(
    uint32_t n_layers_,
    uint32_t n_ctx_max,
    uint32_t kv_lora_rank_,
    uint32_t max_lookahead,
    ggml_backend_dev_t device,
    void * compute_stream,
    void * draft_stream_,
    uint32_t max_draft_,
    ggml_type  cache_type_k,
    uint32_t   key_dim)
{
    n_layers       = n_layers_;
    kv_lora_rank   = kv_lora_rank_;
    gpu_compute_stream = compute_stream;
    draft_stream        = draft_stream_;
    max_draft = max_draft_ > 0 ? max_draft_ : 6;
    current_draft = max_draft;

    if (!pool.init(n_layers_, n_ctx_max, kv_lora_rank_, max_lookahead, device, cache_type_k, key_dim)) {
        fprintf(stderr, "hybrid_orchestrator: VRAM pool init failed\n");
        return false;
    }

    stages.resize(n_layers_);

    for (int i = 0; i < 2; i++) {
        cudaEventCreateWithFlags(
            reinterpret_cast<cudaEvent_t *>(&tma_events[i]),
            cudaEventDisableTiming);
    }

    // Create a separate CUDA stream for async flash_attn
    cudaStreamCreateWithFlags(
        reinterpret_cast<cudaStream_t *>(&attn_stream),
        cudaStreamNonBlocking);
    cudaEventCreateWithFlags(
        reinterpret_cast<cudaEvent_t *>(&attn_event),
        cudaEventDisableTiming);

    // Register with the global fattn module so the CUDA backend can redirect
    ggml_cuda_attn_set_async_ctx(
        reinterpret_cast<cudaStream_t>(attn_stream),
        reinterpret_cast<cudaEvent_t>(attn_event));

    // Pre-allocate pinned CPU buffers for flash_attn output (one per layer)
    // Size: head_dim * sizeof(float) = 128 * 4 = 512 bytes per slot
    attn_pinned.resize(n_layers_, nullptr);
    for (uint32_t i = 0; i < n_layers_; i++) {
        cudaHostAlloc(&attn_pinned[i], 512, cudaHostAllocDefault);
    }

    fprintf(stderr, "hybrid_orchestrator: ready (%u layers, rank=%u, attn-stream=%p)\n",
            n_layers_, kv_lora_rank_, (void*)attn_stream);
    return true;
}

void hybrid_orchestrator::free_all() {
    pool.free_all();
    // Unregister from the fattn module
    ggml_cuda_attn_set_async_ctx(nullptr, nullptr);
    for (int i = 0; i < 2; i++) {
        if (tma_events[i]) {
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(tma_events[i]));
            tma_events[i] = nullptr;
        }
    }
    if (attn_stream) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(attn_stream));
        attn_stream = nullptr;
    }
    if (attn_event) {
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(attn_event));
        attn_event = nullptr;
    }
    for (auto * buf : attn_pinned) {
        if (buf) cudaFreeHost(buf);
    }
    attn_pinned.clear();
    stages.clear();
}

void hybrid_orchestrator::on_tma_enqueued(uint32_t layer) {
    auto & s = stages[layer];

    float * dst = pool.slot_ptr(layer, current_token + layer);

    bool ok = ggml_tma_enqueue_h2d_1d(
        dst,
        s.tma.cpu_src,
        s.tma.bytes,
        gpu_compute_stream,
        tma_events[tma_head & 1]);

    if (!ok) {
        fprintf(stderr, "hybrid: TMA fallback at layer %u\n", layer);
    }

    s.tma.enqueued = true;
    tma_head++;
    s.phase = hybrid_phase::TMA_ENQUEUED;
}

bool hybrid_orchestrator::wait_tma_event() {
    // Make the GPU compute stream wait for the oldest TMA transfer.
    // CPU returns immediately — no blocking.
    // The stream will not execute flash_attn until TMA data arrives.
    if (tma_head < 2) return true; // nothing to wait for yet
    cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(tma_events[(tma_head - 2) & 1]);
    cudaError_t err = cudaStreamWaitEvent(
        (cudaStream_t)gpu_compute_stream, ev, 0);
    return err == cudaSuccess;
}

void hybrid_orchestrator::on_gpu_attn_done(uint32_t layer) {
    // Non-blocking: CPU enqueues wait dependency on GPU stream,
    // then immediately returns to compute MoE FFN on CPU.
    wait_tma_event();
    stages[layer].phase = hybrid_phase::GPU_ATTN_DONE;
}

void hybrid_orchestrator::trace_tma_verify(uint32_t layer) {
    auto & s = stages[layer];
    float * dst = pool.slot_ptr(layer, current_token + layer);

    float readback[4];
    cudaMemcpy(readback, dst, sizeof(readback), cudaMemcpyDeviceToHost);

    fprintf(stderr,
        "hybrid[TMA] layer=%u pos=%u head=%d src=%p dst=%p "
        "bytes=%llu enq=%d rb=[%f %f %f %f] phase=%d\n",
        layer, current_token + layer, tma_head,
        (void*)s.tma.cpu_src, (void*)dst,
        (unsigned long long)s.tma.bytes,
        (int)s.tma.enqueued,
        readback[0], readback[1], readback[2], readback[3],
        (int)s.phase);
}

void hybrid_orchestrator::patch_graph_for_mla(
    ggml_cgraph * gf,
    ggml_backend_sched_t sched,
    ggml_backend_t gpu_backend)
{
    if (!gf || !sched || !gpu_backend) return;
    int patched_attn = 0;
    int patched_cpy  = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_FLASH_ATTN_EXT &&
            ggml_backend_supports_op(gpu_backend, node))
        {
            ggml_backend_sched_set_tensor_backend(sched, node, gpu_backend);

            // Mark for async execution: op_params[4] = 1 signals the CUDA
            // flash_attn handler to record a fence event and return
            // immediately.  Slot 4 is free (0=scale, 1=max_bias,
            // 2=logit_softcap, 3=precision).
            node->op_params[4] = 1;

            patched_attn++;
        }
        // Move KV-cache write ops (cpy_k) to GPU — K cache is in VRAM,
        // CPU backend cannot write to it.  Tensor name patterns:
        //   DeepSeek-V3:  blk.N.k_cache_latent
        //   Kimi-k2.6:    blk.N.k_cache_latent (MLA), blk.N.v_cache_compressed (KDA)
        if (node->op == GGML_OP_CPY &&
            node->dst &&
            (strstr(node->dst->name, "k_cache_latent") ||
             strstr(node->dst->name, "v_cache_compressed")) &&
            ggml_backend_supports_op(gpu_backend, node))
        {
            ggml_backend_sched_set_tensor_backend(sched, node, gpu_backend);
            patched_cpy++;
        }
    }
    fprintf(stderr, "hybrid: patched %d flash-attn + %d cpy_k ops → GPU\n",
            patched_attn, patched_cpy);
}

void hybrid_orchestrator::hybrid_gpu_fence() {
    if (attn_event) {
        cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(attn_event));
    }
}

#else // !GGML_USE_CUDA

bool hybrid_orchestrator::init(
    uint32_t, uint32_t, uint32_t, uint32_t,
    ggml_backend_dev_t, void *, void *,
    uint32_t, ggml_type, uint32_t)
{
    fprintf(stderr, "hybrid_orchestrator: CUDA not available\n");
    return false;
}

void hybrid_orchestrator::free_all() { stages.clear(); }
void hybrid_orchestrator::on_tma_enqueued(uint32_t) {}
bool hybrid_orchestrator::wait_tma_event() { return true; }
void hybrid_orchestrator::on_gpu_attn_done(uint32_t) {}
void hybrid_orchestrator::trace_tma_verify(uint32_t) {}
void hybrid_orchestrator::patch_graph_for_mla(ggml_cgraph *, ggml_backend_sched_t, ggml_backend_t) {}
void hybrid_orchestrator::hybrid_gpu_fence() {}

#endif // GGML_USE_CUDA

void hybrid_orchestrator::on_norm_done(uint32_t layer, const int32_t expert_ids[2]) {
    auto & s = stages[layer];
    s.phase = hybrid_phase::NORM_DONE;

    s.prefetch.expert_ids[0] = expert_ids[0];
    s.prefetch.expert_ids[1] = expert_ids[1];
    s.prefetch.prefetched    = true;
}

void hybrid_orchestrator::on_kv_compressed(uint32_t layer, const float * c_tkv_cpu) {
    auto & s = stages[layer];
    s.phase = hybrid_phase::KV_COMPRESSED;

    s.tma.cpu_src  = c_tkv_cpu;
    s.tma.bytes    = static_cast<uint64_t>(kv_lora_rank) * 1;
    s.tma.enqueued = false;
}

void hybrid_orchestrator::on_merge_done(uint32_t layer) {
    stages[layer].phase = hybrid_phase::MERGE_DONE;
}

void hybrid_orchestrator::advance_token() {
    current_token++;
    current_layer = 0;
}

uint32_t hybrid_orchestrator::verify_and_rollback(
    llama_context * ctx_tgt,
    const llama_tokens & target_tokens)
{
    uint32_t n_match = 0;
    for (uint32_t i = 0; i < draft.lookahead_buffer.size() && i < target_tokens.size(); i++) {
        if (target_tokens[i] == draft.lookahead_buffer[i]) {
            n_match++;
        } else {
            break;
        }
    }

    verify.accepted    = n_match;
    verify.rejected_at = n_match;
    verify.n_draft     = static_cast<uint32_t>(draft.lookahead_buffer.size());
    verify.total_attempts++;
    verify.total_accepted += n_match;

    if (n_match < draft.lookahead_buffer.size()) {
        llama_memory_seq_rm(
            llama_get_memory(ctx_tgt),
            0,
            static_cast<llama_pos>(current_token + n_match + 1),
            -1);
    }

    // Adaptive lookahead: EMA-based acceptance rate tracking
    float step_rate = draft.lookahead_buffer.size() > 0
        ? (float)n_match / (float)draft.lookahead_buffer.size()
        : 1.0f;
    accept_rate_ema = 0.9f * accept_rate_ema + 0.1f * step_rate;
    if (accept_rate_ema > 0.60f && current_draft < max_draft) {
        current_draft = max_draft;
    } else if (accept_rate_ema < 0.30f && current_draft > 3) {
        current_draft = 3;
    }

    draft.lookahead_buffer.erase(
        draft.lookahead_buffer.begin() + n_match,
        draft.lookahead_buffer.end());
    draft.in_flight = false;

    return n_match;
}

void hybrid_orchestrator::start_draft_batch(std::function<void()> draft_fn) {
    if (draft.in_flight) return;
    draft.lookahead_buffer.clear();
    draft_fn();
    draft.in_flight = true;
}
