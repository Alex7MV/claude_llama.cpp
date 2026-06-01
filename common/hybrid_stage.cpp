#include "hybrid_stage.h"
#include <cstdio>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#include "ggml-cuda/tma-transfer.h"

bool hybrid_orchestrator::init(
    uint32_t n_layers_,
    uint32_t n_ctx_max,
    uint32_t kv_lora_rank_,
    uint32_t max_lookahead,
    ggml_backend_dev_t device,
    void * compute_stream,
    void * draft_stream_)
{
    n_layers       = n_layers_;
    kv_lora_rank   = kv_lora_rank_;
    gpu_compute_stream = compute_stream;
    draft_stream        = draft_stream_;

    if (!pool.init(n_layers_, n_ctx_max, kv_lora_rank_, max_lookahead, device)) {
        fprintf(stderr, "hybrid_orchestrator: VRAM pool init failed\n");
        return false;
    }

    stages.resize(n_layers_);

    for (int i = 0; i < 2; i++) {
        cudaEventCreateWithFlags(
            reinterpret_cast<cudaEvent_t *>(&tma_events[i]),
            cudaEventDisableTiming);
    }

    fprintf(stderr, "hybrid_orchestrator: ready (%u layers, rank=%u)\n",
            n_layers_, kv_lora_rank_);
    return true;
}

void hybrid_orchestrator::free_all() {
    pool.free_all();
    for (int i = 0; i < 2; i++) {
        if (tma_events[i]) {
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(tma_events[i]));
            tma_events[i] = nullptr;
        }
    }
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
    if (tma_head < 2) return true; // nothing to wait for yet
    cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(tma_events[(tma_head - 2) & 1]);
    // non-blocking probe
    for (int spin = 0; spin < 64; spin++) {
        cudaError_t err = cudaEventQuery(ev);
        if (err == cudaSuccess) return true;
        if (err != cudaErrorNotReady) return false;
        __builtin_ia32_pause();
    }
    // fallback to blocking sync
    cudaError_t err = cudaEventSynchronize(ev);
    return err == cudaSuccess;
}

void hybrid_orchestrator::on_gpu_attn_done(uint32_t layer) {
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
    int patched = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_FLASH_ATTN_EXT &&
            ggml_backend_supports_op(gpu_backend, node))
        {
            ggml_backend_sched_set_tensor_backend(sched, node, gpu_backend);
            patched++;
        }
    }
    if (patched > 0) {
        fprintf(stderr, "hybrid: patched %d flash-attn ops → GPU\n", patched);
    }
}

#else // !GGML_USE_CUDA

bool hybrid_orchestrator::init(
    uint32_t, uint32_t, uint32_t, uint32_t,
    ggml_backend_dev_t, void *, void *)
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

    if (n_match < draft.lookahead_buffer.size()) {
        llama_memory_seq_rm(
            llama_get_memory(ctx_tgt),
            0,
            static_cast<llama_pos>(current_token + n_match + 1),
            -1);
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
