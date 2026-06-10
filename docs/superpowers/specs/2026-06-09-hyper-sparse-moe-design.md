# Hyper-Sparse MoE: Dynamic Expert Skipping for EPYC+5090 Hybrid Pipeline

**Date:** 2026-06-09
**Models:** DeepSeek-V3 (256 experts, top-8), Kimi-K2.6 (576 experts, top-9)
**Hardware:** AMD EPYC Turin (DDR5) + RTX 5090 (sm_90+, TMA capable)

## Problem

The DDR5 bandwidth wall on EPYC Turin limits inference throughput to ~8.15 t/s for a 1T MoE model. MoE FFN accounts for >90% of DDR5 traffic (loading expert weights). Standard top-k routing loads all model weights for the top-k experts per token, even when the routing distribution is highly concentrated on a small subset.

## Solution: Decoupled Routing + Cumulative Weight Thresholding

Move MoE routing from Phase C (FFN) to Phase A (QKV), compute a ubatch-level cumulative weight threshold, skip low-contribution experts entirely, and renormalize remaining expert weights.

### Architecture

```
Phase A (stream 0):
  QKV proj ──→ scratch.QKV ──→ signal(cda_qkv)
  MoE route ─→ scratch.moe_ids (cached expert IDs)
             → scratch.moe_weights (cached weights, before renormalization)
             → cumulative_threshold_kernel (GPU, 1 warp)
               → scratch.expert_mask (skip bitmask, uint64_t[])
               → p->remap_tensor (original ID → compact slot)
               → renormalize moe_weights in-place
             → signal(cda_moe)

Prefetch (stream 2):
  wait(cda_moe)
  read expert_mask + remap_tensor (D2H, < 72 bytes)
  for each non-skipped expert e:
    TMA/cudaMemcpyAsync(dst + remap[e] * sz, src + e * sz, sz, H2D)
  signal(e_prefetch_done)

Phase B (stream 1):
  wait(cda_qkv) → flash_attn(argmax(top_k)) → scratch.ffn_inp
                                               signal(cda_attn)

Phase C (stream 0):
  wait(cda_attn)
  wait(e_prefetch_done)   (if prefetch active)
  remapped_ids = get_rows(p->remap_tensor, scratch.moe_ids)
  FFN(up/gate/down, cur, remapped_ids, scratch.moe_weights)
    (no routing recomputed — cached from Phase A)
```

### Key Data Structures

```cpp
struct llama_layer_scratch {
    // ... existing QKV/attn scratch ...
    struct ggml_tensor * moe_ids;       // [n_expert_used, n_tokens] i32
    struct ggml_tensor * moe_weights;   // [n_expert_used, n_tokens] f32
    struct ggml_tensor * expert_mask;   // byte buffer, ceil(n_expert/64)*8 bytes
    struct ggml_tensor * moe_remap;     // [n_expert] i32 (per-slot, written by threshold)
    struct ggml_tensor * moe_kept_count; // [1] i32
};
```

### Cumulative Threshold Algorithm (GPU, 1 block × 32 threads)

1. Load per-expert contribution totals from scratch.moe_weights across ubatch
2. Bitonic sort experts by total weight descending (shared memory)
3. Cumulative sum until 0.95 of total weight reached (min floor=3 experts)
4. Build skip_mask: bit=1 → skip this expert
5. Build remap table: kept experts → contiguous slots 0..kept_count-1
6. Renormalize kept-expert weights per-token: w'[k] = w[k] / Σ_kept w[k]

### Prefetch Changes

- Buffer increased: 16 → 128 slots (`LLAMA_PIPELINE_PREFETCH_MAX_EXPERTS`)
- Compact buffer: slot 0 = first kept expert, slot 1 = second, etc.
- MUL_MAT_ID uses remapped IDs (via ggml_get_rows) to index compact slots
- TMA preferred for H2D on RTX 5090 (sm_90+), fallback cudaMemcpyAsync

### Phase C FFN with Cached Routing

New `build_moe_ffn` overload:
```cpp
ggml_tensor * build_moe_ffn(cur, up_exps, gate_exps, down_exps,
    n_expert, n_expert_used, type_op, il,
    cached_ids, cached_weights, ...);
```
- Skips: gate_inp matmul, softmax, argsort_top_k, weight extraction, norm
- Reads cached IDs (already remapped to compact slots) and renormalized weights
- Gate/up/down MMID → activation → down MMID → mul_by_weights → sum

### Performance Expectations

| Metric | DeepSeek-V3 (256×8) | Kimi-K2.6 (576×9) |
|--------|---------------------|--------------------|
| Expected skip rate | 40-55% | 50-70% |
| DDR5 bandwidth saved | 3-4 GB/s of ~8 GB/s | 4-5.5 GB/s |
| Estimated speedup | 1.4-1.8x | 1.7-2.5x |
| Additional GPU compute | <1% of Phase A | <1% of Phase A |
| Additional GPU memory | 64KB per slot | 288KB per slot |

### Risk Mitigation

- **Routing quality**: Using pre-attention hidden state (ffn-normed) instead of post-attention — attention delta is small (<10% of hidden state)
- **Cumulative error**: Per-token renormalization preserves Σ weights = 1.0 → no distribution shift
- **Prefetch overflow**: If kept_count > 128, fallback to the original 16-slot prefetch for overflow
- **Cold start**: First ubatch uses full routing (no threshold) to warm up stats

### Synchronization

| Barrier | Signal at | Wait at | Purpose |
|---------|-----------|---------|---------|
| `cda_qkv[L]` | Phase A QKV done | Phase B start | QKV scratch ready |
| `cda_moe[L]` | Phase A threshold done | Prefetch start | moe_scratch ready |
| `cda_attn[L]` | Phase B done | Phase C start | ffn_inp ready |
| `e_prefetch_done` | Prefetch H2D done | Phase C FFN start | VRAM weights ready |
