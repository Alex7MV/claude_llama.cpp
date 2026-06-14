# Hardware Validation Checklist — Per-Layer Async Pipeline

## Prerequisites
- RTX 5090 (CUDA 12.8+)
- DeepSeek V3 model (GGUF format)
- Branch: `feat/hyper-sparse-v1`, tip: `f109e6390`

## Build
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

## Validation Steps

### 1. Compilation check
- [ ] Builds cleanly with `-DGGML_CUDA=ON`
- [ ] No warnings on `pipeline-prefetch.cu` or `llama-context.cpp`

### 2. Output correctness
- [ ] Set `LLAMA_LOG=1` and run with a short prompt
- [ ] Compare logits against Phase 3 (monolithic) baseline — values match within numerical precision

### 3. Token-level latency (vs Phase 3 baseline)
- [ ] Measure `--timings` with Phase 4 (current) on a 256-token generation
- [ ] Switch to Phase 3 by reverting `f109e6390` + `c3aa6fe5c` (or cherry-pick onto pre-Step-4 base)
- [ ] Compare `predict` time per token at same context length

**Expected:** Phase 4 should show token latency reduction proportional to H2D transfer overlap (typically 10–30% for MoE-heavy models).

### 4. Edge cases to test
- [ ] Single-token prompt (no warm-up prefetch layer)
- [ ] Very short context (1–2 tokens) — all skipped/expert mask edge paths
- [ ] `--no-warmup` flag behavior (triggers `warmup_did_prefetch == false` path)
- [ ] `--cont-batching` with 2+ concurrent sequences (different kv_cache slicing)

### 5. Diagnostics (if performance is worse than Phase 3)
If Phase 4 is slower, the likely culprit is CPU overhead from calling `build_graph` + `sched_reset` + `sched_alloc_graph` 58× per decode (~11ms total). Run with `--verbose` and check:

- Time spent in `build_graph` loop vs inside `graph_compute`
- Profiler trace showing gaps between GPU kernel launches (CPU scheduling bottleneck)

### 6. Potential follow-up optimizations
- **Per-layer graph caching:** If `sched_reset`/`sched_alloc_graph` overhead > 0.1ms each per layer, cache the per-layer `ggml_cgraph` objects and reuse them with `ggml_graph_update()` instead of rebuilding.
- **`set_inputs` skip for Phase 2:** Already implemented — verify it's actually saving the 57 dead memcpys (use nsys trace).

## Key files
| File | Role |
|------|------|
| `src/llama-context.cpp` (Phase 1a/1b/4 loop) | Main pipeline logic, all fix commits |
| `ggml/src/ggml-cuda/pipeline-prefetch.cu` | GPU skip-prefetch kernel, compilation fixes applied |
| `ggml/include/ggml-cuda.h` | Public API for skip-prefetch (event + host-mask params) |
| `src/models/kimi-linear.cpp` | Layer loop with `build_layer_only` guard |
