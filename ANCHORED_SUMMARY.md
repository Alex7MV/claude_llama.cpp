# Session Summary: DeepSeek V3/R1 Three-Stage Pipeline + Phase 2 MoE Graph Caching

## Current Focus
Implement non-intrusive, software-level graph caching / reuse for Phase 2 MoE so that token generation can skip `model.build_graph()`, `cascade_force_moe_consumers()`, and `ggml_backend_sched_alloc_graph()` on every token after the first.

## Goal
- Persist a valid Phase 2 `ggml_cgraph` across decode steps so the scheduler can replay the same graph topology without allocator work.
- Keep `ggml-backend` core unmodified; use a dedicated `sched_phase2` that is isolated from Phase 1 (attention/routing).
- Dynamic expert indices and sparse masks must update correctly each token without layout drift.
- Server counter `n_reused` must increment on cache hits.
- Target: unlock >10 t/s generation on DeepSeek V3/R1 class MoE models.

## Constraints & Preferences
- No changes inside `ggml/` backends; all work lives in `src/llama-context.cpp`, `src/moe-hijacker.{h,cpp}`, and `src/llama-context.h`.
- `sched_phase2` is created in `sched_reserve()` with a 10k node reserve and is separate from the main scheduler.
- On Token 2+: skip `build_graph()`, `cascade_force_moe_consumers()`, and `ggml_backend_sched_alloc_graph()`.
- Graph reuse is only legal for single-token (`ubatch.n_tokens == 1`) decode steps.
- Shallow capture must share tensor objects with the freshly-built graph so framework-owned backend metadata (`extra`, `buffer`, view offsets) stays consistent.

## Progress
### Done
- **Phase 2 MoE graph cache structure:**
  - `moe::phase2_graph_cache` stores a persistent `ggml_cgraph*` plus the `ggml_context` that owns it.
  - `moe::capture_phase2_graph()` uses `ggml_new_graph_custom()` to allocate a fully-initialized graph (valid `visited_hash_set`, `use_counts`, `nodes`/`leafs` arrays) while keeping the original tensor pointers.
  - `release()` frees only the cgraph context; tensor objects are owned by the framework and are not deleted.
- **Shallow capture implemented in `src/moe-hijacker.cpp`:**
  - Removed memberwise deep-copy of `ggml_tensor`.
  - Removed `tensor_map`, `find_by_name()`, and `owned_tensors`.
  - `phase2_graph_cache::remap()` is now an identity no-op because the same tensor objects are shared.
- **BUILD / REPLAY split in `src/llama-context.cpp:1858-2001`:**
  - **Path A (Token 1 BUILD):** creates `sched_phase2`, resets it, builds Phase 2 graph, captures a shallow persistent copy, runs `cascade_force_moe_consumers()` on the persistent graph, calls `ggml_backend_sched_alloc_graph()` once, then computes.
  - **Path B (Token 2+ REPLAY):** reuses `phase2_cache.persistent_gf`, calls `ggml_backend_sched_graph_compute_async()` only (no allocator, no rebuild), increments `n_reused`.
  - `phase2_cache.valid` is only set to `true` after a *single-token* BUILD; multi-token prompt builds leave the cache invalid, so the next generation token rebuilds with the correct shape.
- **Cascade placement fixed:** `cascade_force_moe_consumers()` now runs on the persistent graph so backend assignments are stored against the same tensor pointers seen by `ggml_backend_sched_alloc_graph()`.
- **Build verified:** `cmake --build build -j$(nproc)` completed successfully (target user repeated the build separately after source edits).

### In Progress
- Runtime verification with a real DeepSeek MoE model to confirm:
  - No `SIGFPE` / `SIGABRT` during Phase 2 compute.
  - `n_reused` increments after the first generation token.
  - Latency improves relative to the rebuild-every-token path.

### Blocked
- No real `.gguf` model present in the workspace tree, so runtime verification cannot proceed without the user providing a model path or test command.

## Key Decisions
- **Shallow capture, not tensor copy:** A memberwise copy of `ggml_tensor` destroys backend-private `extra`/`buffer`/view metadata and was the root cause of earlier `SIGFPE` in `ggml_graph_compute_thread`. By sharing tensor objects and copying only the `ggml_cgraph` structure, those internals remain exactly as `model.build_graph()` produced them.
- **Context-owned cgraph:** Manually allocating `ggml_cgraph` with `new char[]` leaves `visited_hash_set`/`use_counts` uninitialized, which `ggml_backend_sched_graph_compute_async()` reads. Using `ggml_new_graph_custom()` fixes this properly.
- **Cascade on the persistent graph:** Running `cascade_force_moe_consumers()` on the original transient graph stores backend keys against tensors that the persistent allocator never sees; switching to the persistent graph before cascade fixes the backend assignment mismatch.
- **Replay restricted to single-token graphs:** The persistent allocation is only valid for the exact shape used during BUILD. Setting `phase2_cache.valid = true` only inside `if (do_cuda)` (which is true only for `ubatch.n_tokens == 1`) prevents reusing a prompt-shaped graph for generation.
- **No remapping required:** Because tensor objects are shared, `res->inputs` and `res->t_logits`/`res->t_embd` already reference the same tensors as `phase2_cache.persistent_gf`.

## Remaining Risks in `src/llama-context.cpp` (as of latest edit)
1. **Dead remap block (lines ~1914-1988):** `phase2_cache.remap()` is an identity no-op, so the entire block is harmless but misleading. It adds per-token overhead *only* during the BUILD path and is a maintenance hazard if someone later reintroduces a non-identity remap.
2. **`force_idxs_to_cpu_p2()` timing:** On REPLAY this is called *after* allocation and before `set_inputs()`. Changing tensor backend assignments after the splits are locked should keep inputs CPU-resident; if `ggml_backend_sched` instead re-evaluates placement, persistent GPU buffers already allocated for those tensors could become mismatched.
3. **Static-buffer hijack after allocation:** `scan_and_update_snapshots()` overwrites `t->data` of hijacked MoE outputs with the pre-allocated static buffer address *after* `ggml_backend_sched_alloc_graph()`. The scheduler may have already allocated output buffers for those tensors; compute then writes to the static buffer. This is intentional but assumes the static buffer is addressable by the consumer backend and that the scheduler-allocated buffer is simply unused.
4. **Cache invalidation is narrow:** The cache is never invalidated by KV-cache defragmentation, `set_causal_attn`, LoRA adapter changes, context resize, or `graph_reuse_disable`. Any of these could change the Phase 2 graph topology while `phase2_cache.valid` stays `true`.
5. **`cascade_force_moe_consumers` op-code magic numbers:** Uses hard-coded op ids `37` and `2` for `ADD`/`VIEW`; if `ggml_op` enum shifts, the cascade will silently stop forcing GPU placement.
6. **No shape check for dynamic tensors:** Input tensors such as `out_ids` may vary in `ne[0]` with `n_outputs`. The persistent allocation uses the first token's shape; future single-token calls are assumed to have the same shape for all Phase 2 tensors.
7. **Shared tensor lifetime:** `phase2_cache.persistent_gf` references tensors allocated inside `res->get_ctx()`. Reuse depends on `res->reset()` being skipped on every REPLAY call. If any future code path calls `res->reset()` between tokens while the cache is valid, the persistent graph will hold dangling tensor pointers.

## Next Steps
1. Run a real DeepSeek MoE model (`llama-cli` or `llama-server`) and confirm:
   - `capture_phase2_graph` log line appears once.
   - `n_reused` starts incrementing after the first generated token.
   - No crash in `ggml_graph_compute_thread` / `ggml-cuda` fusion paths.
2. Measure decode t/s vs. the rebuild-every-token fallback.
3. Remove the dead remap block and the now-unnecessary `remap()` method once reuse is proven.
4. Add explicit cache invalidation triggers for `graph_reuse_disable`, KV-cache defrag, and adapter changes.
5. Replace magic op ids in `cascade_force_moe_consumers()` with named `GGML_OP_*` constants.

## Historical Pipeline Work
The previous session implemented the three-stage per-layer async pipeline. That work is preserved below for context.

---

## Goal (original pipeline)
Implement three-stage per-layer async pipeline for DeepSeek V3/R1 that dispatches QKV proj / flash_attn / MoE FFN across 2 CUDA streams with event synchronization and a persistent scratch buffer for inter-phase data.

## Constraints & Preferences (original pipeline)
- Pipeline scheduler is DeepSeek-specific, lives in `src/pipeline-sched.{h,cpp}` (not `ggml/` or `src/models/`)
- Uses `ggml_backend_cuda_set_stream()` + `set_stream` function pointer for CUDA stream switching (caller provides hooks, avoids hard dependency on ggml-cuda.h)
- Persistent scratch buffer (separate from gallocr-managed main buffer) bridges data between phases across scheduler reset boundaries
- Three separate `ggml_cgraph *` per layer (one per phase), built at init time, dispatched at runtime via `ggml_backend_sched_graph_compute_async`
- Integration point is `process_ubatch()` in `llama-context.cpp` (not `server-context.cpp`) — the actual full compute flow: `update_slots` → `llama_decode` → `process_ubatch`
- `llama_pipeline_setup` struct used to extract pipeline-relevant tensors from graph builder before `llm_graph_context` is destroyed
- All input data (embd, pos, DSA masks/indices) must be copied to persistent scratch before pipeline compute — DSA tensors in `res->get_ctx()` are recreated after `res->reset()` + `build_graph()` at new addresses, invalidating pipeline graph references
- Pipeline init must happen INSIDE `build_graph()` because `llm_graph_context` (which holds the graph builder context with tensor metadata) is destroyed when `build_graph()` returns, but its DSA input objects and tensors live in `llm_graph_result` which is owned by the caller and persists

## Progress (original pipeline)
### Done
- **Task 1 (`ggml_backend_cuda_set_stream`):** Declaration in `ggml-cuda.h:50-54`, implementation in `ggml-cuda.cu:4806-4817`
- **Task 2 (per-layer builder):** `llama_build_deepseek32_layer()` in `deepseek32.cpp:511-808`, declared in `models.h:1049-1059`
- **Task 3 (Pipeline Scheduler):** `pipeline-sched.h` + `pipeline-sched.cpp` — three phase graphs per layer, persistent scratch buffer, event-based dispatch, scheduler-reset-safe allocation
- **CMakeLists.txt:** added `pipeline-sched.cpp` and `../ggml/include` include path
- **Infrastructure:**
  - Added `bool deepseek_pipeline` flag to `llama_cparams` (`llama-cparams.h`)
  - Added `struct llama_pipeline_setup` and virtual `get_pipeline_setup()` accessor on `llm_graph_context` (`llama-graph.h`)
  - DeepSeek32 `graph` constructor saves pipeline tensors as member fields
  - Added `get_pipeline_setup()` override to `graph` in `models.h`
  - Modified `llama_model::build_graph()` to accept `llama_pipeline_setup *` + `llama_pipeline_sched **` out-params
- **Pipeline-sched fixes:**
  - `inpL_embd` now a persistent scratch tensor, added `scratch_inp_pos` for position tensor copy
  - Added `gf_output` / `ctx_output` for output head graph
  - Added `llama_pipeline_sched_copy_inputs()` with `src_dsa` parameter for DSA tensor copy
  - Added `llama_pipeline_sched_compute_output_head()` for post-pipeline output head
  - Added persistent `inp_attn_persist` DSA input object with scratch tensor pointers
  - Added DSA scratch tensors (`scratch_k_rot_lid`, `scratch_k_idxs_lid`, `scratch_kq_mask_lid`, `scratch_k_idxs_mla`, `scratch_kq_mask_mla`) in scratch buffer
  - `k_rot_lid` filled once at init via `ggml_backend_tensor_copy()` (stable Hadamard matrix)
  - Added `ctx_persist` ggml_context for DSA/metadata; `ctx_output` for output head graph; `inp_attn_persist` member
  - `llama_pipeline_sched_free()` now frees persistent DSA object and output head context
  - Added `ggml_backend_sched_synchronize()` before each `reset()` in compute loop for correctness
  - Added `saved_inp_pos` / `saved_inp_attn_dsa` fields for copy_inputs to reference original main-graph tensors
  - Pointer saving in `llama_pipeline_sched_init()` after all setup is done
- **llama-context.h changes:**
  - Added `#include "pipeline-sched.h"` (already present)
  - Added `struct llama_pipeline_sched * deepseek_pipeline = nullptr;` field to `llama_context`
- **llama-model.h:** Added `struct llama_pipeline_sched *` forward declaration and `p_pipeline` param to `build_graph()`
- **llama-model.cpp:** Added `#include "pipeline-sched.h"` (guarded by `LLAMA_DEEPSEEK_PIPELINE`); pipeline init now inside `build_graph()` where `llm_graph_context` is alive, creating the pipeline scheduler using the GPU backend from `params.sched`, the model reference, the live graph context, and extracted pipeline setup data
- **llama-context.cpp integration (process_ubatch):**
  - Passes `&deepseek_pipeline` to `build_graph()` for inline pipeline creation
  - All pipeline code guarded by `#ifdef LLAMA_DEEPSEEK_PIPELINE`
  - After `set_inputs()`, copies embed/pos/DSA data to persistent scratch via `copy_inputs`
  - Computes all layers via `llama_pipeline_sched_compute()`
  - Computes output head via `llama_pipeline_sched_compute_output_head()`
  - Variable scope issues resolved — no out-of-scope `pipe_setup_storage` references

### Done (Task 5 - Expert Prefetch)
- **Pipeline struct additions**: `has_moe[]` per-layer MoE indicator, `prefetch_gate/up/down` GPU destination tensors, `model_gate_exps/up_exps/down_exps[]` original weight refs, `e_prefetch_done` completion event, `prefetch_fn` callback with user_data
- **Init**: Allocates GPU prefetch buffers (sized for `n_expert_used` experts × 3 weight types), saves model weight tensor references, detects MoE layers, rebuilds phase C graphs for MoE layers using prefetch buffer tensors instead of model weight tensors
- **Compute loop**: After QKV phase (stream 0), calls `prefetch_fn` with GPU top_k tensor (from DSA indexer), model weight tensor pointers, prefetch destination buffers, and completion event. The callback runs on a dedicated transfer stream concurrently with flash_attn (stream 1). Before FFN phase, event synchronization ensures prefetch is complete.
- **Free**: Cleans up `prefetch_buf`, `e_prefetch_done`
- **Callback design**: `llama_pipeline_expert_prefetch_fn` receives `dst_tensors[3]` (GPU), `src_tensors[3]` (CPU), `slice_bytes[3]`, `top_k` GPU tensor, layer index, completion event, user_data. The callback handles GPU top_k readback via `cudaMemcpyDeviceToHost` and async H2D copies of selected expert slices. This separation lets the pipeline scheduler (C++ file) orchestrate timing while the CUDA implementation handles device-specific operations.
- **API**: `llama_pipeline_sched_init()` new `prefetch_fn` / `prefetch_user_data` parameters. New setter: `llama_pipeline_sched_set_prefetch_fn()` for post-init configuration. All guarded by `#ifdef LLAMA_DEEPSEEK_PIPELINE` in integration code.
- **`build_phase_c()`**: Updated signature with optional `prefetch_gate/up/down` tensor overrides. When provided, MoE weight tensors point to GPU-resident prefetch buffers instead of CPU model weights. Scheduler sees GPU-resident tensors and skips its own copy.

### Blocked (original pipeline)
- Cannot run cmake to verify compilation (cmake not in PATH on this machine)
- Prefetch callback implementation requires CUDA code (`cudaMemcpyDeviceToHost` for top_k readback, `cudaMemcpyAsync` for H2D copies) - must be provided by CUDA-capable caller via `llama_pipeline_sched_set_prefetch_fn()`

## Key Decisions (original pipeline)
- **Persistent DSA scratch tensors, not original DSA references**: After `res->reset()` + `build_graph()`, new DSA tensor objects are created in `res->get_ctx()` at different addresses. Pipeline phase graphs (built at init) have stale tensor pointers. Solution: create persistent scratch DSA tensors, copy data via `ggml_backend_tensor_copy()` each decode step, and reference scratch tensors via a persistent `inp_attn_persist` object.
- **Pipeline init inside `build_graph()`**: The `llm_graph_context` is destroyed when `build_graph()` returns, but the pipeline init needs the live context to access tensor metadata. Solution: add `llama_pipeline_sched **` out-param to `build_graph()`, create pipeline inside the function while `llm` is alive, and return the pointer. The pipeline survives because it manages its own memory, ggml_contexts, and graphs independent of the now-destroyed `llm_graph_context`.
- **`ctx0` = `res->get_ctx()` persistence**: Tensors created in `ctx0` persist because `ctx0` is `res->ctx_compute.get()`, not a temporary context. But `res->reset()` creates a new `ctx_compute` and clears `inputs` (destroying old DSA objects), so the tensor pointers change.
- **`ggml_gallocr_is_allocated()` skips tensors with `data != NULL`**: Input tensors in CPU buffer survive scheduler reset because their `data` pointer stays set. But GPU-buffer input tensors (like `inpL_embd`) can be overwritten by re-allocation — must copy to persistent scratch.
- **Three separate sub-graphs per layer**: QKV→flash_attn→FFN dependency chain is serial, not fork-join. Three explicit graphs are required.
- **`ggml_cpy` bridges phase outputs to scratch**: Each phase ends with `ggml_cpy` from gallocr-managed result to persistent scratch tensor.
- **Synchronize before every reset**: `ggml_backend_sched_synchronize()` required before each `reset()` to prevent main-buffer memory reuse races between phases. Currently makes all phases sequential — stream parallelism needs separate gallocr instances or a single pre-allocated graph.
- **Output head as separate post-pipeline graph**: Built on-the-fly after pipeline compute, uses same scheduler, writes to `res->t_logits`.
- **Saved pointers in pipeline struct**: `saved_inp_pos` and `saved_inp_attn_dsa` are set during init and used by `copy_inputs` to reference the original main-graph tensors. Valid for graph reuse case (same `res`, same `inputs`); re-set on graph rebuild (new `inputs`, pipeline re-init).

## Next Steps (original pipeline)
1. **Task 6: CDA Optimization** — Blackwell CDA barriers for sub-us sync
2. Provide CUDA prefetch implementation: write `ggml_backend_cuda_prefetch_experts()` in `ggml-cuda.cu` that reads DSA top_k from GPU, selects unique expert IDs, and launches `cudaMemcpyAsync` H2D copies on a dedicated transfer stream

## Critical Context
- `ggml_backend_sched_graph_compute_async()` skips allocation if `sched->is_alloc == true`. After each phase, `ggml_backend_sched_reset()` sets `is_alloc = false` so the next phase's graph gets properly allocated.
- `ggml_gallocr_init_tensor()` skips tensors with `data != NULL`. Scratch tensors have pre-set `data` in persistent buffer → gallocr never re-allocates them.
- **Both GPU-buffer AND ctx0-allocated input pointers can go stale**: GPU-buffer inputs (embd, pos) get overwritten by re-allocation. `res->get_ctx()` DSA tensors get recreated after `res->reset()`, changing their addresses. Both must be copied to persistent scratch.
- **`ggml_backend_tensor_copy()`**: Used for GPU-to-GPU or CPU-to-GPU copy from main-buffer input tensors to scratch buffers. Called between `set_inputs()` and first pipeline `compute()`.
- **`llm_graph_context` destroyed after `build_graph()` returns**: Pipeline init that needs the context must happen INSIDE `build_graph()`. Solution: `llama_pipeline_sched **` out-param.
- **`res->inputs` cleared by `res->reset()`**: DSA input objects are destroyed during graph rebuild. Pipeline re-init captures new DSA input. For graph reuse, the same `inputs` are preserved, so saved pointers remain valid.
- `ggml_dyn_tallocr_reset()` frees talloc chunks but does NOT free the underlying GPU buffer memory. However, `ggml_gallocr_reserve_n_impl()` calls `ggml_dyn_tallocr_reset()` then rebuilds chunks from scratch, potentially overlapping with old input tensor data locations.
- `set_input_kq_mask()` asserts `ggml_backend_buffer_is_host(dst->buffer)`. Standard CUDA backend buffer is NOT a host buffer, but input tensors are allocated on the CPU backend which IS host-accessible. Scratch buffers are regular GPU buffers — DSA data must be copied via `ggml_backend_tensor_copy()`, NOT by calling `set_input_kq_mask` on scratch tensors.

## Relevant Files
- `src/moe-hijacker.h` — `phase2_graph_cache`, capture/release API
- `src/moe-hijacker.cpp` — shallow capture via `ggml_new_graph_custom()`, cascade forcing, static buffer hijack
- `src/llama-context.h` — `sched_phase2`, `phase2_cache`, `n_reused`
- `src/llama-context.cpp:1816-2035` — Phase 2 BUILD/REPLAY split, `capture_phase2_graph`, `cascade_force_moe_consumers`, `scan_and_update_snapshots`
- `src/pipeline-sched.h` — pipeline scheduler struct + API (DSA scratch tensors, `inp_attn_persist`, `saved_inp_pos`, `saved_inp_attn_dsa`, output head fields)
- `src/pipeline-sched.cpp` — fully rewritten: persistent DSA tensors, `copy_inputs` with DSA copy, output head, proper free, pointer saving in init
- `src/models/deepseek32.cpp` — graph constructor saves pipeline tensors as members
- `src/models/models.h:1033-1043` — `llama_model_deepseek32::graph` has `m_pipe_*` fields + `get_pipeline_setup()` override
- `src/llama-graph.h` — `struct llama_pipeline_setup`, virtual `get_pipeline_setup()` on `llm_graph_context`
- `src/llama-cparams.h` — `bool deepseek_pipeline` flag
- `src/llama-model.h:629` — `build_graph()` signature with `llama_pipeline_setup *` + `llama_pipeline_sched **` params
- `src/llama-model.cpp:2132` — `build_graph()` with pipeline init inside (guarded by `LLAMA_DEEPSEEK_PIPELINE`)
- `ggml/src/ggml-backend.cpp:1823-1903` — `sched_reset`/`sched_alloc_graph`/`graph_compute_async` internals
- `ggml/include/ggml-backend.h:71,116` — `ggml_backend_tensor_copy` / `copy_async` declarations
