# Session Summary: DeepSeek V3/R1 Three-Stage Pipeline

## Goal
Implement three-stage per-layer async pipeline for DeepSeek V3/R1 that dispatches QKV proj / flash_attn / MoE FFN across 2 CUDA streams with event synchronization and a persistent scratch buffer for inter-phase data.

## Constraints & Preferences
- Pipeline scheduler is DeepSeek-specific, lives in `src/pipeline-sched.{h,cpp}` (not `ggml/` or `src/models/`)
- Uses `ggml_backend_cuda_set_stream()` + `set_stream` function pointer for CUDA stream switching (caller provides hooks, avoids hard dependency on ggml-cuda.h)
- Persistent scratch buffer (separate from gallocr-managed main buffer) bridges data between phases across scheduler reset boundaries
- Three separate `ggml_cgraph *` per layer (one per phase), built at init time, dispatched at runtime via `ggml_backend_sched_graph_compute_async`
- Integration point is `process_ubatch()` in `llama-context.cpp` (not `server-context.cpp`) — the actual full compute flow: `update_slots` → `llama_decode` → `process_ubatch`
- `llama_pipeline_setup` struct used to extract pipeline-relevant tensors from graph builder before `llm_graph_context` is destroyed
- All input data (embd, pos, DSA masks/indices) must be copied to persistent scratch before pipeline compute — DSA tensors in `res->get_ctx()` are recreated after `res->reset()` + `build_graph()` at new addresses, invalidating pipeline graph references
- Pipeline init must happen INSIDE `build_graph()` because `llm_graph_context` (which holds the graph builder context with tensor metadata) is destroyed when `build_graph()` returns, but its DSA input objects and tensors live in `llm_graph_result` which is owned by the caller and persists

## Progress
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

### Done (new: Task 5 - Expert Prefetch)
- **Pipeline struct additions**: `has_moe[]` per-layer MoE indicator, `prefetch_gate/up/down` GPU destination tensors, `model_gate_exps/up_exps/down_exps[]` original weight refs, `e_prefetch_done` completion event, `prefetch_fn` callback with user_data
- **Init**: Allocates GPU prefetch buffers (sized for `n_expert_used` experts × 3 weight types), saves model weight tensor references, detects MoE layers, rebuilds phase C graphs for MoE layers using prefetch buffer tensors instead of model weight tensors
- **Compute loop**: After QKV phase (stream 0), calls `prefetch_fn` with GPU top_k tensor (from DSA indexer), model weight tensor pointers, prefetch destination buffers, and completion event. The callback runs on a dedicated transfer stream concurrently with flash_attn (stream 1). Before FFN phase, event synchronization ensures prefetch is complete.
- **Free**: Cleans up `prefetch_buf`, `e_prefetch_done`
- **Callback design**: `llama_pipeline_expert_prefetch_fn` receives `dst_tensors[3]` (GPU), `src_tensors[3]` (CPU), `slice_bytes[3]`, `top_k` GPU tensor, layer index, completion event, user_data. The callback handles GPU top_k readback via `cudaMemcpyDeviceToHost` and async H2D copies of selected expert slices. This separation lets the pipeline scheduler (C++ file) orchestrate timing while the CUDA implementation handles device-specific operations.
- **API**: `llama_pipeline_sched_init()` new `prefetch_fn` / `prefetch_user_data` parameters. New setter: `llama_pipeline_sched_set_prefetch_fn()` for post-init configuration. All guarded by `#ifdef LLAMA_DEEPSEEK_PIPELINE` in integration code.
- **`build_phase_c()`**: Updated signature with optional `prefetch_gate/up/down` tensor overrides. When provided, MoE weight tensors point to GPU-resident prefetch buffers instead of CPU model weights. Scheduler sees GPU-resident tensors and skips its own copy.

### Blocked
- Cannot run cmake to verify compilation (cmake not in PATH on this machine)
- Prefetch callback implementation requires CUDA code (`cudaMemcpyDeviceToHost` for top_k readback, `cudaMemcpyAsync` for H2D copies) - must be provided by CUDA-capable caller via `llama_pipeline_sched_set_prefetch_fn()`

## Key Decisions
- **Persistent DSA scratch tensors, not original DSA references**: After `res->reset()` + `build_graph()`, new DSA tensor objects are created in `res->get_ctx()` at different addresses. Pipeline phase graphs (built at init) have stale tensor pointers. Solution: create persistent scratch DSA tensors, copy data via `ggml_backend_tensor_copy()` each decode step, and reference scratch tensors via a persistent `inp_attn_persist` object.
- **Pipeline init inside `build_graph()`**: The `llm_graph_context` is destroyed when `build_graph()` returns, but the pipeline init needs the live context to access tensor metadata. Solution: add `llama_pipeline_sched **` out-param to `build_graph()`, create pipeline inside the function while `llm` is alive, and return the pointer. The pipeline survives because it manages its own memory, ggml_contexts, and graphs independent of the now-destroyed `llm_graph_context`.
- **`ctx0` = `res->get_ctx()` persistence**: Tensors created in `ctx0` persist because `ctx0` is `res->ctx_compute.get()`, not a temporary context. But `res->reset()` creates a new `ctx_compute` and clears `inputs` (destroying old DSA objects), so the tensor pointers change.
- **`ggml_gallocr_is_allocated()` skips tensors with `data != NULL`**: Input tensors in CPU buffer survive scheduler reset because their `data` pointer stays set. But GPU-buffer input tensors (like `inpL_embd`) can be overwritten by new gallocr allocations — must copy to persistent scratch.
- **Three separate sub-graphs per layer**: QKV→flash_attn→FFN dependency chain is serial, not fork-join. Three explicit graphs are required.
- **`ggml_cpy` bridges phase outputs to scratch**: Each phase ends with `ggml_cpy` from gallocr-managed result to persistent scratch tensor.
- **Synchronize before every reset**: `ggml_backend_sched_synchronize()` required before each `reset()` to prevent main-buffer memory reuse races between phases. Currently makes all phases sequential — stream parallelism needs separate gallocr instances or a single pre-allocated graph.
- **Output head as separate post-pipeline graph**: Built on-the-fly after pipeline compute, uses same scheduler, writes to `res->t_logits`.
- **Saved pointers in pipeline struct**: `saved_inp_pos` and `saved_inp_attn_dsa` are set during init and used by `copy_inputs` to reference the original main-graph tensors. Valid for graph reuse case (same `res`, same `inputs`); re-set on graph rebuild (new `inputs`, pipeline re-init).

## Next Steps
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
- `src/pipeline-sched.h` — pipeline scheduler struct + API (now includes DSA scratch tensors, `inp_attn_persist`, `saved_inp_pos`, `saved_inp_attn_dsa`, output head fields)
- `src/pipeline-sched.cpp` — fully rewritten: persistent DSA tensors, `copy_inputs` with DSA copy, output head, proper free, pointer saving in init
- `src/models/deepseek32.cpp` — graph constructor saves pipeline tensors as members
- `src/models/models.h:1033-1043` — `llama_model_deepseek32::graph` has `m_pipe_*` fields + `get_pipeline_setup()` override
- `src/llama-graph.h` — `struct llama_pipeline_setup`, virtual `get_pipeline_setup()` on `llm_graph_context`
- `src/llama-cparams.h` — `bool deepseek_pipeline` flag
- `src/llama-model.h:629` — `build_graph()` signature with `llama_pipeline_setup *` + `llama_pipeline_sched **` params
- `src/llama-model.cpp:2132` — `build_graph()` with pipeline init inside (guarded by `LLAMA_DEEPSEEK_PIPELINE`)
- `src/llama-context.h` — `#include "pipeline-sched.h"`, `struct llama_pipeline_sched * deepseek_pipeline` field
- `src/llama-context.cpp` — `process_ubatch()` integration: pass `&deepseek_pipeline` to `build_graph()`, `copy_inputs`/`compute`/`compute_output_head` block (guarded by `LLAMA_DEEPSEEK_PIPELINE`)
- `ggml/src/ggml-backend.cpp:1823-1903` — `sched_reset`/`sched_alloc_graph`/`graph_compute_async` internals
- `ggml/include/ggml-backend.h:71,116` — `ggml_backend_tensor_copy` / `copy_async` declarations
