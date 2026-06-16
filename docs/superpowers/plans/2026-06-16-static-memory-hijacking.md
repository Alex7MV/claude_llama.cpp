# Static Memory Hijacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate CUDA Graph data corruption in MoE Phase 2 by hijacking tensor->data pointers to a static VRAM buffer, preserving 10.99 t/s throughput.

**Architecture:** Three CUDA-only structs (`phase2_hijack`, `phase2_inject`, `phase2_guard`) are added to `llama_context`. A 4.08 MB VRAM buffer is allocated once at model init, partitioned into 61 layers × 9 tensors at compile-time offsets. Before CUDA Graph capture, all 549 tensor->data pointers are hijacked; before each replay, they are restored and input data is injected via `cudaMemcpyAsync`.

**Tech Stack:** C++17, CUDA, ggml

**Spec:** `docs/superpowers/specs/2026-06-16-static-memory-hijacking-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `src/llama-context.h` | Struct definitions + context members | ~30 lines |
| `src/llama-context.cpp` | Init/destroy + process_ubatch integration + cgraph scanner | ~130 lines |

---

### Task 1: Add `phase2_hijack` struct definition to header

**Files:**
- Modify: `src/llama-context.h`

- [ ] **Step 1: Add struct definition after `llama_moe_weight_cache` closing brace**

Insert the following between line 109 (`};` closing `llama_moe_weight_cache`) and line 110 (blank line before `struct llama_context`):

```cpp
#ifdef GGML_USE_CUDA

#define H2_FFN_DIM       2048
#define H2_N_EMBD        7168
#define H2_ALIGN         256
#define H2_ALIGN_UP(x)   (((x) + (H2_ALIGN - 1)) & ~(H2_ALIGN - 1))

#define H2_SZ_INP        H2_ALIGN_UP(H2_N_EMBD * 2)         // 14,336
#define H2_SZ_IDS        H2_ALIGN_UP(8 * 4)                 // 256
#define H2_SZ_WEIGHTS    H2_ALIGN_UP(8 * 4)                 // 256
#define H2_SZ_RESIDUAL   H2_ALIGN_UP(H2_N_EMBD * 2)         // 14,336
#define H2_SZ_GATE       H2_ALIGN_UP(H2_FFN_DIM * 2)        // 4,096
#define H2_SZ_UP         H2_ALIGN_UP(H2_FFN_DIM * 2)        // 4,096
#define H2_SZ_SILU       H2_ALIGN_UP(H2_FFN_DIM * 2)        // 4,096
#define H2_SZ_DOWN       H2_ALIGN_UP(H2_N_EMBD * 2)         // 14,336
#define H2_SZ_OUT        H2_ALIGN_UP(H2_N_EMBD * 2)         // 14,336

#define H2_OFF_INP       0x00000
#define H2_OFF_IDS       0x03800
#define H2_OFF_WEIGHTS   0x03900
#define H2_OFF_RESIDUAL  0x03A00
#define H2_OFF_GATE      0x07200
#define H2_OFF_UP        0x08200
#define H2_OFF_SILU      0x09200
#define H2_OFF_DOWN      0x0A200
#define H2_OFF_OUT       0x0DA00

#define H2_LAYER_STRIDE  0x11200   // 70,144 bytes
#define H2_N_LAYERS      61

struct phase2_hijack {
    static constexpr int N_TENSOR_TYPES = 9;
    static constexpr int MAX_SNAPSHOTS = H2_N_LAYERS * N_TENSOR_TYPES; // 549

    struct entry {
        ggml_tensor * t;
        void        * static_addr;
    };

    void * base = nullptr;
    entry  snapshots[MAX_SNAPSHOTS];
    int    snapshot_count = 0;
    bool   captured = false;
    cudaGraphExec_t cuda_graph_exec = nullptr;

    void* addr(int il, size_t off) {
        return (char*)base + il * H2_LAYER_STRIDE + off;
    }

    void init() {
        cudaMalloc(&base, H2_N_LAYERS * H2_LAYER_STRIDE);
    }

    void destroy() {
        if (cuda_graph_exec) { cudaGraphExecDestroy(cuda_graph_exec); cuda_graph_exec = nullptr; }
        if (base)           { cudaFree(base); base = nullptr; }
    }

    void hijack_one(ggml_tensor * t, void * addr) {
        assert(snapshot_count < MAX_SNAPSHOTS);
        snapshots[snapshot_count].t           = t;
        snapshots[snapshot_count].static_addr = addr;
        t->data = addr;
        snapshot_count++;
    }

    void restore_all() {
        for (int i = 0; i < snapshot_count; i++)
            snapshots[i].t->data = snapshots[i].static_addr;
        asm volatile("" ::: "memory");
    }

    // Walk Phase 2 cgraph, hijack all matching tensors into static buffer.
    // Called once during capture (after alloc_graph, before cudaStreamBeginCapture).
    void scan_and_hijack(ggml_cgraph * gf);

    // Walk Phase 2 cgraph (rebuilt each replay), update snapshot entries
    // with current tensor pointers. Followed by restore_all().
    // Called during replay (after alloc_graph, before cudaMemcpyAsync).
    void scan_and_update_snapshots(ggml_cgraph * gf);
};

struct phase2_inject {
    void * host_ffn_inp[H2_N_LAYERS];

    void init() {
        for (int i = 0; i < H2_N_LAYERS; i++)
            cudaHostAlloc(&host_ffn_inp[i], H2_SZ_INP, cudaHostAllocDefault);
    }

    void destroy() {
        for (int i = 0; i < H2_N_LAYERS; i++)
            if (host_ffn_inp[i]) { cudaFreeHost(host_ffn_inp[i]); host_ffn_inp[i] = nullptr; }
    }

    void fill_layer(int il, const void * ffn_inp_src) {
        memcpy(host_ffn_inp[il], ffn_inp_src, H2_SZ_INP);
    }

    void inject_all(const phase2_hijack & hijack, cudaStream_t stream) {
        for (int il = 0; il < H2_N_LAYERS; il++) {
            cudaMemcpyAsync(hijack.addr(il, H2_OFF_INP),     host_ffn_inp[il], H2_SZ_INP,     cudaMemcpyHostToDevice, stream);
            // moe_ids and moe_weights are small (32 bytes each) — injected from stack
        }
    }

    int    host_moe_ids[H2_N_LAYERS][8];
    float  host_moe_w[H2_N_LAYERS][8];

    void inject_ids_and_weights(int il, const int * ids, const float * w) {
        memcpy(host_moe_ids[il], ids, H2_SZ_IDS);
        memcpy(host_moe_w[il],   w,   H2_SZ_WEIGHTS);
    }

    void inject_ids_and_weights_all(const phase2_hijack & hijack, cudaStream_t stream) {
        for (int il = 0; il < H2_N_LAYERS; il++) {
            cudaMemcpyAsync(hijack.addr(il, H2_OFF_IDS),     host_moe_ids[il], H2_SZ_IDS,     cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(hijack.addr(il, H2_OFF_WEIGHTS), host_moe_w[il],   H2_SZ_WEIGHTS, cudaMemcpyHostToDevice, stream);
        }
    }
};

struct phase2_guard {
    cudaEvent_t phase1_done_event = nullptr;

    void init() { cudaEventCreate(&phase1_done_event); }
    void destroy() { if (phase1_done_event) { cudaEventDestroy(phase1_done_event); phase1_done_event = nullptr; } }

    void record(cudaStream_t stream) {
        cudaEventRecord(phase1_done_event, stream);
    }
};

#endif // GGML_USE_CUDA
```

- [ ] **Step 2: Verify header compiles**

```bash
cmake --build build --target llama 2>&1 | Select-String -Pattern "error"
```

Expected: no errors from `llama-context.h`.

- [ ] **Step 3: Commit**

```bash
git add src/llama-context.h
git commit -m "feat: add phase2_hijack, phase2_inject, phase2_guard structs to header"
```

---

### Task 2: Add struct members to `llama_context` and declare `h2_init()`

**Files:**
- Modify: `src/llama-context.h`

- [ ] **Step 1: Add members inside `#ifdef LLAMA_DEEPSEEK_PIPELINE` block in `llama_context`**

Insert after line 418 (`void init_moe_weight_cache();`) and before the `#endif` at line 419:

```cpp
#ifdef GGML_USE_CUDA
    phase2_hijack   h2_hijack;
    phase2_inject   h2_inject;
    phase2_guard    h2_guard;

    void h2_init();
    void h2_destroy();
#endif
```

- [ ] **Step 2: Verify header compiles**

```bash
cmake --build build --target llama 2>&1 | Select-String -Pattern "error"
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/llama-context.h
git commit -m "feat: add h2_hijack/h2_inject/h2_guard members and h2_init/h2_destroy to llama_context"
```

---

### Task 3: Implement `h2_init()` and `h2_destroy()` + cgraph scanner

**Files:**
- Modify: `src/llama-context.cpp`

- [ ] **Step 1: Add `#include <cassert>` if not already present**

Check if `#include <cassert>` exists near line 34-38. If not, add it:

```cpp
#include <cassert>
```

- [ ] **Step 2: Add CUDA forward declarations for cudaMalloc/cudaFree/cudaHostAlloc/cudaFreeHost/cudaEventCreate/cudaEventDestroy/cudaEventRecord**

After the existing `#ifdef GGML_USE_CUDA` block at lines 40-54, add:

```cpp
// Static memory hijacking: additional CUDA API forward declarations
extern "C" {
    int cudaMalloc(void ** devPtr, size_t size);
    int cudaFree(void * devPtr);
    int cudaHostAlloc(void ** pHost, size_t size, unsigned int flags);
    int cudaFreeHost(void * ptr);
    int cudaEventCreate(void ** event);
    int cudaEventDestroy(void * event);
    int cudaEventRecord(void * event, void * stream);
}
constexpr int cudaMemcpyHostToDevice = 1;
constexpr int cudaHostAllocDefault   = 0;
```

- [ ] **Step 3: Implement `h2_init()` — call from constructor**

Add function definition before `llama_context::llama_context()` constructor (around line 70):

```cpp
#ifdef GGML_USE_CUDA
void llama_context::h2_init() {
    h2_hijack.init();
    h2_inject.init();
    h2_guard.init();
}

void llama_context::h2_destroy() {
    h2_hijack.destroy();
    h2_inject.destroy();
    h2_guard.destroy();
}
#endif
```

- [ ] **Step 4: Call `h2_init()` at end of constructor**

In `llama_context::llama_context()` constructor, find the closing brace (~line 434). Before the closing brace, add:

```cpp
#ifdef LLAMA_DEEPSEEK_PIPELINE
#ifdef GGML_USE_CUDA
    h2_init();
#endif
#endif
```

- [ ] **Step 5: Call `h2_destroy()` in destructor**

In `llama_context::~llama_context()`, after the `moe_weight_cache` cleanup block (after line 477, the `#endif` for `LLAMA_DEEPSEEK_PIPELINE`), add:

```cpp
#ifdef LLAMA_DEEPSEEK_PIPELINE
#ifdef GGML_USE_CUDA
    h2_destroy();
#endif
#endif
```

- [ ] **Step 6: Implement `scan_and_hijack()` — cgraph walker for capture**

Add after the `h2_init()`/`h2_destroy()` definitions:

```cpp
void phase2_hijack::scan_and_hijack(ggml_cgraph * gf) {
    snapshot_count = 0;
    int layer_idx = 0;
    int tensor_pos = 0; // 0..8 within current layer

    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * dst = ggml_graph_node(gf, i);

        // Identify tensor by operation type and dst ne[] dimensions.
        // Phase 2 graph order per layer (for n_tokens=1, deterministic):
        //
        //   ffn_inp     INPUT,  ne=[7168, 1],   F16
        //   moe_ids     INPUT,  ne=[8, 1],      I32
        //   moe_weights INPUT,  ne=[8, 1],      F32
        //   residual    INPUT,  ne=[7168, 1],   F16
        //   gate_out    MUL_MAT_ID, ne=[2048, X], F16  (X depends on expert batching)
        //   up_out      MUL_MAT_ID, ne=[2048, X], F16
        //   silu_mul    MUL,        ne=[2048, X], F16  (SILU(gate)*up)
        //   down_out    MUL_MAT_ID, ne=[7168, 1], F16
        //   embd_out    result (last in layer),  ne=[7168, 1], F16

        if (layer_idx >= H2_N_LAYERS) break;

        int64_t ne0 = dst->ne[0];

        // ---- ffn_inp: leaf INPUT with n_embd ----
        if (tensor_pos == 0 && (dst->flags & GGML_TENSOR_FLAG_INPUT) && ne0 == H2_N_EMBD && dst->type == GGML_TYPE_F16) {
            hijack_one(dst, addr(layer_idx, H2_OFF_INP));
            tensor_pos++; continue;
        }

        // ---- moe_ids: leaf INPUT with n_expert_used ----
        if (tensor_pos == 1 && (dst->flags & GGML_TENSOR_FLAG_INPUT) && ne0 == 8 && dst->type == GGML_TYPE_I32) {
            hijack_one(dst, addr(layer_idx, H2_OFF_IDS));
            tensor_pos++; continue;
        }

        // ---- moe_weights: leaf INPUT ----
        if (tensor_pos == 2 && (dst->flags & GGML_TENSOR_FLAG_INPUT) && ne0 == 8 && dst->type == GGML_TYPE_F32) {
            hijack_one(dst, addr(layer_idx, H2_OFF_WEIGHTS));
            tensor_pos++; continue;
        }

        // ---- residual: leaf INPUT with n_embd (second occurrence after moe_weights) ----
        if (tensor_pos == 3 && (dst->flags & GGML_TENSOR_FLAG_INPUT) && ne0 == H2_N_EMBD && dst->type == GGML_TYPE_F16) {
            hijack_one(dst, addr(layer_idx, H2_OFF_RESIDUAL));
            tensor_pos++; continue;
        }

        // ---- gate_out: MUL_MAT_ID with ne[0] == ffn_dim ----
        if (tensor_pos == 4 && dst->op == GGML_OP_MUL_MAT_ID && ne0 == H2_FFN_DIM) {
            hijack_one(dst, addr(layer_idx, H2_OFF_GATE));
            tensor_pos++; continue;
        }

        // ---- up_out: MUL_MAT_ID with ne[0] == ffn_dim (second occurrence) ----
        if (tensor_pos == 5 && dst->op == GGML_OP_MUL_MAT_ID && ne0 == H2_FFN_DIM) {
            hijack_one(dst, addr(layer_idx, H2_OFF_UP));
            tensor_pos++; continue;
        }

        // ---- silu_mul: MUL where one src is SILU node ----
        if (tensor_pos == 6 && dst->op == GGML_OP_MUL && ne0 == H2_FFN_DIM) {
            hijack_one(dst, addr(layer_idx, H2_OFF_SILU));
            tensor_pos++; continue;
        }

        // ---- down_out: MUL_MAT_ID with ne[0] == n_embd ----
        if (tensor_pos == 7 && dst->op == GGML_OP_MUL_MAT_ID && ne0 == H2_N_EMBD) {
            hijack_one(dst, addr(layer_idx, H2_OFF_DOWN));
            tensor_pos++; continue;
        }

        // ---- embd_out: last non-INPUT in layer, ne[0] == n_embd ----
        if (tensor_pos == 8 && !(dst->flags & GGML_TENSOR_FLAG_INPUT) && ne0 == H2_N_EMBD) {
            hijack_one(dst, addr(layer_idx, H2_OFF_OUT));
            tensor_pos = 0;
            layer_idx++;
        }
    }
}

void phase2_hijack::scan_and_update_snapshots(ggml_cgraph * gf) {
    int idx = 0;
    int layer_idx = 0;
    int tensor_pos = 0;

    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * dst = ggml_graph_node(gf, i);
        if (layer_idx >= H2_N_LAYERS) break;

        int64_t ne0 = dst->ne[0];
        bool match = false;

        if      (tensor_pos == 0 && (dst->flags & GGML_TENSOR_FLAG_INPUT) && ne0 == H2_N_EMBD && dst->type == GGML_TYPE_F16) match = true;
        else if (tensor_pos == 1 && (dst->flags & GGML_TENSOR_FLAG_INPUT) && ne0 == 8 && dst->type == GGML_TYPE_I32) match = true;
        else if (tensor_pos == 2 && (dst->flags & GGML_TENSOR_FLAG_INPUT) && ne0 == 8 && dst->type == GGML_TYPE_F32) match = true;
        else if (tensor_pos == 3 && (dst->flags & GGML_TENSOR_FLAG_INPUT) && ne0 == H2_N_EMBD && dst->type == GGML_TYPE_F16) match = true;
        else if (tensor_pos == 4 && dst->op == GGML_OP_MUL_MAT_ID && ne0 == H2_FFN_DIM) match = true;
        else if (tensor_pos == 5 && dst->op == GGML_OP_MUL_MAT_ID && ne0 == H2_FFN_DIM) match = true;
        else if (tensor_pos == 6 && dst->op == GGML_OP_MUL && ne0 == H2_FFN_DIM) match = true;
        else if (tensor_pos == 7 && dst->op == GGML_OP_MUL_MAT_ID && ne0 == H2_N_EMBD) match = true;
        else if (tensor_pos == 8 && !(dst->flags & GGML_TENSOR_FLAG_INPUT) && ne0 == H2_N_EMBD) { match = true; layer_idx++; }

        if (match && idx < snapshot_count) {
            snapshots[idx].t = dst;
            idx++;
            tensor_pos = (tensor_pos == 8) ? 0 : tensor_pos + 1;
        }
    }
}
```

- [ ] **Step 4: Verify build compiles**

```bash
cmake --build build --target llama 2>&1 | Select-String -Pattern "error"
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/llama-context.cpp
git commit -m "feat: implement h2_init/h2_destroy with cgraph scanner for phase2 hijack"
```

---

### Task 4: Modify CAPTURE path in `process_ubatch()`

**Files:**
- Modify: `src/llama-context.cpp` (around lines 1750-1781)

- [ ] **Step 1: Add hijack call before `cudaStreamBeginCapture`**

In the capture block (line 1750-1781), after `ggml_backend_sched_alloc_graph` (line 1747) and `res->set_inputs` (line 1748), and BEFORE `cudaStreamBeginCapture` (line 1755), insert:

```cpp
                            // Hijack tensor->data to static buffer addresses BEFORE capture
                            h2_hijack.scan_and_hijack(phase2_gf);
```

- [ ] **Step 2: Save cuda_graph_exec into h2_hijack instead of moe_weight_cache**

Replace line 1762:
```cpp
                                            moe_weight_cache.cuda_graph_exec = (void*)ex;
```
with:
```cpp
                                            h2_hijack.cuda_graph_exec = ex;
                                            h2_hijack.captured = true;
```

Replace line 1763 (remove the redundant `moe_weight_cache.cuda_graph_captured = true` since it's no longer the source of truth):
```cpp
                                            moe_weight_cache.cuda_graph_captured = true;
```
→ Delete this line.

- [ ] **Step 3: Verify build compiles**

```bash
cmake --build build --target llama 2>&1 | Select-String -Pattern "error"
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/llama-context.cpp
git commit -m "feat: hijack tensor data to static buffer before cudaGraphBeginCapture"
```

---

### Task 5: Modify REPLAY path in `process_ubatch()`

**Files:**
- Modify: `src/llama-context.cpp` (around lines 1716-1736)

- [ ] **Step 1: Replace replay block to use h2_hijack + h2_inject + h2_guard**

Replace the entire replay block (lines 1716-1736):

```cpp
                    if (do_cuda && h2_hijack.captured) {
                        // REPLAY — build graph fresh, restore tensor addresses,
                        // inject input data from host, launch CUDA graph.

                        // Build Phase 2 graph; model.build_graph also populates res->t_logits/t_embd
                        res->reset();
                        ggml_backend_sched_reset(sched.get());
                        moe_weight_cache.build_layer_only = -1;
                        moe_weight_cache.build_phase = 2;
                        phase2_gf = model.build_graph(gparams, nullptr, nullptr, &moe_weight_cache);
                        if (!phase2_gf) { ret = GGML_STATUS_FAILED; return nullptr; }
                        ggml_backend_sched_reset(sched.get());
                        force_idxs_to_cpu();
                        if (!ggml_backend_sched_alloc_graph(sched.get(), phase2_gf)) { ret = GGML_STATUS_ALLOC_FAILED; return nullptr; }

                        ggml_backend_t be = ggml_backend_sched_get_backend(sched.get(), 0);
                        if (res->t_logits) ggml_backend_sched_set_tensor_backend(sched.get(), res->t_logits, be);
                        if (res->t_embd)   ggml_backend_sched_set_tensor_backend(sched.get(), res->t_embd,   be);

                        // 1. Restore all tensor->data to static addresses
                        h2_hijack.scan_and_update_snapshots(phase2_gf);
                        h2_hijack.restore_all();

                        // 2. Fill host buffers with Phase 1 results
                        //    ffn_inp is read from GPU scratch buffers after threshold readback
                        //    moe_ids and moe_weights come from the threshold output on CPU
                        for (int il = 0; il < H2_N_LAYERS; il++) {
                            // ffn_inp: already read back to CPU in Phase 1 D2H step (line ~1561)
                            // For now, read directly from GPU scratch_ffn_inp[il] via sync cudaMemcpy
                            if (moe_weight_cache.scratch_ffn_inp[il]) {
                                cudaMemcpy(h2_inject.host_ffn_inp[il],
                                           moe_weight_cache.scratch_ffn_inp[il]->data,
                                           H2_SZ_INP, cudaMemcpyDeviceToHost);
                            }
                            // moe_ids / moe_weights available on CPU from threshold processing
                            // Filled by existing code — we snapshot here:
                            // (exact source depends on threshold readback structure;
                            //  engineer verifies with Phase 1 output tensor names)
                        }

                        // 3. Record phase1 completion on the CUDA stream
                        void* st = ggml_backend_cuda_get_stream_ptr(gpu, 0);
                        ggml_backend_cuda_set_stream(gpu, 0);
                        h2_guard.record(st);

                        // 4. Async H2D injection: host → static GPU buffer
                        h2_inject.inject_all(h2_hijack, st);
                        h2_inject.inject_ids_and_weights_all(h2_hijack, st);

                        // 5. Launch CUDA Graph
                        ggml_backend_sched_synchronize(sched.get());
                        int cu = cudaGraphLaunch(h2_hijack.cuda_graph_exec, st);
                        if (cu != cudaSuccess) fprintf(stderr, "cuda_replay: %s\n", cudaGetErrorString(cu));
                        ggml_backend_sched_synchronize(sched.get());

                        // 6. Save logits data pointer for direct readback
                        if (res->t_logits)
                            moe_weight_cache.phase2_logits_data = res->t_logits->data;
                    } else {
```

Note: The `else {` on the last line above connects to the existing capture/non-cuda block starting at line 1737. Keep the existing code from line 1737-1787 unchanged.

Also remove the old `moe_weight_cache.cuda_graph_captured` check on line 1716 — it's now `h2_hijack.captured`. The variable `moe_weight_cache.cuda_graph_captured` continues to exist but is never checked again (the `h2_hijack.captured` is the single source of truth after capture).

- [ ] **Step 2: Update the else-condition on line 1737**

Change line 1737 from:
```cpp
                    } else {
```
This `else` branch should remain. Verify it still properly serves as the fallback for non-captured/non-cuda execution (the existing code at lines 1737-1787 handles both capture and non-cuda paths).

- [ ] **Step 3: Verify build compiles**

```bash
cmake --build build --target llama 2>&1 | Select-String -Pattern "error"
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/llama-context.cpp
git commit -m "feat: implement phase2 CUDA graph replay with static memory hijacking"
```

---

### Task 6: Runtime verification

**Files:**
- None (manual verification)

- [ ] **Step 1: Build and run inference**

```bash
cmake --build build --target llama-cli
.\build\bin\Release\llama-cli.exe -m <model.gguf> -p "Hello" -n 10 -ngl 99
```

Expected: runs without crashes, no "cuda_replay: ..." error messages, output is coherent (not garbage).

- [ ] **Step 2: Verify capture message appears once**

Check that `fprintf(stderr, "cuda: captured\n");` fires exactly once (on the first token), then all subsequent tokens use the replay path without errors.

- [ ] **Step 3: Verify throughput >= baseline**

Expected: >= 10.99 t/s.

- [ ] **Step 4: Commit (if verified)**

```bash
git add src/llama-context.cpp
git commit -m "verify: static memory hijacking passes smoke test at baseline throughput"
```