# MoE Module Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the MoE static-hijack/cascade/prefetch infrastructure from `src/llama-context.cpp` into 3 clean, production-grade C++ modules under `src/` with zero semantic changes.

**Architecture:** Approach A — free-function extraction. `moe-static-bunker` holds the 3 inert data structs (phase2_hijack, inject, guard) with init/destroy. `moe-hijacker` provides free functions operating on those structs for graph scanning, cascade forcing, and D2D copying. `moe-prefetcher` is a new async background worker class with EPYC core affinity for expert H2D transfers. `llama_context` owns instances and orchestrates calls in `process_ubatch`.

**Tech Stack:** C++17, CUDA runtime API (cudaMalloc, cudaMemcpyAsync, cudaEvent*), pthread (Linux), ggml backends.

## Global Constraints

- All 3 new modules guard with `#ifdef GGML_USE_CUDA` only (no LLAMA_DEEPSEEK_PIPELINE guard — structs are inert data without it)
- Use `.h` extension matching existing project conventions (no `.hpp`)
- Flat in `src/`, not a subdirectory
- Zero semantic changes to the 4 preserved edge cases (get_tensor_async fallback, Direct DMA, cascade GPU forcing, Expert IDs isolation)
- Namespace `moe` for hijacker and prefetcher; bunker structs remain in global scope (backward compat)
- Build must compile with `-DGGML_CUDA=ON -DLLAMA_DEEPSEEK_PIPELINE=ON`

---

### Task 1: Create `src/moe-static-bunker.h`

**Files:**
- Create: `src/moe-static-bunker.h`

**Interfaces:**
- Produces: `phase2_hijack`, `phase2_inject`, `phase2_guard` structs; `H2_ALIGN`, `H2_ALIGN_UP`, `H2_N_LAYERS_MAX` macros

- [ ] **Step 1: Write the header file**

```cpp
#pragma once

// Static memory hijacking infrastructure for MoE Phase 2.
// Contains inert data structures (POD) — no graph-walking logic.
// Graph scanning / cascade forcing lives in moe-hijacker.h.

#ifdef GGML_USE_CUDA

#include <cstddef>

#define H2_ALIGN         256
#define H2_ALIGN_UP(x)   (((x) + (H2_ALIGN - 1)) & ~(H2_ALIGN - 1))
#define H2_N_LAYERS_MAX  128

struct phase2_hijack {
    static constexpr const char * TENSOR_NAMES[7] = {
        "ffn_moe_gate_up",
        "ffn_moe_gate",
        "ffn_moe_up",
        "ffn_moe_swiglu",
        "ffn_moe_down",
        "ffn_moe_weighted",
        "ffn_moe_out",
    };
    enum TypeId : int {
        T_GATE_UP   = 0,
        T_GATE      = 1,
        T_UP        = 2,
        T_SWIGLU    = 3,
        T_DOWN      = 4,
        T_WEIGHTED  = 5,
        T_OUT       = 6,
        N_TYPES     = 7,
    };

    struct slot {
        void * addr;
        size_t size;
        int    parent;
        ptrdiff_t offset;
        void * orig_data;
    };

    void * buffer = nullptr;
    size_t buffer_size = 0;
    int    n_layers = 0;

    slot * slots = nullptr;
    int    n_slots = 0;
    int    slot_scan_count = 0;

    bool   captured = false;
    void * cuda_graph_exec = nullptr;
    void * stream = nullptr;

    void init(int n_moe_layers);
    void destroy();

    int slot_idx(int il, int type_id) const { return il * N_TYPES + type_id; }

private:
    void allocate_slots(int max_il);
};

struct phase2_inject {
    void * host_moe_ids[H2_N_LAYERS_MAX];
    float * host_moe_w[H2_N_LAYERS_MAX];

    void init();
    void destroy();
    void fill_layer(int il, const int * ids, const float * w);
    void inject_all(void * stream);
};

struct phase2_guard {
    void * phase1_done_event = nullptr;

    void init();
    void destroy();
    void record(void * stream);
    void wait(void * stream);
};

#endif // GGML_USE_CUDA
```

- [ ] **Step 2: Commit**

```bash
git add src/moe-static-bunker.h
git commit -m "feat: add moe-static-bunker.h — MoE static memory data structures (#huanglawsded)"
```

---

### Task 2: Create `src/moe-static-bunker.cpp`

**Files:**
- Create: `src/moe-static-bunker.cpp`

**Interfaces:**
- Consumes: `moe-static-bunker.h` (structs from Task 1)
- Produces: Init/destroy/allocate implementations for phase2_hijack, phase2_inject, phase2_guard

- [ ] **Step 1: Write the implementation file**

```cpp
#ifdef GGML_USE_CUDA

#include "moe-static-bunker.h"

#include <cstdio>
#include <cstring>

// CUDA runtime API forward declarations
extern "C" {
    int  cudaMalloc(void ** devPtr, size_t size);
    int  cudaFree(void * devPtr);
    int  cudaHostAlloc(void ** pHost, size_t size, unsigned int flags);
    int  cudaFreeHost(void * ptr);
    int  cudaEventCreate(void ** event);
    int  cudaEventDestroy(void * event);
    int  cudaEventRecord(void * event, void * stream);
    int  cudaStreamWaitEvent(void * stream, void * event, unsigned int flags);
    int  cudaStreamCreate(void ** stream);
    int  cudaStreamDestroy(void * stream);
    int  cudaStreamSynchronize(void * stream);
    int  cudaGraphExecDestroy(void * graphExec);
    int  cudaGraphDestroy(void * graph);
    const char * cudaGetErrorString(int);
}
constexpr int cudaHostAllocDefault   = 0;

// --- phase2_hijack ---

void phase2_hijack::init(int n_moe_layers) {
    n_layers = n_moe_layers;
    n_slots = n_layers * N_TYPES;
    slots = new slot[n_slots]();
    if (!slots) { fprintf(stderr, "phase2_hijack: OOM for slots\n"); return; }
    cudaStream_t s;
    cudaStreamCreate(&s);
    stream = s;
    captured = false;
    buffer = nullptr;
    buffer_size = 0;
}

void phase2_hijack::destroy() {
    if (cuda_graph_exec) { cudaGraphExecDestroy((void*)cuda_graph_exec); cuda_graph_exec = nullptr; }
    if (buffer)           { cudaFree(buffer); buffer = nullptr; }
    if (stream)           { cudaStreamSynchronize((void*)stream); cudaStreamDestroy((void*)stream); stream = nullptr; }
    delete[] slots; slots = nullptr;
    n_slots = 0; n_layers = 0;
}

void phase2_hijack::allocate_slots(int /*max_il*/) {
    size_t total = 0;
    for (int i = 0; i < n_slots; i++) {
        if (slots[i].size > 0 && slots[i].parent == -1) {
            total = H2_ALIGN_UP(total);
            total += slots[i].size;
        }
    }
    if (total == 0) {
        fprintf(stderr, "phase2_hijack: no slots to allocate\n");
        return;
    }
    int e = cudaMalloc(&buffer, total);
    if (e != 0) {
        fprintf(stderr, "phase2_hijack: cudaMalloc(%zu) failed: %s\n", total, cudaGetErrorString(e));
        buffer = nullptr;
        return;
    }
    buffer_size = total;
    size_t offset = 0;
    for (int i = 0; i < n_slots; i++) {
        if (slots[i].size > 0 && slots[i].parent == -1) {
            offset = H2_ALIGN_UP(offset);
            slots[i].addr = (char*)buffer + offset;
            offset += slots[i].size;
        }
    }
    fprintf(stderr, "phase2_hijack: allocated %zu byte buffer (%d types x %d layers)\n",
            total, N_TYPES, n_layers);
}

// --- phase2_inject ---

void phase2_inject::init() {
    for (int i = 0; i < H2_N_LAYERS_MAX; i++) {
        cudaHostAlloc((void**)&host_moe_ids[i],  sizeof(int)   * 32, cudaHostAllocDefault);
        cudaHostAlloc((void**)&host_moe_w[i],    sizeof(float) * 32, cudaHostAllocDefault);
    }
}

void phase2_inject::destroy() {
    for (int i = 0; i < H2_N_LAYERS_MAX; i++) {
        if (host_moe_ids[i])  { cudaFreeHost(host_moe_ids[i]);  host_moe_ids[i]  = nullptr; }
        if (host_moe_w[i])    { cudaFreeHost(host_moe_w[i]);    host_moe_w[i]    = nullptr; }
    }
}

void phase2_inject::fill_layer(int il, const int * ids, const float * w) {
    if (il >= H2_N_LAYERS_MAX) return;
    memcpy(host_moe_ids[il], ids, sizeof(int) * 32);
    memcpy(host_moe_w[il],   w,   sizeof(float) * 32);
}

void phase2_inject::inject_all(void * stream) {
    (void)stream;
}

// --- phase2_guard ---

void phase2_guard::init() {
    cudaEventCreate(&phase1_done_event);
}

void phase2_guard::destroy() {
    if (phase1_done_event) { cudaEventDestroy(phase1_done_event); phase1_done_event = nullptr; }
}

void phase2_guard::record(void * stream) {
    cudaEventRecord(phase1_done_event, stream);
}

void phase2_guard::wait(void * stream) {
    cudaStreamWaitEvent((void*)stream, (void*)phase1_done_event, 0);
}

#endif // GGML_USE_CUDA
```

- [ ] **Step 2: Commit**

```bash
git add src/moe-static-bunker.cpp
git commit -m "feat: add moe-static-bunker.cpp — MoE static memory init/destroy (#huanglawsded)"
```

---

### Task 3: Update `src/llama-context.h`

**Files:**
- Modify: `src/llama-context.h` — remove lines 111-221 (3 structs + macros), add 3 includes after line 13, add 2 member fields after line 535

**Interfaces:**
- Consumes: `moe-static-bunker.h` (Task 1), `moe-hijacker.h` (Task 4), `moe-prefetcher.h` (Task 6)
- Produces: `llama_context` with `moe_prefetch` + `moe_prefetch_started` members; `h2_hijack`, `h2_inject`, `h2_guard` now declared via bunker header

- [ ] **Step 1: Add includes after existing `#include "pipeline-sched.h"` (line 13)**

After line 13 (`#include "pipeline-sched.h"`), add:

```cpp
#ifdef GGML_USE_CUDA
#include "moe-static-bunker.h"
#endif
```

Note: The hijacker and prefetcher includes will be added in Tasks 4 and 6 respectively.

- [ ] **Step 2: Remove struct definitions (lines 111-221)**

Delete from `// Static memory hijacking for MoE Phase 2 CUDA Graph.` (line 111) through `#endif // GGML_USE_CUDA` (line 221). This removes:
- The comment block (lines 111-115)
- `#ifdef GGML_USE_CUDA` guard (line 116)
- `H2_ALIGN`, `H2_ALIGN_UP`, `H2_N_LAYERS_MAX` macros (lines 118-120)
- `struct phase2_hijack` with its nested `slot`, `TypeId`, `TENSOR_NAMES`, all fields/methods (lines 122-200)
- `struct phase2_inject` (lines 202-210)
- `struct phase2_guard` (lines 212-219)
- Closing `#endif // GGML_USE_CUDA` (line 221)

- [ ] **Step 3: Verify header compiles** (pre-check — actual build at Task 10)

No compilation yet (need .cpp changes). This is a logical checkpoint.

- [ ] **Step 5: Commit**

```bash
git add src/llama-context.h
git commit -m "refactor: extract MoE structs from llama-context.h to moe-static-bunker.h (#huanglawsded)"
```

---

### Task 4: Create `src/moe-hijacker.h`

**Files:**
- Create: `src/moe-hijacker.h`

**Interfaces:**
- Consumes: `phase2_hijack` from `moe-static-bunker.h` (Task 1)
- Produces: `moe::match_hijack_name`, `moe::scan_and_hijack`, `moe::scan_and_update_snapshots`, `moe::copy_data_to_static`, `moe::cascade_force_moe_consumers`, `moe::restore_all`

- [ ] **Step 1: Write the header**

```cpp
#pragma once

// Graph scanning and cascade forcing for MoE Phase 2.
// Operates on phase2_hijack structs via reference parameters.
// All functions are in namespace moe.

#ifdef GGML_USE_CUDA

#include <utility>

struct phase2_hijack;
struct ggml_cgraph;

namespace moe {

std::pair<int,int> match_hijack_name(phase2_hijack & h2, const char * name);

int  scan_and_hijack(phase2_hijack & h2, ggml_cgraph * gf);

bool scan_and_update_snapshots(phase2_hijack & h2, ggml_cgraph * gf);

void restore_all(phase2_hijack & h2);

void copy_data_to_static(phase2_hijack & h2, void * cuda_stream);

void cascade_force_moe_consumers(
    phase2_hijack & h2,
    ggml_cgraph * gf,
    void * sched,
    void * gpu_backend);

} // namespace moe

#endif // GGML_USE_CUDA
```

- [ ] **Step 2: Add hijacker include to llama-context.h**

In `src/llama-context.h`, after the bunker include added in Task 3 Step 1, add:

```cpp
#ifdef GGML_USE_CUDA
#include "moe-static-bunker.h"
#include "moe-hijacker.h"    // NEW
#endif
```

- [ ] **Step 3: Commit**

```bash
git add src/moe-hijacker.h src/llama-context.h
git commit -m "feat: add moe-hijacker.h — MoE graph scanning function declarations (#huanglawsded)"
```

---

### Task 5: Create `src/moe-hijacker.cpp`

**Files:**
- Create: `src/moe-hijacker.cpp`

**Interfaces:**
- Consumes: `moe-static-bunker.h` (Task 1), `moe-hijacker.h` (Task 4), `ggml.h` (for ggml_cgraph/ggml_tensor)
- Produces: All graph scanning and cascade implementations

- [ ] **Step 1: Write the implementation**

```cpp
#ifdef GGML_USE_CUDA

#include "moe-hijacker.h"
#include "moe-static-bunker.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

// CUDA forward declarations used by copy_data_to_static
extern "C" {
    int cudaMemcpyAsync(void * dst, const void * src, size_t count, int kind, void * stream);
}
constexpr int cudaMemcpyDefault = 4;

namespace moe {

std::pair<int,int> match_hijack_name(phase2_hijack & h2, const char * name) {
    if (!name || !name[0]) return {-1, -1};
    const char * dash = strrchr(name, '-');
    if (!dash) return {-1, -1};
    int il = 0;
    for (const char * p = dash + 1; *p; p++) {
        if (*p < '0' || *p > '9') return {-1, -1};
        il = il * 10 + (*p - '0');
    }
    size_t prefix_len = (size_t)(dash - name);
    for (int t = 0; t < h2.N_TYPES; t++) {
        size_t pat_len = strlen(h2.TENSOR_NAMES[t]);
        if (prefix_len == pat_len && memcmp(name, h2.TENSOR_NAMES[t], pat_len) == 0) {
            return {t, il};
        }
    }
    return {-1, -1};
}

int scan_and_hijack(phase2_hijack & h2, ggml_cgraph * gf) {
    h2.slot_scan_count = 0;
    int total_nodes = ggml_graph_n_nodes(gf);
    int total_leafs = ggml_graph_n_leafs(gf);
    int total_t = total_leafs + total_nodes;
    int max_il = 0;

    struct Match { ggml_tensor * t; int type_id; int il; void * orig; };
    std::vector<Match> matches, views;

    for (int i = 0; i < total_t; i++) {
        ggml_tensor * t;
        if (i < total_leafs) { t = ggml_graph_leaf(gf, i); } else { t = ggml_graph_node(gf, i - total_leafs); }
        auto [type_id, il] = match_hijack_name(h2, t->name);
        if (type_id < 0 || type_id >= h2.N_TYPES) continue;
        if (il < 0 || il >= h2.n_layers) continue;
        if (il > max_il) max_il = il;

        size_t sz = ggml_type_size(t->type);
        for (int d = 0; d < 4; d++) sz *= t->ne[d] > 0 ? t->ne[d] : 1;

        int s_idx = h2.slot_idx(il, type_id);
        if (s_idx < 0 || s_idx >= h2.n_slots) continue;

        if (h2.slots[s_idx].size == 0) {
            h2.slots[s_idx].size = H2_ALIGN_UP(sz);
            h2.slots[s_idx].addr = nullptr;
            h2.slots[s_idx].parent = -1;
            h2.slots[s_idx].offset = 0;
        }

        if (t->view_src) {
            ptrdiff_t off = (char*)t->data - (char*)t->view_src->data;
            h2.slots[s_idx].offset = off;
            h2.slots[s_idx].parent = -2;
            Match m = {t, type_id, il, t->data};
            views.push_back(m);
        } else {
            h2.slots[s_idx].parent = -1;
            Match m = {t, type_id, il, t->data};
            matches.push_back(m);
        }
    }

    int total_matched = (int)(matches.size() + views.size());
    if (total_matched == 0) {
        fprintf(stderr, "phase2_hijack: no tensors matched by name!\n");
        return 0;
    }

    if (!h2.buffer) {
        h2.allocate_slots(max_il + 1);
        if (!h2.buffer) return 0;
    }

    int skipped = 0;

    for (auto & m : matches) {
        int s_idx = h2.slot_idx(m.il, m.type_id);
        if (s_idx < 0 || s_idx >= h2.n_slots || !h2.slots[s_idx].addr) { skipped++; continue; }
        h2.slots[s_idx].orig_data = m.orig;
        m.t->data = h2.slots[s_idx].addr;
        h2.slot_scan_count++;
    }

    for (auto & m : views) {
        if (!m.t->view_src) { skipped++; continue; }
        auto [pid, pl] = match_hijack_name(h2, m.t->view_src->name);
        if (pid < 0 || pid >= h2.N_TYPES) { skipped++; continue; }
        int p_idx = h2.slot_idx(pl, pid);
        if (p_idx < 0 || p_idx >= h2.n_slots || !h2.slots[p_idx].addr) { skipped++; continue; }
        int s_idx = h2.slot_idx(m.il, m.type_id);
        void * view_addr = (char*)h2.slots[p_idx].addr + h2.slots[s_idx].offset;
        if (s_idx >= 0 && s_idx < h2.n_slots) {
            h2.slots[s_idx].addr = view_addr;
            h2.slots[s_idx].orig_data = m.orig;
        }
        m.t->data = view_addr;
        h2.slot_scan_count++;
    }

    fprintf(stderr, "phase2_hijack: scan_and_hijack -> %d/%d matched, %d skipped (%d layers)\n",
            h2.slot_scan_count, (int)(matches.size() + views.size()), skipped, max_il + 1);
    return h2.slot_scan_count;
}

bool scan_and_update_snapshots(phase2_hijack & h2, ggml_cgraph * gf) {
    int total_nodes = ggml_graph_n_nodes(gf);
    int total_leafs = ggml_graph_n_leafs(gf);
    int total_t = total_leafs + total_nodes;
    int found = 0;

    for (int i = 0; i < total_t; i++) {
        ggml_tensor * t;
        if (i < total_leafs) { t = ggml_graph_leaf(gf, i); } else { t = ggml_graph_node(gf, i - total_leafs); }
        auto [type_id, il] = match_hijack_name(h2, t->name);
        if (type_id < 0 || type_id >= h2.N_TYPES) continue;
        if (il < 0 || il >= h2.n_layers) continue;

        int s_idx = h2.slot_idx(il, type_id);
        if (s_idx < 0 || s_idx >= h2.n_slots) continue;

        void * target = h2.slots[s_idx].addr;
        if (!target) continue;

        if (t->data != target) {
            h2.slots[s_idx].orig_data = t->data;
            t->data = target;
        }
        found++;
    }

    return found > 0;
}

void restore_all(phase2_hijack & h2) {
    (void)h2;
    asm volatile("" ::: "memory");
}

void copy_data_to_static(phase2_hijack & h2, void * cuda_stream) {
    for (int i = 0; i < h2.n_slots; i++) {
        if (h2.slots[i].addr && h2.slots[i].orig_data && h2.slots[i].size > 0) {
            cudaMemcpyAsync(h2.slots[i].addr, h2.slots[i].orig_data,
                h2.slots[i].size, cudaMemcpyDefault, (void*)cuda_stream);
        }
    }
}

void cascade_force_moe_consumers(
    phase2_hijack & h2,
    ggml_cgraph * gf,
    void * sched,
    void * gpu_backend)
{
    extern void ggml_backend_sched_set_tensor_backend(void * sched, ggml_tensor * t, void * backend);

    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op != 37 && t->op != 2) continue;  // VIEW=37, ADD=2
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; s++) {
            auto [sid, sil] = match_hijack_name(h2, t->src[s]->name);
            if (sid >= 0) {
                ggml_backend_sched_set_tensor_backend(sched, t, gpu_backend);
                break;
            }
        }
    }
}

} // namespace moe

#endif // GGML_USE_CUDA
```

**Note about `ggml_backend_sched_set_tensor_backend`:** This function is declared in `ggml-backend.h` and linked from ggml. The `extern` forward declaration at the top of `cascade_force_moe_consumers` ensures the linker resolves it. The actual signature is `void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t, ggml_tensor *, ggml_backend_t)`.

- [ ] **Step 2: Commit**

```bash
git add src/moe-hijacker.cpp
git commit -m "feat: add moe-hijacker.cpp — MoE graph scan/cascade/copy implementations (#huanglawsded)"
```

---

### Task 6: Create `src/moe-prefetcher.h`

**Files:**
- Create: `src/moe-prefetcher.h`
- Modify: `src/llama-context.h` — add `#include "moe-prefetcher.h"` and member fields

**Interfaces:**
- Consumes: `ggml.h` (forward declarations), standard library
- Produces: `moe::prefetch_work_item`, `moe::moe_prefetcher` class

- [ ] **Step 1: Write the header**

```cpp
#pragma once

// Async background worker for expert H2D prefetch on EPYC 9V74.
// Pins to management cores 64-75, uses dedicated CUDA stream.
// Overlaps with main-thread Phase 2 graph build / sched alloc.

#ifdef GGML_USE_CUDA

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

struct ggml_tensor;

namespace moe {

struct prefetch_work_item {
    ggml_tensor * dst_gate   = nullptr;
    ggml_tensor * dst_up     = nullptr;
    ggml_tensor * dst_down   = nullptr;
    ggml_tensor * src_gate   = nullptr;
    ggml_tensor * src_up     = nullptr;
    ggml_tensor * src_down   = nullptr;
    ggml_tensor * expert_mask = nullptr;
    ggml_tensor * moe_remap   = nullptr;
    void *        host_mask   = nullptr;
    void *        host_remap  = nullptr;
    size_t        slice_bytes[3] = {0, 0, 0};
    int           max_kept     = 0;
    void *        prefetch_done = nullptr;
    void *        stream       = nullptr;
    void *        gpu_backend  = nullptr;
};

class moe_prefetcher {
public:
    moe_prefetcher();
    ~moe_prefetcher();

    moe_prefetcher(const moe_prefetcher &) = delete;
    moe_prefetcher & operator=(const moe_prefetcher &) = delete;

    bool start();
    void stop();

    void launch_prefetch(
        const std::vector<prefetch_work_item> & items,
        void * completion_event);

    void wait_prefetch_fence(void * completion_event);

    void * get_stream() const { return h2d_stream_; }

private:
    void worker_loop();
    static bool pin_to_management_cores();

    std::thread                worker_;
    std::atomic<bool>          running_{false};
    std::atomic<bool>          stop_requested_{false};

    std::mutex                 work_mutex_;
    std::condition_variable    work_cv_;
    bool                       work_ready_{false};
    std::vector<prefetch_work_item> work_items_;
    void *                     completion_event_{nullptr};

    void * h2d_stream_ = nullptr;

    std::atomic<int> fence_{0};
};

} // namespace moe

#endif // GGML_USE_CUDA
```

- [ ] **Step 2: Update llama-context.h**

Add include after the hijacker include:

```cpp
#ifdef GGML_USE_CUDA
#include "moe-static-bunker.h"
#include "moe-hijacker.h"
#include "moe-prefetcher.h"  // NEW
#endif
```

Add members after `phase2_guard h2_guard;` (line 535):

```cpp
#ifdef GGML_USE_CUDA
    moe::moe_prefetcher moe_prefetch;
    bool                moe_prefetch_started = false;
#endif
```

- [ ] **Step 3: Commit**

```bash
git add src/moe-prefetcher.h src/llama-context.h
git commit -m "feat: add moe-prefetcher.h — async expert prefetch worker class (#huanglawsded)"
```

---

### Task 7: Create `src/moe-prefetcher.cpp`

**Files:**
- Create: `src/moe-prefetcher.cpp`

**Interfaces:**
- Consumes: `moe-prefetcher.h` (Task 6), `ggml-cuda.h` (for `pipeline_expert_skip_prefetch`)
- Produces: Prefetcher implementation with EPYC core affinity, thread loop, work dispatch

- [ ] **Step 1: Write the implementation**

```cpp
#ifdef GGML_USE_CUDA

#include "moe-prefetcher.h"

#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdio>

#ifdef __linux__
#include <pthread.h>
#endif

extern "C" {
    int cudaStreamCreate(void ** stream);
    int cudaStreamDestroy(void * stream);
    int cudaStreamSynchronize(void * stream);
}

namespace moe {

moe_prefetcher::moe_prefetcher() {
    // stream created in start()
}

moe_prefetcher::~moe_prefetcher() {
    if (running_.load(std::memory_order_acquire)) {
        stop();
    }
}

bool moe_prefetcher::start() {
    if (running_.load(std::memory_order_acquire)) return true;

    int e = cudaStreamCreate(&h2d_stream_);
    if (e != 0) {
        fprintf(stderr, "moe_prefetcher: cudaStreamCreate failed: %d\n", e);
        return false;
    }

    running_.store(true, std::memory_order_release);
    stop_requested_.store(false, std::memory_order_release);

    try {
        worker_ = std::thread(&moe_prefetcher::worker_loop, this);
    } catch (const std::exception & ex) {
        fprintf(stderr, "moe_prefetcher: thread creation failed: %s\n", ex.what());
        running_.store(false, std::memory_order_release);
        return false;
    }

    fprintf(stderr, "moe_prefetcher: started background worker\n");
    return true;
}

void moe_prefetcher::stop() {
    if (!running_.load(std::memory_order_acquire)) return;

    {
        std::lock_guard<std::mutex> lk(work_mutex_);
        stop_requested_.store(true, std::memory_order_release);
    }
    work_cv_.notify_one();

    if (worker_.joinable()) {
        worker_.join();
    }

    if (h2d_stream_) {
        cudaStreamSynchronize(h2d_stream_);
        cudaStreamDestroy(h2d_stream_);
        h2d_stream_ = nullptr;
    }

    running_.store(false, std::memory_order_release);
    fprintf(stderr, "moe_prefetcher: stopped\n");
}

void moe_prefetcher::launch_prefetch(
    const std::vector<prefetch_work_item> & items,
    void * completion_event)
{
    if (!running_.load(std::memory_order_acquire)) {
        // Fallback: execute synchronously
        for (auto & item : items) {
            const void * host_mask_ptr = item.host_mask;
            const void * host_remap_ptr = item.host_remap;
            if (host_mask_ptr && host_remap_ptr) {
                ggml_tensor * dst_arr[] = { item.dst_gate, item.dst_up, item.dst_down };
                ggml_tensor * src_arr[] = { item.src_gate, item.src_up, item.src_down };
                ggml_backend_cuda_pipeline_expert_skip_prefetch(
                    dst_arr, src_arr, item.slice_bytes,
                    item.expert_mask, item.moe_remap,
                    (const uint64_t *)host_mask_ptr,
                    (const int32_t *)host_remap_ptr,
                    nullptr, completion_event, (ggml_backend_t)item.gpu_backend);
            }
        }
        return;
    }

    {
        std::lock_guard<std::mutex> lk(work_mutex_);
        work_items_ = items;
        completion_event_ = completion_event;
        work_ready_ = true;
    }
    work_cv_.notify_one();
}

void moe_prefetcher::wait_prefetch_fence(void * /*completion_event*/) {
    // Spin-wait on atomic fence — threads that got completion via event
    // will store to fence_ before returning. The caller checks this.
    // In practice, the fence is stored by the worker after all H2D transfers
    // are queued and the completion event is recorded.
    while (fence_.load(std::memory_order_acquire) == 0) {
        // spin
    }
    // Reset for next invocation
    fence_.store(0, std::memory_order_release);
}

void * moe_prefetcher::get_stream() const {
    return h2d_stream_;
}

void moe_prefetcher::worker_loop() {
    pin_to_management_cores();

    while (true) {
        {
            std::unique_lock<std::mutex> lk(work_mutex_);
            work_cv_.wait(lk, [this] {
                return work_ready_ || stop_requested_.load(std::memory_order_acquire);
            });
            if (stop_requested_.load(std::memory_order_acquire)) break;
        }

        // Process work items on the dedicated H2D stream
        for (auto & item : work_items_) {
            if (!item.host_mask || !item.host_remap) continue;

            ggml_tensor * dst_arr[] = { item.dst_gate, item.dst_up, item.dst_down };
            ggml_tensor * src_arr[] = { item.src_gate, item.src_up, item.src_down };

            ggml_backend_cuda_pipeline_expert_skip_prefetch(
                dst_arr, src_arr, item.slice_bytes,
                item.expert_mask, item.moe_remap,
                (const uint64_t *)item.host_mask,
                (const int32_t *)item.host_remap,
                nullptr,
                item.prefetch_done ? item.prefetch_done : completion_event_,
                (ggml_backend_t)item.gpu_backend);
        }

        // Signal completion via atomic fence
        fence_.store(1, std::memory_order_release);

        {
            std::lock_guard<std::mutex> lk(work_mutex_);
            work_ready_ = false;
            work_items_.clear();
        }
    }
}

bool moe_prefetcher::pin_to_management_cores() {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    // EPYC 9V74: 80 physical cores, cores 0-63 = primary compute, 64-75 = management
    for (int core = 64; core <= 75; core++) {
        CPU_SET(core, &cpuset);
    }
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        fprintf(stderr, "moe_prefetcher: pthread_setaffinity_np failed: %d\n", rc);
        return false;
    }
    fprintf(stderr, "moe_prefetcher: pinned to cores 64-75\n");
    return true;
#else
    fprintf(stderr, "moe_prefetcher: core affinity not supported on this platform\n");
    return false;
#endif
}

} // namespace moe

#endif // GGML_USE_CUDA
```

- [ ] **Step 2: Commit**

```bash
git add src/moe-prefetcher.cpp
git commit -m "feat: add moe-prefetcher.cpp — async expert prefetch worker with EPYC affinity (#huanglawsded)"
```

---

### Task 8: Update `src/llama-context.cpp`

**Files:**
- Modify: `src/llama-context.cpp` — remove lines 91-362, update constructor/destructor, update process_ubatch hijack/cascade calls

**Interfaces:**
- Consumes: All 3 modules (Tasks 1-7), `ggml-cuda.h` (existing)
- Produces: Refactored `llama_context` using `moe::` namespace functions instead of struct methods

- [ ] **Step 2: Remove struct method implementations (lines 91-362)**

Delete from `#ifdef GGML_USE_CUDA` / `#ifdef LLAMA_DEEPSEEK_PIPELINE` at line 91 through the closing `#endif // GGML_USE_CUDA` at line 363.

This removes:
- `h2_init()` / `h2_destroy()` (lines 94-107)
- `phase2_hijack::init/destroy` (lines 109-128)
- `phase2_inject::init/destroy/fill_layer/inject_all` (lines 130-157)
- `phase2_guard::init/destroy/record/wait` (lines 159-173)
- `match_name` (lines 177-194)
- `allocate_slots` (lines 196-225)
- `scan_and_hijack` (lines 227-316)
- `scan_and_update_snapshots` (lines 318-346)
- `restore_all` (lines 348-351)
- `copy_data_to_static` (lines 353-360)

- [ ] **Step 3: Update constructor — add prefetcher start**

At line 730 (currently `h2_init();`), replace with:

```cpp
#ifdef GGML_USE_CUDA
    h2_init();
    moe_prefetch_started = moe_prefetch.start();
#endif
```

- [ ] **Step 4: Update destructor — add prefetcher stop**

At line 777 (currently `h2_destroy();`), replace with:

```cpp
#ifdef GGML_USE_CUDA
    if (moe_prefetch_started) moe_prefetch.stop();
    h2_destroy();
#endif
```

- [ ] **Step 5: Update process_ubatch — Phase 2 cascade forcing (lines 2050-2061)**

Replace the inline cascade loop:

```cpp
                        // Cascade: force VIEW (op=37) and ADD (op=2) consumers of matched tensors
                        for (int i = 0; i < ggml_graph_n_nodes(phase2_gf); i++) {
                            ggml_tensor * t = ggml_graph_node(phase2_gf, i);
                            if (t->op != 37 && t->op != 2) continue;
                            for (int s = 0; s < GGML_MAX_SRC && t->src[s]; s++) {
                                auto [sid, sil] = h2_hijack.match_name(t->src[s]->name);
                                if (sid >= 0) {
                                    ggml_backend_sched_set_tensor_backend(sched.get(), t, gpu);
                                    break;
                                }
                            }
                        }
```

With:

```cpp
                        // Cascade: force VIEW (op=37) and ADD (op=2) consumers of matched tensors
                        moe::cascade_force_moe_consumers(h2_hijack, phase2_gf, sched.get(), gpu);
```

- [ ] **Step 6: Update process_ubatch — Phase 2 replay snapshot + copy (line 2070, 2077)**

Replace:
```cpp
                        h2_hijack.scan_and_update_snapshots(phase2_gf);
```
With:
```cpp
                        moe::scan_and_update_snapshots(h2_hijack, phase2_gf);
```

Replace (line 2077):
```cpp
                        h2_hijack.copy_data_to_static(st);
```
With:
```cpp
                        moe::copy_data_to_static(h2_hijack, st);
```

- [ ] **Step 7: Update process_ubatch — Tensor matching in replay path (lines 2040-2049)**

The tensor matching code uses `h2_hijack.match_name(t->name)` — this is now `moe::match_hijack_name(h2_hijack, t->name)`.

Replace:
```cpp
                            auto [tid, il] = h2_hijack.match_name(t->name);
```
(occurs at line 2040 and line 2046) with:
```cpp
                            auto [tid, il] = moe::match_hijack_name(h2_hijack, t->name);
```

- [ ] **Step 8: Commit**

```bash
git add src/llama-context.cpp
git commit -m "refactor: switch llama-context.cpp to moe:: namespace functions (#huanglawsded)"
```

---

### Task 9: Update `src/CMakeLists.txt`

**Files:**
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Add 3 new source files in alphabetical order**

In the `add_library(llama ...)` block, insert alphabetically between `llama-model-saver.cpp` and `llama-model.cpp`:

```cmake
            llama-model-saver.cpp
            moe-hijacker.cpp
            moe-prefetcher.cpp
            moe-static-bunker.cpp
            llama-model.cpp
```

- [ ] **Step 2: Commit**

```bash
git add src/CMakeLists.txt
git commit -m "build: add moe-*.cpp modules to CMakeLists.txt (#huanglawsded)"
```

---

### Task 10: Build Verification

**Files:**
- None modified — verification only

- [ ] **Step 1: Configure cmake**

```bash
mkdir -p build && cd build && cmake .. -DGGML_CUDA=ON -DLLAMA_DEEPSEEK_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release
```

Expected: Configuration succeeds, no cmake errors.

- [ ] **Step 2: Build the llama target**

```bash
cmake --build build --target llama -j$(nproc)
```

Expected: Compilation succeeds with zero errors. Warnings should match baseline.

- [ ] **Step 3: Build main executable (for inference test)**

```bash
cmake --build build --target llama-cli -j$(nproc)
```

Expected: Link succeeds.

- [ ] **Step 4: Inference correctness test**

Run 100-token generation on Kimi-K2.7 (or DeepSeek-V3) with seeds 0, 1, 2:
```bash
./build/bin/llama-cli -m <model_path> -p "Hello" -n 100 -s 0 --deepseek-pipeline --moe-two-phase
```

Expected: Output matches baseline token-for-token. Seed 0 should produce identical output to the pre-refactor baseline.

- [ ] **Step 5: Verify preserved edge cases via log inspection**

Check stderr for:
1. `phase2_hijack: allocated <N> byte buffer` — confirms cudaMalloc path
2. `phase2_hijack: scan_and_hijack -> <N>/<M> matched` — confirms tensor matching
3. No `ALARM: Layer <N>, Slot <N>, Bad Index` — confirms expert ID isolation
4. `phase1: <N> us` + `readback_prefetch: <N> us` timing output

- [ ] **Step 6: Commit if verification passes**

```bash
# No code changes — just verification. If all checks pass, no commit needed.
# If fixes were required, commit them with:
# git commit -m "fix: build fixes for moe module decoupling (#huanglawsded)"
```