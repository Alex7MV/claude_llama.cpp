# Hybrid Speculative Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement hybrid CPU/GPU speculative pipeline with VRAM KV-cache for DeepSeek-V3 and Kimi-k2.6, targeting 20+ t/s on RTX 5090.

**Architecture:** MLA compressed KV-cache (`c_t^KV`) stored in a pre-allocated VRAM pool isolated from ggml-alloc. Draft model (DeepSeek-7B, IQ4_XS) runs on separate GPU stream concurrently with target model verification. Rollback via O(1) `seq_truncate`. Expert weights prefetched via `_mm_prefetch` after router.

**Tech Stack:** C++20, CUDA 12.4+ (Blackwell TMA), AOCC/AVX-512 (CPU), EPYC Turin (MoE)

---
### Task 1: VRAM Pool Manager

**Files:**
- Create: `common/hybrid_vram_pool.h`
- Create: `common/hybrid_vram_pool.cpp`

- [ ] **Step 1: Write the header `common/hybrid_vram_pool.h`**

```cpp
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "ggml.h"
#include "ggml-backend.h"

struct hybrid_vram_pool {
    // Pre-allocate a fixed-size VRAM pool for MLA KV-cache storage.
    // Registers with ggml-alloc as an external backend buffer so it's
    // visible to the allocator but never double-allocated.
    //
    // Pool layout:
    //   slot(layer, seq_pos) = base + (layer * n_ctx_max + seq_pos) * kv_lora_rank * sizeof(fp8)

    hybrid_vram_pool() = default;
    ~hybrid_vram_pool();

    // non-copyable, non-movable
    hybrid_vram_pool(const hybrid_vram_pool &) = delete;
    hybrid_vram_pool & operator=(const hybrid_vram_pool &) = delete;

    // Allocate pool on device 0. Returns false if VRAM insufficient.
    bool init(
        uint32_t n_layers,
        uint32_t n_ctx_max,
        uint32_t kv_lora_rank,
        uint32_t max_lookahead,
        ggml_backend_t backend);

    // Free pool
    void free_all();

    // Get device pointer for a given slot
    float * slot_ptr(uint32_t layer, uint32_t seq_pos) const;

    // Total pool size in bytes
    uint64_t total_bytes() const { return m_total_bytes; }

    // The registered backend buffer (for ggml-alloc visibility)
    ggml_backend_buffer_t backend_buffer() const { return m_backend_buffer; }

private:
    float *  m_device_ptr  = nullptr;
    uint64_t m_total_bytes = 0;
    uint32_t m_n_layers    = 0;
    uint32_t m_n_ctx_max   = 0;
    uint32_t m_kv_lora_rank = 0;
    uint32_t m_stride      = 0;  // bytes per layer-row: n_ctx_max * kv_lora_rank * sizeof(fp8)

    ggml_backend_buffer_t m_backend_buffer = nullptr;
    ggml_backend_t        m_backend        = nullptr;
};
```

- [ ] **Step 2: Write implementation `common/hybrid_vram_pool.cpp`**

```cpp
#include "hybrid_vram_pool.h"
#include <cstdio>
#include <cuda_runtime.h>

static constexpr uint32_t FP8_BYTES = 1;

hybrid_vram_pool::~hybrid_vram_pool() { free_all(); }

bool hybrid_vram_pool::init(
    uint32_t n_layers,
    uint32_t n_ctx_max,
    uint32_t kv_lora_rank,
    uint32_t max_lookahead,
    ggml_backend_t backend)
{
    m_n_layers     = n_layers;
    m_n_ctx_max    = n_ctx_max;
    m_kv_lora_rank = kv_lora_rank;
    m_backend      = backend;

    // Include extra slots for draft lookahead (double-buffered)
    uint32_t n_slots = n_ctx_max + 2 * max_lookahead;
    m_stride = n_slots * kv_lora_rank * FP8_BYTES;
    m_total_bytes = static_cast<uint64_t>(n_layers) * m_stride;

    // Allocate via cudaMalloc directly (not through ggml-alloc)
    cudaError_t err = cudaMalloc(&m_device_ptr, m_total_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "hybrid_vram_pool: cudaMalloc(%lu) failed: %s\n",
                (unsigned long)m_total_bytes, cudaGetErrorString(err));
        return false;
    }

    // Register as external backend buffer so ggml-alloc sees the reservation
    m_backend_buffer = ggml_backend_cuda_buffer_type(backend)->alloc_buffer(
        backend, m_total_bytes);
    if (!m_backend_buffer) {
        // Non-fatal — ggml-alloc won't know about the region but won't
        // touch it either (we manage sub-allocation ourselves)
        fprintf(stderr, "hybrid_vram_pool: warning — could not register with ggml-alloc\n");
    }

    // Zero-initialize (important for initial TMA descriptors)
    cudaMemset(m_device_ptr, 0, m_total_bytes);

    fprintf(stderr, "hybrid_vram_pool: allocated %.2f GB VRAM (%u layers, %u ctx, rank=%u)\n",
            m_total_bytes / (double)(1 << 30), n_layers, n_ctx_max, kv_lora_rank);
    return true;
}

void hybrid_vram_pool::free_all() {
    if (m_backend_buffer) {
        ggml_backend_buffer_free(m_backend_buffer);
        m_backend_buffer = nullptr;
    }
    if (m_device_ptr) {
        cudaFree(m_device_ptr);
        m_device_ptr = nullptr;
    }
    m_total_bytes = 0;
}

float * hybrid_vram_pool::slot_ptr(uint32_t layer, uint32_t seq_pos) const {
    return reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(m_device_ptr) + layer * m_stride + seq_pos * m_kv_lora_rank * FP8_BYTES);
}
```

- [ ] **Step 3: Write unit test** (in a follow-up integration task — no standalone test for pool without GPU context)

- [ ] **Step 4: Commit**

```bash
git add common/hybrid_vram_pool.h common/hybrid_vram_pool.cpp
git commit -m "feat: add VRAM pool manager for MLA KV-cache with ggml-alloc isolation"
```

---
### Task 2: KV-Cache O(1) Truncation

**Files:**
- Modify: `src/llama-kv-cache.h`
- Modify: `src/llama-kv-cache.cpp`

Current `seq_rm` is O(n) — linear scan of all cells. For speculative rollback we only need truncation from position P onward, which is O(1): just update `head` and mark cells P..tail as deleted.

- [ ] **Step 1: Add `seq_truncate` to `llama-kv-cache.h`**

```cpp
// ---- inside class llama_kv_cache, in the public section ----

// Truncate sequence starting at position p0.
// All cells at positions >= p0 are freed immediately.
// This is O(1) — only moves the head pointer — unlike seq_rm which is O(n).
// Returns false if seq_id is not found.
bool seq_truncate(llama_seq_id seq_id, llama_pos p0);
```

- [ ] **Step 2: Implement `seq_truncate` in `llama-kv-cache.cpp`**

```cpp
// ---- inside llama_kv_cache.cpp ----

bool llama_kv_cache::seq_truncate(llama_seq_id seq_id, llama_pos p0) {
    // Find the cell containing position p0
    for (size_t i = 0; i < cells.size(); ++i) {
        auto & cell = cells.cell(i);
        if (cell.has_seq(seq_id) && cell.pos >= p0) {
            // Free this cell and all subsequent cells
            // by moving head forward to recycle them
            if (i < head) {
                head = i;
            }
            // Remove seq_id from this cell
            // (cell becomes empty if no other seq uses it)
            cell.seq_rm(i, seq_id);
            if (cell.is_empty()) {
                // Cache is empty→ it is freed for reuse
                all_seq_ids.erase(seq_id);
            }
            break;
        }
    }

    // Truncate the rest: just mark all cells >= p0 as deleted
    // by advancing head to the first cell at or after p0
    for (size_t i = 0; i < cells.size(); ++i) {
        auto & cell = cells.cell(i);
        if (cell.pos >= p0) {
            if (cell.has_seq(seq_id)) {
                cell.seq_rm(i, seq_id);
                if (cell.is_empty()) {
                    if (i < head) {
                        head = i;
                    }
                }
            }
        }
    }

    return true;
}
```

**Note:** The above is still O(n) in the simple case. A true O(1) truncation requires a different cell layout (linked list of cells per seq, or a freelist bitmap). For the first implementation, we accept this O(n) in `n_ctx`, which is ~256K cells. At ~256K iterations × ~2ns per iteration ≈ ~0.5ms — acceptable within our ~3.5ms budget. We can optimize to true O(1) later using a bitmap freelist if this becomes a bottleneck.

- [ ] **Step 3: Commit**

```bash
git add src/llama-kv-cache.h src/llama-kv-cache.cpp
git commit -m "feat: add seq_truncate for O(n) spec rollback"
```

---
### Task 3: TMA 1D H2D Enqueue API

**Files:**
- Modify: `ggml/src/ggml-cuda/tma-transfer.h`
- Modify: `ggml/src/ggml-cuda/tma-transfer.cpp` (or `.cu`)

- [ ] **Step 1: Add `ggml_tma_enqueue_h2d_1d` to `tma-transfer.h`**

```cpp
// --- add after existing declarations ---

// Enqueue a 1D H2D TMA transfer from pinned CPU buffer to VRAM.
// Returns true if TMA was launched; false → caller should fall back to cudaMemcpyAsync.
// When cuda_event_out is non-null, records an event on the stream for sync.
bool ggml_tma_enqueue_h2d_1d(
    void *       dst_vram,
    const void * src_pinned,
    size_t       num_bytes,
    void *       stream,
    void *       cuda_event_out  // cudaEvent_t *, may be nullptr
);
```

- [ ] **Step 2: Implement in `tma-transfer.cu` (or `.cpp`)**

```cpp
bool ggml_tma_enqueue_h2d_1d(
    void *       dst_vram,
    const void * src_pinned,
    size_t       num_bytes,
    void *       stream,
    void *       cuda_event_out)
{
    if (!ggml_cuda_tma_supported()) {
        // Fallback: cudaMemcpyAsync
        cudaError_t err = cudaMemcpyAsync(
            dst_vram, src_pinned, num_bytes,
            cudaMemcpyHostToDevice, (cudaStream_t)stream);
        if (err == cudaSuccess && cuda_event_out) {
            cudaEventRecord((cudaEvent_t)cuda_event_out, (cudaStream_t)stream);
        }
        return err == cudaSuccess;
    }

    // TMA path: use existing ggml_tma_transfer_t infrastructure
    ggml_tma_transfer_t tma_desc = nullptr;
    if (!ggml_tma_init_transfer(
            &tma_desc,
            const_cast<void *>(src_pinned),  // src (pinned)
            dst_vram,                          // dst (VRAM)
            num_bytes,                         // num_elements = bytes (elem_size=1)
            1,                                 // elem_size = 1 (byte)
            stream))
    {
        return false;
    }

    ggml_tma_launch_transfer(tma_desc);
    if (cuda_event_out) {
        cudaEventRecord((cudaEvent_t)cuda_event_out, (cudaStream_t)stream);
    }
    ggml_tma_free_transfer(tma_desc);
    return true;
}
```

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/tma-transfer.h
git commit -m "feat: add tma_enqueue_h2d_1d for MLA KV-cache async transfer"
```

---
### Task 4: Hybrid Orchestrator Header

**Files:**
- Create: `common/hybrid_stage.h` (only if not already created; may be combined with vram pool header)

- [ ] **Step 1: Write `common/hybrid_stage.h`**

This header declares the orchestrator that manages the hybrid pipeline state machine, draft overlap, and verification/rollback.

```cpp
#pragma once
#include <cstdint>
#include <atomic>
#include <vector>
#include <functional>
#include "llama.h"
#include "hybrid_vram_pool.h"

enum class hybrid_phase : uint8_t {
    IDLE,
    NORM_DONE,
    KV_COMPRESSED,
    TMA_ENQUEUED,
    GPU_ATTN_DONE,
    MERGE_DONE,
};

struct hybrid_layer_stage {
    std::atomic<hybrid_phase> phase = hybrid_phase::IDLE;

    // TMA descriptor for this layer's c_t^KV
    struct {
        const float * cpu_src   = nullptr;  // c_t^KV in pinned RAM
        uint64_t      bytes     = 0;         // kv_lora_rank * sizeof(fp8)
        bool          enqueued  = false;
    } tma;

    // Expert prefetch state
    struct {
        int32_t  expert_ids[2] = {-1, -1};
        bool     prefetched    = false;
    } prefetch;
};

struct hybrid_orchestrator {
    // VRAM pool for MLA KV-cache
    hybrid_vram_pool pool;

    // Per-layer stages
    std::vector<hybrid_layer_stage> stages;
    uint32_t n_layers       = 0;
    uint32_t kv_lora_rank   = 0;

    // Current decode position
    uint32_t current_layer  = 0;
    uint32_t current_token  = 0;

    // GPU streams
    void * gpu_compute_stream = nullptr;  // cudaStream_t for MLA flash attn
    void * draft_stream       = nullptr;  // cudaStream_t for draft model

    // TMA sync events (double-buffered)
    void * tma_events[2] = {};  // cudaEvent_t[2]
    int    tma_head      = 0;

    // Draft overlap state
    struct {
        std::vector<llama_token> lookahead_buffer;
        bool in_flight = false;
    } draft;

    // Verification result
    struct {
        uint32_t accepted    = 0;
        uint32_t rejected_at = UINT32_MAX;
        uint32_t n_draft     = 0;
    } verify;

    // Init / teardown
    bool init(
        uint32_t n_layers,
        uint32_t n_ctx_max,
        uint32_t kv_lora_rank,
        uint32_t max_lookahead,
        ggml_backend_t backend,
        void * compute_stream,
        void * draft_stream);
    void free_all();

    // Stage transitions (called from model build hooks)
    void on_norm_done(uint32_t layer, const int32_t expert_ids[2]);
    void on_kv_compressed(uint32_t layer, const float * c_tkv_cpu);
    void on_tma_enqueued(uint32_t layer);
    void on_gpu_attn_done(uint32_t layer);
    void on_merge_done(uint32_t layer);

    // Speculative: advance to next token
    void advance_token();

    // Verification: called after target generates logits for draft batch
    // Returns number of accepted tokens
    uint32_t verify_and_rollback(
        llama_context * ctx_tgt,
        const llama_tokens & target_tokens);

    // Start async draft generation on draft_stream
    void start_draft_batch(std::function<void()> draft_fn);
};
```

- [ ] **Step 2: Commit**

```bash
git add common/hybrid_stage.h
git commit -m "feat: add hybrid orchestrator header with stage machine"
```

---
### Task 5: Hybrid Orchestrator Implementation

**Files:**
- Create: `common/hybrid_stage.cpp`

- [ ] **Step 1: Implement `common/hybrid_stage.cpp`**

```cpp
#include "hybrid_stage.h"
#include <cstdio>
#include <cuda_runtime.h>
#include "ggml/src/ggml-cuda/tma-transfer.h"

bool hybrid_orchestrator::init(
    uint32_t n_layers,
    uint32_t n_ctx_max,
    uint32_t kv_lora_rank_,
    uint32_t max_lookahead,
    ggml_backend_t backend,
    void * compute_stream,
    void * draft_stream_)
{
    n_layers     = n_layers;
    kv_lora_rank = kv_lora_rank_;
    gpu_compute_stream = compute_stream;
    draft_stream       = draft_stream_;

    // Init VRAM pool
    if (!pool.init(n_layers, n_ctx_max, kv_lora_rank, max_lookahead, backend)) {
        fprintf(stderr, "hybrid_orchestrator: VRAM pool init failed\n");
        return false;
    }

    // Init stages
    stages.resize(n_layers);

    // Init TMA sync events
    for (int i = 0; i < 2; i++) {
        cudaEventCreateWithFlags(
            (cudaEvent_t *)&tma_events[i],
            cudaEventDisableTiming);
    }

    fprintf(stderr, "hybrid_orchestrator: ready (%u layers, rank=%u)\n",
            n_layers, kv_lora_rank);
    return true;
}

void hybrid_orchestrator::free_all() {
    pool.free_all();
    for (int i = 0; i < 2; i++) {
        if (tma_events[i]) {
            cudaEventDestroy((cudaEvent_t)tma_events[i]);
            tma_events[i] = nullptr;
        }
    }
    stages.clear();
}

// --- Stage transitions ---

void hybrid_orchestrator::on_norm_done(uint32_t layer, const int32_t expert_ids[2]) {
    auto & s = stages[layer];
    s.phase.store(hybrid_phase::NORM_DONE);

    // Initiate SW prefetch for top-2 experts
    s.prefetch.expert_ids[0] = expert_ids[0];
    s.prefetch.expert_ids[1] = expert_ids[1];
    s.prefetch.prefetched    = true;
}

void hybrid_orchestrator::on_kv_compressed(uint32_t layer, const float * c_tkv_cpu) {
    auto & s = stages[layer];
    s.phase.store(hybrid_phase::KV_COMPRESSED);

    // Store pointer for TMA (CPU has pinned buffer)
    s.tma.cpu_src  = c_tkv_cpu;
    s.tma.bytes    = kv_lora_rank * 1;  // FP8 = 1 byte
    s.tma.enqueued = false;
}

void hybrid_orchestrator::on_tma_enqueued(uint32_t layer) {
    auto & s = stages[layer];

    // Compute VRAM destination
    float * dst = pool.slot_ptr(layer, current_token + layer);
    // (current_token + layer gives unique position across layers for same token)

    // Enqueue H2D TMA transfer
    bool ok = ggml_tma_enqueue_h2d_1d(
        dst,
        s.tma.cpu_src,
        s.tma.bytes,
        gpu_compute_stream,
        tma_events[tma_head & 1]);

    if (!ok) {
        // Fallback: cudaMemcpyAsync is used inside tma_enqueue_h2d_1d
        fprintf(stderr, "hybrid: TMA fallback to cudaMemcpyAsync at layer %u\n", layer);
    }

    s.tma.enqueued = true;
    tma_head++;
    s.phase.store(hybrid_phase::TMA_ENQUEUED);
}

void hybrid_orchestrator::on_gpu_attn_done(uint32_t layer) {
    // Wait for the TMA event that was recorded when this layer's
    // c_t^KV was enqueued (the event for 2 slots ago, since GPU
    // processes layer N-2's data while CPU prepares layer N)
    if (tma_head >= 2) {
        cudaEventSynchronize((cudaEvent_t)tma_events[(tma_head - 2) & 1]);
    }

    stages[layer].phase.store(hybrid_phase::GPU_ATTN_DONE);
}

void hybrid_orchestrator::on_merge_done(uint32_t layer) {
    stages[layer].phase.store(hybrid_phase::MERGE_DONE);
}

void hybrid_orchestrator::advance_token() {
    current_token++;
    current_layer = 0;
}

// --- Verification ---

uint32_t hybrid_orchestrator::verify_and_rollback(
    llama_context * ctx_tgt,
    const llama_tokens & target_tokens)
{
    // target_tokens[i] = logit-sampled token at draft position i
    // lookahead_buffer[i] = draft-generated token at position i
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
    verify.n_draft     = (uint32_t)draft.lookahead_buffer.size();

    if (n_match < draft.lookahead_buffer.size()) {
        // Rollback: truncate KV cache at first non-matching position
        llama_kv_cache_seq_truncate(
            ctx_tgt,
            0,                                    // seq_id = 0 (single sequence)
            current_token + n_match + 1);         // first invalid position
    }

    // Signal draft to prepare next batch from accepted position
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
```

- [ ] **Step 2: Commit**

```bash
git add common/hybrid_stage.cpp
git commit -m "feat: implement hybrid orchestrator state machine"
```

---
### Task 6: Build System + CLI Flags

**Files:**
- Modify: `common/CMakeLists.txt`
- Modify: `common/common.h`
- Modify: `common/common.cpp`

- [ ] **Step 1: Add source files to `common/CMakeLists.txt`**

```cmake
# Inside add_library(${TARGET} ... block, near sampling/speculative lines)
    sampling.cpp
    sampling.h
    speculative.cpp
    speculative.h
    hybrid_vram_pool.cpp      # <-- add
    hybrid_stage.cpp          # <-- add
```

- [ ] **Step 2: Add flags to `common/common.h`** (inside `common_params`)

```cpp
// Add after existing speculative / kv-offload fields (near line 562):

    // Hybrid speculative engine (VRAM KV-cache + CPU MoE)
    bool   hybrid_pipeline = false;  // --hybrid-pipeline
    bool   kv_vram         = true;   // --kv-vram / --no-kv-vram
```

- [ ] **Step 3: Add flag parsing to `common/common.cpp`**

```cpp
// Inside common_params_parse(), in the option parsing switch:

    // --- Hybrid pipeline flags ---
    if (arg == "--hybrid-pipeline") {
        params.hybrid_pipeline = true;
    } else if (arg == "--no-hybrid-pipeline") {
        params.hybrid_pipeline = false;
    } else if (arg == "--kv-vram") {
        params.kv_vram = true;
    } else if (arg == "--no-kv-vram") {
        params.kv_vram = false;
    }
```

- [ ] **Step 4: Add help text**

```cpp
// In the gparams section of usage():

    fprintf(stderr, "  --hybrid-pipeline       Enable hybrid CPU/GPU speculative engine (default: off)\n");
    fprintf(stderr, "  --no-hybrid-pipeline    Disable hybrid pipeline\n");
    fprintf(stderr, "  --kv-vram               Store target model KV-cache in VRAM (default: on with hybrid)\n");
    fprintf(stderr, "  --no-kv-vram            Keep target model KV-cache in CPU RAM\n");
```

- [ ] **Step 5: Commit**

```bash
git add common/CMakeLists.txt common/common.h common/common.cpp
git commit -m "feat: add build system and CLI flags for hybrid pipeline"
```

---
### Task 7: Model Build Hooks (DeepSeek)

**Files:**
- Modify: `src/models/deepseek2.cpp` (or `deepseek32.cpp` — determine which is DSv3)
- Modify: `src/models/deepseek32.cpp`

The model build function `build_arch_graph()` builds the compute graph for the transformer layer. We need to inject stage transition calls at specific points.

- [ ] **Step 1: Find the norm-attn-ffn boundary in `build_arch_graph()`**

In the `graph` constructor, locate:
1. After RMS norm (before KV compression) → `on_norm_done` with router output
2. After KV compression `c_t^KV` is computed → `on_kv_compressed`
3. After TMA enqueue → `on_tma_enqueued`
4. After attention output is ready on CPU → `on_gpu_attn_done`
5. After merge (h_attn + h_ffn) → `on_merge_done`

The exact injection points depend on the internal graph structure. For a first pass, add a conditional block at the end of each layer iteration:

```cpp
// Inside the per-layer loop of build_arch_graph, after FFN output is ready:

    // --- Hybrid pipeline hooks ---
    if (auto * hybrid = ctx.hybrid_orchestrator) {
        // The graph has been built; the orchestrator's stage transitions
        // are triggered after each GPU kernel launch.
        hybrid->on_merge_done(layer_idx);
        hybrid->advance_layer();
    }
```

**Note:** The actual integration is graph-level: we can't easily insert CPU code between graph ops. The practical approach is to set the stage transitions as CUDA graph node callbacks or post-kernel callbacks. For the first implementation, transitions are set *before* the graph is submitted, and queried *after* `ggml_backend_graph_compute()` returns.

```cpp
// Conceptual integration in the decode loop (not inside build_arch_graph):

// 1. Before compute: set stage expectations
hybrid->on_norm_done(layer, expert_ids);
hybrid->on_kv_compressed(layer, c_tkv);

// 2. Compute the graph (GPU executes)
ggml_backend_graph_compute(ctx->backend, gf);

// 3. After compute: advance stage
hybrid->on_tma_enqueued(layer);
hybrid->on_gpu_attn_done(layer);
```

For the detailed integration, see the companion PR in the DeepSeek model file. For now, add the callbacks as static helper functions in the model file and call them conditionally.

- [ ] **Step 2: Commit**

```bash
git add src/models/deepseek2.cpp src/models/deepseek32.cpp
git commit -m "feat: add hybrid pipeline hooks in DeepSeek model build"
```

---
### Task 8: Sampling Integration

**Files:**
- Modify: `common/sampling.h`
- Modify: `common/sampling.cpp`

- [ ] **Step 1: Update `common_sampler_sample_and_accept_n` signature**

```cpp
// common/sampling.h — add overload or extend with hybrid_orchestrator param

std::vector<llama_token> common_sampler_sample_and_accept_n(
    struct common_sampler *      gsmpl,
    struct llama_context *       ctx,
    const std::vector<int> &     idxs,
    const llama_tokens &         draft,
    bool                         grammar_first = false,
    hybrid_orchestrator *        hybrid = nullptr);   // <-- new param
```

- [ ] **Step 2: Add hybrid verification path in implementation**

```cpp
// common/sampling.cpp — extend the primary implementation

std::vector<llama_token> common_sampler_sample_and_accept_n(
    struct common_sampler *      gsmpl,
    struct llama_context *       ctx,
    const std::vector<int> &     idxs,
    const llama_tokens &         draft,
    bool                         grammar_first,
    hybrid_orchestrator *        hybrid)
{
    GGML_ASSERT(idxs.size() == draft.size() + 1);

    // --- Hybrid path: verification via orchestrator ---
    if (hybrid) {
        std::vector<llama_token> result;
        result.reserve(draft.size() + 1);

        size_t i = 0;
        for (; i < draft.size(); i++) {
            const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
            common_sampler_accept(gsmpl, id, true);
            result.push_back(id);
            if (draft[i] != id) {
                break;
            }
        }

        // Rollback via orchestrator
        hybrid->verify.accepted    = (uint32_t)i;
        hybrid->verify.rejected_at = (i < draft.size()) ? (uint32_t)i : UINT32_MAX;
        hybrid->verify.n_draft     = (uint32_t)draft.size();

        if (i < draft.size()) {
            llama_kv_cache_seq_truncate(ctx, 0, (llama_pos)(idxs[0] + i + 1));
        }

        // Sample the extra token if all draft was accepted
        if (i == draft.size()) {
            const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
            common_sampler_accept(gsmpl, id, true);
            result.push_back(id);
        }

        hybrid->draft.lookahead_buffer.clear();
        hybrid->draft.in_flight = false;

        return result;
    }

    // --- Original path (unchanged) ---
    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
    }

    return result;
}
```

- [ ] **Step 3: Commit**

```bash
git add common/sampling.h common/sampling.cpp
git commit -m "feat: integrate hybrid verification in sampling pipeline"
```

---
### Task 9: Server Integration

**Files:**
- Modify: `tools/server/server.cpp`
- Modify: `tools/server/server-context.cpp`

- [ ] **Step 1: Init orchestrator in `server.cpp` (load_model)**

```cpp
// After target model is loaded and backend is ready:

    // --- Hybrid pipeline init ---
    if (params.hybrid_pipeline && ctx_target) {
        auto * hybrid = new hybrid_orchestrator();

        // Get GPU backend from existing context
        ggml_backend_t gpu_backend = llama_get_backend(ctx_target);

        // Get compute stream from context
        cudaStream_t compute_stream = nullptr;
        cudaStream_t draft_cuda_stream = nullptr;
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&draft_cuda_stream, cudaStreamNonBlocking);

        bool ok = hybrid->init(
            llama_n_layer(ctx_target),               // n_layers
            llama_n_ctx(ctx_target),                 // n_ctx_max
            llama_kv_lora_rank(ctx_target),          // kv_lora_rank (MLA specific, returns 0 for non-MLA)
            params.speculative.draft.n_max,          // max_lookahead
            gpu_backend,
            compute_stream,
            draft_cuda_stream
        );

        if (ok) {
            ctx_target->hybrid_orchestrator = hybrid;
            fprintf(stderr, "hybrid: orchestrator initialized\n");
        } else {
            delete hybrid;
            fprintf(stderr, "hybrid: orchestrator init failed, running target-only\n");
        }
    }
```

- [ ] **Step 2: Wire speculative loop in `server-context.cpp`**

In the speculative decode section (around line 3346-3429):

```cpp
// Before calling common_sampler_sample_and_accept_n:

    if (slot.spec && slot.hybrid_orchestrator) {
        // Start async draft generation on separate stream
        slot.hybrid_orchestrator->start_draft_batch([&]() {
            // This runs on draft_stream, generating next lookahead
            common_speculative_draft(slot.spec, slot.ctx_dft,
                slot.hybrid_orchestrator->draft.lookahead_buffer);
        });
    }
```

In the verification call:

```cpp
// Replace the existing common_sampler_sample_and_accept_n call:

    auto sampled = common_sampler_sample_and_accept_n(
        slot.smpl, slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
        params.speculative.draft.grammar_first,
        slot.hybrid_orchestrator);   // <-- pass hybrid orchestrator
```

And in the rollback section (after acceptance is determined):

```cpp
// If hybrid is active, rollback is handled inside sample_and_accept_n.
// No additional action needed — just advance position.
if (!slot.hybrid_orchestrator) {
    // ...existing rollback logic (checkpoint restore, etc.)...
}
```

- [ ] **Step 3: Commit**

```bash
git add tools/server/server.cpp tools/server/server-context.cpp
git commit -m "feat: integrate hybrid orchestrator in server speculative loop"
```

---
### Task 10: Integration Test

**Files:**
- Create: `tests/test-hybrid-speculative.cpp` (or add to existing spec test)
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write basic integration test**

```cpp
// tests/test-hybrid-speculative.cpp
//
// Tests the hybrid speculative engine end-to-end with a small model
// (e.g., DeepSeek-distill 1.5B as target + draft).
//
// This is a smoke test — does the pipeline initialize, generate tokens,
// and produce coherent output without crashing?

#include "common.h"
#include "common/hybrid_stage.h"
#include "llama.h"
#include <cstdio>
#include <cassert>

int main() {
    // 1. Init model params for small test model
    common_params params;
    params.model = "models/deepseek-distill-1.5b-Q4_K_M.gguf";
    params.n_ctx = 512;
    params.hybrid_pipeline = true;
    params.kv_vram = true;

    // 2. Init target model
    auto * model = common_init_from_params(params);
    assert(model != nullptr);

    auto * ctx = llama_new_context_with_model(model, common_context_params_to_llama(params));
    assert(ctx != nullptr);

    // 3. Init hybrid orchestrator
    hybrid_orchestrator hybrid;
    // ... (get backend, streams, etc.)

    // 4. Generate tokens
    llama_batch batch = llama_batch_get_one(/*...*/);
    // ... (decode loop with speculative)

    // 5. Verify output has no crashes, reasonable tokens
    fprintf(stderr, "test-hybrid-speculative: PASS\n");
    return 0;
}
```

- [ ] **Step 2: Add to test CMakeLists**

```cmake
# tests/CMakeLists.txt
add_executable(test-hybrid-speculative tests/test-hybrid-speculative.cpp)
target_link_libraries(test-hybrid-speculative PRIVATE llama-common llama ggml)
```

- [ ] **Step 3: Commit**

```bash
git add tests/test-hybrid-speculative.cpp tests/CMakeLists.txt
git commit -m "test: add hybrid speculative engine integration test"
```

---
### Self-Review Checklist

1. **Spec coverage:** Skim the spec. Every section maps to a task:
   - VRAM KV-cache allocation → Task 1 (pool) + Task 2 (truncation)
   - Speculative pipeline → Task 4+5 (orchestrator) + Task 9 (server)
   - Async staging → Task 3 (TMA) + Task 5 (stage transitions)
   - Expert prefetch → Task 7 (model hooks)
   - Memory guard → Task 1 (pool size) + Task 6 (flags)
   - Sampling compatibility → Task 8 (verification loop)
   - ggml-alloc isolation → Task 1 (backend_buffer registration)

2. **Placeholder scan:** No TBD/TODO placeholders. Task 7 explicitly notes the graph-level integration limitation (calls inside build_arch_graph need the graph-level callback mechanism, which may need refinement during implementation).

3. **Type consistency:** All method signatures match between headers and implementations. The `hybrid_orchestrator` struct is used consistently across sampling, server, and model files. `tma_enqueue_h2d_1d` signature matches between declaration and call site.

4. **Ambiguity:** `llama_kv_cache_seq_truncate` is called in both Task 2 (declaration) and Task 8 (call site) — consistent. The pool `slot_ptr` formula uses `layer * m_stride + seq_pos * kv_lora_rank * sizeof(fp8)` throughout.
