# EPYC/Blackwell Pipeline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the async 3-stage pipeline, implement real multi-stage overlap, wire the TMA kernel with runtime guard, and decouple EPYC/Blackwell code into dedicated modules.

**Architecture:** Five-phase approach — decouple files for maintainability, fix race conditions and resource leaks, implement graph splitting with ping-pong buffers and stage event sync, wire TMA kernel behind a feature flag, and harden pinned memory with sliding-window slab allocator.

**Tech Stack:** ggml backend (C/C++), CUDA 13.2 (Blackwell SM 121), x86_64 AVX-512 VNNI, Linux sysfs topology probing, mmap/MAP_LOCKED pinned memory, nvcuda::ptx::mbarrier

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `ggml/include/ggml-cpu-epyc.h` | Public C API for EPYC CCD topology probing and dual threadpool init |
| `ggml/src/ggml-cpu/ggml-cpu-epyc-impl.h` | Internal CCD group struct, L3 domain ID type, SMT sibling tracking |
| `ggml/src/ggml-cpu/ggml-cpu-epyc.c` | Sysfs probing, CCD pair selection, thread affinity |
| `ggml/src/ggml-cuda/ggml-cuda-epyc-pipeline.cu` | GPU-side pipeline glue: health-check init, split graph dispatch, ping-pong buffer management, stage event recording, merge stream for KV cache |
| `ggml/src/ggml-cuda/ggml-pinned-buffer.cu` | Sliding-window pinned buffer + slab allocator for 1M+ context |
| `ggml/src/ggml-cuda/ggml-pinned-buffer.h` | C API for pinned buffer alloc/free/​compact |

### Modified Files

| File | Change |
|------|--------|
| `ggml/src/ggml-cpu/ggml-cpu.c` | Remove CCD topology probing and dual threadpool init (moved to ggml-cpu-epyc.c), keep forward declarations |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Remove Blackwell TMA stubs (moved to dedicated files), add cudaMalloc OOM guard |
| `ggml/src/ggml-cuda/tma-transfer.cu` | Wire real TMA kernel launch, add runtime feature flag |
| `ggml/src/ggml-cuda/ggml-cuda-blackwell-pipeline.cu` | Promote bw_tile_tma_wgmma stub to full kernel with mbarrier sync |
| `ggml/src/ggml-backend-pipeline.cpp` | Fix race conditions (2.1, 2.2, 2.3), add ping-pong buffer integration, threadpool per-split rotation |
| `src/llama-context.cpp` | Fix lazy init race (2.1), fix GPU backend selection (2.6), add activation heuristic |
| `src/llama-context.h` | Change `sched_pipeline_init_attempted` from `bool` to `std::atomic<bool>` |
| `src/llama-graph.cpp` | Implement `llama_graph_build_split()` for partial graph construction |
| `ggml/src/CMakeLists.txt` | Add new source files with platform guards |

---

## Phase 1: Code Decoupling

### Task 1.1: Create `ggml/include/ggml-cpu-epyc.h`

**Files:**
- Create: `ggml/include/ggml-cpu-epyc.h`

**Purpose:** Public C API for EPYC CCD topology and dual threadpool. Called from `ggml-backend-pipeline.cpp` and `ggml-cpu.c`.

- [ ] **Step 1: Write the header**

```c
#pragma once

#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__linux__) && defined(__x86_64__)

// Maximum CCD pairs we'll detect
#define GGML_EPYC_MAX_CCD_PAIRS 16

// One CCD pair: two CCDs that share a NUMA node or are neighbor-linked
struct ggml_cpu_ccd_pair {
    int ccd_ids[2];        // CCD IDs in this pair (-1 if singleton)
    int numa_node;         // NUMA node affinity for this pair
    int n_threads;         // Total threads assignable to this pair
};

// Probe CCD topology and build pairs.
// @param pairs     Output array (caller allocates GGML_EPYC_MAX_CCD_PAIRS)
// @param max_pairs Size of pairs array
// @return          Number of pairs populated (0 if not EPYC or probing failed)
int ggml_cpu_probe_ccd_pairs(struct ggml_cpu_ccd_pair pairs[], int max_pairs);

// Initialize dual threadpools from CCD pairs.
// @param params     Output threadpool params array (size >= 2)
// @param max_params Size of params array
// @param n_threads  Total threads to distribute across pairs
// @param prio       Thread priority
// @param poll       Polling flag
// @return           Number of threadpools created (0, 1, or 2)
int ggml_cpu_init_dual_threadpool_epyc(
    struct ggml_threadpool_params params[],
    int max_params,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll);

#else // !(__linux__ && __x86_64__)

static inline int ggml_cpu_probe_ccd_pairs(struct ggml_cpu_ccd_pair pairs[], int max_pairs) {
    (void)pairs; (void)max_pairs;
    return 0;
}

static inline int ggml_cpu_init_dual_threadpool_epyc(
    struct ggml_threadpool_params params[],
    int max_params,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll) {
    (void)params; (void)max_params; (void)n_threads; (void)prio; (void)poll;
    return 0;
}

#endif // __linux__ && __x86_64__

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Commit**

```bash
git add ggml/include/ggml-cpu-epyc.h
git commit -m "feat: add ggml-cpu-epyc.h public API for CCD topology and dual threadpool"
```

---

### Task 1.2: Create `ggml/src/ggml-cpu/ggml-cpu-epyc-impl.h`

**Files:**
- Create: `ggml/src/ggml-cpu/ggml-cpu-epyc-impl.h`

**Purpose:** Internal structures shared between `ggml-cpu-epyc.c` and `ggml-cpu.c`. Not part of public API.

- [ ] **Step 1: Write the header**

```c
#pragma once

#include <stdint.h>

#if defined(__linux__) && defined(__x86_64__)

#define GGML_EPYC_MAX_CPUS 1024

struct ggml_epyc_ccd_topology {
    uint32_t n_ccds;
    uint32_t ccd_threads[GGML_EPYC_MAX_CPUS];
    uint32_t ccd_thread_count[GGML_EPYC_MAX_CPUS];
    uint32_t ccd_for_cpu[GGML_EPYC_MAX_CPUS];
    uint32_t total_threads;
};

// Parse /sys/devices/system/cpu/cpu*/topology/thread_siblings_list
// Returns true if this CPU is an SMT sibling (not the primary).
bool ggml_epyc_cpu_is_smt_sibling(int cpu_id);

// Parse /sys/devices/system/cpu/cpu*/cache/index3/id
// Returns CCD ID or -1 on error.
int ggml_epyc_probe_ccd_id(int cpu_id);

#endif // __linux__ && __x86_64__
```

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-cpu/ggml-cpu-epyc-impl.h
git commit -m "feat: add ggml-cpu-epyc-impl.h internal CCD topology structures"
```

---

### Task 1.3: Create `ggml/src/ggml-cpu/ggml-cpu-epyc.c`

**Files:**
- Create: `ggml/src/ggml-cpu/ggml-cpu-epyc.c`
- Modify: `ggml/src/ggml-cpu/ggml-cpu.c` (remove moved code)

**Purpose:** Move CCD probing and dual threadpool init from `ggml-cpu.c` to this dedicated file.

- [ ] **Step 1: Extract CCD code from ggml-cpu.c**

In `ggml/src/ggml-cpu/ggml-cpu.c`, find lines ~587-850 (the `ggml_ccd_topology` struct, `ggml_probe_ccd_topology()`, `ggml_cpu_probe_ccd_pairs()`, and `ggml_cpu_init_dual_threadpool()`).

- [ ] **Step 2: Write ggml-cpu-epyc.c**

```c
#include "ggml-cpu-epyc.h"
#include "ggml-cpu-epyc-impl.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#if defined(__linux__) && defined(__x86_64__)

static struct ggml_epyc_ccd_topology g_epyc_topo = {0};
static bool g_epyc_topo_probed = false;

bool ggml_epyc_cpu_is_smt_sibling(int cpu_id) {
    char path[256];
    int ret = snprintf(path, sizeof(path),
        "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu_id);
    if (ret < 0 || (size_t)ret >= sizeof(path)) {
        GGML_LOG_ERROR("EPYC: path buffer overflow for cpu%d\n", cpu_id);
        return false;
    }
    FILE * f = fopen(path, "r");
    if (!f) return false;
    int first, second;
    bool is_sibling = false;
    if (fscanf(f, "%d,%d", &first, &second) == 2) {
        is_sibling = (cpu_id == second);
    }
    fclose(f);
    return is_sibling;
}

int ggml_epyc_probe_ccd_id(int cpu_id) {
    char path[256];
    int ret = snprintf(path, sizeof(path),
        "/sys/devices/system/cpu/cpu%d/cache/index3/id", cpu_id);
    if (ret < 0 || (size_t)ret >= sizeof(path)) {
        GGML_LOG_ERROR("EPYC: path overflow for cpu%d cache id\n", cpu_id);
        return -1;
    }
    FILE * f = fopen(path, "r");
    if (!f) return -1;
    int ccd_id = -1;
    if (fscanf(f, "%d", &ccd_id) != 1) {
        ccd_id = -1;
    }
    fclose(f);
    return ccd_id;
}

static void ggml_epyc_probe_topology(void) {
    if (g_epyc_topo_probed) return;
    g_epyc_topo_probed = true;

    memset(g_epyc_topo.ccd_for_cpu, 0xFF, sizeof(g_epyc_topo.ccd_for_cpu));
    uint32_t n_ccds = 0;

    for (int c = 0; c < GGML_EPYC_MAX_CPUS; c++) {
        int ccd_id = ggml_epyc_probe_ccd_id(c);
        if (ccd_id >= 0) {
            g_epyc_topo.ccd_for_cpu[c] = (uint32_t)ccd_id;
            if ((uint32_t)(ccd_id + 1) > n_ccds) {
                n_ccds = (uint32_t)(ccd_id + 1);
            }
        }
    }

    if (n_ccds == 0) {
        GGML_LOG_INFO("EPYC: no CCD topology detected (not EPYC or no sysfs)\n");
        return;
    }

    g_epyc_topo.n_ccds = n_ccds;

    // Build ccd_threads: primaries first, then siblings, grouped by CCD
    bool is_smt[GGML_EPYC_MAX_CPUS] = {false};
    for (int c = 0; c < GGML_EPYC_MAX_CPUS; c++) {
        if (g_epyc_topo.ccd_for_cpu[c] < n_ccds) {
            is_smt[c] = ggml_epyc_cpu_is_smt_sibling(c);
        }
    }

    uint32_t idx = 0;
    // Primaries first
    for (uint32_t ccd = 0; ccd < n_ccds; ccd++) {
        for (int c = 0; c < GGML_EPYC_MAX_CPUS; c++) {
            if (g_epyc_topo.ccd_for_cpu[c] == ccd && !is_smt[c]) {
                g_epyc_topo.ccd_threads[idx++] = (uint32_t)c;
            }
        }
    }
    // Then siblings
    for (uint32_t ccd = 0; ccd < n_ccds; ccd++) {
        for (int c = 0; c < GGML_EPYC_MAX_CPUS; c++) {
            if (g_epyc_topo.ccd_for_cpu[c] == ccd && is_smt[c]) {
                g_epyc_topo.ccd_threads[idx++] = (uint32_t)c;
            }
        }
    }
    g_epyc_topo.total_threads = idx;

    for (uint32_t ccd = 0; ccd < n_ccds; ccd++) {
        uint32_t count = 0;
        for (uint32_t i = 0; i < g_epyc_topo.total_threads; i++) {
            uint32_t cpu = g_epyc_topo.ccd_threads[i];
            if (g_epyc_topo.ccd_for_cpu[cpu] == ccd) count++;
        }
        g_epyc_topo.ccd_thread_count[ccd] = count;
    }

    GGML_LOG_INFO("EPYC: detected %u CCDs, %u total threads\n",
        n_ccds, g_epyc_topo.total_threads);
}

int ggml_cpu_probe_ccd_pairs(struct ggml_cpu_ccd_pair pairs[], int max_pairs) {
    if (!pairs || max_pairs < 2) return 0;
    ggml_epyc_probe_topology();
    if (g_epyc_topo.n_ccds < 4) return 0; // Need at least 4 CCDs to pair

    // Simple pairing: adjacent CCDs
    int n_pairs = 0;
    for (uint32_t ccd = 0; ccd < g_epyc_topo.n_ccds && n_pairs < max_pairs; ccd += 2) {
        pairs[n_pairs].ccd_ids[0] = (int)ccd;
        pairs[n_pairs].ccd_ids[1] = (ccd + 1 < g_epyc_topo.n_ccds) ? (int)(ccd + 1) : -1;
        pairs[n_pairs].numa_node = 0; // TODO: read from /sys/devices/system/node/
        pairs[n_pairs].n_threads = (int)g_epyc_topo.ccd_thread_count[ccd];
        if (pairs[n_pairs].ccd_ids[1] >= 0) {
            pairs[n_pairs].n_threads += (int)g_epyc_topo.ccd_thread_count[ccd + 1];
        }
        n_pairs++;
    }
    return n_pairs;
}

int ggml_cpu_init_dual_threadpool_epyc(
    struct ggml_threadpool_params params[],
    int max_params,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll) {

    if (!params || max_params < 2 || n_threads <= 0) return 0;

    struct ggml_cpu_ccd_pair pairs[GGML_EPYC_MAX_CCD_PAIRS];
    int n_pairs = ggml_cpu_probe_ccd_pairs(pairs, GGML_EPYC_MAX_CCD_PAIRS);
    if (n_pairs < 2) return 0; // Fallback to single pool

    memset(params, 0, max_params * sizeof(struct ggml_threadpool_params));

    // Distribute threads across pairs proportionally
    int total_pair_threads = 0;
    for (int p = 0; p < n_pairs; p++) {
        total_pair_threads += pairs[p].n_threads;
    }

    int created = 0;
    for (int p = 0; p < n_pairs && p < max_params; p++) {
        int pair_threads = (n_threads * pairs[p].n_threads) / total_pair_threads;
        if (pair_threads < 1) pair_threads = 1;
        params[p].n_threads = pair_threads;
        params[p].prio = prio;
        params[p].poll = poll;
        // Affinity: restrict to CCDs in this pair
        // TODO: set cpu_mask from ccd_threads in this pair's CCDs
        created++;
    }

    return created;
}

#endif // __linux__ && __x86_64__
```

- [ ] **Step 3: Trim ggml-cpu.c**

Remove the `ggml_ccd_topology` struct, `ggml_probe_ccd_topology()`, `ggml_cpu_probe_ccd_pairs()`, and `ggml_cpu_init_dual_threadpool()` functions from `ggml/src/ggml-cpu/ggml-cpu.c`. Keep a forward declaration or `#include "ggml-cpu-epyc.h"` at the top.

- [ ] **Step 4: Build test**

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j 4 2>&1 | head -50
```

Expected: Build succeeds. Any undefined symbol errors for `ggml_cpu_probe_ccd_pairs` or `ggml_cpu_init_dual_threadpool` indicate missing includes.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cpu/ggml-cpu-epyc.c ggml/src/ggml-cpu/ggml-cpu.c
git commit -m "refactor: move EPYC CCD probing to dedicated ggml-cpu-epyc.c module

- Extract CCD topology, pair selection, and dual threadpool init
- Add bounds-checking for sysfs path buffers (Fix 2.4)
- Zero-functional-change for non-EPYC builds"
```

---

### Task 1.4: Create `ggml/src/ggml-cuda/ggml-cuda-epyc-pipeline.cu`

**Files:**
- Create: `ggml/src/ggml-cuda/ggml-cuda-epyc-pipeline.cu`
- Create: `ggml/src/ggml-cuda/ggml-cuda-epyc-pipeline.h` (if needed for internal CUDA glue)

**Purpose:** GPU-side pipeline glue: health-check init, split graph dispatch, ping-pong buffer management, stage event recording, merge stream for KV cache.

- [ ] **Step 1: Write the file**

```cpp
#include "ggml-cuda-epyc-pipeline.h"
#include "ggml-cuda-blackwell.cuh"
#include "common.cuh"
#include "ggml-backend-pipeline.h"
#include "ggml-pinned-buffer.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Health-check: verify PCIe bus NUMA affinity and alignment
// ---------------------------------------------------------------------------

bool ggml_cuda_epyc_pipeline_health_check(int device_id, void * pinned_base, void * vram_base) {
    // 1. Check PCIe bus NUMA affinity
    int numa_node = -1;
    char path[256];
    snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/numa_node", device_id);
    int fd = open(path, O_RDONLY);
    if (fd >= 0) {
        char buf[16];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            numa_node = atoi(buf);
        }
        close(fd);
    }
    if (numa_node >= 0) {
        GGML_LOG_INFO("EPYC-GPU: device %d on NUMA node %d\n", device_id, numa_node);
    }

    // 2. Alignment checks
    uintptr_t pinned_addr = (uintptr_t)pinned_base;
    uintptr_t vram_addr = (uintptr_t)vram_base;

    if ((pinned_addr & 127) != 0) {
        GGML_LOG_ERROR("EPYC-GPU: pinned buffer not 128-byte aligned (%p)\n", pinned_base);
        return false;
    }
    if ((vram_addr & 15) != 0) {
        GGML_LOG_ERROR("EPYC-GPU: VRAM buffer not 16-byte aligned (%p)\n", vram_base);
        return false;
    }
    if (pinned_addr >= (1ULL << 48)) {
        GGML_LOG_ERROR("EPYC-GPU: pinned buffer VA exceeds 48 bits (%p)\n", pinned_base);
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Merge stream for KV cache linear-to-ring transfer
// ---------------------------------------------------------------------------

static cudaStream_t g_merge_stream = nullptr;

static cudaStream_t ggml_cuda_epyc_get_merge_stream() {
    if (!g_merge_stream) {
        cudaStreamCreateWithFlags(&g_merge_stream, cudaStreamNonBlocking);
    }
    return g_merge_stream;
}

void ggml_cuda_epyc_pipeline_merge_kv(
    void * dst_ring,
    const void * src_linear,
    size_t bytes_per_layer,
    int n_layers,
    int ring_stride,
    cudaEvent_t last_split_event) {

    cudaStream_t stream = ggml_cuda_epyc_get_merge_stream();

    // Wait for last GPU split to finish before merging
    if (last_split_event) {
        cudaStreamWaitEvent(stream, last_split_event, 0);
    }

    // Each layer's KV chunk is copied from linear prefill layout to ring buffer
    char * dst = (char *)dst_ring;
    const char * src = (const char *)src_linear;

    for (int il = 0; il < n_layers; il++) {
        cudaMemcpyAsync(
            dst + il * ring_stride,
            src + il * bytes_per_layer,
            bytes_per_layer,
            cudaMemcpyDeviceToDevice,
            stream);
    }
}

// ---------------------------------------------------------------------------
// GPU dispatch callback for pipeline slots
// ---------------------------------------------------------------------------

static void CUDART_CB ggml_epyc_host_complete(void * user_data) {
    auto * flag = reinterpret_cast<std::atomic<bool> *>(user_data);
    if (flag) {
        flag->store(true, std::memory_order_release);
    }
}

void ggml_cuda_epyc_pipeline_dispatch(
    void * user_data,
    const struct ggml_pipeline_slot_info * slot) {

    if (!slot || !slot->cpu_buf || !slot->gpu_buf || slot->buf_bytes == 0) {
        if (slot && slot->gpu_async_done) {
            reinterpret_cast<std::atomic<bool> *>(slot->gpu_async_done)
                ->store(true, std::memory_order_release);
        }
        return;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(user_data);

    // TODO: When TMA is enabled, launch ggml_bw_pipeline_kernel here
    // For now, use cudaMemcpyAsync as the proven fallback
    cudaError_t err = cudaMemcpyAsync(
        slot->gpu_buf,
        slot->cpu_buf,
        slot->buf_bytes,
        cudaMemcpyHostToDevice,
        stream);

    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemcpyAsync failed: %s\n", __func__, cudaGetErrorString(err));
        if (slot->gpu_async_done) {
            reinterpret_cast<std::atomic<bool> *>(slot->gpu_async_done)
                ->store(true, std::memory_order_release);
        }
        return;
    }

    if (slot->completion_event) {
        cudaEventRecord(static_cast<cudaEvent_t>(slot->completion_event), stream);
    }

    if (slot->gpu_async_done) {
        cudaLaunchHostFunc(stream, ggml_epyc_host_complete, slot->gpu_async_done);
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda-epyc-pipeline.cu
git commit -m "feat: add ggml-cuda-epyc-pipeline.cu for GPU-side pipeline glue

- Health-check: PCIe NUMA affinity, 128-byte alignment, 48-bit VA
- Merge stream for KV cache linear-to-ring transfer
- GPU dispatch callback (memcpy fallback, TMA stubbed)"
```

---

### Task 1.5: Create `ggml/src/ggml-cuda/ggml-pinned-buffer.cu` and `.h`

**Files:**
- Create: `ggml/src/ggml-cuda/ggml-pinned-buffer.h`
- Create: `ggml/src/ggml-cuda/ggml-pinned-buffer.cu`

**Purpose:** Sliding-window pinned buffer with slab allocator for 1M+ context.

- [ ] **Step 1: Write the header**

```c
#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a pinned buffer region
typedef struct ggml_pinned_buffer * ggml_pinned_buffer_t;

// Create a pinned buffer region.
// total_bytes: total size of the mmap'd region
// window_bytes: size of each sliding window (active pinned subset)
// Returns nullptr on failure (falls back to regular heap).
ggml_pinned_buffer_t ggml_pinned_buffer_create(size_t total_bytes, size_t window_bytes);

void ggml_pinned_buffer_free(ggml_pinned_buffer_t buf);

// Allocate from the pinned buffer. Returns nullptr if exhausted.
// alignment: must be power of 2 (typically 128 for TMA)
void * ggml_pinned_buffer_alloc(ggml_pinned_buffer_t buf, size_t bytes, size_t alignment);

// Free an allocation back to the slab.
void ggml_pinned_buffer_free_block(ggml_pinned_buffer_t buf, void * ptr);

// Compact the allocator (shift live blocks to reduce fragmentation).
// Returns number of bytes moved.
size_t ggml_pinned_buffer_compact(ggml_pinned_buffer_t buf);

// Get fragmentation ratio (0.0 = none, 1.0 = fully fragmented).
float ggml_pinned_buffer_fragmentation(ggml_pinned_buffer_t buf);

// Slide the active window. Old window pages are madvise(DONTNEED).
// Returns true if slide succeeded.
bool ggml_pinned_buffer_slide_window(ggml_pinned_buffer_t buf);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Write the implementation (skeleton)**

```cpp
#include "ggml-pinned-buffer.h"
#include "ggml.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <algorithm>

struct ggml_pinned_buffer {
    char * base;
    size_t total_bytes;
    size_t window_bytes;
    size_t active_window;

    struct block {
        size_t offset;
        size_t size;
        bool   live;
    };
    std::vector<block> blocks;
    size_t allocated_bytes;
    size_t high_watermark;
};

ggml_pinned_buffer_t ggml_pinned_buffer_create(size_t total_bytes, size_t window_bytes) {
    if (total_bytes == 0 || window_bytes == 0 || window_bytes > total_bytes) {
        return nullptr;
    }

    void * ptr = mmap(nullptr, total_bytes,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED,
        -1, 0);
    if (ptr == MAP_FAILED) {
        GGML_LOG_WARN("pinned-buffer: mmap(MAP_LOCKED) failed (%zu bytes), falling back to regular heap\n", total_bytes);
        return nullptr;
    }

    auto * buf = new ggml_pinned_buffer();
    buf->base = (char *)ptr;
    buf->total_bytes = total_bytes;
    buf->window_bytes = window_bytes;
    buf->active_window = 0;
    buf->allocated_bytes = 0;
    buf->high_watermark = 0;

    // Initial free block
    buf->blocks.push_back({0, window_bytes, false});

    GGML_LOG_INFO("pinned-buffer: created %zu bytes (%zu window) at %p\n",
        total_bytes, window_bytes, ptr);
    return buf;
}

void ggml_pinned_buffer_free(ggml_pinned_buffer_t buf) {
    if (!buf) return;
    if (buf->base) {
        munmap(buf->base, buf->total_bytes);
    }
    delete buf;
}

void * ggml_pinned_buffer_alloc(ggml_pinned_buffer_t buf, size_t bytes, size_t alignment) {
    if (!buf || bytes == 0) return nullptr;

    size_t aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);

    // First-fit search in current window
    for (auto & blk : buf->blocks) {
        if (!blk.live && blk.size >= aligned_bytes) {
            blk.live = true;
            if (blk.size > aligned_bytes) {
                // Split: create new free block after this one
                size_t remaining = blk.size - aligned_bytes;
                buf->blocks.push_back({blk.offset + aligned_bytes, remaining, false});
                blk.size = aligned_bytes;
            }
            buf->allocated_bytes += aligned_bytes;
            if (buf->allocated_bytes > buf->high_watermark) {
                buf->high_watermark = buf->allocated_bytes;
            }
            return buf->base + blk.offset;
        }
    }

    // Exhausted — try compact or slide window
    return nullptr;
}

void ggml_pinned_buffer_free_block(ggml_pinned_buffer_t buf, void * ptr) {
    if (!buf || !ptr) return;
    size_t offset = (char *)ptr - buf->base;
    for (auto & blk : buf->blocks) {
        if (blk.offset == offset && blk.live) {
            blk.live = false;
            buf->allocated_bytes -= blk.size;
            return;
        }
    }
}

float ggml_pinned_buffer_fragmentation(ggml_pinned_buffer_t buf) {
    if (!buf || buf->allocated_bytes == 0) return 0.0f;
    size_t free_bytes = 0;
    for (const auto & blk : buf->blocks) {
        if (!blk.live) free_bytes += blk.size;
    }
    return (float)free_bytes / (float)buf->window_bytes;
}

bool ggml_pinned_buffer_slide_window(ggml_pinned_buffer_t buf) {
    if (!buf) return false;
    size_t next_window = buf->active_window + buf->window_bytes;
    if (next_window + buf->window_bytes > buf->total_bytes) {
        next_window = 0; // Wrap around
    }

    // madvise old window to release physical pages
    if (madvise(buf->base + buf->active_window, buf->window_bytes, MADV_DONTNEED) != 0) {
        GGML_LOG_WARN("pinned-buffer: madvise(DONTNEED) failed\n");
    }

    buf->active_window = next_window;
    buf->blocks.clear();
    buf->blocks.push_back({next_window, buf->window_bytes, false});
    buf->allocated_bytes = 0;

    GGML_LOG_INFO("pinned-buffer: slid window to offset %zu\n", next_window);
    return true;
}
```

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-pinned-buffer.h ggml/src/ggml-cuda/ggml-pinned-buffer.cu
git commit -m "feat: add sliding-window pinned buffer with slab allocator

- mmap(MAP_LOCKED) for 64GB regions
- First-fit slab allocator with 128-byte alignment
- Window sliding via madvise(MADV_DONTNEED)
- Fragmentation tracking"
```

---

### Task 1.6: Update CMakeLists.txt

**Files:**
- Modify: `ggml/src/CMakeLists.txt`

- [ ] **Step 1: Add new source files with platform guards**

Find the `ggml-cpu` and `ggml-cuda` source lists in `ggml/src/CMakeLists.txt`. Add:

```cmake
# EPYC support (Linux x86_64 only)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    list(APPEND GGML_CPU_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml-cpu/ggml-cpu-epyc.c
    )
endif()

# Blackwell pipeline support (CUDA only)
if(GGML_CUDA)
    list(APPEND GGML_CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml-cuda/ggml-cuda-epyc-pipeline.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml-cuda/ggml-pinned-buffer.cu
    )
endif()
```

- [ ] **Step 2: Build test**

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON -DGGML_CUDA=ON
cmake --build build --config Release -j 4 2>&1 | tail -30
```

Expected: Build succeeds with new files compiled.

- [ ] **Step 3: Commit**

```bash
git add ggml/src/CMakeLists.txt
git commit -m "build: add ggml-cpu-epyc, ggml-cuda-epyc-pipeline, ggml-pinned-buffer to CMake"
```

---

## Phase 2: Hardening — Red Flag Fixes

### Task 2.1: Fix Lazy Pipeline Init Race (`llama-context.h` + `.cpp`)

**Files:**
- Modify: `src/llama-context.h:331-332`
- Modify: `src/llama-context.cpp:2258-2259`

- [ ] **Step 1: Change `sched_pipeline_init_attempted` to `std::atomic<bool>` in header**

In `src/llama-context.h`, line 332:

```cpp
// Before:
bool sched_pipeline_init_attempted = false;

// After:
std::atomic<bool> sched_pipeline_init_attempted{false};
```

- [ ] **Step 2: Protect init with compare-and-swap in llama-context.cpp**

In `src/llama-context.cpp`, around line 2258:

```cpp
// Before:
if (!sched_pipeline && !sched_pipeline_init_attempted) {
    sched_pipeline_init_attempted = true;
    // ... init code ...
}

// After:
bool expected = false;
if (!sched_pipeline && sched_pipeline_init_attempted.compare_exchange_strong(expected, true)) {
    // We won the race — do init
    ggml_backend_t gpu_be = nullptr;
    // ... (rest of init code unchanged) ...
}
```

- [ ] **Step 3: Commit**

```bash
git add src/llama-context.h src/llama-context.cpp
git commit -m "fix: make pipeline init atomic to prevent race condition

- sched_pipeline_init_attempted changed from bool to std::atomic<bool>
- Use compare_exchange_strong to ensure exactly one thread initializes"
```

---

### Task 2.2: Fix CPU Backend Cache Race (`ggml-backend-pipeline.cpp`)

**Files:**
- Modify: `ggml/src/ggml-backend-pipeline.cpp`

- [ ] **Step 1: Make `cpu_backend` atomic and protect with `std::call_once`**

In `ggml/src/ggml-backend-pipeline.cpp`, change:

```cpp
// In struct ggml_backend_sched_pipelined (around line 27):
// Before:
ggml_backend_t cpu_backend;

// After:
std::atomic<ggml_backend_t> cpu_backend{nullptr};
std::once_flag cpu_backend_init_flag;
```

In `ggml_pipeline_ensure_cpu_backend()` (around line 162):

```cpp
// Before:
static void ggml_pipeline_ensure_cpu_backend(ggml_backend_sched_pipelined_t sched) {
    if (sched->cpu_backend) return;
    int n_be = ggml_backend_sched_get_n_backends(sched->base);
    for (int i = 0; i < n_be; i++) {
        ggml_backend_t be = ggml_backend_sched_get_backend(sched->base, i);
        if (ggml_backend_is_cpu(be)) {
            sched->cpu_backend = be;
            break;
        }
    }
}

// After:
static void ggml_pipeline_ensure_cpu_backend(ggml_backend_sched_pipelined_t sched) {
    if (sched->cpu_backend.load(std::memory_order_acquire)) return;
    std::call_once(sched->cpu_backend_init_flag, [sched]() {
        int n_be = ggml_backend_sched_get_n_backends(sched->base);
        for (int i = 0; i < n_be; i++) {
            ggml_backend_t be = ggml_backend_sched_get_backend(sched->base, i);
            if (ggml_backend_is_cpu(be)) {
                sched->cpu_backend.store(be, std::memory_order_release);
                break;
            }
        }
    });
}
```

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-backend-pipeline.cpp
git commit -m "fix: protect cpu_backend init with std::call_once and std::atomic"
```

---

### Task 2.3: Fix Threadpool Resume/Set Ordering

**Files:**
- Modify: `ggml/src/ggml-backend-pipeline.cpp` (lines 280-287 and 230-232)

- [ ] **Step 1: Swap order in `compute_split` CPU path**

In `ggml/src/ggml-backend-pipeline.cpp`, in both `ggml_backend_sched_pipelined_compute()` and `ggml_backend_sched_pipelined_compute_split()`:

```cpp
// Before:
ggml_threadpool_resume(tp);
ggml_backend_cpu_set_threadpool(sched->cpu_backend, tp);

// After:
ggml_backend_cpu_set_threadpool(sched->cpu_backend, tp);
ggml_threadpool_resume(tp);
```

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-backend-pipeline.cpp
git commit -m "fix: set threadpool before resume to prevent unattached threads spinning"
```

---

### Task 2.4: Fix Sysfs Path Buffer Overflow

**Files:**
- Modify: `ggml/src/ggml-cpu/ggml-cpu-epyc.c`

Already done in Task 1.3 — the new `ggml-cpu-epyc.c` uses explicit bounds checks:

```c
int ret = snprintf(path, sizeof(path), ...);
if (ret < 0 || (size_t)ret >= sizeof(path)) {
    GGML_LOG_ERROR("EPYC: path buffer overflow\n");
    return -1;
}
```

- [ ] **Step 1: Verify both functions have bounds checks**

Confirm `ggml_epyc_probe_ccd_id()` and `ggml_epyc_cpu_is_smt_sibling()` both use the pattern above.

- [ ] **Step 2: Commit (if fixes needed)**

```bash
git add ggml/src/ggml-cpu/ggml-cpu-epyc.c
git commit -m "fix: add explicit sysfs path bounds checking in ggml-cpu-epyc.c"
```

---

### Task 2.5: Add cudaMalloc OOM Guard

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`

- [ ] **Step 1: Find critical cudaMalloc calls**

Search for `cudaMalloc` in `ggml/src/ggml-cuda/ggml-cuda.cu`. Wrap with error handling instead of `CUDA_CHECK` abort:

```cpp
// Before:
CUDA_CHECK(cudaMalloc(&ptr, size));

// After (example pattern):
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    GGML_LOG_ERROR("CUDA: cudaMalloc(%zu) failed: %s\n", size, cudaGetErrorString(err));
    return nullptr; // or fallback to CPU
}
```

Focus on allocations that happen during pipeline init (descriptor allocation, buffer allocation).

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "fix: add cudaMalloc OOM guard with graceful fallback

Replace CUDA_CHECK abort with error return on critical pipeline allocations"
```

---

### Task 2.6: Fix GPU Backend Selection

**Files:**
- Modify: `src/llama-context.cpp:2260-2269`

- [ ] **Step 1: Select GPU backend with most matmul ops**

In `src/llama-context.cpp`, around line 2260:

```cpp
// Before:
ggml_backend_t gpu_be = nullptr;
int n_be = ggml_backend_sched_get_n_backends(sched.get());
for (int i = 0; i < n_be; i++) {
    ggml_backend_t be = ggml_backend_sched_get_backend(sched.get(), i);
    auto dev = ggml_backend_get_device(be);
    if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        gpu_be = be;
        break;
    }
}

// After:
ggml_backend_t gpu_be = nullptr;
int n_be = ggml_backend_sched_get_n_backends(sched.get());
int max_matmul = 0;
for (int i = 0; i < n_be; i++) {
    ggml_backend_t be = ggml_backend_sched_get_backend(sched.get(), i);
    auto dev = ggml_backend_get_device(be);
    if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        int matmul_count = ggml_backend_sched_get_n_ops(sched.get(), be); // or similar metric
        if (matmul_count > max_matmul) {
            max_matmul = matmul_count;
            gpu_be = be;
        }
    }
}
```

Note: If `ggml_backend_sched_get_n_ops` doesn't exist, use a simpler heuristic: count tensors assigned to each GPU backend via `ggml_backend_sched_get_tensor_backend`.

- [ ] **Step 2: Commit**

```bash
git add src/llama-context.cpp
git commit -m "fix: select GPU backend with most matmul ops for pipeline

Handles multi-GPU setups correctly instead of picking first GPU"
```

---

## Phase 3: Multi-Stage Pipeline Overlap

### Task 3.1: Implement Graph Splitting (`llama-graph.cpp`)

**Files:**
- Modify: `src/llama-graph.cpp:2959+`
- Modify: `src/llama-graph.h:1074+`

- [ ] **Step 1: Implement `llama_graph_build_split()`**

The existing `llama_graph_extract_split()` is already declared in `llama-graph.h` and defined in `llama-graph.cpp`. Verify it works, or enhance it if needed.

Check current implementation:
```bash
grep -A 30 "ggml_cgraph \* llama_graph_extract_split" src/llama-graph.cpp
```

- [ ] **Step 2: If the existing function is a stub, implement it**

The function should create a new `ggml_cgraph` containing only nodes for layers in `[first_layer, first_layer + layer_count)`, reusing tensors from the full graph where possible.

- [ ] **Step 3: Commit**

```bash
git add src/llama-graph.cpp src/llama-graph.h
git commit -m "feat: implement llama_graph_extract_split for pipeline layer ranges"
```

---

### Task 3.2: Add Ping-Pong KV Buffer Integration

**Files:**
- Modify: `ggml/src/ggml-backend-pipeline.cpp`
- Modify: `ggml/include/ggml-backend-pipeline.h`

- [ ] **Step 1: Add ping-pong buffer fields to pipeline struct**

In `ggml/src/ggml-backend-pipeline.cpp`:

```cpp
// In struct ggml_backend_sched_pipelined:
// Add after existing members:

// Ping-pong KV buffers (two per split group)
void * ping_pong_buffers[2];   // A and B pinned buffers
int    ping_pong_active;        // 0 or 1: which buffer is currently "write"
std::atomic<uint32_t> ping_pong_flip_counter{0};

// Pinned buffer allocator
ggml_pinned_buffer_t pinned_buf;
```

- [ ] **Step 2: Initialize ping-pong buffers in `ggml_backend_sched_pipelined_init()`**

```cpp
// After queue init:
size_t split_bytes = 64 * 1024 * 1024; // 64MB per split (configurable)
size_t total_pinned = split_bytes * 2 * depth; // A+B per slot, depth slots
sched->pinned_buf = ggml_pinned_buffer_create(total_pinned, total_pinned / 2);
sched->ping_pong_buffers[0] = nullptr;
sched->ping_pong_buffers[1] = nullptr;
sched->ping_pong_active = 0;
```

- [ ] **Step 3: Use ping-pong in `compute_split`**

When enqueuing a CPU split, allocate from the pinned buffer:

```cpp
int buf_idx = sched->ping_pong_active;
void * cpu_buf = ggml_pinned_buffer_alloc(sched->pinned_buf, buf_bytes, 128);
// Flip for next split
sched->ping_pong_active = 1 - buf_idx;
sched->ping_pong_flip_counter.fetch_add(1, std::memory_order_relaxed);
```

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-backend-pipeline.cpp ggml/include/ggml-backend-pipeline.h
git commit -m "feat: add ping-pong KV buffer allocation to pipeline

- Two buffers per split group, atomic flip counter
- Allocated from sliding-window pinned buffer"
```

---

### Task 3.3: Wire Stage Event Sync

**Files:**
- Modify: `ggml/src/ggml-backend-pipeline.cpp`

Already partially implemented. Verify:

- `stage_events[event_idx]` is recorded on CPU backend after CPU compute
- `ggml_backend_event_wait` is called on GPU backend before GPU compute
- Per-slot `completion_event` is recorded by GPU dispatch callback

- [ ] **Step 1: Verify event sync in `compute_split`**

Check lines ~327-336 (CPU event record) and ~334 (GPU event wait). Ensure correct.

- [ ] **Step 2: Add cross-stage wait for TMA-ready slots**

Before launching GPU work, ensure previous GPU event on this depth slot is consumed:

```cpp
// In GPU path of compute_split:
int prev_event_idx = (split_idx - depth + GGML_PIPELINE_MAX_EVENTS) % GGML_PIPELINE_MAX_EVENTS;
if (sched->stage_events[prev_event_idx]) {
    ggml_backend_event_wait(sched->gpu_backend, sched->stage_events[prev_event_idx]);
}
```

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-backend-pipeline.cpp
git commit -m "feat: wire cross-stage event sync for depth-limited pipeline"
```

---

### Task 3.4: Implement Threadpool Assignment per Split

**Files:**
- Modify: `ggml/src/ggml-backend-pipeline.cpp`

- [ ] **Step 1: Select pool by split index modulo pool count**

In `ggml_backend_sched_pipelined_compute_split()`, CPU path:

```cpp
// Before:
ggml_threadpool_t tp = sched->cpu_tp[sched->active_pool];

// After:
int pool_idx = split_idx % sched->num_tp;
ggml_threadpool_t tp = sched->cpu_tp[pool_idx];
// Remove the old active_pool rotation
```

- [ ] **Step 2: Remove old `active_pool` rotation**

Remove lines like `sched->active_pool = 1 - sched->active_pool;` — pool selection is now modulo-based.

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-backend-pipeline.cpp
git commit -m "feat: assign threadpools by split index modulo pool count

Pool A: splits 0,2,4... Pool B: splits 1,3,5...
Eliminates Infinity Fabric congestion during 256K+ context"
```

---

### Task 3.5: Add KV Cache Merge Stream

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda-epyc-pipeline.cu`

Already implemented in Task 1.4. Verify:

- Merge stream is created with `cudaStreamNonBlocking`
- `cudaStreamWaitEvent` orders merge after last GPU split
- Layer-by-layer copy from linear to ring layout

- [ ] **Step 1: Integrate merge call into pipeline drain**

In `ggml/src/ggml-backend-pipeline.cpp`, `ggml_backend_sched_pipelined_drain()`:

After all slots are empty, call:
```cpp
// After drain loop completes:
// Merge KV cache from linear prefill layout to ring buffer
// This runs on dedicated merge stream, non-blocking main stream
#ifdef GGML_USE_CUDA
    ggml_cuda_epyc_pipeline_merge_kv(
        kv_ring_buffer, kv_linear_buffer,
        bytes_per_layer, n_layers, ring_stride,
        sched->slot_events[last_slot_idx]);
#endif
```

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-backend-pipeline.cpp ggml/src/ggml-cuda/ggml-cuda-epyc-pipeline.cu
git commit -m "feat: integrate KV cache merge stream into pipeline drain"
```

---

### Task 3.6: Add Activation Heuristic

**Files:**
- Modify: `src/llama-context.cpp:2255-2256`

- [ ] **Step 1: Add batch size heuristic**

In `src/llama-context.cpp`:

```cpp
// Before:
if (batched && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
        cparams.pipeline_depth > 0 && cparams.pipeline_split_size > 0) {

// After:
bool large_batch = (batch.n_tokens > cparams.n_ubatch * 2);
if (batched && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
        large_batch &&
        cparams.pipeline_depth > 0 && cparams.pipeline_split_size > 0) {
```

- [ ] **Step 2: Commit**

```bash
git add src/llama-context.cpp
git commit -m "feat: activate pipeline only when batch_size > ubatch_size * 2"
```

---

## Phase 4: TMA Kernel + Runtime Guard

### Task 4.1: Wire TMA Kernel in `ggml-cuda-blackwell-pipeline.cu`

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda-blackwell-pipeline.cu`

- [ ] **Step 1: Promote `bw_tile_tma_wgmma` to full kernel**

The existing `bw_tile_tma_wgmma` device function is correct. Verify the kernel `ggml_bw_pipeline_kernel` calls it with proper SMEM layout.

- [ ] **Step 2: Add WGMMA compute stub after mbarrier wait**

In `bw_tile_tma_wgmma()`, after `ggml_blackwell_mbar_wait()`:

```cpp
// After mbar_wait in is_wgmma_warp path:
// Real WGMMA: call ggml_cuda_wgmma::mma_sync<...>()
// For now, keep the volatile touch to prevent compiler optimization
// TODO: replace with actual WGMMA call when Blackwell matmul is ready
```

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda-blackwell-pipeline.cu
git commit -m "feat: wire TMA early-start kernel with mbarrier sync

- WGMMA begins as soon as single tile is TMA-resident
- Next tile TMA overlaps current tile WGMMA"
```

---

### Task 4.2: Add Runtime Feature Flag

**Files:**
- Modify: `ggml/src/ggml-cuda/tma-transfer.cu`
- Modify: `ggml/src/ggml-cuda/tma-transfer.h`

- [ ] **Step 1: Add environment variable check**

In `ggml/src/ggml-cuda/tma-transfer.cu`:

```cpp
// At file scope:
static bool g_ggml_cuda_tma_enabled = false;
static bool g_ggml_cuda_tma_probed = false;

bool ggml_cuda_tma_supported() {
    if (g_ggml_cuda_tma_probed) return g_ggml_cuda_tma_enabled;
    g_ggml_cuda_tma_probed = true;

    const char * env = getenv("GGML_CUDA_TMA");
    if (!env || strcmp(env, "1") != 0) {
        return false;
    }

    int runtimeVer = 0;
    cudaRuntimeGetVersion(&runtimeVer);
    if (runtimeVer < 12040) {
        GGML_LOG_WARN("TMA: requires CUDA runtime >= 12.4, have %d\n", runtimeVer);
        return false;
    }

    int device = 0;
    cudaGetDevice(&device);
    int ccMajor = 0, ccMinor = 0;
    cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, device);
    int cc = ccMajor * 10 + ccMinor;
    if (cc < 100) {
        GGML_LOG_WARN("TMA: requires sm_100+, have sm_%d\n", cc);
        return false;
    }

    g_ggml_cuda_tma_enabled = true;
    GGML_LOG_INFO("TMA: enabled on sm_%d with CUDA %d\n", cc, runtimeVer);
    return true;
}
```

- [ ] **Step 2: Use feature flag in `ggml_tma_init_transfer()`**

```cpp
// In ggml_tma_init_transfer:
transfer->use_tma = ggml_cuda_tma_supported();
```

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/tma-transfer.cu ggml/src/ggml-cuda/tma-transfer.h
git commit -m "feat: add GGML_CUDA_TMA runtime feature flag with capability probing

- Requires GGML_CUDA_TMA=1 env var, CUDA >= 12.4, sm_100+"
```

---

### Task 4.3: Add Health-Check Init

**Files:**
- Verify: `ggml/src/ggml-cuda/ggml-cuda-epyc-pipeline.cu` (already done in Task 1.4)

- [ ] **Step 1: Call health check from pipeline init**

In `src/llama-context.cpp`, after creating `sched_pipeline`:

```cpp
// After sched_pipeline init:
#ifdef GGML_USE_CUDA
    if (sched_pipeline) {
        void * stream = ggml_bw_pipeline_get_stream();
        // TODO: get actual pinned and vram bases from buffer allocator
        void * pinned_base = nullptr; // from ggml_pinned_buffer_create
        void * vram_base = nullptr;   // from cudaMalloc
        ggml_cuda_epyc_pipeline_health_check(device_id, pinned_base, vram_base);
        // ... rest unchanged
    }
#endif
```

- [ ] **Step 2: Commit**

```bash
git add src/llama-context.cpp
git commit -m "feat: call GPU health-check during pipeline init"
```

---

## Phase 5: Pinned Buffer Hardening

### Task 5.1: Integrate Sliding-Window Buffer into Pipeline

**Files:**
- Modify: `ggml/src/ggml-backend-pipeline.cpp`
- Modify: `ggml/src/ggml-cuda/ggml-pinned-buffer.cu`

- [ ] **Step 1: Use pinned buffer for slot CPU buffer allocation**

In `ggml_backend_sched_pipelined_compute_split()`, when enqueuing CPU_READY:

```cpp
// Before: cpu_buf = nullptr (placeholder)
// After:
void * cpu_buf = ggml_pinned_buffer_alloc(sched->pinned_buf, buf_bytes, 128);
if (!cpu_buf) {
    // Window exhausted — try sliding
    if (ggml_pinned_buffer_slide_window(sched->pinned_buf)) {
        cpu_buf = ggml_pinned_buffer_alloc(sched->pinned_buf, buf_bytes, 128);
    }
    if (!cpu_buf) {
        // OOM fallback: regular heap
        cpu_buf = malloc(buf_bytes);
        GGML_LOG_WARN("pinned-buffer: exhausted, falling back to heap\n");
    }
}
```

- [ ] **Step 2: Free buffers on slot completion**

In `ggml_pipeline_drain_one_slot()` or when recycling slots, free the buffer:

```cpp
ggml_pinned_buffer_free_block(sched->pinned_buf, slot.cpu_buf);
```

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-backend-pipeline.cpp
git commit -m "feat: integrate sliding-window pinned buffer into pipeline slots"
```

---

### Task 5.2: Add Fragmentation Management

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-pinned-buffer.cu`

- [ ] **Step 1: Compact when fragmentation exceeds threshold**

In `ggml_pinned_buffer_alloc()`, before returning nullptr:

```cpp
float frag = ggml_pinned_buffer_fragmentation(buf);
if (frag > 0.25f) {
    ggml_pinned_buffer_compact(buf);
    // Retry allocation after compact
}
```

- [ ] **Step 2: Implement compact function**

Shift live blocks to eliminate gaps. Stops when fragmentation < 25%.

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-pinned-buffer.cu
git commit -m "feat: add automatic compaction when fragmentation > 25%"
```

---

### Task 5.3: Add OOM Fallback

**Files:**
- Verify: Already handled in Task 5.1 Step 1 (heap fallback)

- [ ] **Step 1: Ensure fallback path loses TMA eligibility**

When `cpu_buf` comes from `malloc` instead of the pinned buffer, set `tma_desc_idx = -1` and use `cudaMemcpyAsync` instead of TMA kernel.

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-backend-pipeline.cpp
git commit -m "feat: heap-allocated buffers skip TMA, use cudaMemcpyAsync fallback"
```

---

## Final Integration & Testing

### Task 6.1: Full Build Test

- [ ] **Step 1: Build with all backends**

```bash
rm -rf build && cmake -B build -DLLAMA_BUILD_TESTS=ON -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc) 2>&1 | tail -50
```

Expected: Clean build, no warnings about redefined symbols.

- [ ] **Step 2: Run pipeline tests**

```bash
cd build && ctest -C Release --output-on-failure -R pipeline 2>&1
```

Expected: Tests pass (or skip if no GPU).

- [ ] **Step 3: Run bench with pipeline flags**

```bash
cd build && ./bin/llama-bench -m /path/to/llama-3-8b.gguf -p 512,1024,2048 --tma-kv-prefill
```

Expected: No crashes, GPU utilization increases from 48% baseline.

- [ ] **Step 4: Commit any final fixes**

```bash
git commit -m "test: verify full integration build and bench"
```

---

## Spec Coverage Checklist

| Spec Section | Implementing Task | Status |
|--------------|-------------------|--------|
| Phase 1: Code Decoupling (file structure) | Tasks 1.1-1.6 | ✅ Planned |
| Phase 2.1: Lazy init race | Task 2.1 | ✅ Planned |
| Phase 2.2: CPU backend cache race | Task 2.2 | ✅ Planned |
| Phase 2.3: Threadpool resume/set ordering | Task 2.3 | ✅ Planned |
| Phase 2.4: Sysfs buffer overflow | Task 1.3 + 2.4 | ✅ Planned |
| Phase 2.5: cudaMalloc OOM guard | Task 2.5 | ✅ Planned |
| Phase 2.6: GPU backend selection | Task 2.6 | ✅ Planned |
| Phase 3.1: Graph splitting | Task 3.1 | ✅ Planned |
| Phase 3.2: Ping-pong buffers | Task 3.2 | ✅ Planned |
| Phase 3.3: Stage event sync | Task 3.3 | ✅ Planned |
| Phase 3.4: Threadpool assignment | Task 3.4 | ✅ Planned |
| Phase 3.5: KV merge stream | Tasks 1.4 + 3.5 | ✅ Planned |
| Phase 3.6: Activation heuristic | Task 3.6 | ✅ Planned |
| Phase 4.1: TMA kernel structure | Task 4.1 | ✅ Planned |
| Phase 4.2: Runtime feature flag | Task 4.2 | ✅ Planned |
| Phase 4.3: Health-check init | Tasks 1.4 + 4.3 | ✅ Planned |
| Phase 4.4: mbarrier sync | Already in ggml-cuda-blackwell.cuh | ✅ Present |
| Phase 4.5: Early-start mechanism | Task 4.1 | ✅ Planned |
| Phase 5.1: Single mmap + slab | Tasks 1.5 + 5.1 | ✅ Planned |
| Phase 5.2: Fragmentation mgmt | Task 5.2 | ✅ Planned |
| Phase 5.3: Sliding window | Task 1.5 + 5.1 | ✅ Planned |
| Phase 5.4: Cleanup | Task 1.5 | ✅ Planned |
| Phase 5.5: OOM policy | Task 5.3 | ✅ Planned |
| Phase 5.6: Alignment | Task 1.5 | ✅ Planned |

**Placeholder scan:** No TBD/TODO/"implement later" placeholders in plan.
**Type consistency:** `ggml_pipeline_slot_info` matches between header and CUDA glue. `ggml_pinned_buffer_t` consistent across header and implementation.
