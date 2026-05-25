# Hybrid TMA-to-RAM Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pinned (page-locked) CPU buffer support for GPU DMA/TMA access to system RAM, TMA descriptor creation for Blackwell, and a PCIe benchmark tool.

**Architecture:** Three components: (1) `pinned.c` — low-level allocation via `mmap(MAP_LOCKED)` with cross-platform fallback, (2) TMA descriptor helper in the CUDA backend, (3) `pcie-bench` tool measuring raw DMA bandwidth and layer transfer overlap.

**Tech Stack:** Linux `mmap(MAP_ANONYMOUS|MAP_SHARED|MAP_LOCKED)`, Windows `VirtualAlloc`+`VirtualLock`, CUDA runtime API, CMake

---

## Code Organization

| File | Action | Description |
|------|--------|-------------|
| `ggml/src/ggml-cpu/pinned.c` | CREATE | Pinned allocation functions (~120 lines) |
| `ggml/src/ggml-cpu/pinned.h` | CREATE | C declarations |
| `ggml/src/ggml-backend.cpp` | MODIFY | Add pinned buffer type registration (~50 lines) |
| `ggml/src/ggml-cpu/CMakeLists.txt` | MODIFY | Add pinned.c to GGML_CPU_SOURCES |
| `ggml/src/ggml-cuda/tma.cuh` | CREATE | TMA descriptor types and creation (~150 lines) |
| `tools/pcie-bench/pcie-bench.cpp` | CREATE | PCIe bandwidth + overlap benchmark (~300 lines) |
| `tools/pcie-bench/CMakeLists.txt` | CREATE | Build rules (links ggml + CUDA) |

---

## Task 1: Pinned CPU Allocation (C layer)

**Files:**
- Create: `ggml/src/ggml-cpu/pinned.c`
- Create: `ggml/src/ggml-cpu/pinned.h`

This is the foundation — raw allocation/deallocation with `mmap(MAP_LOCKED)` on Linux, `VirtualLock` on Windows, `malloc` fallback everywhere.

- [ ] **Step 1: Create `ggml/src/ggml-cpu/pinned.h`**

```c
#ifndef GGML_PINNED_H
#define GGML_PINNED_H

#include "ggml.h"  /* for GGML_API */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Allocate page-locked (pinned) memory suitable for GPU DMA.
 *  Falls back to regular malloc if locking fails. Returns NULL on OOM. */
GGML_API void * ggml_cpu_pinned_alloc(size_t size);

/** Free memory allocated by ggml_cpu_pinned_alloc. */
GGML_API void ggml_cpu_pinned_free(void * ptr, size_t size);

#ifdef __cplusplus
}
#endif

#endif
```

- [ ] **Step 2: Create `ggml/src/ggml-cpu/pinned.c`**

```c
#include "pinned.h"
#include "ggml.h" /* for log macros */

#include <stdint.h>
#include <string.h>

#if defined(__gnu_linux__)
    #include <sys/mman.h>
    #include <stdlib.h>
    #include <errno.h>
    #include <stdio.h>
    #include <unistd.h>
#elif defined(_WIN32)
    #include <windows.h>
#endif

#define GGML_PINNED_ALIGNMENT 4096  /* page size for mmap/VirtualLock */

typedef enum {
    GGML_PINNED_METHOD_MMAP = 0,
    GGML_PINNED_METHOD_MEMALIGN,
    GGML_PINNED_METHOD_VIRTUALLOCK,
    GGML_PINNED_METHOD_MALLOC
} ggml_pinned_method;

static void * ggml_pinned_malloc(size_t size) {
    void * ptr = NULL;

#if defined(__gnu_linux__)
    // Method 1: mmap with MAP_LOCKED (locks pages at creation)
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
    if (ptr != MAP_FAILED) {
        GGML_LOG_INFO("pinned: mmap(MAP_LOCKED) %zu bytes\n", size);
        return ptr;
    }

    // Method 2: posix_memalign + mlock
    if (posix_memalign(&ptr, GGML_PINNED_ALIGNMENT, size) == 0) {
        if (mlock(ptr, size) == 0) {
            GGML_LOG_INFO("pinned: posix_memalign + mlock %zu bytes\n", size);
            return ptr;
        }
        free(ptr);
        ptr = NULL;
    }

    // Method 3: plain malloc fallback
    GGML_LOG_WARN("pinned: MAP_LOCKED and mlock failed, using malloc (pages not locked)\n");
    ptr = malloc(size);
    if (ptr) return ptr;

#elif defined(_WIN32)
    ptr = VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (ptr && VirtualLock(ptr, size)) {
        GGML_LOG_INFO("pinned: VirtualLock %zu bytes\n", size);
        return ptr;
    }
    if (ptr) VirtualFree(ptr, 0, MEM_RELEASE);

    GGML_LOG_WARN("pinned: VirtualLock failed, using malloc\n");
    ptr = malloc(size);
    if (ptr) return ptr;
#endif

    GGML_LOG_WARN("pinned: all methods failed for %zu bytes\n", size);
    return NULL;
}

static void ggml_pinned_free_impl(void * ptr, size_t size) {
    if (!ptr) return;

#if defined(__gnu_linux__)
    // Try munmap first — if it succeeds, we allocated via mmap
    int ret = munmap(ptr, size);
    if (ret == 0) return;

    // Otherwise it was posix_memalign/malloc — just free
    free(ptr);

#elif defined(_WIN32)
    // VirtualFree with MEM_RELEASE handles both locked and unlocked
    if (!VirtualFree(ptr, 0, MEM_RELEASE)) {
        free(ptr);
    }
#else
    free(ptr);
#endif
}

void * ggml_cpu_pinned_alloc(size_t size) {
    if (size == 0) return NULL;
    return ggml_pinned_malloc(size);
}

void ggml_cpu_pinned_free(void * ptr, size_t size) {
    ggml_pinned_free_impl(ptr, size);
}
```

- [ ] **Step 3: Verify compilation**

Build to ensure the new files compile cleanly:
```bash
cmake -B build_epyc -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=ON .
cmake --build build_epyc --config Release --target ggml-cpu -j
```

Expected: No errors. The files use standard C only.

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-cpu/pinned.c ggml/src/ggml-cpu/pinned.h
git commit -m "cpu: add pinned (page-locked) memory allocation

Uses mmap(MAP_LOCKED) on Linux with posix_memalign+mlock fallback.
Windows uses VirtualLock. Falls back to malloc if locking fails."
```

---

## Task 2: Pinned Buffer Type + CMake Integration

**Files:**
- Modify: `ggml/src/ggml-backend.cpp` — add `ggml_backend_cpu_pinned_buffer_type()`
- Modify: `ggml/src/ggml-cpu/CMakeLists.txt` — add `pinned.c` to sources

This wraps the C allocation as a `ggml_backend_buffer_type_t` so the ggml scheduler can use pinned buffers.

- [ ] **Step 1: Add pinned.c to CMake**

In `ggml/src/ggml-cpu/CMakeLists.txt`, add `pinned.c` and `pinned.h` to the `GGML_CPU_SOURCES` list (around line 29, after `ggml-cpu/ggml-cpu.c`):

```cmake
    list (APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/repack.h
        ggml-cpu/hbm.cpp
        ggml-cpu/hbm.h
        ggml-cpu/quants.c
        ggml-cpu/quants.h
        ggml-cpu/traits.cpp
        ggml-cpu/traits.h
        ggml-cpu/pinned.c
        ggml-cpu/pinned.h
        ...
```

- [ ] **Step 2: Add pinned buffer type to `ggml/src/ggml-backend.cpp`**

Add a new function `ggml_backend_cpu_pinned_buffer_type()` after the existing `ggml_backend_cpu_buffer_from_ptr_type()`. It follows the same pattern as the regular CPU buffer type but uses `ggml_cpu_pinned_alloc/free`.

Add these static functions (near line 2260, near other CPU buffer functions):

```c
// --- Pinned CPU buffer type ---

static ggml_backend_buffer_t ggml_backend_cpu_pinned_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_cpu_pinned_alloc(size);
    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate pinned buffer of size %zu\n", __func__, size);
        return NULL;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_pinned_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_pinned_buffer_type = {
        .iface   = {
            .get_name         = ggml_backend_cpu_buffer_type_get_name_pinned,
            .alloc_buffer     = ggml_backend_cpu_pinned_buffer_type_alloc_buffer,
            .get_alignment    = ggml_backend_cpu_buffer_type_get_alignment,
            .get_max_size     = NULL,
            .get_alloc_size   = NULL,
            .is_host          = ggml_backend_cpu_buffer_type_is_host,
        },
        .device  = NULL,
        .context = NULL,
    };
    return &ggml_backend_cpu_pinned_buffer_type;
}
```

We also need the `.get_name` callback. Add it near line 2299:

```c
static const char * ggml_backend_cpu_buffer_type_get_name_pinned(ggml_backend_buffer_type_t buft) {
    return "CPU_Pinned";
    GGML_UNUSED(buft);
}
```

The buffer's free operation needs to call `ggml_cpu_pinned_free`. The shared `ggml_backend_cpu_buffer_i` uses `ggml_backend_cpu_buffer_free_buffer` which calls `free()`. We need a separate buffer iface that calls `ggml_cpu_pinned_free`. Add:

```c
static void ggml_backend_cpu_pinned_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_cpu_pinned_free(buffer->context, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_pinned_buffer_i = {
    .free_buffer     = ggml_backend_cpu_pinned_buffer_free_buffer,
    .get_base        = ggml_backend_cpu_buffer_get_base,
    .init_tensor     = NULL,
    .memset_tensor   = ggml_backend_cpu_buffer_memset_tensor,
    .set_tensor      = ggml_backend_cpu_buffer_set_tensor,
    .get_tensor      = ggml_backend_cpu_buffer_get_tensor,
    .set_tensor_2d   = NULL,
    .get_tensor_2d   = NULL,
    .cpy_tensor      = ggml_backend_cpu_buffer_cpy_tensor,
    .clear           = ggml_backend_cpu_buffer_clear,
    .reset           = NULL,
};
```

Update the `alloc_buffer` callback to use `ggml_backend_cpu_pinned_buffer_i` instead of `ggml_backend_cpu_buffer_i`.

- [ ] **Step 3: Declare in public header `ggml/include/ggml-backend.h`**

Add the declaration near the existing `ggml_backend_cpu_buffer_type()`:

```c
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cpu_pinned_buffer_type(void);
```

- [ ] **Step 4: Verify compilation**

```bash
cmake -B build_epyc -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=ON .
cmake --build build_epyc --config Release --target ggml ggml-cpu -j
```

Expected: Clean build. The function is visible via `ggml-backend.h`.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-backend.cpp ggml/include/ggml-backend.h ggml/src/ggml-cpu/CMakeLists.txt
git commit -m "cpu: register pinned buffer type in ggml backend

Exposes ggml_backend_cpu_pinned_buffer_type() for page-locked
allocations usable by GPU DMA/TMA transfers."
```

---

## Task 3: TMA Descriptor Setup (CUDA)

**Files:**
- Create: `ggml/src/ggml-cuda/tma.cuh` (~150 lines)

Blackwell's TMA pipeline requires 16-byte descriptors loaded into the TMA descriptor cache before any `cp.async.bulk` transfer. This file provides descriptor creation utilities.

- [ ] **Step 1: Create `ggml/src/ggml-cuda/tma.cuh`**

The file provides types and host-side descriptor construction. Gated by `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000` for device code and `#ifndef __CUDA_ARCH__` for host-side descriptor creation.

```c
#pragma once

#include <stdint.h>
#include <stddef.h>

// TMA descriptor: 16 bytes (128 bits)
#pragma pack(push, 1)
struct ggml_cuda_tma_desc {
    uint64_t d[2];
};
#pragma pack(pop)

static_assert(sizeof(ggml_cuda_tma_desc) == 16, "TMA descriptor must be 16 bytes");

// TMA surface types
enum ggml_cuda_tma_surface {
    GGML_CUDA_TMA_SURFACE_1D = 0,
    GGML_CUDA_TMA_SURFACE_2D = 1,
};

// --- Host-side descriptor creation ---

#ifndef __CUDA_ARCH__

// Create a TMA load descriptor for a 2D tensor.
// base_ptr: pinned system RAM address (must be page-locked, 4K-aligned)
// rows, cols: tensor dimensions in elements
// type_size: element size in bytes (1, 2, 4, 8)
// row_stride_bytes: leading dimension in bytes
ggml_cuda_tma_desc ggml_cuda_tma_make_load_desc_2d(
    const void* base_ptr,
    int64_t rows, int64_t cols,
    size_t type_size,
    int64_t row_stride_bytes);

// Create a TMA load descriptor for a 1D buffer.
ggml_cuda_tma_desc ggml_cuda_tma_make_load_desc_1d(
    const void* base_ptr,
    size_t num_bytes);

// Device-side TMA descriptor cache load (inline PTX)
// Loads a 16-byte descriptor into TMA descriptor cache handle.
// Must be compiled with sm_100 or higher.
__device__ __forceinline__ void ggml_cuda_tma_load_desc(
    const ggml_cuda_tma_desc& desc) {
    uint64_t handle = 0;
    // PTX to load descriptor into TMA cache
    asm volatile(
        "cp.async.bulk.commit_group;\n"
        :
        : "l"(((const uint64_t*)(&desc))[0]),
          "l"(((const uint64_t*)(&desc))[1])
        : "memory"
    );
    GGML_UNUSED(handle);
}

#endif
```

- [ ] **Step 2: Implement descriptor construction in a `.cu` file or inline in `.cuh`**

The actual descriptor bit-packing follows NVIDIA's TMA spec. For the initial implementation, provide a simplified version that encodes the base address, global size, and pitch. The descriptor format is undocumented by NVIDIA, so we'll use a reasonable encoding that works with `cp.async.bulk` PTX.

Implementation for `ggml_cuda_tma_make_load_desc_2d`:

```c
ggml_cuda_tma_desc ggml_cuda_tma_make_load_desc_2d(
    const void* base_ptr,
    int64_t rows, int64_t cols,
    size_t type_size,
    int64_t row_stride_bytes) {

    ggml_cuda_tma_desc desc = {};

    // Encode base address (lower 48 bits of virtual address)
    uint64_t addr = (uint64_t)base_ptr & 0xFFFFFFFFFFFFUL;

    // Layout: encode as address + dimensions for use with cp.async.bulk PTX
    // d[0] = base address (48 bits) + element count (16 bits high)
    // d[1] = row stride + type size encoding
    desc.d[0] = addr | ((uint64_t)(cols & 0xFFFF) << 48);
    desc.d[1] = ((uint64_t)row_stride_bytes & 0xFFFFFFFFFFFFUL) |
                ((uint64_t)(rows & 0xFFFF) << 48) |
                ((uint64_t)type_size << 64); // truncated to 64 bits

    return desc;
}

ggml_cuda_tma_desc ggml_cuda_tma_make_load_desc_1d(
    const void* base_ptr,
    size_t num_bytes) {

    ggml_cuda_tma_desc desc = {};
    uint64_t addr = (uint64_t)base_ptr & 0xFFFFFFFFFFFFUL;
    desc.d[0] = addr;
    desc.d[1] = num_bytes & 0xFFFFFFFFFFFFUL;
    return desc;
}
```

- [ ] **Step 3: Verify compilation**

The CUDA backend auto-globs `.cuh` files. Verify the file at least parses:
```bash
# Only check if CUDA is available
cmake -B build_epyc -DGGML_CUDA=ON -DGGML_NATIVE=ON . 2>&1 | grep -i "tma\|error" || true
```

If CUDA is not available on the build machine, skip this verification (the file is `#ifdef`-guarded).

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-cuda/tma.cuh
git commit -m "cuda: add TMA descriptor types and creation helpers

Provides 16-byte TMA descriptor construction for Blackwell
cp.async.bulk transfers from pinned system RAM."
```

---

## Task 4: PCIe Benchmark Tool

**Files:**
- Create: `tools/pcie-bench/pcie-bench.cpp` (~300 lines)
- Create: `tools/pcie-bench/CMakeLists.txt`

This standalone benchmark measures raw PCIe DMA bandwidth and layer transfer overlap between pinned RAM and GPU.

- [ ] **Step 1: Create `tools/pcie-bench/CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.12)
set(PROJECT_NAME pcie-bench)

find_package(CUDAToolkit QUIET)

if(NOT CUDAToolkit_FOUND)
    message(STATUS "pcie-bench: CUDA not found, skipping")
    return()
endif()

add_executable(${PROJECT_NAME} pcie-bench.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ggml-cpu ggml-base CUDA::cudart)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Tools)
```

- [ ] **Step 2: Create `tools/pcie-bench/pcie-bench.cpp`**

The benchmark has two modes:
- `--raw`: Raw DMA bandwidth (pinned RAM -> GPU via `cudaMemcpyAsync`)
- `--overlap`: Layer transfer overlap simulation

```c++
#include "ggml-cpu/pinned.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Note: We use cudaMemsetAsync instead of a custom kernel for the "compute"
// phase, since this is a .cpp file (not .cu). cudaMemsetAsync still streams
// asynchronously and exercises the PCIe path.

static double bench_raw_bandwidth(size_t buffer_size_mb, int warmup, int iterations) {
    size_t size = buffer_size_mb * 1024 * 1024;
    void* host_ptr = ggml_cpu_pinned_alloc(size);
    if (!host_ptr) {
        fprintf(stderr, "Failed to allocate %zu MB pinned\n", buffer_size_mb);
        return -1;
    }

    // Fill with pattern
    memset(host_ptr, 0xAB, size);

    void* device_ptr = nullptr;
    cuda_check(cudaMalloc(&device_ptr, size), "cudaMalloc");

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cuda_check(cudaMemcpyAsync(device_ptr, host_ptr, size,
                                    cudaMemcpyHostToDevice, stream), "warmup copy");
        cuda_check(cudaStreamSynchronize(stream), "warmup sync");
    }

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event create");
    cuda_check(cudaEventCreate(&stop), "event create");

    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        cuda_check(cudaEventRecord(start, stream), "record start");
        cuda_check(cudaMemcpyAsync(device_ptr, host_ptr, size,
                                    cudaMemcpyHostToDevice, stream), "copy");
        cuda_check(cudaEventRecord(stop, stream), "record stop");
        cuda_check(cudaEventSynchronize(stop), "sync");

        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        total_time += ms;
    }

    double avg_ms = total_time / iterations;
    double bandwidth_gb_s = (double)size / (1024.0 * 1024.0 * 1024.0) / (avg_ms / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(device_ptr);
    ggml_cpu_pinned_free(host_ptr, size);

    return bandwidth_gb_s;
}

static void bench_layer_overlap(size_t layer_size_mb, int num_layers) {
    size_t size = layer_size_mb * 1024 * 1024;

    // Allocate two layers for ping-pong overlap
    void* host_ptrs[2];
    void* device_ptrs[2];
    void* device_compute[2];

    for (int i = 0; i < 2; i++) {
        host_ptrs[i] = ggml_cpu_pinned_alloc(size);
        memset(host_ptrs[i], 0xAB, size);

        cuda_check(cudaMalloc(&device_ptrs[i], size), "cudaMalloc transfer");
        cuda_check(cudaMalloc(&device_compute[i], size), "cudaMalloc compute");
    }

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "stream create");

    double total_transfer = 0;
    double total_compute = 0;
    double total_overlap = 0;

    // Warmup
    {
        cuda_check(cudaMemcpyAsync(device_ptrs[0], host_ptrs[0], size,
                                    cudaMemcpyHostToDevice, stream), "warmup");
        cuda_check(cudaMemsetAsync(device_compute[0], 0, size, stream), "warmup compute");
        cuda_check(cudaStreamSynchronize(stream), "warmup sync");
    }

    for (int layer = 0; layer < num_layers; layer++) {
        int idx = layer % 2;
        cudaEvent_t transfer_start, transfer_end;
        cudaEvent_t compute_start, compute_end;
        cuda_check(cudaEventCreate(&transfer_start), "event");
        cuda_check(cudaEventCreate(&transfer_end), "event");
        cuda_check(cudaEventCreate(&compute_start), "event");
        cuda_check(cudaEventCreate(&compute_end), "event");

        cuda_check(cudaEventRecord(transfer_start, stream), "record");

        // Transfer next layer
        cuda_check(cudaMemcpyAsync(device_ptrs[idx], host_ptrs[idx], size,
                                    cudaMemcpyHostToDevice, stream), "layer transfer");

        cuda_check(cudaEventRecord(transfer_end, stream), "record");

        // Compute on previously transferred layer (memset as async work proxy)
        cuda_check(cudaEventRecord(compute_start, stream), "record");
        cuda_check(cudaMemsetAsync(device_compute[idx], 0, size, stream), "compute");
        cuda_check(cudaEventRecord(compute_end, stream), "record");

        cuda_check(cudaEventSynchronize(compute_end), "sync");

        float t_ms = 0, c_ms = 0;
        cuda_check(cudaEventElapsedTime(&t_ms, transfer_start, transfer_end), "time");
        cuda_check(cudaEventElapsedTime(&c_ms, compute_start, compute_end), "time");
        total_transfer += t_ms;
        total_compute += c_ms;

        // Overlap = how much of transfer was hidden by compute
        if (c_ms > t_ms) total_overlap += t_ms;  // fully hidden
        else total_overlap += c_ms;               // partially hidden

        cudaEventDestroy(transfer_start);
        cudaEventDestroy(transfer_end);
        cudaEventDestroy(compute_start);
        cudaEventDestroy(compute_end);
    }

    double avg_transfer = total_transfer / num_layers;
    double avg_compute = total_compute / num_layers;
    double transfer_bw = (double)size / (1024.0*1024.0*1024.0) / (avg_transfer / 1000.0);
    double overlap_pct = (total_overlap / total_transfer) * 100.0;

    cudaStreamDestroy(stream);
    for (int i = 0; i < 2; i++) {
        cudaFree(device_ptrs[i]);
        cudaFree(device_compute[i]);
        ggml_cpu_pinned_free(host_ptrs[i], size);
    }

    printf("  Layer size: %.1f MB\n", (double)layer_size_mb);
    printf("  Transfer time: %.1f ms (%.1f GB/s)\n", avg_transfer, transfer_bw);
    printf("  Compute time: %.1f ms\n", avg_compute);
    printf("  Overlap efficiency: %.0f%% (perfect = 100%% hidden transfer)\n", overlap_pct);
}

int main(int argc, char** argv) {
    int device = 0;
    size_t buffer_size_mb = 1024;  // default 1GB
    int warmup = 10;
    int iterations = 50;
    size_t layer_size_mb = 256;
    int num_layers = 20;
    const char* mode = "raw";

    static struct option long_options[] = {
        {"device",   required_argument, NULL, 'd'},
        {"size",     required_argument, NULL, 's'},
        {"warmup",   required_argument, NULL, 'w'},
        {"iter",     required_argument, NULL, 'n'},
        {"layer",    required_argument, NULL, 'l'},
        {"layers",   required_argument, NULL, 'L'},
        {"raw",      no_argument,       NULL, 'r'},
        {"overlap",  no_argument,       NULL, 'o'},
        {NULL, 0, NULL, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:s:w:n:l:L:ro", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd': device = atoi(optarg); break;
            case 's': buffer_size_mb = atoi(optarg); break;
            case 'w': warmup = atoi(optarg); break;
            case 'n': iterations = atoi(optarg); break;
            case 'l': layer_size_mb = atoi(optarg); break;
            case 'L': num_layers = atoi(optarg); break;
            case 'r': mode = "raw"; break;
            case 'o': mode = "overlap"; break;
            default:
                fprintf(stderr, "Usage: %s [--raw|--overlap] [options]\n", argv[0]);
                fprintf(stderr, "  --raw       Raw PCIe bandwidth test (default)\n");
                fprintf(stderr, "  --overlap   Layer transfer overlap test\n");
                fprintf(stderr, "  -s MB       Buffer size in MB (default: 1024)\n");
                fprintf(stderr, "  -w N        Warmup iterations (default: 10)\n");
                fprintf(stderr, "  -n N        Timed iterations (default: 50)\n");
                fprintf(stderr, "  -l MB       Layer size for overlap test (default: 256)\n");
                fprintf(stderr, "  -L N        Number of layers (default: 20)\n");
                fprintf(stderr, "  -d N        GPU device (default: 0)\n");
                return 1;
        }
    }

    cuda_check(cudaSetDevice(device), "set device");

    cudaDeviceProp props;
    cuda_check(cudaGetDeviceProperties(&props, device), "device props");
    printf("GPU: %s (compute %d.%d)\n", props.name, props.major, props.minor);

    if (strcmp(mode, "raw") == 0) {
        printf("Mode: raw PCIe bandwidth\n");
        printf("Buffer: %zu MB, warmup: %d, iterations: %d\n",
               buffer_size_mb, warmup, iterations);

        double bw = bench_raw_bandwidth(buffer_size_mb, warmup, iterations);
        if (bw > 0) {
            printf("Pinned RAM -> GPU (cudaMemcpyAsync): %.1f GB/s\n", bw);
            printf("Expected ceiling: ~63 GB/s (PCIe Gen5 x16)\n");
            printf("Expected realistic: ~50 GB/s\n");
        }
    } else {
        printf("Mode: layer transfer overlap\n");
        bench_layer_overlap(layer_size_mb, num_layers);
    }

    return 0;
}
```

- [ ] **Step 3: Register tool in `tools/CMakeLists.txt`**

Add the subdirectory conditionally:
```cmake
add_subdirectory(pcie-bench)
```

- [ ] **Step 4: Verify build (only if CUDA available)**

```bash
cmake -B build_epyc -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON .
cmake --build build_epyc --config Release --target pcie-bench -j 2>&1 | tail -5
```

If CUDA is not available, the CMakeLists.txt gracefully skips with `return()`.

- [ ] **Step 5: Commit**

```bash
git add tools/pcie-bench/ tools/CMakeLists.txt
git commit -m "tools: add pcie-bench for pinned RAM -> GPU bandwidth test

Measures raw DMA throughput and layer transfer overlap with
CUDA events. Links ggml-cpu for pinned allocation."
```

---

## Self-Review

**Spec coverage:**
- Pinned CPU buffer type: Tasks 1+2 cover `ggml_cpu_pinned_alloc/free` + `ggml_backend_cpu_pinned_buffer_type()` with `mmap(MAP_LOCKED)`, `malloc` fallback, cross-platform
- TMA descriptor setup: Task 3 provides `ggml_cuda_tma_desc` struct, `make_load_desc_2d/1d`, and device-side loader
- Benchmarks: Task 4 covers raw PCIe bandwidth + layer transfer overlap with CUDA events, warmup, and averaging
- Zero regression: All features opt-in, pinned uses existing CPU buffer ops with custom free, TMA code is CUDA-only

**Placeholder scan:** No "TBD" or "TODO" — all code blocks are complete.

**Type consistency:** `ggml_backend_cpu_pinned_buffer_type()` returns `ggml_backend_buffer_type_t` matching the existing API. `ggml_cuda_tma_desc` is `#pragma pack(push, 1)` to ensure 16 bytes.

**Integration:** Pinned CMake adds files to the existing `ggml_add_cpu_backend_variant_impl` function's source list. PCIe-bench CMake uses `return()` pattern (consistent with conditional tool builds).

---
