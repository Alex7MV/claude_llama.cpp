# EPYC/Blackwell Pipeline Hardening — Production-Grade 3-Stage Async Pipeline

> **Prepared for:** AMD EPYC 9V74 (80-core Zen 4) + NVIDIA RTX 5090 (Blackwell SM 121)
> **Target:** 600GB+ models, 256K–1M+ context, zero-bubble execution
> **Current state:** Functional 3-stage pipeline with CCD threadpool rotation, TMA kernel stubbed, `depth`/`split_size` unused, race conditions in lazy init, EPYC/Blackwell code mixed into core files, GPU utilization stuck at 48%
> **Approach:** Refactor and harden existing pipeline — keep working orchestrator/queue, decouple EPYC/Blackwell into dedicated modules, implement real TMA kernel, CCD-local slab allocator, sliding-window pinned buffer

**Goal:** Harden the async pipeline, implement real multi-stage overlap, wire the TMA kernel with runtime guard, and decouple EPYC/Blackwell code into dedicated modules.

**Architecture:** Five-phase approach — decouple files for maintainability, fix race conditions and resource leaks, implement graph splitting with ping-pong buffers and stage event sync, wire TMA kernel behind a feature flag, and harden pinned memory with sliding-window slab allocator.

**Tech Stack:** ggml backend (C/C++), CUDA 13.2 (Blackwell SM 121), x86_64 AVX-512 VNNI, Linux sysfs topology probing, mmap/MAP_LOCKED pinned memory, nvcuda::ptx::mbarrier

---

## Phase 1: Code Decoupling

Move EPYC-specific and Blackwell-specific code out of core files into dedicated modules. Each module has one clear responsibility and a well-defined interface.

### Target File Structure

```
ggml/
 ├── include/
 │   ├── ggml-backend-pipeline.h       (extended — stage API, unchanged skeleton)
 │   └── ggml-cpu-epyc.h               (NEW — EPYC CCD topology + dual threadpool C API)
 ├── src/
 │   ├── ggml-cpu/
 │   │   ├── ggml-cpu.c                (trimmed — remove CCD probing, dual pool init)
 │   │   ├── ggml-cpu-epyc.c           (NEW — sysfs probing, CCD pair selection, affinity)
 │   │   └── ggml-cpu-epyc-impl.h      (NEW — CCD group struct, L3 domain ID, SMT tracking)
 │   └── ggml-cuda/
 │       ├── ggml-cuda.cu               (trimmed — remove Blackwell TMA stubs)
 │       ├── tma-transfer.cu            (enhanced — real TMA kernel + runtime guard)
 │       ├── tma-transfer.h             (enhanced — TMA transfer C API)
 │       ├── ggml-cuda-blackwell.cuh    (kept — TMA desc, mbarrier, WGMMA stubs)
 │       ├── ggml-cuda-blackwell-pipeline.cu (kept — host launcher, health check)
 │       └── ggml-cuda-epyc-pipeline.cu (NEW — GPU-side pipeline glue, merge stream)
```

### Module Responsibilities

| Module | Responsibility | Interface |
|--------|---------------|-----------|
| `ggml-cpu-epyc.h/c` | CCD topology via sysfs, dual threadpool creation, CCD pair selection | `ggml_probe_ccd_pairs()`, `ggml_cpu_init_dual_threadpool_epyc()` |
| `ggml-cpu-epyc-impl.h` | Internal CCD group struct, L3 domain ID, SMT sibling tracking | None (internal) |
| `ggml-cuda-blackwell.cuh` | TMA descriptor types, mbarrier wrappers, WGMMA Frag stubs | Device-side inline functions |
| `ggml-cuda-blackwell-pipeline.cu` | Host launcher, health-check init, SMEM validation | `ggml_bw_pipeline_launch()`, `ggml_bw_pipeline_validate_device()` |
| `ggml-cuda-epyc-pipeline.cu` | GPU-side pipeline: split graph dispatch, ping-pong buffer management, stage event recording, merge stream for KV cache | `ggml_cuda_epyc_pipeline_dispatch()`, `ggml_cuda_epyc_pipeline_merge_kv()` |
| `ggml-backend-pipeline.cpp` | Orchestrator only — stages, events, depth, rotation | Existing C API |

### Compilation Guards

- EPYC code compiled only on `__linux__ && __x86_64__`
- Blackwell code compiled only when `CUDA_ARCH >= 1000`
- CMake updated with new source files; existing `ggml-cpu.c` and `ggml-cuda.cu` untouched except for removed EPYC/BW code

---

## Phase 2: Hardening — Red Flag Fixes

Six fixes ordered by severity. Each is self-contained and testable.

### 2.1 — Lazy pipeline init race (`llama-context.cpp`)

Replace `bool sched_pipeline_init_attempted` with `std::atomic<bool>`. Protect `sched_pipeline` pointer init with `std::call_once`. The pipeline pointer becomes `std::shared_ptr<ggml_backend_sched_pipelined>` so concurrent readers never see a partial pointer.

### 2.2 — CPU backend cache race (`ggml-backend-pipeline.cpp`)

`sched->cpu_backend` becomes `std::atomic<ggml_backend_t*>`. Init protected by `std::call_once` stored in the pipeline struct. Scan-backend logic runs exactly once.

### 2.3 — Threadpool resume/set ordering (`ggml-backend-pipeline.cpp`)

Swap order: `ggml_backend_cpu_set_threadpool(backend, tp)` first, then `ggml_threadpool_resume(tp)`. Threads become active only after attachment to the backend's work queue.

### 2.4 — Sysfs path buffer overflow (`ggml-cpu-epyc.c`)

Replace `GGML_ASSERT(snprintf(...) < sizeof(path))` with explicit bounds check: `if (ret >= sizeof(path)) { log_error; return failure; }`. Safe in release builds.

### 2.5 — cudaMalloc OOM guard (`ggml-cuda.cu`)

Wrap critical allocation with return code check. On failure: log error, return `nullptr`, caller falls back to CPU. No `CUDA_CHECK` abort.

### 2.6 — GPU backend selection (`llama-context.cpp`)

Select the backend that the scheduler assigned the most matmul ops to, instead of picking the first GPU. This handles multi-GPU correctly.

---

## Phase 3: Multi-Stage Pipeline Overlap

The core throughput gain. Graph splitting, ping-pong buffers, and stage event sync.

### 3.1 — Graph Splitting

New function `llama_graph_build_split()` builds a partial `ggml_cgraph` for a layer range `[first_layer, first_layer + layer_count)`. Each split graph contains its own norm → QKV → RoPE → attention → FFN → residual chain, wired to the KV cache slice for that range. Splits are independent within a single prefill batch.

Given `split_size = 8` layers and a 64-layer model, the graph splits into 8 groups. Pipeline `depth = 3` allows 3 groups in-flight simultaneously:

```
Time -> |---S0---|---S1---|---S2---|---S0---|---S1---|---S2---|
CPU     | CG 0  | CG 1  | CG 2  | CG 3  | CG 4  | CG 5  |
TMA/GPU |       | T0+G0 |       | T1+G1 |       | T2+G2 |
```

### 3.2 — Ping-Pong KV Buffers

Two pinned RAM buffers (A, B) per split group. GPU reads from the read buffer while CPU writes KV to the write buffer. A `std::atomic<uint32_t>` flip counter tracks current buffer. TMA transfers from write buffer to VRAM. Buffers allocated from the sliding window pinned allocator (Section 5).

### 3.3 — Stage Event Sync

CUDA events mark boundaries. CPU compute for split N waits on `stage_events[N % depth]`. After GPU dispatch, `cudaEventRecord(event, stream)` signals completion. GPU-side mbarrier provides cross-stage sync within the kernel.

### 3.4 — Threadpool Assignment per Split

Pipeline owns N pools (one per CCD pair). Split I uses `pools[I % pool_count]`. Pool A processes Splits 0, 3, 6 — independent of Pool B on Splits 1, 4, 7. This prevents Infinity Fabric congestion during 256K+ context processing.

### 3.5 — KV Cache Double-Buffer & Merge Stream

Prefill uses a linear, append-only KV layout. Each split writes to a contiguous chunk. After prefill, chunks merge into the main ring buffer via `memmove` on a **dedicated merge CUDA stream** (not the main infer stream), so generation of the first token starts immediately on pipeline drain. The merge stream is ordered after the last GPU split via `cudaStreamWaitEvent()`.

### 3.6 — Pipeline Depth Enforcement

`pipeline_depth = 3` limits in-flight splits. Backpressure blocks dispatch when trying to start Split N+depth while Split N is still on GPU. Prevents VRAM pressure.

### 3.7 — Activation Heuristic

Pipeline activates when `batch_size > ubatch_size * 2`. Small batches use the standard path — overhead would dominate.

---

## Phase 4: TMA Kernel + Runtime Guard

### 4.1 — TMA Kernel Structure

`ggml-cuda-blackwell.cuh` defines store and load kernels wrapping `cp.async.bulk` PTX. Guarded with `#if __CUDA_ARCH__ >= 1000 && defined(GGML_CUDA_HAS_WGMMA)`. The existing `bw_tile_tma_wgmma` device function is promoted to a full kernel that loops over tiles within a slot.

### 4.2 — Runtime Feature Flag

- Env var `GGML_CUDA_TMA=0|1` (default `0` = memcpy)
- `ggml_cuda_tma_supported()` probes at init: `runtimeGetVersion() >= 12400`, `deviceArch >= sm_90`, `GGML_CUDA_TMA=1`
- When enabled, `ggml_tma_init_transfer` builds descriptors with alignment checks (`pinned_addr % 128 == 0`, `vram_addr % 16 == 0`)
- Transfer dispatches TMA kernel; otherwise `cudaMemcpyAsync`
- Descriptor copy is synchronous (`cudaMemcpy`) during init — no implicit stream ordering

### 4.3 — Health-Check Init (`ggml-cuda-epyc-pipeline.cu`)

On pipeline start, verify:
1. RTX 5090 PCIe bus matches EPYC NUMA node affinity (read `/sys/class/drm/card*/device/numa_node`)
2. Pinned RAM window is 128-byte aligned (TMA requirement)
3. VRAM target is 16-byte aligned
4. `pinned_addr < (1ULL << 48)` — 48-bit VA assertion

Log warnings for non-fatal mismatches, abort on alignment failures.

### 4.4 — mbarrier Sync

TMA completion signaled via mbarrier. `__mbarrier_arrive_and_expect_tx` and `__mbarrier_wait` wrappers already exist in `ggml-cuda-blackwell.cuh`. Pipeline stage events wired through mbarrier on GPU side.

### 4.5 — Early-Start Mechanism

The key zero-bubble insight: WGMMA begins as soon as a single tile is TMA-resident, not the whole layer. The mbarrier transaction count is set to ONE tile's bytes. The WGMMA loop iterates with per-tile mbarrier wait. Next tile's TMA can overlap with current tile's WGMMA because each tile uses the same SMEM slot sequentially within the block.

Shared memory layout per tile slot:
```
[0..7]     mbarrier (uint64_t)
[16..K)    K tile buffer  (tile_m * tile_n * type_size bytes)
[K..K+V)   V tile buffer  (same size)
(padded to 16-byte alignment)
```

Tile dimensions (conservative for 128 KB SMEM budget per SM):
- `TILE_M = 64` tokens
- `TILE_N = 128` head_dim (BF16 = 256 bytes per token)
- `TILE_K = 64` (chunk of sequence length for attention reduction)

KV tile bytes (K or V, not both): `64 * 128 * 2 = 16,384 bytes`
Total SMEM per tile: `8 + 2 * 16,384 = 32,776 bytes` (well under 128 KB)

---

## Phase 5: Pinned Buffer Hardening (Sliding Window for 1M+ Context)

### 5.1 — Single Large mmap + Slab Allocator

One massive pinned region (64GB) at init with `mmap(MAP_LOCKED | MAP_ANONYMOUS | MAP_PRIVATE)`. Internal allocation uses offset-based slab with 128-byte alignment. Avoids repeated `mlock` syscalls and `RLIMIT_MEMLOCK` limits. Ping-pong: two regions for zero-stop writes.

### 5.2 — Fragmentation Management

Tracks `allocated_bytes`, `high_watermark`, `fragmentation_ratio`. Compact pass shifts live blocks when fragmentation exceeds 25%. Amortized cost — prefill splits have predictable lifetimes (allocated at split start, freed at GPU_DONE).

### 5.3 — Sliding Window Logic

For 1M+ context, the pinned region is divided into fixed-size windows (e.g., 256K tokens each). Active window is pinned; inactive windows are `madvise(MADV_DONTNEED)` to release physical pages without unmapping. Window switch is triggered when the current window's free space drops below a threshold. This prevents the OS from holding 64GB of physical RAM when only 8GB is actively used.

### 5.4 — Cleanup

Single `munmap` per region on destroy. Threadpool join before destroy prevents dangling pointers.

### 5.5 — OOM Policy

When both regions exhausted: fall back to regular heap with warning. Tensor computes normally, loses TMA eligibility. No OOM crash.

### 5.6 — Alignment

Every allocation rounded to 128 bytes. Free list is index-based (not pointer-linked) to simplify compact pass.

---

## Testing Strategy

| Phase | Tests |
|-------|-------|
| Decoupling | Build succeeds with/without new files; `ctest` passes; no symbol conflicts |
| Hardening | ASan/valgrind on init paths; race detection on lazy init; sysfs overflow with long CPU names |
| Pipeline overlap | Golden-path: pipelined output matches non-pipelined for small model (Llama-3-8B); per-stage timings in bench tool |
| TMA kernel | TMA disabled = memcpy path; TMA enabled = same output; alignment check catches unaligned buffers |
| Pinned buffer | Allocate/deallocate stress test; 1M context sim; compact pass correctness; OOM fallback |

---

## Self-Review

- **Placeholder scan:** No TBD/TODO placeholders in spec. All code paths specified.
- **Internal consistency:** File structure matches module responsibilities. Phase ordering is correct (decouple → harden → overlap → TMA → pinned buffer). Merge stream in Phase 3 aligns with dedicated stream in Section 3.5.
- **Scope check:** Single plan, 5 phases. Phases build on each other (decoupling enables targeted hardening, which enables safe overlap). Cohesive enough for one implementation plan.
- **Ambiguity check:** "split_size = 8 layers" is concrete. "depth = 3" is explicit. Activation heuristic (`batch_size > ubatch_size * 2`) is a specific formula. OOM policy is defined. Alignment requirements are numeric. Tile dimensions are specified.
