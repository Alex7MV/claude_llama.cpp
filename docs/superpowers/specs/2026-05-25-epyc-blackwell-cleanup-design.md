# EPYC/Blackwell Pipelined Engine — Cleanup & Performance Hardening

> **Prepared for:** AMD EPYC 9V74 (80-core Zen 4) + NVIDIA RTX 5090 (Blackwell SM 120)
> **Target:** 600GB+ models, 256K context, zero-waste engineering
> **Current state:** Functional 3-stage pipeline with CCD threadpool rotation, but TMA kernel unimplemented, `depth`/`split_size` unused, race conditions in lazy init, EPYC/Blackwell code mixed into core files

**Goal:** Harden the async pipeline, implement real multi-stage overlap, stub the TMA kernel with runtime guard, and decouple EPYC/Blackwell code into dedicated modules.

**Architecture:** Four-phase approach — decouple files for maintainability, fix race conditions and resource leaks, implement graph splitting with ping-pong buffers and stage event sync, then stub TMA behind a feature flag.

**Tech Stack:** ggml backend (C/C++), CUDA 12.4+, x86_64 AVX-512 VNNI, Linux sysfs topology probing, mmap/MAP_LOCKED pinned memory

---

## Phase 1: Code Decoupling

Move EPYC-specific and Blackwell-specific code out of core files into dedicated modules. Each module has one clear responsibility and a well-defined interface.

### Target File Structure

```
ggml/
 ├── include/
 │   ├── ggml-backend-pipeline.h       (extended — stage API)
 │   └── ggml-cpu-epyc.h               (NEW — EPYC CCD topology + dual threadpool)
 ├── src/
 │   ├── ggml-cpu/
 │   │   ├── ggml-cpu.c                (trimmed — core CPU backend only)
 │   │   ├── ggml-cpu-epyc.c           (NEW — CCD probing, dual pool init, affinity)
 │   │   └── ggml-cpu-epyc-impl.h      (NEW — EPYC internals: CCD group struct)
 │   └── ggml-cuda/
 │       ├── ggml-cuda.cu               (trimmed — core CUDA backend)
 │       ├── tma-transfer.cu            (enhanced — TMA kernel + guard)
 │       ├── ggml-cuda-blackwell.cuh    (NEW — TMA desc, WGMMA stubs, mbarrier)
 │       └── ggml-cuda-epyc-pipeline.cu (NEW — GPU-side pipeline glue + health check)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `ggml-cpu-epyc.h/c` | CCD topology probing via sysfs, dual threadpool creation, CCD pair selection. Exposes `ggml_probe_ccd_pairs()`, `ggml_cpu_init_dual_threadpool_epyc()`. |
| `ggml-cpu-epyc-impl.h` | Internal CCD group struct, L3 domain ID type, SMT sibling tracking. |
| `ggml-cuda-blackwell.cuh` | TMA descriptor types (store/load), mbarrier wrappers, WGMMA Frag type stubs. Compiled only when `CUDA_ARCH >= 1000`. |
| `ggml-cuda-epyc-pipeline.cu` | GPU-side pipeline: health-check init (PCIe bus + 128-byte alignment), split graph dispatch, ping-pong buffer management, stage event recording. Merge stream for KV cache linear-to-ring transfer. |
| `ggml-backend-pipeline.h/c` | Orchestrator only — knows stages, events, depth, rotation. Delegates EPYC/BW details to dedicated modules. |

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

### Split Model

Given `split_size = 8` layers and a 64-layer model, the graph splits into 8 groups. Pipeline depth=3 allows 3 groups in-flight simultaneously:

```
Time -> |---S0---|---S1---|---S2---|---S0---|---S1---|---S2---|
CPU     | CG 0  | CG 1  | CG 2  | CG 3  | CG 4  | CG 5  |
TMA/GPU |       | T0+G0 |       | T1+G1 |       | T2+G2 |
```

### Graph Splitting

New function `llama_graph_build_split()` builds a partial `ggml_cgraph` for a layer range `[first_layer, first_layer + layer_count)`. Each split graph contains its own norm -> QKV -> RoPE -> attention -> FFN -> residual chain, wired to the KV cache slice for that range. Splits are independent within a single prefill batch.

### Ping-Pong KV Buffers

Two pinned RAM buffers (A, B) per split group. GPU reads from the read buffer while CPU writes KV to the write buffer. A `std::atomic<uint32_t>` flip counter tracks current buffer. TMA transfers from write buffer to VRAM. Buffers allocated from the sliding window pinned allocator (Section 5).

### Stage Event Sync

CUDA events mark boundaries. CPU compute for split N waits on `stage_events[N % depth]`. After GPU dispatch, `cudaEventRecord(event, stream)` signals completion. GPU-side mbarrier provides cross-stage sync.

### Threadpool Assignment per Split

Pipeline owns N pools (one per CCD pair). Split I uses `pools[I % pool_count]`. Pool A processes Splits 0, 3, 6 — independent of Pool B on Splits 1, 4, 7.

### KV Cache Double-Buffer

Prefill uses a linear, append-only KV layout. Each split writes to a contiguous chunk. After prefill, chunks merge into the main ring buffer via `memmove` on a **dedicated merge CUDA stream** (not the main infer stream), so generation of the first token starts immediately on pipeline drain. The merge stream is ordered after the last GPU split via `cudaStreamWaitEvent()`.

### Pipeline Depth Enforcement

`pipeline_depth = 3` limits in-flight splits. Backpressure blocks dispatch when trying to start Split N+depth while Split N is still on GPU. Prevents VRAM pressure.

### Activation Heuristic

Pipeline activates when `batch_size > ubatch_size * 2`. Small batches use the standard path — overhead would dominate.

---

## Phase 4: TMA Stub + Runtime Guard

### TMA Kernel Structure

`ggml-cuda-blackwell.cuh` defines store and load kernels wrapping `__cp_async_bulk_uniform_raw_tma_crypto_zero_copy_mem_first_pass`. Guarded with `#if __CUDA_ARCH__ >= 1000`. A matching load kernel uses the load intrinsic.

### Runtime Feature Flag

- Env var `GGML_CUDA_TMA=0|1` (default `0` = memcpy)
- `ggml_cuda_tma_supported()` probes at init: `runtimeGetVersion() >= 12400`, `deviceArch >= sm_90`, `GGML_CUDA_TMA=1`
- When enabled, `ggml_tma_init_transfer` builds descriptors with alignment checks (`pinned_addr % 128 == 0`, `vram_addr % 16 == 0`)
- Transfer dispatches TMA kernel; otherwise `cudaMemcpyAsync`
- Descriptor copy is synchronous (`cudaMemcpy`) during init — no implicit stream ordering

### Health-Check Init (`ggml-cuda-epyc-pipeline.cu`)

On pipeline start, verify:
1. RTX 5090 PCIe bus matches EPYC NUMA node affinity (read `/sys/class/drm/card*/device/numa_node`)
2. Pinned RAM window is 128-byte aligned (TMA requirement)
3. VRAM target is 16-byte aligned
4. `pinned_addr < (1ULL << 48)` — 48-bit VA assertion

Log warnings for non-fatal mismatches, abort on alignment failures.

### Descriptor Safety

48-bit VA truncation stays for current hardware. `GGML_ASSERT(addr < (1ULL << 48))` at init fails loudly on 5-level paging systems.

### mbarrier Sync

TMA completion signaled via mbarrier. `__mbarrier_arrive_and_expect_tx` and `__mbarrier_wait` wrappers in `ggml-cuda-blackwell.cuh`. Pipeline stage events wired through mbarrier on GPU side.

---

## Phase 5: Pinned Buffer Hardening (Sliding Window for 1M+ Context)

### Single Large mmap + Slab Allocator

One massive pinned region (64GB) at init with `mmap(MAP_LOCKED)`. Internal allocation uses offset-based slab with 128-byte alignment. Avoids repeated `mlock` syscalls and `RLIMIT_MEMLOCK` limits. Ping-pong: two regions for zero-stop writes.

### Fragmentation Management

Tracks `allocated_bytes`, `high_watermark`, `fragmentation_ratio`. Compact pass shifts live blocks when fragmentation exceeds 25%. Amortized cost — prefill splits have predictable lifetimes.

### Cleanup

Single `munmap` per region on destroy. Threadpool join before destroy prevents dangling pointers.

### OOM Policy

When both regions exhausted: fall back to regular heap with warning. Tensor computes normally, loses TMA eligibility. No OOM crash.

### Alignment

Every allocation rounded to 128 bytes. Free list is index-based (not pointer-linked) to simplify compact pass.

---

## Testing Strategy

| Phase | Tests |
|-------|-------|
| Decoupling | Build succeeds with/without new files; `ctest` passes; no symbol conflicts |
| Hardening | ASan/valgrind on init paths; race detection on lazy init; sysfs overflow with long CPU names |
| Pipeline overlap | Golden-path: pipelined output matches non-pipelined for small model (Llama-3-8B); per-stage timings in bench tool |
| TMA stub | TMA disabled = memcpy path; TMA enabled = same output; alignment check catches unaligned buffers |
| Pinned buffer | Allocate/deallocate stress test; 1M context sim; compact pass correctness; OOM fallback |

---

## Self-Review

- **Placeholder scan:** No TBD/TODO placeholders in spec. All code paths specified.
- **Internal consistency:** File structure matches module responsibilities. Phase ordering is correct (decouple -> harden -> overlap -> TMA). Merge stream in Phase 3 aligns with dedicated stream in Section 3.
- **Scope check:** Single plan, 4 phases + pinned buffer hardening. Phases build on each other (decoupling enables targeted hardening, which enables safe overlap). Cohesive enough for one implementation plan.
- **Ambiguity check:** "split_size = 8 layers" is concrete. "depth = 3" is explicit. Activation heuristic (`batch_size > ubatch_size * 2`) is a specific formula. OOM policy is defined. Alignment requirements are numeric.
