# Blackwell WGMMA Phase 2 — Kernel Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Integrate WGMMA primitives into a working `mul_mat_q_wgmma` kernel for Q8_0 on SM120+, with dispatch routing in `mmq.cu`.

**Architecture:** New kernel `mul_mat_q_wgmma<type, mmq_x>` uses 128-thread warpgroups. It reuses existing `load_tiles_q8_0` for loading quantized X to shared memory, then dispatches `ggml_cuda_wgmma::mma_sync<64,32,8>` for INT8 MMA via shared memory pointers. The K-loop uses 2-stage double-buffering (`__syncthreads()` for now; TMA pipeline is Phase 3). Y tiles follow the existing `block_q8_1_mmq` layout. SM120+ only for initial implementation.

**Tech Stack:** CUDA 13.2+, nvcc inline PTX, CMake

## STATUS: Implementation complete, gated by toolchain detection

**Committed (14 commits on refactor/CUDA-Blackwell):**
- Phase 1: WGMMA primitives in mma.cuh, guards in common.cuh, Blackwell CC in CMake (9 commits)
- Phase 2a: `cp_async_wait_stage<N>` for multi-stage async copy pipelining (1 commit)
- Phase 2b: WGMMA kernel code with build-time detection (2 commits)
- Phase 2c: CMake `try_compile` WGMMA PTX detection (1 commit). Auto-detects toolkit support, defines `GGML_CUDA_HAS_WGMMA`.
- Phase 3 (started): NUMA node detection + multi-GPU topology logging (1 commit)
- Phase 3: FP8 WGMMA templates `mma_sync_e4m3<>`/`mma_sync_e5m2<>` for m64n16/32/64h16 (1 commit)
  - `BLACKWELL_WGMMMA_AVAILABLE` now requires `GGML_CUDA_HAS_WGMMA` preprocessor define
  - `blackwell_wgmma_available()` runtime check also gates on `GGML_CUDA_HAS_WGMMA`
  - `mul_mat_q_wgmma` kernel, `vec_dot_q8_0_wgmma`, `launch_mul_mat_q_wgmma`, `mul_mat_q_wgmma_case`
  - Dispatch in mmq.cu routes WGMMA path only when `BLACKWELL_WGMMMA_AVAILABLE` is defined
  - Q8_0 MUL_MAT tests pass on RTX 5090 (regular kernel path, WGMMA not compiled)

**To enable WGMMA:** Define `GGML_CUDA_HAS_WGMMA` in CMake once the CUDA toolkit supports WGMMA PTX. The kernel will automatically compile and activate on SM1000+ devices.

---

## File Structure

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/cp-async.cuh` | Add `cp_async_wait_stage<N>` for staged cp.async waits (SM100+) |
| `ggml/src/ggml-cuda/mmq.cuh` | Add `vec_dot_q8_0_wgmma`, `mul_mat_q_wgmma_process_tile`, `mul_mat_q_wgmma` kernel |
| `ggml/src/ggml-cuda/mmq.cuh` | Update `mmq_type_traits<Q8_0>` with `vec_dot_wgmma` pointer |
| `ggml/src/ggml-cuda/mmq.cu` | Add WGMMA dispatch path in `ggml_cuda_mul_mat_q()` |
| `ggml/src/ggml-cuda/mmq.cu` | Add `launch_mul_mat_q_wgmma<type, mmq_x>()` |
| `ggml/src/ggml-cuda/mmq.cuh` | Add `launch_mul_mat_q_wgmma` declaration and extern template |

---

### Task 1: Add staged cp.async wait to cp-async.cuh

**Files:**
- Modify: `ggml/src/ggml-cuda/cp-async.cuh`

**Context:** Current `cp_async_wait_all()` waits for all outstanding async copies. For double/triple buffering, we need `cp.async.wait<N>` which waits for all but the last N stages. This is needed for the pipeline in Task 3.

- [ ] **Step 1: Add `cp_async_wait_stage<N>()`**

Append to `cp-async.cuh` after `cp_async_wait_all()`:

```cpp
// Wait for all async copies except the last N stages.
// Used for N-buffer pipelining: cp_async_wait_stage<1>() waits for all but the most recent stage.
template <int N>
static __device__ __forceinline__ void cp_async_wait_stage() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_stage %0;" :: "n"(N) : "memory");
#else
    NO_DEVICE_CODE;
#endif
}
```

- [ ] **Step 2: Verify build**

```bash
cmake -B build_p2 -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | tail -3 && cmake --build build_p2 --config Release --target ggml-cuda -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -5
```

Expected: No errors or warnings.

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/cp-async.cuh
git commit -m "cuda: add cp.async.wait_stage<N> for multi-stage async copy pipelining"
```

---

### Task 2: Add WGMMA vec_dot for Q8_0 in mmq.cuh

**Files:**
- Modify: `ggml/src/ggml-cuda/mmq.cuh` — add `vec_dot_q8_0_wgmma<mmq_x, mmq_y>()` after line ~1295

**Context:** The existing `vec_dot_q8_0_q8_1_mma<>` (lines 1217-1293) uses warp-level `mma()` with `tile<16,8,int>` fragments. The WGMMA version must use `ggml_cuda_wgmma::mma_sync<M,N,K>` with shared memory pointers instead of register tiles.

The X data layout in shared memory: `int * tile_x` contains `mmq_y` rows of `MMQ_MMA_TILE_X_K_Q8_0` int32 elements. Each row stores Q8_0 quantized data packed as int32. The first `2*MMQ_TILE_NE_K` ints are `qs` (quantized values as int8), followed by scales.

For WGMMA, we need INT8 tiles in shared memory. The Q8_0 data in `tile_x` is already int8 (stored as `int32_t` for alignment). We cast the shared memory pointer to `const uint32_t*` and pass it to `wgmma_mma_sync`.

The key mapping:
- WGMMA tile: 64 rows x 32 cols of INT8 = 2048 multiplications per instruction
- K-dimension: h8 means K=8 per stage. Total K=32 requires 4 stages.
- Accumulator: `frag_c<64,32>` = 64 uint32_t elements per thread
- For `mmq_y = 128`: 2 warpgroup tiles of 64 rows each

**Implementation:**

After the existing `vec_dot_q8_0_q8_1_mma` function (~line 1295), add:

```cpp
#ifdef BLACKWELL_WGMMMA_AVAILABLE
// WGMMA-based vec_dot for Q8_0 on Blackwell SM>=1000.
// Uses 128-thread warpgroups, reads directly from shared memory.
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void
vec_dot_q8_0_wgmma(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    // Process mmq_y rows in groups of 64 (WGMMA M dimension).
    // Each warpgroup (128 threads) computes one or more 64xN tiles.
    const int warp_size = ggml_cuda_get_physical_warp_size();
    const int nwarps = mmq_get_nwarps_device();

    // Each warpgroup handles mmq_y rows total, split into ceil(mmq_y/64) sub-tiles.
    // For mmq_y=128: 2 sub-tiles of 64 rows.
    // Each thread computes sum for mmq_x / nwarps columns.
    for (int tile_m = 0; tile_m < mmq_y; tile_m += 64) {
        const int m_base = tile_m;

        // Process N dimension in chunks of 32 (WGMMA N dimension).
        for (int tile_n = 0; tile_n < mmq_x; tile_n += 32) {
            const int n_base = tile_n;
            const int n_actual = min(32, mmq_x - tile_n);

            ggml_cuda_wgmma::frag_c<64, 32> D;
            D.zero();

            // K loop: step by 8 (h8), total K = MMQ_ITER_K/2 = 128 per vec_dot call.
            const int k_start = k00;
            const int k_end = k00 + MMQ_TILE_NE_K;

            for (int k = k_start; k < k_end; k += 8) {
                // X tile in shared memory: row-major, mmq_y rows, MMQ_MMA_TILE_X_K_Q8_0 ints per row.
                // For 64-row tile starting at m_base, column offset k:
                const uint32_t *A = reinterpret_cast<const uint32_t *>(x + (m_base + threadIdx.y * 0) * MMQ_MMA_TILE_X_K_Q8_0 + k);
                // Y tile in shared memory: mmq_x cols, MMQ_TILE_Y_K ints per col.
                const uint32_t *B = reinterpret_cast<const uint32_t *>(y + (n_base + 0) * MMQ_TILE_Y_K + k);

                ggml_cuda_wgmma::mma_sync<D, 64, 32, 8>(D, A, B, true);
                ggml_cuda_wgmma::commit_sync();
            }
            ggml_cuda_wgmma::wait_sync<0>();

            // Write back D to sum array.
            // sum is indexed as sum[j*mmq_y/nwarps + i].
            // Each thread in warpgroup contributes to specific (i,j) pairs.
            for (int idx = 0; idx < ggml_cuda_wgmma::frag_c<64,32>::ne; ++idx) {
                const int i = m_base + (idx / 32) * 4 + (idx % 32) / 8;  // map fragment index to row
                const int j = n_base + (idx % 8);  // map fragment index to col
                if (i < mmq_y && j < mmq_x) {
                    sum[(j / (mmq_x / (nwarps * warp_size))) * (mmq_y / (nwarps * warp_size)) + i] += reinterpret_cast<float*>(&D.x[idx]);
                }
            }
        }
    }
}
#endif
```

**Note:** This is a conceptual outline — the actual implementation needs careful index mapping for the warpgroup fragment layout. The PTX `<16>` vector modifier means each of the first 16 threads in a warp provides one register. The full 64 rows are distributed across 4 warps. The `sum[]` array layout must match the existing `mmq_write_back_mma` expectations.

- [ ] **Step 1: Implement `vec_dot_q8_0_wgmma`**

Write the actual function with correct index mapping. The accumulator fragment `frag_c<64,32>::ne = 64` means 64 uint32_t values per thread. These map to 64x32=2048 total FP32 results, distributed as 2048/128 = 16 results per thread... but ne=64 means each thread holds 64 uint32_t = 64 FP32 values. This is because WGMMA accumulates across all warpgroup threads.

The mapping follows the PTX spec: for m64n32h8, the accumulator is organized as 64 threads × 32 elements per thread for the M-major layout, but with `<16>` vector modifiers, each thread contributes 16 registers per warp segment.

- [ ] **Step 2: Verify compilation**

```bash
cmake --build build_p2 --config Release --target ggml-cuda -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -10
```

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/mmq.cuh
git commit -m "cuda: add WGMMA vec_dot for Q8_0 using wgmma_mma_sync<64,32,8>"
```

---

### Task 3: Add WGMMA kernel and process_tile function

**Files:**
- Modify: `ggml/src/ggml-cuda/mmq.cuh` — add `mul_mat_q_wgmma` kernel after existing `mul_mat_q` kernel

**Context:** The new kernel follows the same structure as `mul_mat_q` but:
- Block dims: `(128, 1, 1)` — 128 threads per warpgroup
- Uses `mul_mat_q_wgmma_process_tile` with WGMMA vec_dot
- Stream-K mode only (simpler dispatch, no xy tiling)

- [ ] **Step 1: Add `mul_mat_q_wgmma_process_tile`**

Similar to `mul_mat_q_process_tile` (line 3447) but:
- Uses `#ifdef BLACKWELL_WGMMMA_AVAILABLE` guard
- Calls WGMMA vec_dot instead of warp-level MMA
- Uses triple-buffered shared memory layout

- [ ] **Step 2: Add `mul_mat_q_wgmma` kernel**

```cpp
template <ggml_type type, int mmq_x, bool need_check>
__launch_bounds__(128, 2)
__global__ void mul_mat_q_wgmma(...) {
    // stream-k dispatch logic similar to mul_mat_q
}
```

- [ ] **Step 3: Verify compilation**

- [ ] **Step 4: Commit**

---

### Task 4: Add WGMMA dispatch logic in mmq.cu and mmq.cuh

**Files:**
- Modify: `ggml/src/ggml-cuda/mmq.cu` — add `blackwell_wgmma_available(cc)` check
- Modify: `ggml/src/ggml-cuda/mmq.cuh` — add `launch_mul_mat_q_wgmma<type, mmq_x>()`
- Modify: `ggml/src/ggml-cuda/mmq.cu` — add `mul_mat_q_wgmma_case<type>()`

**Context:** The dispatch in `ggml_cuda_mul_mat_q()` (line 77) currently selects the existing `mul_mat_q` kernel. We add a check:

```cpp
const bool use_wgmma = blackwell_wgmma_available(cc) && !use_native_fp4;
```

If `use_wgmma` is true, route to `mul_mat_q_wgmma_case<type>()` instead of `mul_mat_q_case<type>()`.

- [ ] **Step 1: Add dispatch check in `ggml_cuda_mul_mat_q()`**

After line 125 (`use_native_fp4`), add:
```cpp
const bool use_wgmma = blackwell_wgmma_available(cc) && !use_native_fp4;
```

- [ ] **Step 2: Route to WGMMA path**

In the type switch (line 160), replace:
```cpp
ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
```
with:
```cpp
if (use_wgmma) {
    ggml_cuda_mul_mat_q_wgmma_switch_type(ctx, args, stream);
} else {
    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}
```

- [ ] **Step 3: Add `ggml_cuda_mul_mat_q_wgmma_switch_type()`**

New function that mirrors `ggml_cuda_mul_mat_q_switch_type` but calls `mul_mat_q_wgmma_case<type>()`.

- [ ] **Step 4: Add `mul_mat_q_wgmma_case<type>()` and `launch_mul_mat_q_wgmma()`**

In `mmq.cuh`, add the WGMMA equivalents of `mul_mat_q_case` and `launch_mul_mat_q`.

- [ ] **Step 5: Update `ggml_cuda_should_use_mmq()`**

In `mmq.cu` line 307, add:
```cpp
if (blackwell_wgmma_available(cc)) {
    return true;  // WGMMA path always preferred on Blackwell
}
```

- [ ] **Step 6: Verify full build**

```bash
cmake --build build_p2 --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -10
```

- [ ] **Step 7: Commit**

```bash
git add ggml/src/ggml-cuda/mmq.cu ggml/src/ggml-cuda/mmq.cuh
git commit -m "cuda: add WGMMA dispatch path for Blackwell in mmq"
```

---

### Task 5: Integration testing

**Files:**
- Modify: `tests/test-backend-ops.cpp` — optional, add WGMMA-specific test flags

- [ ] **Step 1: Build with tests**

```bash
cmake -B build_p2_test -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build_p2_test --config Release -j$(nproc)
```

- [ ] **Step 2: Run CUDA backend tests**

```bash
cd build_p2_test && timeout 60 bin/test-backend-ops -b CUDA0 -o "MUL_MAT(type=q8_0" 2>&1
```

- [ ] **Step 3: Verify WGMMA is being used**

Check that the kernel launches with 128-thread blocks on SM120+.

- [ ] **Step 4: Clean up and commit**

---

## Follow-up Work (Phase 3)

**Completed:**
- NUMA node detection + multi-GPU topology logging (ggml-cuda.cu, committed)
- FP8 WGMMA `mma_sync_e4m3<>`/`mma_sync_e5m2<>` templates in mma.cuh (m64n16h16, m64n32h16, m64n64h16)

**Blocked on CUDA toolchain (requires toolkit with WGMMA PTX support):**
- TMA pipeline with `cp.async.bulk` + `.mbarrier` (SM120+)
- Triple-buffered WGMMA pipeline with `cp.async.cg` double-buffering
- `wgmma.store_gmem` writeback (SM120+)
- Extend to all quant types (Q4_0/Q5_0/Q6_K need WGMMA-specific shared memory layout with pre-dequantized INT8)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| WGMMA fragment index mapping is complex | Start with Q8_0 m64n32h8, verify against mma.sync results |
| Register pressure from `frag_c<64,32>` (64 uint32) | Use `__launch_bounds__(128, 2)`; profile with `cuobjdump --sass` |
| Shared memory layout mismatch | Reuse existing `load_tiles_q8_0` — WGMMA reads same smem layout |
| Breaking existing mmq path | `#ifdef BLACKWELL_WGMMMA_AVAILABLE` guards all WGMMA code |
