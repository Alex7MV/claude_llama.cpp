# Blackwell WGMMA Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add WGMMA primitives (register fragments + PTX intrinsics + pipeline control) to `mma.cuh`, expand CUDA architecture config for consumer Blackwell (SM100/101), and add a runtime availability check — all guarded by `__CUDA_ARCH__ >= 1000`.

**Architecture:** New `#if __CUDA_ARCH__ >= 1000` block in `mma.cuh` defines `ggml_cuda_wgmma::` namespace with `frag_c<>` structs (explicit `uint32_t[N]` arrays) and `wgmma_mma_sync<>`/`wgmma_commit_sync`/`wgmma_wait<>` PTX wrappers. No kernel changes in Phase 1; the existing `mma.sync` path is untouched. Phase 2 (planned separately) integrates these primitives into a new `mul_mat_q_wgmma` kernel.

**Tech Stack:** CUDA 12.8+ (tested with 13.2), nvcc PTX inline assembly, CMake

---

## File Structure

| File | Responsibility |
|------|---------------|
| `ggml/src/ggml-cuda/common.cuh` | New CC constant `GGML_CUDA_CC_BLACKWELL_WG = 1000`, compile-time guard `BLACKWELL_WGMMMA_AVAILABLE`, runtime check `blackwell_wgmma_available(cc)` |
| `ggml/src/ggml-cuda/mma.cuh` | New `#if __CUDA_ARCH__ >= 1000` block: `ggml_cuda_wgmma::` namespace, `frag_c<>` structs, `wgmma_mma_sync<>`, `wgmma_commit_sync`, `wgmma_wait<>`, `wgmma_fence` |
| `ggml/src/ggml-cuda/CMakeLists.txt` | Add `100a-real` and `101a-real` architectures for CUDA 13.0+ (consumer Blackwell) |
| `tests/test-backend-ops.cpp` | (Phase 2 only — not in this plan) |

**Files NOT modified in Phase 1:** `mmq.cuh`, `mmq.cu`, `cp-async.cuh`, `mmf.cu`, `ggml-cuda.cu` — these are Phase 2/3 work.

---

### Task 1: Add consumer Blackwell architectures to CMakeLists.txt

**Files:**
- Modify: `ggml/src/ggml-cuda/CMakeLists.txt:40-55`

**Context:** The CMake architecture list (lines 40-55) currently adds `120a-real` for CUDA >= 12.8 and `121a-real` for CUDA >= 12.9. We need to add `100a-real` (SM100 — RTX 5090) and `101a-real` (SM101 — RTX 5060 Ti). These require CUDA 13.0+ for tooling support. The existing 12X-to-12Xa fixup pass (lines 75-92) does not cover 10X, so we must use the "a" suffix explicitly.

- [ ] **Step 1: Add architecture entries after existing Blackwell entries**

In `ggml/src/ggml-cuda/CMakeLists.txt`, after line 55 (`endif()` for the 121a check), add:

```cmake
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100a-real 101a-real)
endif()
```

- [ ] **Step 2: Verify CMake configuration succeeds**

```bash
cd /home/alexmv2025/projects/cuda/fast/claude_llama.cpp && rm -rf build_test_cmake && cmake -B build_test_cmake -DGGML_CUDA=ON 2>&1 | grep -iE "(cuda|arch|error|warn)" && rm -rf build_test_cmake
```

Expected: No errors. CMake should find CUDA 13.2 and add `100a-real;101a-real` to the architecture list.

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/CMakeLists.txt
git commit -m "cmake: add consumer Blackwell architectures 100a/101a for CUDA 13.0+"
```

---

### Task 2: Add GGML_CUDA_CC_BLACKWELL_WG constant and feature guards in common.cuh

**Files:**
- Modify: `ggml/src/ggml-cuda/common.cuh:55-60` (constants)
- Modify: `ggml/src/ggml-cuda/common.cuh:282-288` (compile-time guards)
- Modify: `ggml/src/ggml-cuda/common.cuh:356-359` (runtime checks)

**Context:** The project uses `GGML_CUDA_CC_BLACKWELL = 1200` for SM120a datacenter Blackwell. Consumer Blackwell (SM100/101) has CC = 1000/1010. We need a constant that covers *all* Blackwell for the WGMMA path. The existing `BLACKWELL_MMA_AVAILABLE` guard is scoped to SM120+ (`>= 1200 && < 1300`). We add `BLACKWELL_WGMMMA_AVAILABLE` for `>= 1000 && < 1300` to cover SM100/101/120/121.

- [ ] **Step 1: Add the constant after line 60 (GGML_CUDA_CC_RUBIN)**

In `common.cuh`, after the line `#define GGML_CUDA_CC_RUBIN           1300`, add:

```cpp
// Consumer Blackwell (SM100/101) also has WGMMA; use 1000 as the WGMMA availability floor
#define GGML_CUDA_CC_BLACKWELL_WG        1000
```

- [ ] **Step 2: Add the compile-time guard after line 284**

In `common.cuh`, after the existing `BLACKWELL_MMA_AVAILABLE` guard (lines 282-284), add:

```cpp
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL_WG && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
#    define BLACKWELL_WGMMMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL_WG
```

- [ ] **Step 3: Add the runtime check function after line 359**

In `common.cuh`, after the existing `blackwell_mma_available()` function, add:

```cpp
static bool blackwell_wgmma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL_WG &&
           ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_RUBIN;
}
```

- [ ] **Step 4: Verify compilation — host-only build**

```bash
cd /home/alexmv2025/projects/cuda/fast/claude_llama.cpp && cmake -B build_verify -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | tail -5 && cmake --build build_verify --config Release -j$(nproc) 2>&1 | tail -10 && rm -rf build_verify
```

Expected: Build succeeds with no new warnings. The `BLACKWELL_WGMMMA_AVAILABLE` guard is a pure `#define` — no code references it yet, so it won't trigger unused-code warnings.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/common.cuh
git commit -m "cuda: add BLACKWELL_WGMMMA_AVAILABLE guard for SM >= 1000 and runtime check"
```

---

### Task 3: Add WGMMA register fragment structs in mma.cuh

**Files:**
- Modify: `ggml/src/ggml-cuda/mma.cuh` — new `#if __CUDA_ARCH__ >= 1000` block appended after the existing `BLACKWELL_MMA_AVAILABLE` block (after line ~1310, end of `mma_block_scaled_fp4`)

**Context:** WGMMA PTX requires accumulator fragments to be explicit `uint32_t` arrays of fixed sizes — one element per (M*N/32) result per thread. The spec defines: `frag_c_m16n8`, `frag_c_m16n16`, `frag_c_m64n16`, `frag_c_m64n32`, `frag_c_m64n64`, `frag_c_m64n128`. Each has a zero-initialized `uint32_t x[N]` member, a static `M`/`N` constant, and `ne` (elements per thread).

- [ ] **Step 1: Add fragment structs in new namespace block**

In `mma.cuh`, after the closing `#endif // BLACKWELL_MMA_AVAILABLE` of the existing `mma_block_scaled_fp4` template (around line ~1154), add the following block. Place it *before* the existing `mma(tile<16, 8, float> &, ...)` free functions to keep the file organized by architecture:

```cpp
// ---------------------------------------------------------------------------------------------------------
// Blackwell Warpgroup MMA (WGMMA) primitives — SM 100/101/120/121
// Accurate register mapping: explicit uint32_t arrays, no half2/float reinterpret
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warpgroup-matrix-instructions-wgmma

#if __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1300

namespace ggml_cuda_wgmma {

    // Accumulator fragments: uint32_t x[ne], zero-init, one struct per MxN tile
    // ne = (M * N / 32) elements per thread in a 32-thread warp of a 128-thread warpgroup
    template <int M, int N>
    struct frag_c {
        static constexpr int ne = (M * N) / 32; // 32 threads per warp in nvcc inline asm context
        alignas(16) uint32_t x[ne] = {};

        __device__ __forceinline__ frag_c() = default;

        template <int other_M, int other_N>
        __device__ __forceinline__ frag_c(const frag_c<other_M, other_N> & other) {
            for (int i = 0; i < ne; ++i) {
                x[i] = i < frag_c<other_M, other_N>::ne ? other.x[i] : 0;
            }
        }

        __device__ __forceinline__ void zero() {
            for (int i = 0; i < ne; ++i) x[i] = 0;
        }
    };

    // Explicit specializations for tile sizes used by WGMMA PTX
    // m16 variants (single warp within warpgroup)
    template <> struct frag_c<16, 8>  { static constexpr int ne = 4;  alignas(16) uint32_t x[ne] = {}; __device__ __forceinline__ frag_c() = default; __device__ __forceinline__ void zero() { for(int i=0;i<ne;++i) x[i]=0; } };
    template <> struct frag_c<16, 16> { static constexpr int ne = 8;  alignas(16) uint32_t x[ne] = {}; __device__ __forceinline__ frag_c() = default; __device__ __forceinline__ void zero() { for(int i=0;i<ne;++i) x[i]=0; } };
    template <> struct frag_c<16, 32> { static constexpr int ne = 16; alignas(16) uint32_t x[ne] = {}; __device__ __forceinline__ frag_c() = default; __device__ __forceinline__ void zero() { for(int i=0;i<ne;++i) x[i]=0; } };
    template <> struct frag_c<16, 64> { static constexpr int ne = 32; alignas(16) uint32_t x[ne] = {}; __device__ __forceinline__ frag_c() = default; __device__ __forceinline__ void zero() { for(int i=0;i<ne;++i) x[i]=0; } };

    // m64 variants (full warpgroup, 4 warps)
    template <> struct frag_c<64, 16> { static constexpr int ne = 32;  alignas(16) uint32_t x[ne] = {}; __device__ __forceinline__ frag_c() = default; __device__ __forceinline__ void zero() { for(int i=0;i<ne;++i) x[i]=0; } };
    template <> struct frag_c<64, 32> { static constexpr int ne = 64;  alignas(16) uint32_t x[ne] = {}; __device__ __forceinline__ frag_c() = default; __device__ __forceinline__ void zero() { for(int i=0;i<ne;++i) x[i]=0; } };
    template <> struct frag_c<64, 64> { static constexpr int ne = 128; alignas(16) uint32_t x[ne] = {}; __device__ __forceinline__ frag_c() = default; __device__ __forceinline__ void zero() { for(int i=0;i<ne;++i) x[i]=0; } };
    template <> struct frag_c<64, 128>{ static constexpr int ne = 256; alignas(16) uint32_t x[ne] = {}; __device__ __forceinline__ frag_c() = default; __device__ __forceinline__ void zero() { for(int i=0;i<ne;++i) x[i]=0; } };

    // m128 variants (full warpgroup, large tiles)
    template <> struct frag_c<128, 64> { static constexpr int ne = 256; alignas(16) uint32_t x[ne] = {}; __device__ __forceinline__ frag_c() = default; __device__ __forceinline__ void zero() { for(int i=0;i<ne;++i) x[i]=0; } };
    template <> struct frag_c<128, 128>{ static constexpr int ne = 512; alignas(16) uint32_t x[ne] = {}; __device__ __forceinline__ frag_c() = default; __device__ __forceinline__ void zero() { for(int i=0;i<ne;++i) x[i]=0; } };

} // namespace ggml_cuda_wgmma

#endif // __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1300
```

**Note on `ne` calculation:** WGMMA operates on a warpgroup of 128 threads (4 warps). For an MxN tile with FP32 accumulator, total elements = M*N. Each thread holds M*N/128 elements. However, nvcc schedules PTX per-warp, so each *warp* holds M*N/32 elements. The `frag_c` struct is per-thread in the PTX context, but the `wgmma.mma_sync` PTX uses vector operand modifiers (`%0<16>`) to spread registers across the warpgroup. The `ne` values above match the number of `uint32_t` registers each thread contributes to the accumulator fragment in the inline asm constraint.

- [ ] **Step 2: Verify compilation**

```bash
cd /home/alexmv2025/projects/cuda/fast/claude_llama.cpp && cmake -B build_fragments -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | tail -3 && cmake --build build_fragments --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -10 && rm -rf build_fragments
```

Expected: No errors. No warnings. The structs are templates with explicit specializations — they won't instantiate unless referenced, but the `#if __CUDA_ARCH__` guard ensures nvcc compiles them for the target CC.

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/mma.cuh
git commit -m "cuda: add WGMMA register fragment structs for Blackwell SM>=1000"
```

---

### Task 4: Add WGMMA PTX intrinsics (mma_sync, commit_sync, wait_sync, fence) in mma.cuh

**Files:**
- Modify: `ggml/src/ggml-cuda/mma.cuh` — append to the `#if __CUDA_ARCH__ >= 1000` block from Task 3, inside `namespace ggml_cuda_wgmma`

**Context:** The WGMMA PTX instructions follow the pattern:
- `wgmma.mma_sync.sync.aligned.m<M>n<N>h<K>.s32.s8.s8.s32` — INT8 MMA (for Q4_K/Q8_0 dequant path)
- `wgmma.commit_sync.sync` — commit all outstanding WGMMA operations
- `wgmma.wait_sync.sync N` — wait for N most recent commits
- `wgmma.fence.sync` — memory fence for shared memory visibility
- `wgmma.ld_matrix.sync` — load shared memory tile into LD matrix cache (SM120+ only)

We implement the INT8 variant first (covers all existing K-quant types). FP8 variants (`e4m3`, `e5m2`) are Phase 3.

- [ ] **Step 1: Add PTX intrinsics after the fragment structs**

Inside the `ggml_cuda_wgmma` namespace, after the `frag_c` specializations, add:

```cpp
namespace ggml_cuda_wgmma {

    // ------------------------------------------------------------------
    // Pipeline control: commit, wait, fence
    // ------------------------------------------------------------------

    __device__ __forceinline__ void commit_sync() {
        asm volatile("wgmma.commit_sync.sync;" ::: "memory");
    }

    template <int N> // N = number of most recent commits to wait for
    __device__ __forceinline__ void wait_sync() {
        asm volatile("wgmma.wait_sync.sync %0;" : :: "n"(N) : "memory");
    }

    __device__ __forceinline__ void fence() {
        asm volatile("wgmma.fence.sync;" ::: "memory");
    }

    __device__ __forceinline__ void mem_sync() {
        asm volatile("membar.cta;" ::: "memory");
    }

    // ------------------------------------------------------------------
    // WGMMA MMA — INT8 input, FP32 accumulator
    // Template params: M, N, K (K = stage size, typically 4, 8, 16)
    // A: shared memory pointer, __ld_matrix__ qualified, 16-byte aligned
    // B: shared memory pointer, __ld_matrix__ qualified, 16-byte aligned
    // D: accumulator fragment (read-modify-write)
    // scale_d: true = use D as initial accumulator, false = zero-init
    // ------------------------------------------------------------------

    template <int M, int N, int K>
    __device__ __forceinline__ void
    mma_sync(frag_c<M, N> & D,
             const uint32_t * __ld_matrix__ A,
             const uint32_t * __ld_matrix__ B,
             bool scale_d = true) {
        if constexpr (M == 16 && K == 4) {
            asm volatile(
                "wgmma.mma_sync.sync.aligned.m16n8h4.s32.s8.s8.s32 "
                "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8], [%9], {%0, %1, %2, %3, %4, %5, %6, %7}, %10;"
                : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3]),
                  "+r"(D.x[4]), "+r"(D.x[5]), "+r"(D.x[6]), "+r"(D.x[7])
                : "l"(reinterpret_cast<uint64_t>(A)),
                  "l"(reinterpret_cast<uint64_t>(B)),
                  "r"(scale_d ? 1 : 0)
                : "memory"
            );
        } else if constexpr (M == 16 && N == 16 && K == 4) {
            // m16n16h4 — 16x16 output, K=4 INT8 elements per stage
            const uint32_t * __ld_matrix* B1 = B + 0;
            const uint32_t * __ld_matrix* B2 = B + 8; // stride for B tile, column-major
            // Two m16n8h4 operations to cover n=16
            asm volatile(
                "wgmma.mma_sync.sync.aligned.m16n8h4.s32.s8.s8.s32 "
                "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8], [%9], {%0, %1, %2, %3, %4, %5, %6, %7}, %10;"
                : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3]),
                  "+r"(D.x[4]), "+r"(D.x[5]), "+r"(D.x[6]), "+r"(D.x[7])
                : "l"(reinterpret_cast<uint64_t>(A)),
                  "l"(reinterpret_cast<uint64_t>(B1)),
                  "r"(scale_d ? 1 : 0)
                : "memory"
            );
            asm volatile(
                "wgmma.mma_sync.sync.aligned.m16n8h4.s32.s8.s8.s32 "
                "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8], [%9], {%0, %1, %2, %3, %4, %5, %6, %7}, 1;"
                : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3]),
                  "+r"(D.x[4]), "+r"(D.x[5]), "+r"(D.x[6]), "+r"(D.x[7])
                : "l"(reinterpret_cast<uint64_t>(A)),
                  "l"(reinterpret_cast<uint64_t>(B2)),
                  "r"(1)
                : "memory"
            );
        } else if constexpr (M == 64 && N == 16 && K == 8) {
            // m64n16h8 — warpgroup-wide, 64x16 output, K=8 per stage
            // Uses vector modifier <16> for 16 registers per warp
            // Total: 32 elements per thread = 4 warps * 8 elems
            uint32_t * dx = D.x;
            asm volatile(
                "wgmma.mma_sync.sync.aligned.m64n16h8.s32.s8.s8.s32 "
                "{%0<16>}, [%1], [%2], {%3<16>}, %4;"
                : "+r"(dx[0]), "+r"(dx[1]), "+r"(dx[2]), "+r"(dx[3]),
                  "+r"(dx[4]), "+r"(dx[5]), "+r"(dx[6]), "+r"(dx[7]),
                  "+r"(dx[8]), "+r"(dx[9]), "+r"(dx[10]), "+r"(dx[11]),
                  "+r"(dx[12]), "+r"(dx[13]), "+r"(dx[14]), "+r"(dx[15]),
                  "+r"(dx[16]), "+r"(dx[17]), "+r"(dx[18]), "+r"(dx[19]),
                  "+r"(dx[20]), "+r"(dx[21]), "+r"(dx[22]), "+r"(dx[23]),
                  "+r"(dx[24]), "+r"(dx[25]), "+r"(dx[26]), "+r"(dx[27]),
                  "+r"(dx[28]), "+r"(dx[29]), "+r"(dx[30]), "+r"(dx[31])
                : "l"(reinterpret_cast<uint64_t>(A)),
                  "l"(reinterpret_cast<uint64_t>(B)),
                  "r"(scale_d ? 1 : 0)
                : "memory"
            );
        } else if constexpr (M == 64 && N == 32 && K == 8) {
            // m64n32h8 — 64x32 output, two m64n16h8 passes
            uint32_t * dx = D.x;
            const uint32_t * __ld_matrix* B1 = B + 0;
            const uint32_t * __ld_matrix* B2 = B + 16;
            asm volatile(
                "wgmma.mma_sync.sync.aligned.m64n16h8.s32.s8.s8.s32 "
                "{%0<16>}, [%1], [%2], {%3<16>}, %4;"
                : "+r"(dx[0]), "+r"(dx[1]), "+r"(dx[2]), "+r"(dx[3]),
                  "+r"(dx[4]), "+r"(dx[5]), "+r"(dx[6]), "+r"(dx[7]),
                  "+r"(dx[8]), "+r"(dx[9]), "+r"(dx[10]), "+r"(dx[11]),
                  "+r"(dx[12]), "+r"(dx[13]), "+r"(dx[14]), "+r"(dx[15]),
                  "+r"(dx[16]), "+r"(dx[17]), "+r"(dx[18]), "+r"(dx[19]),
                  "+r"(dx[20]), "+r"(dx[21]), "+r"(dx[22]), "+r"(dx[23]),
                  "+r"(dx[24]), "+r"(dx[25]), "+r"(dx[26]), "+r"(dx[27]),
                  "+r"(dx[28]), "+r"(dx[29]), "+r"(dx[30]), "+r"(dx[31]),
                  "+r"(dx[32]), "+r"(dx[33]), "+r"(dx[34]), "+r"(dx[35]),
                  "+r"(dx[36]), "+r"(dx[37]), "+r"(dx[38]), "+r"(dx[39]),
                  "+r"(dx[40]), "+r"(dx[41]), "+r"(dx[42]), "+r"(dx[43]),
                  "+r"(dx[44]), "+r"(dx[45]), "+r"(dx[46]), "+r"(dx[47]),
                  "+r"(dx[48]), "+r"(dx[49]), "+r"(dx[50]), "+r"(dx[51]),
                  "+r"(dx[52]), "+r"(dx[53]), "+r"(dx[54]), "+r"(dx[55]),
                  "+r"(dx[56]), "+r"(dx[57]), "+r"(dx[58]), "+r"(dx[59]),
                  "+r"(dx[60]), "+r"(dx[61]), "+r"(dx[62]), "+r"(dx[63])
                : "l"(reinterpret_cast<uint64_t>(A)),
                  "l"(reinterpret_cast<uint64_t>(B1)),
                  "r"(scale_d ? 1 : 0)
                : "memory"
            );
            asm volatile(
                "wgmma.mma_sync.sync.aligned.m64n16h8.s32.s8.s8.s32 "
                "{%0<16>}, [%1], [%2], {%3<16>}, 1;"
                : "+r"(dx[0]), "+r"(dx[1]), "+r"(dx[2]), "+r"(dx[3]),
                  "+r"(dx[4]), "+r"(dx[5]), "+r"(dx[6]), "+r"(dx[7]),
                  "+r"(dx[8]), "+r"(dx[9]), "+r"(dx[10]), "+r"(dx[11]),
                  "+r"(dx[12]), "+r"(dx[13]), "+r"(dx[14]), "+r"(dx[15]),
                  "+r"(dx[16]), "+r"(dx[17]), "+r"(dx[18]), "+r"(dx[19]),
                  "+r"(dx[20]), "+r"(dx[21]), "+r"(dx[22]), "+r"(dx[23]),
                  "+r"(dx[24]), "+r"(dx[25]), "+r"(dx[26]), "+r"(dx[27]),
                  "+r"(dx[28]), "+r"(dx[29]), "+r"(dx[30]), "+r"(dx[31]),
                  "+r"(dx[32]), "+r"(dx[33]), "+r"(dx[34]), "+r"(dx[35]),
                  "+r"(dx[36]), "+r"(dx[37]), "+r"(dx[38]), "+r"(dx[39]),
                  "+r"(dx[40]), "+r"(dx[41]), "+r"(dx[42]), "+r"(dx[43]),
                  "+r"(dx[44]), "+r"(dx[45]), "+r"(dx[46]), "+r"(dx[47]),
                  "+r"(dx[48]), "+r"(dx[49]), "+r"(dx[50]), "+r"(dx[51]),
                  "+r"(dx[52]), "+r"(dx[53]), "+r"(dx[54]), "+r"(dx[55]),
                  "+r"(dx[56]), "+r"(dx[57]), "+r"(dx[58]), "+r"(dx[59]),
                  "+r"(dx[60]), "+r"(dx[61]), "+r"(dx[62]), "+r"(dx[63])
                : "l"(reinterpret_cast<uint64_t>(A)),
                  "l"(reinterpret_cast<uint64_t>(B2)),
                  "r"(1)
                : "memory"
            );
        } else if constexpr (M == 64 && N == 64 && K == 8) {
            // m64n64h8 — four m64n16h8 passes
            uint32_t * dx = D.x;
            for (int n = 0; n < 4; ++n) {
                const uint32_t * __ld_matrix* Bn = B + n * 16;
                bool sc = (n == 0 && !scale_d) ? false : (n == 0 ? scale_d : true);
                asm volatile(
                    "wgmma.mma_sync.sync.aligned.m64n16h8.s32.s8.s8.s32 "
                    "{%0<16>}, [%1], [%2], {%3<16>}, %4;"
                    : "+r"(dx[0]), "+r"(dx[1]), "+r"(dx[2]), "+r"(dx[3]),
                      "+r"(dx[4]), "+r"(dx[5]), "+r"(dx[6]), "+r"(dx[7]),
                      "+r"(dx[8]), "+r"(dx[9]), "+r"(dx[10]), "+r"(dx[11]),
                      "+r"(dx[12]), "+r"(dx[13]), "+r"(dx[14]), "+r"(dx[15]),
                      "+r"(dx[16]), "+r"(dx[17]), "+r"(dx[18]), "+r"(dx[19]),
                      "+r"(dx[20]), "+r"(dx[21]), "+r"(dx[22]), "+r"(dx[23]),
                      "+r"(dx[24]), "+r"(dx[25]), "+r"(dx[26]), "+r"(dx[27]),
                      "+r"(dx[28]), "+r"(dx[29]), "+r"(dx[30]), "+r"(dx[31]),
                      "+r"(dx[32]), "+r"(dx[33]), "+r"(dx[34]), "+r"(dx[35]),
                      "+r"(dx[36]), "+r"(dx[37]), "+r"(dx[38]), "+r"(dx[39]),
                      "+r"(dx[40]), "+r"(dx[41]), "+r"(dx[42]), "+r"(dx[43]),
                      "+r"(dx[44]), "+r"(dx[45]), "+r"(dx[46]), "+r"(dx[47]),
                      "+r"(dx[48]), "+r"(dx[49]), "+r"(dx[50]), "+r"(dx[51]),
                      "+r"(dx[52]), "+r"(dx[53]), "+r"(dx[54]), "+r"(dx[55]),
                      "+r"(dx[56]), "+r"(dx[57]), "+r"(dx[58]), "+r"(dx[59]),
                      "+r"(dx[60]), "+r"(dx[61]), "+r"(dx[62]), "+r"(dx[63]),
                      "+r"(dx[64]), "+r"(dx[65]), "+r"(dx[66]), "+r"(dx[67]),
                      "+r"(dx[68]), "+r"(dx[69]), "+r"(dx[70]), "+r"(dx[71]),
                      "+r"(dx[72]), "+r"(dx[73]), "+r"(dx[74]), "+r"(dx[75]),
                      "+r"(dx[76]), "+r"(dx[77]), "+r"(dx[78]), "+r"(dx[79]),
                      "+r"(dx[80]), "+r"(dx[81]), "+r"(dx[82]), "+r"(dx[83]),
                      "+r"(dx[84]), "+r"(dx[85]), "+r"(dx[86]), "+r"(dx[87]),
                      "+r"(dx[88]), "+r"(dx[89]), "+r"(dx[90]), "+r"(dx[91]),
                      "+r"(dx[92]), "+r"(dx[93]), "+r"(dx[94]), "+r"(dx[95]),
                      "+r"(dx[96]), "+r"(dx[97]), "+r"(dx[98]), "+r"(dx[99]),
                      "+r"(dx[100]), "+r"(dx[101]), "+r"(dx[102]), "+r"(dx[103]),
                      "+r"(dx[104]), "+r"(dx[105]), "+r"(dx[106]), "+r"(dx[107]),
                      "+r"(dx[108]), "+r"(dx[109]), "+r"(dx[110]), "+r"(dx[111]),
                      "+r"(dx[112]), "+r"(dx[113]), "+r"(dx[114]), "+r"(dx[115]),
                      "+r"(dx[116]), "+r"(dx[117]), "+r"(dx[118]), "+r"(dx[119]),
                      "+r"(dx[120]), "+r"(dx[121]), "+r"(dx[122]), "+r"(dx[123]),
                      "+r"(dx[124]), "+r"(dx[125]), "+r"(dx[126]), "+r"(dx[127])
                    : "l"(reinterpret_cast<uint64_t>(A)),
                      "l"(reinterpret_cast<uint64_t>(Bn)),
                      "r"(sc ? 1 : 0)
                    : "memory"
                );
            }
        }
    }

} // namespace ggml_cuda_wgmma
```

**Important implementation notes:**
- The `__ld_matrix__` pointer qualifier tells nvcc these pointers reference the LD matrix cache (shared memory, 16-byte aligned). This is required for WGMMA PTX.
- The `reinterpret_cast<uint64_t>()` converts the `__ld_matrix__` pointer to a 64-bit register for the PTX `[%reg]` constraint.
- The vector modifier `<16>` in the PTX tells the assembler to expect 16 consecutive registers for that operand list segment.
- For `N > 16` tiles (n32, n64), we decompose into multiple `m64n16h8` operations with advancing B pointers. This follows the CUTLASS convention.
- The `scale_d` parameter: `1` = use existing D values as accumulator (D += A@B), `0` = treat D as zero (D = A@B).

- [ ] **Step 2: Verify compilation**

```bash
cd /home/alexmv2025/projects/cuda/fast/claude_llama.cpp && cmake -B build_wgmma -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | tail -3 && cmake --build build_wgmma --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -20 && rm -rf build_wgmma
```

Expected: Build succeeds. If nvcc reports PTX syntax errors, it means the WGMMA instruction names or constraint formats are incorrect for CUDA 13.2 — adjust the PTX strings and retry.

- [ ] **Step 3: Verify PTX generation (if build succeeds)**

If the build produces a CUDA object file, extract and inspect PTX:

```bash
# Find the compiled CUDA fatbin
find build_wgmma -name "*.o" -path "*ggml-cuda*" | head -3 | while read f; do
    echo "=== $f ===" && cuobjdump -pxx "$f" 2>/dev/null | grep -i "wgmma" | head -5
done && rm -rf build_wgmma
```

Expected: If compiled for CC >= 1000, the PTX should contain `wgmma.mma_sync`, `wgmma.commit_sync`, and `wgmma.wait_sync` instructions. If no CC >= 1000 was in the build (because the machine doesn't have the GPU), the `#if __CUDA_ARCH__` guard prevents instantiation — this is expected.

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-cuda/mma.cuh
git commit -m "cuda: add WGMMA PTX intrinsics (mma_sync, commit_sync, wait_sync, fence) for Blackwell"
```

---

### Task 5: Integration verification — full build with tests

**Files:** (no new files — verification only)

**Context:** After adding the WGMMA primitives, we verify that the full project builds with tests enabled and that the existing test suite still passes on the CPU backend (since WGMMA code is guarded by `__CUDA_ARCH__ >= 1000` and won't execute without the GPU).

- [ ] **Step 1: Full CMake configure with tests**

```bash
cd /home/alexmv2025/projects/cuda/fast/claude_llama.cpp && rm -rf build_final && cmake -B build_final -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | tail -5
```

Expected: Configure succeeds. CMake should list all architectures including `100a-real;101a-real;120a-real;121a-real` (for CUDA 13.2).

- [ ] **Step 2: Build**

```bash
cmake --build build_final --config Release -j$(nproc) 2>&1 | tail -20
```

Expected: Build succeeds with no errors. No new warnings from `mma.cuh` or `common.cuh`.

- [ ] **Step 3: Run CPU-only tests (regression check)**

```bash
cd build_final && ctest -C Release --output-on-failure -R "test-backend-ops" 2>&1 | tail -20
```

Expected: Existing tests pass. The WGMMA code is in a guarded namespace and not referenced by any kernel yet, so it cannot affect existing test results.

- [ ] **Step 4: Clean up build directory**

```bash
rm -rf build_final
```

- [ ] **Step 5: Final commit check**

```bash
git status && git log --oneline -5
```

Expected: 3 commits — CMake, common.cuh, mma.cuh. Working tree clean.

---

## Follow-up Work (Phases 2 and 3 — separate plans)

**Phase 2: TMA Pipeline + mmq Kernel** (next plan to write)
- `cp-async.cuh`: TMA descriptor, `tma_load_bulk<>`, `.mbarrier` (SM120+); `cp_async_wait_stage` (SM100)
- `mmq.cuh`: new `mul_mat_q_wgmma<>` kernel with triple-buffered pipeline
- `mmq.cu`: dispatch logic, tile selection, TMA descriptor setup
- `test-backend-ops.cpp`: WGMMA vs mma.sync correctness comparison

**Phase 3: NUMA Polish + FP8** (later plan)
- `ggml-cuda.cu`: NUMA detection + logging in `ggml_cuda_init()`
- `mma.cuh`: FP8 WGMMA templates (`e4m3`, `e5m2`)
- `mma.cuh`: SM120+ `wgmma.store_gmem` writeback

---

## Risk Mitigation

| Risk | Mitigation in Phase 1 |
|------|----------------------|
| nvcc rejects WGMMA PTX syntax | Task 4 Step 2 catches this immediately; adjust PTX strings per CUDA 13.2 PTX ISA docs |
| `__ld_matrix__` qualifier not supported | Falls back to plain pointer if nvcc < 13.0; Phase 1 compiles but defers ld.matrix to Phase 2 |
| Register pressure from large `frag_c` | Structs are template specializations — they only instantiate when referenced (Phase 2) |
| Breaking existing build | `#if __CUDA_ARCH__ >= 1000` guard isolates all changes; zero impact on CC < 1000 |
