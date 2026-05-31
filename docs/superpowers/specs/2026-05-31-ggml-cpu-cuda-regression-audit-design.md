# Deep Audit & Regression Recovery for GGML (CPU/CUDA)

**Date:** 2026-05-31
**Branch:** `fix/ggml-cpu-cuda`
**Baseline:** `5429d89f8f31e3fee14c834a68af3a0ebe56a3d6` (2026-05-22)
**Target:** 1T model (DeepSeek V4 Pro) on dual-socket EPYC + 600GB+ RAM

## 1. Problem

After merging upstream llama.cpp changes into a high-performance fork optimized for DeepSeek V4 Pro, suspected performance regressions in:

- AOCC-optimized SIMD paths overwritten or improperly merged
- zen4/AVX-512 kernel flags misconfigured
- MoE expert routing bottlenecked by MUL_MAT_ID dispatch
- CUDA memory-staging optimizations lost for large weights
- Threadpool/barrier synchronization overhead increased

**Preliminary baseline audit found:** `GGML_USE_AOCC` and AOCC-specific `#ifdef` blocks do not exist in production code — optimizations were planned in design docs but never implemented. This spec covers both recovery of merged-out code AND initial implementation of planned optimizations.

## 2. Decomposition

The work is split into 4 independent phases, each with its own spec → plan → implementation cycle:

| Phase | Scope | Depends On |
|-------|-------|------------|
| Phase 1 — Audit | Static diff + flags check against baseline | — (first) |
| Phase 2 — AOCC Kernel Pack | `-march=znver4`, prefetch, GGML_USE_AOCC, AVX-512 VNNI | Phase 1 findings |
| Phase 3 — MoE Pipeline | `MUL_MAT_ID` async dispatch, NUMA-aware expert weights | Phase 1 |
| Phase 4 — CUDA Regression | ggml-cuda diff, EPYC pipeline recovery | Phase 1 |

**This spec covers only Phase 1 (Audit).**

## 3. Architecture: Audit Tool

A self-contained Python utility at `tools/regression/audit.py` that:

1. Accepts baseline commit hash and current HEAD
2. Runs 4 passes (Flags + config.h + NUMA, Compiler diagnostic, Code diff, Runtime)
3. Outputs structured report to `tools/regression/report/`

```
tools/regression/
├── audit.py              # CLI entry point
├── pass_flags.py         # Build flag comparison
├── pass_code.py          # Structured code diff classifier
├── pass_runtime.py       # Optional microbenchmark runner
├── report/               # Output directory (gitignored)
│   ├── flags_diff.txt
│   ├── code_diff_classified.txt
│   └── summary.html
└── report.html           # Flat HTML with inline diff, no JS
```

### 3.1 CLI

```bash
python tools/regression/audit.py <baseline_hash> [--head HEAD] [--files FILE...]
```

- `--head`: if omitted, uses `HEAD`
- `--files`: if omitted, uses the full hot-path file list (see §4.2)
- `--runtime`: if set, runs microbenchmarks (requires build at both baselines)
- Output: `tools/regression/report/summary.html`

## 4. Pass 1: Build Flags

### 4.1 What to check

Extract these from `ggml/CMakeLists.txt` and `ggml/src/ggml-cpu/CMakeLists.txt` at both revisions:

| Flag | Source |
|------|--------|
| `GGML_AVX512` | option default + #cmakedefine propagation to config.h |
| `GGML_AVX512_VNNI` | same |
| `GGML_AVX512_BF16` | same |
| `GGML_AVX_VNNI` | same |
| `GGML_FMA` | same |
| `GGML_F16C` | same |
| `GGML_AMX_TILE` / `GGML_AMX_INT8` / `GGML_AMX_BF16` | same |
| zen4 variant | `ggml_add_cpu_backend_variant(zen4 ...)` — present, lost, or altered flags |
| `-mavx512*` / `/arch:AVX512` | compiler flag strings in CMakeLists.txt |
| `GGML_CPU_ALL_VARIANTS` | option default |
| CUDA_ARCH_FLAGS | `ggml/src/ggml-cuda/CMakeLists.txt` — sm_80, sm_90a |

### 4.2 config.h propagation check

CMake options set in CMakeLists.txt are meaningless unless they reach the preprocessor via `#cmakedefine` in a config header. The audit must verify the **generated config.h** (or `build.ninja`) at both revisions:

```
ggml/build/ggml-config.h  (after cmake configure)
  ↓
  #cmakedefine GGML_AVX512        → #define GGML_AVX512 1   or  /* #undef GGML_AVX512 */
  #cmakedefine GGML_AVX512_VNNI
  #cmakedefine GGML_AVX512_BF16
  #cmakedefine GGML_FMA
  #cmakedefine GGML_F16C
  #cmakedefine GGML_NUMA
```

Detection logic:
```python
def check_config_h(baseline, head):
    # Read generated ggml-config.h from a build directory at each revision
    # Return any #cmakedefine that was set in CMakeLists.txt but NOT #define'd in config.h
    # This catches silent propagation failures (e.g. missing target_compile_definitions)
```

If no build directory exists at either revision, fall back to checking the **template** `ggml/include/ggml-config.h.in` for the presence of `#cmakedefine GGML_AVX512` lines — a missing template entry means the define can NEVER reach the compiler.

### 4.3 GGML_NUMA flag check (EPYC dual-socket critical)

On a dual-socket EPYC with 600GB+ RAM, `GGML_NUMA` is **mandatory**. Without it:
- All memory allocates on one NUMA node
- Cross-socket access goes through Infinity Fabric with 2-3× latency
- Thread affinity is random, defeating cache locality

Check both:

| Check | Source | Severity |
|-------|--------|----------|
| `option(GGML_NUMA ...)` in `ggml/CMakeLists.txt` | CMake option default | critical |
| `#cmakedefine GGML_NUMA` in config header | Generated config.h | critical |
| `ggml_numa_init()` call site | `ggml/src/ggml-cpu/ggml-cpu.c` — must be reachable at startup | critical |
| `ggml_set_numa_thread_affinity()` is called for each thread | threadpool init code | warning |

### 4.4 Detection logic

```python
# pseudocode
def check_flag_diff(baseline, head):
    report = []
    for cmake_file in CMAKE_FILES:
        base_flags = parse_option_blocks(checkout(baseline, cmake_file))
        head_flags = parse_option_blocks(checkout(head, cmake_file))
        for flag in base_flags | head_flags:
            if base_flags.get(flag) != head_flags.get(flag):
                report.append({
                    "flag": flag,
                    "baseline": base_flags.get(flag, "NOT FOUND"),
                    "head":     head_flags.get(flag, "NOT FOUND"),
                    "file":     cmake_file,
                    "severity": "critical" if flag in CRITICAL_FLAGS else "warning",
                })
    return report
```

Critical flags: `GGML_AVX512`, `GGML_AVX512_VNNI`, `GGML_AVX512_BF16`, `GGML_FMA`, `GGML_NUMA`, `zen4`.

## 5. Pass 2: Structured Code Diff

### 5.1 Files to scan

| File | Priority | Check |
|------|----------|-------|
| `ggml/src/ggml-cpu/ops.cpp` | **CRITICAL** | `mul_mat`, `mul_mat_id`, `get_rows`, `rms_norm`, `soft_max`, `rope` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | **CRITICAL** | `ggml_graph_compute`, `ggml_barrier`, threadpool, NUMA init |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | CRITICAL | backend_init, backend_reg, features, buffer types |
| `ggml/src/ggml-cpu/vec.h` | CRITICAL | dot product, quantized ops (Q4_0..Q8_0, Q2_K..Q6_K) |
| `ggml/src/ggml-cpu/vec.cpp` | CRITICAL | same |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h` | WARN | type traits, SIMD fallbacks |
| `ggml/src/ggml-cuda/ggml-cuda-epyc-pipeline.cu` | CRITICAL | pipeline health check, async dispatch |
| `ggml/src/ggml-cuda/CMakeLists.txt` | CRITICAL | CUDA_ARCH_FLAGS |
| `ggml/src/ggml-cuda/*.cu` | WARN | staging, async copy, stream sync |
| `ggml/src/ggml-cpu/CMakeLists.txt` | CRITICAL | arch variant, AVX-512 flag strings |

### 5.2 Classification

```python
CLASSIFIER_RULES = {
    "ops.cpp": {
        "CRITICAL": [
            "ggml_compute_forward_mul_mat",
            "vec_dot",
            "ggml_compute_forward_mul_mat_id",
            "ggml_compute_forward_get_rows",
            "ggml_compute_forward_rms_norm",
            "ggml_compute_forward_soft_max",
            "ggml_compute_forward_rope",
        ],
        "WARN": [
            "ggml_compute_forward_dup",
            "ggml_compute_forward_add",
            "ggml_compute_forward_cpy",
        ],
    },
    "ggml-cpu.c": {
        "CRITICAL": [
            "ggml_graph_compute",
            "ggml_graph_compute_kickoff",
            "ggml_barrier",
            "ggml_graph_compute_secondary_thread",
            "ggml_cpu_init",
            "ggml_numa_init",
            "ggml_cpu_set_numa_interleave",
        ],
        "WARN": [
            "ggml_cpu_try_fuse_ops",
        ],
    },
    "ggml-cpu.cpp": {
        "CRITICAL": [
            "ggml_backend_cpu_init",
            "ggml_backend_cpu_reg",
        ],
        "WARN": [
            "ggml_backend_cpu_get_features",
            "ggml_backend_cpu_get_extra_bufts",
        ],
    },
    "vec.h": {
        "CRITICAL": [
            "ggml_vec_dot_q4_0_q8_0",
            "ggml_vec_dot_q4_1_q8_1",
            "ggml_vec_dot_q5_0_q8_0",
            "ggml_vec_dot_q5_1_q8_1",
            "ggml_vec_dot_q8_0_q8_0",
            "ggml_vec_dot_q2_K_q8_K",
            "ggml_vec_dot_q3_K_q8_K",
            "ggml_vec_dot_q4_K_q8_K",
            "ggml_vec_dot_q5_K_q8_K",
            "ggml_vec_dot_q6_K_q8_K",
        ],
    },
}

CUDA_CRITICAL_PATTERNS = [
    "ggml_cuda_op_mul_mat",
    "ggml_cuda_op_get_rows",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaStreamSynchronize",
    "pipeline_prefetch",
    "async_copy",
    "cp.async",
]
```

### 5.3 Output format for each function

```
CRITICAL: ggml_compute_forward_mul_mat (ops.cpp:2450)
──────────────────────────────────────────────
- 309:   const int64_t ne10 = src1->ne[0];  // ORIGINAL
+ 312:   const int64_t ne10 = params->ne10;  // HEAD (changed param access)

- 410:   for (int64_t i = 0; i < ne00; i++) { vec_dot(...) }
+ 420:   for (int64_t i = 0; i < ne01; i++) { vec_dot(...) }
                               ^^ loop bound changed — potential performance impact

VERDICT: Loop bound differs. Baseline iterates over ne00, HEAD over ne01.
         If ne00 != ne01, this changes the number of vec_dot calls.
```

## 6. Compiler Environment Diagnostic

Before any code analysis, the audit script must capture the compiler version at **both revisions**. A regression is often caused by the build environment switching compilers (e.g., from AOCC 5.2.0 to system GCC 14) without any code change.

### 6.1 What to capture

```bash
# Compiler identity
cc  --version 2>&1 | head -3    # or 'aocc --version'
c++ --version 2>&1 | head -3

# Effective flags for a real compile
echo '#include <cstdint>' | cc  -mavx512f -E -dM - 2>&1 | grep -E '__AVX512|__FMA__|__SSE'
# This shows WHICH AVX-512 subfeatures the compiler actually enables
```

### 6.2 Diagnostic output

```
== Compiler ==
  Baseline (5429d89):
    cc:  AOCC 5.2.0 (AMD Optimizing Clang)
    c++: AOCC 5.2.0 (AMD Optimizing Clang)
    __AVX512F__: 1  __AVX512CD__: 1  __AVX512BW__: 1  __AVX512DQ__: 1  __AVX512VL__: 0

  HEAD (dc22a8d1):
    cc:  gcc 14.2.1 (GCC)
    c++: g++ 14.2.1 (GCC)
    __AVX512F__: 1  __AVX512CD__: 1  __AVX512BW__: 1  __AVX512DQ__: 1  __AVX512VL__: 0

  ⚠ COMPILER CHANGED: AOCC 5.2.0 → GCC 14.2.1
    This alone explains most performance differences.
    AOCC generates better vector code for AMD Zen cores.
```

### 6.3 Detection logic

```python
def check_compiler(build_dir):
    if not build_dir:
        # No build available — try PATH
        return {
            "cc_version":   subprocess.run(["cc", "--version"], ...),
            "cxx_version":  subprocess.run(["c++", "--version"], ...),
        }
    # Build dir exists — read compile_commands.json for exact compiler path
    with open(f"{build_dir}/compile_commands.json") as f:
        cc_cmd = json.load(f)[0]["command"]
        # Extract compiler path from the first compile command
```

Severity rules:

| Condition | Severity |
|-----------|----------|
| Same compiler + same version | info |
| Same compiler, different minor version | warning |
| Different compiler family (AOCC→GCC, GCC→Clang) | **critical** |
| Compiler missing from PATH | warning + expected path hint |

## 7. Pass 3: Runtime Microbenchmark (Optional)

Run only when Pass 1 or Pass 2 finds a suspicious change.

### 6.1 Build both baselines

```bash
git worktree add /tmp/baseline-checkout 5429d89f8f31e3fee14c834a68af3a0ebe56a3d6
cd /tmp/baseline-checkout && cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_AVX512=ON ...
cmake --build build --target bench
cp build/bin/bench /tmp/baseline-bench

cd /path/to/fork
cmake -B build-fix -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_AVX512=ON ...
cmake --build build-fix --target bench
cp build-fix/bin/bench /tmp/head-bench
```

### 6.2 Benchmark functions

```
Ops:     mul_mat (128,128)×(128,4096), mul_mat_id, get_rows, rms_norm
Quant:   vec_dot per Q4_0..Q6_K, Q8_0
Latency: single-thread, then N-thread (EPYC: 2×64 = 128 threads)
```

### 6.3 Comparison

```bash
/tmp/baseline-bench --ops mul_mat,get_rows,rms_norm --repeat 100
/tmp/head-bench    --ops mul_mat,get_rows,rms_norm --repeat 100
# report: % difference per op
```

## 7. Error Handling & Edge Cases

| Scenario | Behavior |
|----------|----------|
| Baseline commit not found | Error: `fatal: not a valid commit` |
| File doesn't exist in baseline | Report: "NEW FILE" with all content as added |
| File doesn't exist in HEAD | Report: "DELETED" with all content as removed |
| Zero diff (no changes) | Report: "No changes found for <file>" |
| `--head` not specified | Use `HEAD` |
| `--runtime` without build | Warning + skip runtime pass |
| Network unavailable | Only static analysis runs |
| config.h not generated (no build dir) | Fall back to `ggml-config.h.in` template check |
| Compiler not found on PATH | Warning + hint: "Expected AOCC 5.2.0 at /opt/AMD/aocc" |
| GGML_NUMA not found in config.h | Critical: "NUMA disabled — 2-socket EPYC will suffer Infinity Fabric penalty" |

## 8. Testing

| Test | Method |
|------|--------|
| audit.py runs without error | `python -m pytest tools/regression/ --doctest-modules` |
| Correct flag extraction | Unit test with known CMakeLists.txt fixtures |
| Correct function classification | Unit test with synthetic diffs |
| Empty diff handling | `git diff baseline baseline` → zero changes reported |
| Large diff (1000+ lines) | Integration test with baseline vs HEAD |
| `--runtime` skip | `--runtime` without build binaries → warning + no crash |
| Compiler version capture | Unit test with fake `cc --version` output fixture |
| config.h propagation | Unit test with known config.h.in template |

## 9. Future-Proofing

The audit tool is designed to be reusable for ALL future merges:

```bash
# Before every upstream merge:
python tools/regression/audit.py <pre-merge-baseline> --head origin/upstream/master
```

This makes future regression detection a 1-command operation, satisfying the "Zero-Regret Merging" constraint from the brief.
