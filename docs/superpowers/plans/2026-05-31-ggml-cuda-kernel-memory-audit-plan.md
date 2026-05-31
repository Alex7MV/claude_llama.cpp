# Phase 4: CUDA Kernel & Memory Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `tools/regression/audit.py` with CUDA-specific regression detection: H2D bandwidth benchmark, NVCC flags diff, MMVQ kernel prefetch analysis.

**Architecture:** Plugin architecture: `audit.py --cuda` dynamic-imports `pass_cuda_*.py` modules. CUDA-heavy work (H2D benchmark) lives in a standalone `cuda_bench.cu` binary, called as subprocess. Report dict gets a `report["passes"]["cuda_*"]` key, rendered as a dedicated section in `report.html`.

**Tech Stack:** Python 3.11+, CUDA C (nvcc), pytest, git

---

### Task 1: Test fixtures for CUDA passes

**Files:**
- Create: `tools/regression/tests/fixtures/CMakeLists.txt.cuda.full`
- Create: `tools/regression/tests/fixtures/CMakeLists.txt.cuda.stripped`
- Create: `tools/regression/tests/fixtures/cuda_bench.good.json`
- Create: `tools/regression/tests/fixtures/cuda_bench.slow.json`
- Create: `tools/regression/tests/fixtures/diff.mmvq.prefetch.txt`
- Create: `tools/regression/tests/fixtures/diff.mmvq.clean.txt`

- [ ] **Step 1: Create CMakeLists.txt.cuda.full**

Full CUDA CMakeLists.txt with all flags and archs

```txt
# Full CUDA config — baseline expected state
option(GGML_CUDA_FA "enable flash attention" ON)
option(GGML_CUDA_FORCE_MMQ "force mmq kernels" OFF)
option(GGML_CUDA_GRAPHS "enable CUDA graphs" OFF)
option(GGML_CUDA_NCCL "enable NCCL" ON)
option(GGML_CUDA_NO_VMM "disable VMM" OFF)
option(GGML_CUDA_COMPRESSION_MODE "link compression" "size")

set(CMAKE_CUDA_ARCHITECTURES "75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real 121a-real 100a-real 101a-real")
set_property(TARGET ggml-cuda PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_compile_options(ggml-cuda PRIVATE -use_fast_math -extended-lambda)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    target_compile_options(ggml-cuda PRIVATE -compress-mode=size)
endif()

# WGMMA detection
if(GGML_CUDA_HAS_WGMMA)
    target_compile_definitions(ggml-cuda PRIVATE GGML_CUDA_HAS_WGMMA=1)
endif()
```

- [ ] **Step 2: Create CMakeLists.txt.cuda.stripped**

Same but missing critical flags (regression scenario):

```txt
# Stripped CUDA config — regression scenario
option(GGML_CUDA_FA "enable flash attention" OFF)
option(GGML_CUDA_FORCE_MMQ "force mmq kernels" OFF)
option(GGML_CUDA_NCCL "enable NCCL" ON)

set(CMAKE_CUDA_ARCHITECTURES "75-virtual 80-virtual 86-real 89-real 90-virtual")
target_compile_options(ggml-cuda PRIVATE -extended-lambda)
```

- [ ] **Step 3: Create cuda_bench.good.json**

```json
{
  "gpu_name": "NVIDIA RTX 5090",
  "compute_capability": "12.1",
  "num_devices": 1,
  "h2d_bandwidth_gbs": 58.2,
  "h2d_max_size_bytes": 1073741824,
  "p2p_enabled": false,
  "p2p_bandwidth_gbs": null
}
```

- [ ] **Step 4: Create cuda_bench.slow.json**

```json
{
  "gpu_name": "NVIDIA RTX 5090",
  "compute_capability": "12.1",
  "num_devices": 1,
  "h2d_bandwidth_gbs": 12.3,
  "h2d_max_size_bytes": 1073741824,
  "p2p_enabled": false,
  "p2p_bandwidth_gbs": null
}
```

- [ ] **Step 5: Create diff.mmvq.prefetch.txt**

A diff that removes `#pragma unroll` from the mmvq kernel loop (regression):

```diff
--- a/ggml/src/ggml-cuda/mmvq.cu
+++ b/ggml/src/ggml-cuda/mmvq.cu
@@ -120,15 +120,14 @@ static __global__ void mul_mat_vec_q4_0_q8_0(
     const int row = blockIdx.x * blockDim.y + threadIdx.y;
     if (row >= nrows) return;

-    #pragma unroll
     for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
         const int ib = row * ncols + col;
         float sum = 0.0f;
         #pragma unroll
         for (int j = 0; j < QK4_0; j += 4) {
-            // prefetch registers: load 4 values ahead
-            const float4 v = v4_load(&x[ib].qs[j]);
-            sum += v.x * (float)y[ib].qs[j] + v.y * (float)y[ib].qs[j+1]
-                 + v.z * (float)y[ib].qs[j+2] + v.w * (float)y[ib].qs[j+3];
+            // fallback: scalar load
+            for (int k = 0; k < 4; k++)
+                sum += (float)(x[ib].qs[j+k] & 0x0F) * (float)y[ib].qs[j+k];
         }
     }
```

- [ ] **Step 6: Create diff.mmvq.clean.txt**

A clean additive-only diff (no regression):

```diff
--- a/ggml/src/ggml-cuda/mmvq.cu
+++ b/ggml/src/ggml-cuda/mmvq.cu
@@ -130,6 +130,10 @@ static __global__ void mul_mat_vec_q4_0_q8_0(
     const int row = blockIdx.x * blockDim.y + threadIdx.y;
     if (row >= nrows) return;

+    // new safety check for out-of-bounds
+    if (threadIdx.x >= blockDim.x) return;
+    if (row >= nrows) return;
+
     for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
         const int ib = row * ncols + col;
         float sum = 0.0f;
```

- [ ] **Step 7: Commit**

```bash
git add tools/regression/tests/fixtures/CMakeLists.txt.cuda.full tools/regression/tests/fixtures/CMakeLists.txt.cuda.stripped tools/regression/tests/fixtures/cuda_bench.good.json tools/regression/tests/fixtures/cuda_bench.slow.json tools/regression/tests/fixtures/diff.mmvq.prefetch.txt tools/regression/tests/fixtures/diff.mmvq.clean.txt
git commit -m "test: add CUDA pass fixtures for Phase 4"
```

---

### Task 2: cuda_bench.cu — standalone CUDA benchmark binary

**Files:**
- Create: `tools/regression/cuda_bench.cu`

- [ ] **Step 1: Write cuda_bench.cu**

This binary probes GPU capabilities and runs H2D bandwidth benchmarks. Outputs JSON to stdout.

```c
// tools/regression/cuda_bench.cu
// Standalone CUDA H2D bandwidth benchmark.
// Compile: nvcc -O3 -o cuda_bench cuda_bench.cu
// Run:     ./cuda_bench [--p2p]

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static const char *cuda_err(cudaError_t e) {
    return cudaGetErrorString(e);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cuda_err(err)); \
        exit(1); \
    } \
} while(0)

// Bandwidth test sizes
static const size_t test_sizes[] = {1UL << 20, 10UL << 20, 100UL << 20, 1UL << 30};
static const int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

// Returns H2D bandwidth in GB/s for the largest tested size
static double bench_h2d(size_t max_size, int num_devices) {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    // Query device memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    // Use at most half of free memory
    size_t alloc_size = max_size;
    if (alloc_size > free_mem / 2) {
        alloc_size = free_mem / 2;
    }
    // Round down to test_sizes
    for (int i = num_sizes - 1; i >= 0; i--) {
        if (test_sizes[i] <= alloc_size) {
            alloc_size = test_sizes[i];
            break;
        }
    }
    if (alloc_size < (1UL << 20)) {
        alloc_size = 1UL << 20;  // minimum 1 MB
    }

    // Allocate pinned host memory
    void *host_buf = NULL;
    CUDA_CHECK(cudaHostAlloc(&host_buf, alloc_size, cudaHostAllocDefault));
    // Touch pages
    memset(host_buf, 0xAB, alloc_size);

    // Allocate device memory
    void *dev_buf = NULL;
    CUDA_CHECK(cudaMalloc(&dev_buf, alloc_size));

    // Warmup
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemcpy(dev_buf, host_buf, alloc_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double best_sec = 1e100;
    const int num_trials = 10;
    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(dev_buf, host_buf, alloc_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double sec = ms / 1000.0;
        if (sec > 0 && sec < best_sec) best_sec = sec;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dev_buf));
    CUDA_CHECK(cudaFreeHost(host_buf));

    // Bandwidth in GB/s: (bytes / sec) / 1e9
    return (double)alloc_size / best_sec / 1e9;
}

int main(int argc, char **argv) {
    int do_p2p = (argc > 1 && strcmp(argv[1], "--p2p") == 0);

    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // H2D benchmark
    double h2d_gbs = bench_h2d(1UL << 30, num_devices);

    // P2P (reserved)
    int p2p_enabled = 0;
    double p2p_gbs = 0.0;
    if (do_p2p && num_devices > 1) {
        for (int i = 0; i < num_devices && !p2p_enabled; i++) {
            for (int j = 0; j < num_devices; j++) {
                if (i == j) continue;
                int can = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can, i, j));
                if (can) { p2p_enabled = 1; break; }
            }
        }
        // TODO: measure P2P bandwidth in future
    }

    printf("{\n");
    printf("  \"gpu_name\": \"%s\",\n", prop.name);
    printf("  \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
    printf("  \"num_devices\": %d,\n", num_devices);
    printf("  \"h2d_bandwidth_gbs\": %.1f,\n", h2d_gbs);
    printf("  \"h2d_max_size_bytes\": %zu,\n", (size_t)(1UL << 30));
    printf("  \"p2p_enabled\": %s,\n", p2p_enabled ? "true" : "false");
    printf("  \"p2p_bandwidth_gbs\": %s\n", p2p_enabled ? "0.0" : "null");
    printf("}\n");

    return 0;
}
```

- [ ] **Step 2: Compile and verify**

Run: `nvcc -O3 -o tools/regression/cuda_bench tools/regression/cuda_bench.cu`
Expected: no errors, binary `tools/regression/cuda_bench` exists

Run: `./tools/regression/cuda_bench`
Expected: JSON output with `h2d_bandwidth_gbs` field showing a positive number

- [ ] **Step 3: Commit**

```bash
git add tools/regression/cuda_bench.cu
git commit -m "feat: add cuda_bench.cu — standalone H2D bandwidth benchmark"
```

---

### Task 3: pass_cuda_flags.py — CUDA CMake flags diff pass

**Files:**
- Create: `tools/regression/pass_cuda_flags.py`
- Create: `tools/regression/tests/test_pass_cuda_flags.py`

- [ ] **Step 1: Write the failing tests**

```python
# tools/regression/tests/test_pass_cuda_flags.py
from tools.regression.pass_cuda_flags import (
    parse_cuda_options,
    check_cuda_architectures,
    cmp_cuda_flags,
)

FIXTURE_DIR = "tools/regression/tests/fixtures"


def test_parse_cuda_options_finds_flags():
    cmake_text = """
option(GGML_CUDA_FA "enable flash attention" ON)
option(GGML_CUDA_FORCE_MMQ "force mmq" OFF)
option(GGML_CUDA_NCCL "enable NCCL" ON)
"""
    result = parse_cuda_options(cmake_text)
    assert result["GGML_CUDA_FA"] == "ON"
    assert result["GGML_CUDA_FORCE_MMQ"] == "OFF"
    assert result["GGML_CUDA_NCCL"] == "ON"


def test_parse_cuda_options_empty():
    result = parse_cuda_options("# no options")
    assert result == {}


def test_check_cuda_architectures_ok():
    archs = "75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real 121a-real 100a-real 101a-real"
    problems = check_cuda_architectures(archs)
    assert len(problems) == 0


def test_check_cuda_architectures_missing_90a():
    archs = "75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real"
    problems = check_cuda_architectures(archs)
    # sm_121a, sm_100a, sm_101a expected but missing → warning for each
    assert len(problems) == 3
    for p in problems:
        assert p["severity"] == "warning"


def test_cmp_cuda_flags_all_match():
    full = open(f"{FIXTURE_DIR}/CMakeLists.txt.cuda.full").read()
    findings = cmp_cuda_flags(full, full)
    # no mismatches when same text
    mismatch_flags = [f for f in findings if f.get("status", "") == "mismatch"]
    assert len(mismatch_flags) == 0


def test_cmp_cuda_flags_detects_stripped():
    full = open(f"{FIXTURE_DIR}/CMakeLists.txt.cuda.full").read()
    stripped = open(f"{FIXTURE_DIR}/CMakeLists.txt.cuda.stripped").read()
    findings = cmp_cuda_flags(full, stripped)
    mismatch_flags = [f for f in findings if f.get("status", "") == "mismatch"]
    # Should detect: GGML_CUDA_FA (ON→OFF), missing sm_120a, sm_121a, sm_100a, sm_101a,
    # missing -use_fast_math, missing -compress-mode
    assert len(mismatch_flags) >= 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/regression/tests/test_pass_cuda_flags.py -v`
Expected: FAIL with "ModuleNotFoundError" or "function not defined"

- [ ] **Step 3: Write minimal implementation**

```python
# tools/regression/pass_cuda_flags.py
"""Pass: CUDA CMake flags diff between baseline and head."""

import re
import subprocess


def parse_cuda_options(cmake_text: str) -> dict:
    """Parse option() calls from CUDA CMakeLists.txt content."""
    result = {}
    for m in re.finditer(r'^\s*option\s*\(\s*(\w+)\s+.*?\s+(ON|OFF)\s*\)', cmake_text, re.MULTILINE):
        result[m.group(1)] = m.group(2)
    return result


def check_cuda_architectures(arch_str: str) -> list:
    """Check that key CUDA architectures are present. Returns list of finding dicts."""
    expected = {"120a-real", "121a-real", "100a-real", "101a-real"}
    actual = set(arch_str.split())
    findings = []
    for arch in sorted(expected):
        if arch not in actual:
            findings.append({
                "flag": f"CMAKE_CUDA_ARCHITECTURES: {arch}",
                "status": "missing",
                "severity": "warning",
            })
    return findings


def _has_compile_flag(text: str, flag: str) -> bool:
    """Check if target_compile_options contains the given flag."""
    # Collect all lines between target_compile_options(ggml-cuda PRIVATE ...)
    in_block = False
    block_lines = []
    for line in text.split("\n"):
        if "target_compile_options(ggml-cuda PRIVATE" in line:
            in_block = True
            block_lines.append(line)
            continue
        if in_block:
            block_lines.append(line)
            if ")" in line and not line.strip().startswith("target_compile_options"):
                break
    full_block = " ".join(block_lines)
    return flag in full_block


def _find_option(text: str, name: str) -> str:
    """Find the value of a specific CMake option."""
    for m in re.finditer(r'^\s*option\s*\(' + re.escape(name) + r'\s+.*?\s+(ON|OFF)\s*\)', text, re.MULTILINE):
        return m.group(1)
    return "NOT SET"


def cmp_cuda_flags(baseline_text: str, head_text: str) -> list:
    """Compare CUDA flags between baseline and head CMakeLists.txt content."""
    findings = []

    # Compare options
    baseline_opts = parse_cuda_options(baseline_text)
    head_opts = parse_cuda_options(head_text)
    all_opts = set(baseline_opts.keys()) | set(head_opts.keys())
    for opt in sorted(all_opts):
        b = baseline_opts.get(opt, "NOT SET")
        h = head_opts.get(opt, "NOT SET")
        if b != h:
            findings.append({
                "flag": opt,
                "baseline": b,
                "head": h,
                "severity": "warning" if h == "NOT SET" else "info",
            })

    # Compare architectures
    basearch = re.search(r'set\s*\(\s*CMAKE_CUDA_ARCHITECTURES\s+"([^"]+)"', baseline_text)
    headarch = re.search(r'set\s*\(\s*CMAKE_CUDA_ARCHITECTURES\s+"([^"]+)"', head_text)
    if basearch and headarch and basearch.group(1) != headarch.group(1):
        findings.append({
            "flag": "CMAKE_CUDA_ARCHITECTURES",
            "baseline": basearch.group(1),
            "head": headarch.group(1),
            "severity": "critical",
        })
    if headarch:
        findings.extend(check_cuda_architectures(headarch.group(1)))

    # Compare compile flags
    for flag in ["-use_fast_math", "-compress-mode"]:
        b_has = _has_compile_flag(baseline_text, flag)
        h_has = _has_compile_flag(head_text, flag)
        if b_has and not h_has:
            findings.append({
                "flag": f"compile_flag: {flag}",
                "baseline": "present",
                "head": "missing",
                "severity": "critical",
            })

    # Check WGMMA
    b_wgmma = "GGML_CUDA_HAS_WGMMA" in baseline_text
    h_wgmma = "GGML_CUDA_HAS_WGMMA" in head_text
    if b_wgmma and not h_wgmma:
        findings.append({
            "flag": "GGML_CUDA_HAS_WGMMA",
            "baseline": "present",
            "head": "missing",
            "severity": "critical",
        })

    return findings


def run_cuda_flags_pass(baseline: str, head: str, critical: list, warnings: list) -> list:
    """Run the CUDA flags pass: compare CUDA CMakeLists.txt flags."""
    findings = []
    try:
        baselinesh = subprocess.run(
            ["git", "show", f"{baseline}:ggml/src/ggml-cuda/CMakeLists.txt"],
            capture_output=True, text=True, check=True)
        headsh = subprocess.run(
            ["git", "show", f"{head}:ggml/src/ggml-cuda/CMakeLists.txt"],
            capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        findings.append({"flag": "git_show", "severity": "error",
                         "baseline": "?", "head": "?",
                         "detail": "could not read CMakeLists.txt at one or both revisions"})
        return findings

    findings = cmp_cuda_flags(baselinesh.stdout, headsh.stdout)

    for f in findings:
        if f.get("severity") == "critical":
            critical.append(f"CUDA flags: {f['flag']} (baseline={f.get('baseline','?')}, head={f.get('head','?')})")
        elif f.get("severity") == "warning":
            warnings.append(f"CUDA flags: {f['flag']}")

    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/regression/tests/test_pass_cuda_flags.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/regression/pass_cuda_flags.py tools/regression/tests/test_pass_cuda_flags.py
git commit -m "feat: pass_cuda_flags — CUDA CMake flags diff pass"
```

---

### Task 4: pass_cuda_memory.py — H2D bandwidth verification pass

**Files:**
- Create: `tools/regression/pass_cuda_memory.py`
- Create: `tools/regression/tests/test_pass_cuda_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
# tools/regression/tests/test_pass_cuda_memory.py
import json
from tools.regression.pass_cuda_memory import (
    load_cuda_bench_json,
    check_h2d_bandwidth,
    compute_pcie_limit,
    run_cuda_memory_pass,
)

FIXTURE_DIR = "tools/regression/tests/fixtures"


def test_load_good_json():
    data = json.load(open(f"{FIXTURE_DIR}/cuda_bench.good.json"))
    assert data["h2d_bandwidth_gbs"] == 58.2
    assert data["num_devices"] == 1


def test_load_slow_json():
    data = json.load(open(f"{FIXTURE_DIR}/cuda_bench.slow.json"))
    assert data["h2d_bandwidth_gbs"] == 12.3


def test_compute_pcie_gen5():
    limit = compute_pcie_limit("12.1")  # Blackwell = Gen5
    assert limit == 63.0


def test_compute_pcie_gen4():
    limit = compute_pcie_limit("8.9")  # Ampere = Gen4
    assert limit == 31.5


def test_compute_pcie_gen3():
    limit = compute_pcie_limit("7.0")  # Volta = Gen3
    assert limit == 15.8


def test_compute_pcie_unknown():
    limit = compute_pcie_limit("0.0")
    assert limit is None


def test_check_h2d_good():
    data = {"h2d_bandwidth_gbs": 58.2, "compute_capability": "12.1"}
    findings = check_h2d_bandwidth(data)
    num_critical = sum(1 for f in findings if f.get("severity") == "critical")
    num_warning = sum(1 for f in findings if f.get("severity") == "warning")
    assert num_critical == 0
    assert num_warning == 0


def test_check_h2d_slow():
    data = {"h2d_bandwidth_gbs": 12.3, "compute_capability": "12.1"}
    findings = check_h2d_bandwidth(data)
    num_critical = sum(1 for f in findings if f.get("severity") == "critical")
    assert num_critical >= 1


def test_check_h2d_mid():
    # 45 GB/s on Gen5 (12.1) = 71% threshold → warning, not critical
    data = {"h2d_bandwidth_gbs": 45.0, "compute_capability": "12.1"}
    findings = check_h2d_bandwidth(data)
    num_critical = sum(1 for f in findings if f.get("severity") == "critical")
    num_warning = sum(1 for f in findings if f.get("severity") == "warning")
    assert num_critical == 0
    assert num_warning >= 1


def test_run_pass_no_binary():
    critical, warnings = [], []
    findings = run_cuda_memory_pass("HEAD", "HEAD", critical, warnings, binary_path="nonexistent_binary")
    assert len(findings) > 0
    # Should report "binary not found" as info
    assert any("not found" in str(f.get("detail", "")) for f in findings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/regression/tests/test_pass_cuda_memory.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Write minimal implementation**

```python
# tools/regression/pass_cuda_memory.py
"""Pass: CUDA H2D bandwidth verification via cuda_bench binary."""

import json
import os
import subprocess


def compute_pcie_limit(compute_capability: str) -> float | None:
    """Return expected PCIe bandwidth limit (GB/s) based on compute capability.

    Gen5 (Blackwell 12.x): 63 GB/s theoretical
    Gen4 (Ampere 8.x, Ada 8.9): 31.5 GB/s
    Gen3 (Volta 7.x, Turing 7.5): 15.8 GB/s
    """
    try:
        major, minor = compute_capability.split(".")
        sm = int(major) * 10 + int(minor)
    except (ValueError, AttributeError):
        return None
    if sm >= 120:
        return 63.0  # Blackwell: PCIe Gen5 x16
    elif sm >= 80:
        return 31.5  # Ampere/Ada: PCIe Gen4 x16
    elif sm >= 70:
        return 15.8  # Volta/Turing: PCIe Gen3 x16
    return None


def load_cuda_bench_json(path: str) -> dict | None:
    """Run cuda_bench binary and parse JSON output."""
    if not os.path.isfile(path):
        return None
    try:
        result = subprocess.run([path], capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            json.JSONDecodeError, FileNotFoundError):
        return None


def check_h2d_bandwidth(data: dict) -> list:
    """Check H2D bandwidth against expected PCIe limit."""
    findings = []
    bw = data.get("h2d_bandwidth_gbs", 0.0)
    cc = data.get("compute_capability", "0.0")
    limit = compute_pcie_limit(cc)

    if limit is None:
        findings.append({
            "flag": "h2d_bandwidth_pcie_limit",
            "baseline": str(limit),
            "head": f"{bw:.1f} GB/s",
            "severity": "info",
            "detail": f"Unknown PCIe generation for CC {cc}",
        })
        return findings

    pct = bw / limit * 100.0
    if bw < limit * 0.5:
        findings.append({
            "flag": "h2d_bandwidth",
            "baseline": f">={limit * 0.5:.0f} GB/s",
            "head": f"{bw:.1f} GB/s ({pct:.0f}% of {limit:.0f} GB/s limit)",
            "severity": "critical",
            "detail": f"H2D bandwidth {bw:.1f} GB/s is below 50% of PCIe {limit:.0f} GB/s limit",
        })
    elif bw < limit * 0.8:
        findings.append({
            "flag": "h2d_bandwidth",
            "baseline": f">={limit * 0.8:.0f} GB/s",
            "head": f"{bw:.1f} GB/s ({pct:.0f}% of {limit:.0f} GB/s limit)",
            "severity": "warning",
            "detail": f"H2D bandwidth {bw:.1f} GB/s is below 80% of PCIe {limit:.0f} GB/s limit",
        })
    else:
        findings.append({
            "flag": "h2d_bandwidth",
            "baseline": f">={limit * 0.5:.0f} GB/s",
            "head": f"{bw:.1f} GB/s ({pct:.0f}% of {limit:.0f} GB/s limit)",
            "severity": "pass",
            "detail": f"H2D bandwidth {bw:.1f} GB/s meets PCIe {limit:.0f} GB/s limit",
        })

    return findings


def run_cuda_memory_pass(baseline: str, head: str, critical: list, warnings: list, binary_path: str | None = None) -> list:
    """Run the CUDA memory pass: execute cuda_bench and check H2D bandwidth.
    
    Args:
        baseline: baseline commit hash (unused, for interface compatibility)
        head: head commit hash (unused, for interface compatibility)
        critical: list to append critical findings
        warnings: list to append warnings
        binary_path: path to cuda_bench binary (default: cuda_bench in same directory)
    """
    findings = []
    if binary_path is None:
        binary_path = os.path.join(os.path.dirname(__file__), "cuda_bench")
    data = load_cuda_bench_json(binary_path)
    if data is None:
        findings.append({
            "flag": "cuda_bench",
            "severity": "info",
            "detail": f"cuda_bench binary not found at {binary_path} or failed to execute",
        })
        return findings

    # Record device info
    findings.append({
        "flag": "gpu_info",
        "baseline": f"{data.get('gpu_name', '?')} CC {data.get('compute_capability', '?')}",
        "head": f"{data.get('gpu_name', '?')} CC {data.get('compute_capability', '?')}",
        "severity": "info",
    })

    # Check H2D bandwidth
    bw_findings = check_h2d_bandwidth(data)
    findings.extend(bw_findings)

    for f in bw_findings:
        if f.get("severity") == "critical":
            critical.append(f"CUDA H2D bandwidth: {f['detail']}")
        elif f.get("severity") == "warning":
            warnings.append(f"CUDA H2D bandwidth: {f['detail']}")

    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/regression/tests/test_pass_cuda_memory.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/regression/pass_cuda_memory.py tools/regression/tests/test_pass_cuda_memory.py
git commit -m "feat: pass_cuda_memory — H2D bandwidth verification pass"
```

---

### Task 5: pass_cuda_mmvq.py — MMVQ register-prefetch diff pass

**Files:**
- Create: `tools/regression/pass_cuda_mmvq.py`
- Create: `tools/regression/tests/test_pass_cuda_mmvq.py`

- [ ] **Step 1: Write the failing tests**

```python
# tools/regression/tests/test_pass_cuda_mmvq.py
from tools.regression.pass_cuda_mmvq import (
    classify_mmvq_hunks,
    run_cuda_mmvq_pass,
)

FIXTURE_DIR = "tools/regression/tests/fixtures"


def test_classify_prefetch_removal_critical():
    with open(f"{FIXTURE_DIR}/diff.mmvq.prefetch.txt") as f:
        diff_text = f.read()
    findings = classify_mmvq_hunks(diff_text, "diff.mmvq.prefetch.txt")
    critical = [f for f in findings if f.get("severity") == "critical"]
    assert len(critical) >= 1
    # Should detect #pragma unroll removal
    assert any("unroll" in c.get("detail", "").lower() for c in critical)


def test_classify_prefetch_removal_v4_load():
    with open(f"{FIXTURE_DIR}/diff.mmvq.prefetch.txt") as f:
        diff_text = f.read()
    findings = classify_mmvq_hunks(diff_text, "diff.mmvq.prefetch.txt")
    warning = [f for f in findings if f.get("severity") == "warning"]
    # Should detect v4_load → scalar fallback
    v4_findings = [f for f in warning if "v4_load" in f.get("detail", "")]
    assert len(v4_findings) >= 1


def test_classify_clean_diff_no_issues():
    with open(f"{FIXTURE_DIR}/diff.mmvq.clean.txt") as f:
        diff_text = f.read()
    findings = classify_mmvq_hunks(diff_text, "diff.mmvq.clean.txt")
    critical = [f for f in findings if f.get("severity") == "critical"]
    assert len(critical) == 0


def test_classify_empty_diff():
    findings = classify_mmvq_hunks("", "empty.diff")
    assert len(findings) == 0


def test_run_pass_no_diff():
    critical, warnings = [], []
    # No diff between same revision
    findings = run_cuda_mmvq_pass("HEAD", "HEAD", critical, warnings)
    assert len(findings) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/regression/tests/test_pass_cuda_mmvq.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Write minimal implementation**

```python
# tools/regression/pass_cuda_mmvq.py
"""Pass: MMVQ kernel diff analysis — detect register-prefetch regression."""

import re
import subprocess


def classify_mmvq_hunks(diff_text: str, filename: str) -> list:
    """Analyze diff hunks in mmvq.cu for register-prefetch regressions."""
    findings = []

    # Parse hunks: @@ -start,count +start,count @@
    current_hunk = 0
    hunk_lines = []
    for line in diff_text.split("\n"):
        if line.startswith("@@"):
            if hunk_lines:
                _analyze_hunk(hunk_lines, current_hunk, findings, filename)
            current_hunk += 1
            hunk_lines = []
        elif line.startswith("+") or line.startswith("-"):
            hunk_lines.append(line)

    if hunk_lines:
        _analyze_hunk(hunk_lines, current_hunk, findings, filename)

    return findings


def _analyze_hunk(hunk_lines: list, hunk_id: int, findings: list, filename: str):
    """Analyze a single diff hunk for known regression patterns."""
    removed_lines = [l[1:] for l in hunk_lines if l.startswith("-")]
    added_lines = [l[1:] for l in hunk_lines if l.startswith("+")]
    full_removed = "\n".join(removed_lines)
    full_added = "\n".join(added_lines)

    # Pattern 1: #pragma unroll removed
    if "#pragma unroll" in full_removed and "#pragma unroll" not in full_added:
        findings.append({
            "flag": f"{filename}:hunk_{hunk_id}",
            "severity": "critical",
            "detail": "#pragma unroll removed from loop — register prefetch may be lost",
        })

    # Pattern 2: v4_load → scalar fallback
    if "v4_load" in full_removed and "v4_load" not in full_added:
        findings.append({
            "flag": f"{filename}:hunk_{hunk_id}",
            "severity": "warning",
            "detail": "v4_load vectorized load replaced — register prefetch degraded to scalar",
        })

    # Pattern 3: float4 → float (register packing lost)
    if re.search(r"float4\b", full_removed) and not re.search(r"float4\b", full_added):
        findings.append({
            "flag": f"{filename}:hunk_{hunk_id}",
            "severity": "warning",
            "detail": "float4 register packing removed — may reduce memory throughput",
        })

    # Pattern 4: Loop structure change (tile loop → simple loop)
    if any("for" in l for l in removed_lines) and any("for" in l for l in added_lines):
        # Check if loop structure changed significantly
        removed_loops = [l for l in removed_lines if "for" in l]
        added_loops = [l for l in added_lines if "for" in l]
        if len(removed_loops) != len(added_loops):
            findings.append({
                "flag": f"{filename}:hunk_{hunk_id}",
                "severity": "info",
                "detail": "Loop structure changed — verify tile prefetch is preserved",
            })

    # Pattern 5: __shared__ memory introduced (prefetch target change)
    if "__shared__" in full_added and "__shared__" not in full_removed:
        findings.append({
            "flag": f"{filename}:hunk_{hunk_id}",
            "severity": "warning",
            "detail": "__shared__ memory introduced — prefetch target may have changed from register to shared",
        })


def run_cuda_mmvq_pass(baseline: str, head: str, critical: list, warnings: list) -> list:
    """Run the CUDA MMVQ pass: diff mmvq.cu and check for prefetch regression."""
    findings = []
    try:
        result = subprocess.run(
            ["git", "diff", f"{baseline}..{head}", "--", "ggml/src/ggml-cuda/mmvq.cu"],
            capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        findings.append({
            "flag": "mmvq_diff",
            "severity": "error",
            "detail": "git diff failed for mmvq.cu",
        })
        return findings

    diff_text = result.stdout
    if not diff_text.strip():
        return findings  # no changes — clean

    findings = classify_mmvq_hunks(diff_text, "mmvq.cu")

    for f in findings:
        if f.get("severity") == "critical":
            critical.append(f"MMVQ kernel: {f['detail']}")
        elif f.get("severity") == "warning":
            warnings.append(f"MMVQ kernel: {f['detail']}")

    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/regression/tests/test_pass_cuda_mmvq.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/regression/pass_cuda_mmvq.py tools/regression/tests/test_pass_cuda_mmvq.py
git commit -m "feat: pass_cuda_mmvq — MMVQ register-prefetch diff pass"
```

---

### Task 6: Refactor audit.py — dynamic import of pass_cuda_*.py

**Files:**
- Modify: `tools/regression/audit.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tools/regression/tests/test_audit.py

def test_audit_cuda_flag_imports_passes():
    """Verify --cuda flag causes dynamic import of pass_cuda_* modules."""
    import subprocess
    # Just check that --cuda flag is accepted
    result = subprocess.run(
        ["python", "tools/regression/audit.py", "HEAD", "--cuda", "--head", "HEAD"],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "CUDA" in result.stdout or "cuda" in result.stdout
```

- [ ] **Step 2: Write the audit.py refactoring**

Refactor `audit.py` to add `--cuda` flag and dynamic import of `pass_cuda_*.py` modules:

```python
#!/usr/bin/env python3
"""CLI entry point for GGML regression audit.

Usage:
    python tools/regression/audit.py <baseline_commit> [--head HEAD] [--files FILE...] [--runtime] [--cuda]
"""

import argparse
import html
import importlib
import inspect
import json
import os
import pkgutil
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from pathlib import Path

REPORT_DIR = Path(__file__).parent / "report"


def render_html(report: dict) -> str:
    """Build a flat HTML report from structured findings dict."""
    lines = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>GGML Regression Audit</title>",
             "<style>body{font-family:monospace;max-width:960px;margin:auto;padding:2em}"
             ".critical{color:#c00;font-weight:bold}.warning{color:#c60}table{border-collapse:collapse;width:100%}"
             "td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}"
             "th{background:#eee}</style></head><body>",
             f"<h1>GGML Regression Audit</h1>",
             f"<p>Baseline: {report.get('baseline', '?')} | Head: {report.get('head', '?')}</p>"]

    if report.get("critical"):
        lines += ["<h2 class='critical'>Critical Issues</h2><ul>"]
        for item in report["critical"]:
            lines.append(f"<li>{html.escape(str(item))}</li>")
        lines.append("</ul>")

    passes = report.get("passes", {})
    for name, data in passes.items():
        # Skip cuda_* passes when not in --cuda mode
        if name.startswith("cuda_") and data is None:
            continue
        display_name = name.replace("_", " ").title()
        lines.append(f"<h2>Pass: {html.escape(display_name)}</h2>")
        if data is None:
            lines.append("<p>skipped</p>")
        elif isinstance(data, str):
            lines.append(f"<pre>{html.escape(data)}</pre>")
        elif isinstance(data, list):
            if not data:
                lines.append("<p>No issues found.</p>")
            else:
                lines += ["<table><tr><th>Flag</th><th>Baseline</th><th>Head</th><th>Severity</th></tr>"]
                for row in data:
                    sev = row.get("severity", "info")
                    cls = f" class='{sev}'" if sev in ("critical", "warning") else ""
                    lines.append(f"<tr{cls}><td>{html.escape(str(row.get('flag','')))}</td>"
                                 f"<td>{html.escape(str(row.get('baseline','')))}</td>"
                                 f"<td>{html.escape(str(row.get('head','')))}</td>"
                                 f"<td>{sev}</td></tr>")
                lines.append("</table>")
        else:
            lines.append(f"<pre>{html.escape(str(data))}</pre>")

    lines.append(f"<hr><p>Generated by ggml regression audit</p></body></html>")
    return "\n".join(lines)


def _discover_cuda_passes():
    """Discover pass_cuda_*.py modules in the regression directory.

    Returns dict mapping pass name → (module, run_function).
    """
    cuda_passes = {}
    reg_dir = os.path.dirname(os.path.abspath(__file__))
    for importer, modname, ispkg in pkgutil.iter_modules([reg_dir]):
        if modname.startswith("pass_cuda_") and not ispkg:
            mod = importlib.import_module(f"tools.regression.{modname}")
            # Find the run function
            for name, obj in inspect.getmembers(mod, inspect.isfunction):
                if name.startswith("run_"):
                    cuda_passes[modname] = (mod, obj)
                    break
    return cuda_passes


def main():
    parser = argparse.ArgumentParser(description="GGML regression audit against a known-good baseline")
    parser.add_argument("baseline", help="Baseline commit hash (known-good)")
    parser.add_argument("--head", default="HEAD", help="Head revision to compare (default: HEAD)")
    parser.add_argument("--files", nargs="*", help="Limit audit to specific files (default: all hot-path files)")
    parser.add_argument("--runtime", action="store_true", help="Run runtime microbenchmarks (requires builds)")
    parser.add_argument("--cuda", action="store_true", help="Run CUDA-specific regression passes (requires GPU)")
    parser.add_argument("--cuda-bench", default=None, help="Path to cuda_bench binary (default: tools/regression/cuda_bench)")
    parser.add_argument("--build-dir-baseline", default=None, help="Path to baseline build dir (for config.h)")
    parser.add_argument("--build-dir-head", default=None, help="Path to HEAD build dir (for config.h)")
    args = parser.parse_args()

    report = {"baseline": args.baseline, "head": args.head, "warnings": [], "critical": [], "passes": {}}

    print(f"Baseline: {args.baseline}  Head: {args.head}")

    from tools.regression.pass_flags import run_flags_pass
    report["passes"]["flags"] = run_flags_pass(args.baseline, args.head,
                                                args.build_dir_baseline, args.build_dir_head,
                                                report["critical"], report["warnings"])

    from tools.regression.pass_compiler import run_compiler_pass
    report["passes"]["compiler"] = run_compiler_pass(args.build_dir_baseline, args.build_dir_head,
                                                      report["critical"], report["warnings"])

    from tools.regression.pass_code import run_code_pass
    report["passes"]["code"] = run_code_pass(args.baseline, args.head, args.files,
                                              report["critical"], report["warnings"])

    if args.runtime:
        from tools.regression.pass_runtime import run_runtime_pass
        report["passes"]["runtime"] = run_runtime_pass(args.baseline, args.head,
                                                        report["critical"], report["warnings"])
    else:
        report["passes"]["runtime"] = None

    # CUDA optional passes
    if args.cuda:
        cuda_passes = _discover_cuda_passes()
        if not cuda_passes:
            print("  No CUDA passes discovered (pass_cuda_*.py not found)")
        for modname, (mod, run_fn) in cuda_passes.items():
            pass_name = modname.replace("pass_", "")
            print(f"  Running CUDA pass: {pass_name}")
            sig = inspect.signature(run_fn)
            extra_kwargs = {}
            if "binary_path" in sig.parameters:
                bench_path = args.cuda_bench or os.path.join(os.path.dirname(__file__), "cuda_bench")
                extra_kwargs["binary_path"] = bench_path
            report["passes"][pass_name] = run_fn(
                args.baseline, args.head,
                report["critical"], report["warnings"],
                **extra_kwargs)
    else:
        # Mark CUDA passes as skipped (won't render)
        for modname, (mod, run_fn) in _discover_cuda_passes().items():
            report["passes"][modname.replace("pass_", "")] = None

    html_out = render_html(report)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.html").write_text(html_out, encoding="utf-8")
    print(f"\nReport: {REPORT_DIR / 'report.html'}")

    if report["critical"]:
        print(f"\nCRITICAL: {len(report['critical'])} issues")
    if report["warnings"]:
        print(f"\nWARNINGS: {len(report['warnings'])} issues")
    return 0 if not report["critical"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `python -m pytest tools/regression/tests/ -v`
Expected: all existing tests still PASS

- [ ] **Step 4: Run --cuda flag test**

Run: `python -m pytest tools/regression/tests/test_audit.py::test_audit_cuda_flag_imports_passes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/regression/audit.py tools/regression/tests/test_audit.py
git commit -m "refactor: audit.py — dynamic import of pass_cuda_*.py via --cuda flag"
```

---

### Task 7: Build cuda_bench and run end-to-end on EPYC

**Files:**
- Modify: `tools/regression/cuda_bench.cu` (already created in Task 2)

- [ ] **Step 1: Compile cuda_bench on EPYC**

```bash
cd /home/.../claude_llama.cpp
nvcc -O3 -o tools/regression/cuda_bench tools/regression/cuda_bench.cu
./tools/regression/cuda_bench
```

Expected: JSON output with GPU name, compute capability, and H2D bandwidth in GB/s

- [ ] **Step 2: Run full CUDA audit**

```bash
python tools/regression/audit.py HEAD --cuda
```

Expected: Report generated at `tools/regression/report/report.html` with CUDA section showing bandwidth results

- [ ] **Step 3: Verify against Nvidia GPUs**

- If EPYC has Blackwell (RTX 5090, sm_120a): expected ≥ 55 GB/s H2D on PCIe Gen5×16
- If EPYC has Ampere (RTX 3090, sm_86): expected ≥ 28 GB/s H2D on PCIe Gen4×16
- If bandwidth is below 50% of theoretical → critical finding in report
- If bandwidth is 50-80% → warning finding

- [ ] **Step 4: Commit**

```bash
git add tools/regression/cuda_bench
git commit -m "chore: add prebuilt cuda_bench binary"
```
