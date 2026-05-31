# Phase 4: CUDA Kernel & Memory Audit

**Date:** 2026-05-31
**Branch:** `fix/ggml-cpu-cuda`
**Baseline commit:** `5429d89f8f31e3fee14c834a68af3a0ebe56a3d6`

## Goal

Extend the existing audit infrastructure (`tools/regression/audit.py`) with CUDA-specific regression detection. After any upstream merge, verify that:

1. CUDA compilation flags and target architectures are preserved
2. Host-to-Device memory bandwidth meets PCIe bus limits
3. MMVQ register-prefetch optimizations (early 2026) are intact
4. Peer-to-Peer GPU access topology is correctly detected

## Motivation

Upstream merges routinely touch `ggml-cuda.cu`, `mmvq.cu`, and `CMakeLists.txt` but lack automated regression checks for:

- CUDA arch flags dropping (e.g., `sm_90a` → `sm_90`, losing WGMMA)
- `-use_fast_math` or `-compress-mode` being silently removed
- `cudaHostAlloc` vs regular `malloc` for pinned memory
- Register-prefetch loop structure in MMVQ kernels
- P2P access path selection for multi-GPU topology

A 1-trillion-parameter model (like DeepSeek V4 Pro) is bottlenecked by H2D bandwidth on PCIe Gen5 — any regression in memory transfer path has outsized impact on end-to-end latency.

## Architecture

### Plugin loading (`audit.py`)

- Add `--cuda` flag (boolean, default `False`)
- When set, scan `tools/regression/` for `pass_cuda_*.py` modules
- Dynamic `importlib.import_module()` each discovered module
- Each CUDA pass exposes `run(baseline: str, head: str, findings: list) → None`
- Findings appended to `report["cuda"]` — a new top-level key in the report dict
- HTML renderer iterates `report["cuda"]` and renders a dedicated section after CPU passes

### CUDA benchmark binary (`tools/regression/cuda_bench.cu`)

Standalone `.cu` file compiled with `nvcc`. Outputs JSON to stdout:

```json
{
  "gpu_name": "NVIDIA RTX 5090",
  "compute_capability": "12.1",
  "num_devices": 1,
  "h2d_bandwidth_gbs": 58.2,
  "p2p_enabled": false,
  "p2p_bandwidth_gbs": null
}
```

Compiled once by the audit script if missing, or prebuilt. Keeps CUDA compilation out of Python.

### Error handling

- If `nvcc` not found or `cuda_bench` fails to compile → all CUDA passes emit `severity: info` with `"CUDA runtime not available — skipping"`
- If `cuda_bench` binary exists but crashes → `pass_cuda_memory.py` catches subprocess error, emits `severity: error`
- Partial results still render — a failed H2D benchmark doesn't block flags or MMVQ analysis

## Components

### `pass_cuda_flags.py`

**Input:** `git show <rev>:ggml/src/ggml-cuda/CMakeLists.txt` at baseline and head

**Checks (all vs baseline):**

| Check | What it compares |
|-------|-----------------|
| `CMAKE_CUDA_ARCHITECTURES` | List of sm_XX targets (e.g., `75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real 121a-real 100a-real 101a-real`) |
| `-use_fast_math` | Present in `target_compile_options` |
| `-extended-lambda` | Present in CUDA flags |
| `-compress-mode` | Present (CUDA ≥ 12.8) |
| `GGML_CUDA_HAS_WGMMA` | `try_compile` result preserved |
| `GGML_CUDA_FA` | Option ON/OFF unchanged |

Each check produces a finding with `severity: critical|warning|info`. Any mismatch is at least `warning`.

### `pass_cuda_memory.py`

**Requires:** `cuda_bench` binary (compiled from `cuda_bench.cu`)

**Procedure:**

1. Check if `cuda_bench` exists at `tools/regression/cuda_bench`
2. If not, attempt `nvcc -O3 -o tools/regression/cuda_bench tools/regression/cuda_bench.cu`
3. Run `cuda_bench` → parse JSON
4. Compare `h2d_bandwidth_gbs` against expected PCIe limit:
   - Gen5×16: ≥ 63 GB/s theoretical, ≥ 55 GB/s realistic
   - Gen4×16: ≥ 32 GB/s theoretical, ≥ 28 GB/s realistic
   - Gen3×16: ≥ 16 GB/s theoretical, ≥ 13 GB/s realistic
5. If bandwidth < 50% of theoretical → `critical` finding
   - If bandwidth < 80% but ≥ 50% → `warning`
   - Otherwise → `pass`

**Benchmark protocol** (in `cuda_bench.cu`):
- Allocate `cudaHostAlloc` pinned buffer (1 GB)
- Allocate `cudaMalloc` device buffer (1 GB)
- Warmup: 3 iterations of H2D copy × 4 sizes (1 MB, 10 MB, 100 MB, 1000 MB)
- Timed: 10 iterations per size, using `cudaEvent` timing
- Report best-of-10 bandwidth for largest size that fits in VRAM
- If VRAM insufficient for 1 GB, scale down gracefully and report max tested size

### `pass_cuda_mmvq.py`

**Input:** `git diff <baseline>..<head> -- ggml/src/ggml-cuda/mmvq.cu`

**Analysis:**

- Parse diff hunks
- Classify each hunk by pattern:

| Pattern | Severity | Meaning |
|---------|----------|---------|
| `#pragma unroll` removal | `critical` | Register prefetch loop unrolling lost |
| Register variable → `__shared__` | `warning` | Prefetch target changed from register to shared memory |
| Loop structure change (tile loop, inner reduction) | `warning` | Potential prefetch pipeline change |
| Data type change (e.g., half → float) | `info` | Precision change, may affect throughput |
| New kernel variant added | `info` | Expected change, note only |

- Apply same hunk classification logic as `pass_code.py` (shared `_classify_hunk` helper if possible, otherwise duplicate pattern matching)
- Emit findings with function name context

### `pass_cuda_p2p.py` (reserved)

Detected when `cuda_bench --p2p` returns `num_devices > 1`.

- Calls `cuda_bench --p2p` which probes:
  - `cudaDeviceCanAccessPeer(&can, i, j)` for all device pairs
  - `cudaDeviceEnablePeerAccess` for each pair
  - H2D and D2D bandwidth between peers
- Reports P2P topology matrix and per-pair bandwidth
- If P2P was previously enabled but now disabled → `critical`
- Not implemented in initial pass — stubs return `severity: info` with `"P2P not configured"`

## Data Flow

```
audit.py [--cuda]

  pass_flags.py ────────────────────────────────┐
  pass_compiler.py                                │
  pass_code.py                                    ├──→ report dict
  pass_runtime.py                                 │    {
  pass_cuda_flags.py ── git show / diff ──────────┤      "flags": [...],
  pass_cuda_memory.py ── subprocess(cuda_bench) ──┤      "compiler": [...],
  pass_cuda_mmvq.py ──── git diff mmvq.cu ────────┤      "code": [...],
  pass_cuda_p2p.py ───── subprocess(cuda_bench) ──┤      "runtime": [...],
                                                 │      "cuda": {
                                                 │        "flags": [...],
                                                 │        "memory": [...],
                                                 │        "mmvq": [...],
                                                 │        "p2p": [...]
                                                 │      }
                                                 │    }
                                                 ↓
                                        render_html(report)
                                                 ↓
                                        tools/regression/report/summary.html
```

## File Changes

### New files

| File | Purpose |
|------|---------|
| `tools/regression/cuda_bench.cu` | Standalone CUDA bandwidth & P2P benchmark, outputs JSON |
| `tools/regression/pass_cuda_flags.py` | CUDA CMake flags diff pass |
| `tools/regression/pass_cuda_memory.py` | H2D bandwidth verification pass |
| `tools/regression/pass_cuda_mmvq.py` | MMVQ register-prefetch diff pass |

### Modified files

| File | Change |
|------|--------|
| `tools/regression/audit.py` | Add `--cuda` flag, dynamic import of `pass_cuda_*.py`, extend report dict, pass CUDA flag to sub-passes |
| `tools/regression/pass_flags.py` | Add `ggml/src/ggml-cuda/CMakeLists.txt` to `CMAKE_FILES` when `--cuda` active |
| `tools/regression/tests/` | Add test modules for new passes |
| `tools/regression/tests/fixtures/` | Add `mmvq.cu.*` diff fixtures, `cuda_bench.json` fixture |

## Testing

### Unit tests

| Test module | What it tests |
|------------|---------------|
| `tests/test_pass_cuda_flags.py` | Parse CUDA CMakeLists.txt with various arch lists; detect missing `-use_fast_math` |
| `tests/test_pass_cuda_memory.py` | Parse `cuda_bench` JSON output; verify bandwidth classification thresholds |
| `tests/test_pass_cuda_mmvq.py` | Classify mmvq.cu diff hunks by prefetch pattern; detect `#pragma unroll` removal |

### Fixtures

| Fixture | Content |
|---------|---------|
| `CMakeLists.txt.cuda.full` | Full CUDA CMakeLists.txt with all flags and archs |
| `CMakeLists.txt.cuda.stripped` | Same but missing `-use_fast_math` and `sm_90a` (regression scenario) |
| `cuda_bench.good.json` | JSON with H2D = 58 GB/s (Gen5 expected) |
| `cuda_bench.slow.json` | JSON with H2D = 12 GB/s (Gen3-level, should warn) |
| `diff.mmvq.prefetch.txt` | Diff that removes `#pragma unroll` from mmvq.cu |
| `diff.mmvq.clean.txt` | Diff with only additive changes (no regression) |

### Manual testing

```bash
# Full audit with CUDA
python tools/regression/audit.py <baseline> --cuda

# Standalone CUDA benchmark
nvcc -O3 -o tools/regression/cuda_bench tools/regression/cuda_bench.cu
./tools/regression/cuda_bench
./tools/regression/cuda_bench --p2p
```

## Non-Goals

- GPU kernel correctness testing (covered by `test-backend-ops.cpp` with `-b CUDA`)
- Multi-GPU NCCL topology mapping (too system-specific, changes are expected)
- CUDA graph capture verification (GGML_CUDA_GRAPHS is experimental)
- Flash attention numerical validation (separate test infrastructure)
- Blackwell WGMMA correctness (covered by `ggml-cuda-blackwell-pipeline.cu` tests)
