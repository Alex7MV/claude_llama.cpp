"""Pass 4: Optional runtime microbenchmark.

Builds both revisions, runs bench, compares latency per op.
"""

import subprocess
import tempfile
import re
from pathlib import Path


def build_revision(commit: str, cmake_flags: list[str] = None, build_dir: str = None) -> str:
    if not build_dir:
        build_dir = tempfile.mkdtemp(prefix=f"audit-{commit[:8]}-")
    build_dir = Path(build_dir)
    src_dir = build_dir / "src"

    subprocess.run(["git", "worktree", "add", str(src_dir), commit], check=True, capture_output=True)
    try:
        cmake_cmd = ["cmake", "-B", str(build_dir / "b"), "-S", str(src_dir),
                     "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF",
                     "-DLLAMA_BUILD_BENCH=ON"]
        if cmake_flags:
            cmake_cmd.extend(cmake_flags)
        subprocess.run(cmake_cmd, check=True, capture_output=True)

        subprocess.run(["cmake", "--build", str(build_dir / "b"), "--target", "bench"],
                       check=True, capture_output=True)
        bench_path = build_dir / "b" / "bin" / "bench"
        if bench_path.exists():
            return str(bench_path)
        # Try alternative location (Linux)
        bench_path_alt = build_dir / "b" / "examples" / "bench" / "bench"
        if bench_path_alt.exists():
            return str(bench_path_alt)
        return ""
    finally:
        subprocess.run(["git", "worktree", "remove", str(src_dir)], capture_output=True)


def run_bench(bench_path: str, repeat: int = 100, ops: str = "mul_mat,get_rows,rms_norm") -> dict:
    try:
        r = subprocess.run(
            [bench_path, f"--ops={ops}", f"--repeat={repeat}"],
            capture_output=True, text=True, timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    results = {}
    for line in r.stdout.splitlines():
        m = re.match(r'op=(\w+),avg=([\d.]+)ms', line)
        if m:
            results[m.group(1)] = float(m.group(2))
    return results


def run_runtime_pass(baseline: str, head: str,
                     critical: list = None, warnings: list = None,
                     cmake_flags: list[str] = None) -> str:
    if critical is None:
        critical = []
    if warnings is None:
        warnings = []

    print("Building baseline bench...")
    baseline_bench = build_revision(baseline, cmake_flags)
    if not baseline_bench:
        return "FAILED: baseline build"

    print("Building HEAD bench...")
    head_bench = build_revision(head, cmake_flags)
    if not head_bench:
        return "FAILED: HEAD build"

    print("Running benchmarks...")
    base_results = run_bench(baseline_bench)
    head_results = run_bench(head_bench)

    lines = []
    for op in sorted(set(list(base_results.keys()) + list(head_results.keys()))):
        base_ms = base_results.get(op)
        head_ms = head_results.get(op)
        if base_ms and head_ms:
            diff_pct = ((head_ms - base_ms) / base_ms) * 100
            marker = " **" if abs(diff_pct) > 5 else ""
            lines.append(f"{op:20s} baseline={base_ms:8.3f}ms  head={head_ms:8.3f}ms  "
                         f"delta={diff_pct:+7.2f}%{marker}")
            if abs(diff_pct) > 10:
                critical.append(f"{op}: {diff_pct:+.1f}% regression (baseline={base_ms:.3f}ms, head={head_ms:.3f}ms)")
            elif abs(diff_pct) > 5:
                warnings.append(f"{op}: {diff_pct:+.1f}% change")
        elif base_ms:
            lines.append(f"{op:20s} baseline={base_ms:8.3f}ms  head=NOT FOUND")
        elif head_ms:
            lines.append(f"{op:20s} baseline=NOT FOUND  head={head_ms:8.3f}ms")

    return "\n".join(lines)
