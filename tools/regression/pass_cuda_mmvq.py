"""Pass: MMVQ kernel diff analysis — detect register-prefetch regression."""

import re
import subprocess


def classify_mmvq_hunks(diff_text: str, filename: str) -> list:
    """Analyze diff hunks in mmvq.cu for register-prefetch regressions."""
    findings = []

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


def _analyze_hunk(
    hunk_lines: list, hunk_id: int, findings: list, filename: str
):
    """Analyze a single diff hunk for known regression patterns."""
    removed_lines = [l[1:] for l in hunk_lines if l.startswith("-")]
    added_lines = [l[1:] for l in hunk_lines if l.startswith("+")]
    full_removed = "\n".join(removed_lines)
    full_added = "\n".join(added_lines)

    # Pattern 1: #pragma unroll removed
    if "#pragma unroll" in full_removed and "#pragma unroll" not in full_added:
        findings.append(
            {
                "flag": f"{filename}:hunk_{hunk_id}",
                "severity": "critical",
                "detail": "#pragma unroll removed from loop — register prefetch may be lost",
            }
        )

    # Pattern 2: v4_load → scalar fallback
    if "v4_load" in full_removed and "v4_load" not in full_added:
        findings.append(
            {
                "flag": f"{filename}:hunk_{hunk_id}",
                "severity": "warning",
                "detail": "v4_load vectorized load replaced — register prefetch degraded to scalar",
            }
        )

    # Pattern 3: float4 → float (register packing lost)
    if re.search(r"float4\b", full_removed) and not re.search(
        r"float4\b", full_added
    ):
        findings.append(
            {
                "flag": f"{filename}:hunk_{hunk_id}",
                "severity": "warning",
                "detail": "float4 register packing removed — may reduce memory throughput",
            }
        )

    # Pattern 4: Loop structure change
    removed_loops = [l for l in removed_lines if "for" in l]
    added_loops = [l for l in added_lines if "for" in l]
    if removed_loops and added_loops and len(removed_loops) != len(added_loops):
        findings.append(
            {
                "flag": f"{filename}:hunk_{hunk_id}",
                "severity": "info",
                "detail": "Loop structure changed — verify tile prefetch is preserved",
            }
        )

    # Pattern 5: __shared__ memory introduced
    if "__shared__" in full_added and "__shared__" not in full_removed:
        findings.append(
            {
                "flag": f"{filename}:hunk_{hunk_id}",
                "severity": "warning",
                "detail": "__shared__ memory introduced — prefetch target may have changed from register to shared",
            }
        )


def run_cuda_mmvq_pass(
    baseline: str, head: str, critical: list, warnings: list
) -> list:
    """Run the CUDA MMVQ pass: diff mmvq.cu and check for prefetch regression."""
    findings = []
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                f"{baseline}..{head}",
                "--",
                "ggml/src/ggml-cuda/mmvq.cu",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        findings.append(
            {
                "flag": "mmvq_diff",
                "severity": "error",
                "detail": "git diff failed for mmvq.cu",
            }
        )
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
