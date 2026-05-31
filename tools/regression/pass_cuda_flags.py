"""Pass: CUDA CMake flags diff between baseline and head."""

import re
import subprocess


def parse_cuda_options(cmake_text: str) -> dict:
    """Parse option() calls from CUDA CMakeLists.txt content."""
    result = {}
    for m in re.finditer(
        r"^\s*option\s*\(\s*(\w+)\s+.*?\s+(ON|OFF)\s*\)",
        cmake_text,
        re.MULTILINE,
    ):
        result[m.group(1)] = m.group(2)
    return result


def check_cuda_architectures(arch_str: str) -> list:
    """Check that key CUDA architectures are present. Returns list of finding dicts."""
    expected = {"120a-real", "121a-real", "100a-real", "101a-real"}
    actual = set(arch_str.split())
    findings = []
    for arch in sorted(expected):
        if arch not in actual:
            findings.append(
                {
                    "flag": f"CMAKE_CUDA_ARCHITECTURES: {arch}",
                    "status": "missing",
                    "severity": "warning",
                }
            )
    return findings


def _has_compile_flag(text: str, flag: str) -> bool:
    """Check if target_compile_options contains the given flag."""
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
            findings.append(
                {
                    "flag": opt,
                    "baseline": b,
                    "head": h,
                    "severity": "warning" if h == "NOT SET" else "info",
                }
            )

    # Compare architectures
    basearch = re.search(
        r'set\s*\(\s*CMAKE_CUDA_ARCHITECTURES\s+"([^"]+)"', baseline_text
    )
    headarch = re.search(
        r'set\s*\(\s*CMAKE_CUDA_ARCHITECTURES\s+"([^"]+)"', head_text
    )
    if basearch and headarch and basearch.group(1) != headarch.group(1):
        findings.append(
            {
                "flag": "CMAKE_CUDA_ARCHITECTURES",
                "baseline": basearch.group(1),
                "head": headarch.group(1),
                "severity": "critical",
            }
        )
    if headarch:
        findings.extend(check_cuda_architectures(headarch.group(1)))

    # Compare compile flags
    for flag in ["-use_fast_math", "-compress-mode"]:
        b_has = _has_compile_flag(baseline_text, flag)
        h_has = _has_compile_flag(head_text, flag)
        if b_has and not h_has:
            findings.append(
                {
                    "flag": f"compile_flag: {flag}",
                    "baseline": "present",
                    "head": "missing",
                    "severity": "critical",
                }
            )

    # Check WGMMA
    b_wgmma = "GGML_CUDA_HAS_WGMMA" in baseline_text
    h_wgmma = "GGML_CUDA_HAS_WGMMA" in head_text
    if b_wgmma and not h_wgmma:
        findings.append(
            {
                "flag": "GGML_CUDA_HAS_WGMMA",
                "baseline": "present",
                "head": "missing",
                "severity": "critical",
            }
        )

    return findings


def run_cuda_flags_pass(
    baseline: str, head: str, critical: list, warnings: list
) -> list:
    """Run the CUDA flags pass: compare CUDA CMakeLists.txt flags."""
    findings = []
    try:
        baselinesh = subprocess.run(
            ["git", "show", f"{baseline}:ggml/src/ggml-cuda/CMakeLists.txt"],
            capture_output=True,
            text=True,
            check=True,
        )
        headsh = subprocess.run(
            ["git", "show", f"{head}:ggml/src/ggml-cuda/CMakeLists.txt"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        findings.append(
            {
                "flag": "git_show",
                "severity": "error",
                "baseline": "?",
                "head": "?",
                "detail": "could not read CMakeLists.txt at one or both revisions",
            }
        )
        return findings

    findings = cmp_cuda_flags(baselinesh.stdout, headsh.stdout)

    for f in findings:
        if f.get("severity") == "critical":
            critical.append(
                f"CUDA flags: {f['flag']} (baseline={f.get('baseline','?')}, head={f.get('head','?')})"
            )
        elif f.get("severity") == "warning":
            warnings.append(f"CUDA flags: {f['flag']}")

    return findings
