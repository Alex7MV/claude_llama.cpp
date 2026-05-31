"""Pass 1: Build flags, variant compile definitions, NUMA runtime check."""

import re
import subprocess
from pathlib import Path

CMAKE_FILES = [
    "ggml/CMakeLists.txt",
    "ggml/src/ggml-cpu/CMakeLists.txt",
]
# Flags that must match between baseline and HEAD.
# Note: GGML_NUMA is not a cmake option in this codebase — it is runtime-detected.
CRITICAL_FLAGS = {"GGML_AVX512", "GGML_AVX512_VNNI", "GGML_AVX512_BF16",
                  "GGML_FMA", "zen4"}


def _git_show(hash: str, path: str) -> str:
    """Return file contents from git at given revision."""
    try:
        return subprocess.run(
            ["git", "show", f"{hash}:{path}"],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return ""


def check_config_h_propagation(config_h_text: str, expected_flags: list) -> dict:
    """Verify which expected flags made it into config.h (utility, not called by pass)."""
    result = {}
    for flag in expected_flags:
        if f"#cmakedefine {flag}" in config_h_text:
            result[flag] = "template_line_present"
        elif re.search(rf'#define\s+{flag}\s+[01]', config_h_text):
            result[flag] = "found"
        elif re.search(rf'/\*\s*#undef\s+{flag}\s*\*/', config_h_text):
            result[flag] = "not found"
        else:
            result[flag] = "not found"
    return result


def parse_cmake_options(text: str) -> dict:
    """Extract option() defaults and set(... CACHE ... FORCE) overrides."""
    flags = {}
    # option(name desc default) — always ON or OFF
    for match in re.finditer(r'^\s*option\s*\(\s*(\w+)\s+[^)]+\s+(ON|OFF)\s*\)', text, re.MULTILINE):
        flags[match.group(1)] = match.group(2)
    # set(name ON|OFF CACHE ... FORCE) — overrides option() defaults
    for match in re.finditer(r'^\s*set\s*\(\s*(\w+)\s+(ON|OFF)\s+CACHE\s', text, re.MULTILINE):
        flags[match.group(1)] = match.group(2)
    return flags


def check_numa(numa_init_called: bool, numa_available_called: bool,
               warnings: list, critical: list):
    """Validate NUMA runtime support (no cmake option — always compiled in)."""
    if not numa_init_called:
        critical.append("NUMA: ggml_numa_init() not found in ggml-cpu.c — thread affinity disabled")
    if not numa_available_called:
        warnings.append("NUMA: numa_available() not detected — runtime NUMA detection may be missing")


def run_flags_pass(baseline: str, head: str,
                   build_dir_baseline: str = None,
                   build_dir_head: str = None,
                   critical: list = None, warnings: list = None) -> list:
    if critical is None:
        critical = []
    if warnings is None:
        warnings = []
    results = []

    for cmake_file in CMAKE_FILES:
        base_text = _git_show(baseline, cmake_file)
        head_text = _git_show(head, cmake_file)
        base_flags = parse_cmake_options(base_text)
        head_flags = parse_cmake_options(head_text)

        all_flags = set(base_flags.keys()) | set(head_flags.keys())
        for flag in sorted(all_flags):
            bv = base_flags.get(flag, "NOT FOUND")
            hv = head_flags.get(flag, "NOT FOUND")
            if bv != hv:
                entry = {"file": cmake_file, "flag": flag, "baseline": bv, "head": hv,
                         "severity": "critical" if flag in CRITICAL_FLAGS else "warning"}
                results.append(entry)
                target = critical if entry["severity"] == "critical" else warnings
                target.append(f"{flag}: {bv} -> {hv} in {cmake_file}")
        for flag in sorted(base_flags.keys() & head_flags.keys()):
            if base_flags[flag] == head_flags[flag]:
                results.append({"file": cmake_file, "flag": flag, "baseline": base_flags[flag],
                                "head": head_flags[flag], "severity": "info"})

    ggml_cpu_c = _git_show(head, "ggml/src/ggml-cpu/ggml-cpu.c")
    numa_init_found = "ggml_numa_init" in ggml_cpu_c
    numa_available_found = "numa_available" in ggml_cpu_c
    check_numa(numa_init_found, numa_available_found, warnings, critical)

    return results
