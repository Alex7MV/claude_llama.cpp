"""Pass 1: Build flags, config.h propagation, NUMA check."""

import re
import subprocess
from pathlib import Path

CMAKE_FILES = [
    "ggml/CMakeLists.txt",
    "ggml/src/ggml-cpu/CMakeLists.txt",
]
CRITICAL_FLAGS = {"GGML_AVX512", "GGML_AVX512_VNNI", "GGML_AVX512_BF16",
                  "GGML_FMA", "GGML_NUMA", "zen4"}


def _git_show(hash: str, path: str) -> str:
    """Return file contents from git at given revision."""
    try:
        return subprocess.run(
            ["git", "show", f"{hash}:{path}"],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return ""


def parse_cmake_options(text: str) -> dict:
    """Extract option(name default_value) from CMakeLists.txt content."""
    flags = {}
    for match in re.finditer(r'^\s*option\s*\(\s*(\w+)\s+[^)]+\s+(ON|OFF)\s*\)', text, re.MULTILINE):
        flags[match.group(1)] = match.group(2)
    return flags


def check_config_h_propagation(config_h_text: str, expected_flags: list) -> dict:
    """Verify which expected flags made it into config.h."""
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


def check_numa(cmake_default: str, config_h_prop: dict,
               numa_init_called: bool, warnings: list, critical: list):
    """Validate NUMA configuration for multi-socket systems."""
    if cmake_default != "ON":
        critical.append("GGML_NUMA: CMake option default is OFF — must be ON for dual-socket EPYC")
    if config_h_prop.get("GGML_NUMA") != "found":
        critical.append("GGML_NUMA: missing from config.h — #cmakedefine propagation broken")
    if not numa_init_called:
        critical.append("GGML_NUMA: ggml_numa_init() not found in ggml-cpu.c — thread affinity disabled")


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

    for label, build_dir in [("baseline", build_dir_baseline), ("head", build_dir_head)]:
        if build_dir and Path(build_dir).exists():
            config_h = Path(build_dir) / "ggml-config.h"
            if config_h.exists():
                text = config_h.read_text()
                prop = check_config_h_propagation(text, list(CRITICAL_FLAGS))
                for flag, status in prop.items():
                    if status not in ("found", "template_line_present"):
                        critical.append(f"{flag}: NOT propagated at {label} revision (config.h: {status})")
                        results.append({"file": "ggml-config.h", "flag": flag, "baseline": "",
                                        "head": "", "severity": "critical"})
            else:
                template_path = Path("ggml/include/ggml-config.h.in")
                if template_path.exists():
                    tmpl_text = template_path.read_text()
                    prop = check_config_h_propagation(tmpl_text, list(CRITICAL_FLAGS))
                    for flag, status in prop.items():
                        if status not in ("template_line_present",):
                            warnings.append(f"{flag}: #cmakedefine missing from ggml-config.h.in")

    numa_cmake = parse_cmake_options(_git_show(head, "ggml/CMakeLists.txt")).get("GGML_NUMA", "NOT FOUND")
    numa_config_h = {"GGML_NUMA": "not found"}
    if build_dir_head and Path(build_dir_head).exists():
        config_h = Path(build_dir_head) / "ggml-config.h"
        if config_h.exists():
            numa_config_h = check_config_h_propagation(config_h.read_text(), ["GGML_NUMA"])
    ggml_cpu_c = _git_show(head, "ggml/src/ggml-cpu/ggml-cpu.c")
    numa_init_found = "ggml_numa_init" in ggml_cpu_c
    check_numa(numa_cmake, numa_config_h, numa_init_found, warnings, critical)
    results.append({"file": "ggml/CMakeLists.txt", "flag": "GGML_NUMA",
                    "baseline": "", "head": numa_cmake,
                    "severity": "critical" if numa_cmake != "ON" else "info"})

    return results
