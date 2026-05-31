# GGML CPU/CUDA Regression Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable `tools/regression/audit.py` that compares `master` against a known-good baseline commit across 4 passes (flags, compiler, code diff, runtime) to identify merged-out optimizations and misconfigured build flags.

**Architecture:** Single CLI entry point dispatches 4 independent pass modules. Each pass writes structured findings to a shared report dict. A flat HTML renderer produces a browsable report.

**Tech Stack:** Python 3.11+ (stdlib only — no pip deps), `subprocess` for git/cmake/compiler calls, `json` for compile_commands.json parsing, `html` for report.

---

## File Structure

```
tools/regression/
├── audit.py                # CLI: parse args, orchestrate passes, render report
├── pass_flags.py           # Build flag extraction, config.h propagation, NUMA check
├── pass_compiler.py        # Compiler version + effective feature macros
├── pass_code.py            # Structured git diff with function-level classification
├── pass_runtime.py         # Optional: build baseline + HEAD, run bench, compare
├── report/                 # Output (gitignored)
│   └── report.html         # Flat HTML, no JS
└── tests/
    ├── fixtures/
    │   ├── CMakeLists.txt.flags        # Minimal CMakeLists.txt with known options
    │   ├── ggml-config.h.in.avx512     # config template with AVX512
    │   ├── ggml-config.h.in.nonuma     # config template without NUMA
    │   ├── diff.q4_dot.txt             # Synthetic diff with vec_dot function change
    │   └── diff.empty.txt              # Empty diff
    ├── test_pass_flags.py
    ├── test_pass_compiler.py
    ├── test_pass_code.py
    └── test_audit.py
```

---

### Task 1: Scaffold CLI entry point + report renderer

**Files:**
- Create: `tools/regression/audit.py`
- Create: `tools/regression/__init__.py`
- Create: `tools/regression/tests/__init__.py`
- Test: `tools/regression/tests/test_audit.py`

- [ ] **Step 1: Write the failing test**

```python
import subprocess, sys

def test_cli_help():
    result = subprocess.run([sys.executable, "tools/regression/audit.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()

def test_cli_missing_baseline():
    result = subprocess.run([sys.executable, "tools/regression/audit.py"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "baseline" in result.stderr.lower()

def test_report_html_structure():
    """Verify the HTML renderer produces valid output."""
    from tools.regression.audit import render_html
    report = {
        "warnings": [],
        "critical": [],
        "passes": {
            "flags": [{"flag": "GGML_AVX512", "baseline": "ON", "head": "OFF", "severity": "critical"}],
            "compiler": "info: AOCC 5.2.0 at both revisions",
            "code": "0 critical diffs, 2 warnings",
            "runtime": None,
        },
    }
    html = render_html(report)
    assert "<html>" in html
    assert "GGML_AVX512" in html
    assert "critical" in html.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tools/regression/tests/test_audit.py::test_cli_help -v
```

Expected: ModuleNotFoundError / ImportError

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""CLI entry point for GGML regression audit.

Usage:
    python tools/regression/audit.py <baseline_commit> [--head HEAD] [--files FILE...] [--runtime]
"""

import argparse
import html
import os
import sys
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
        lines.append(f"<h2>Pass: {html.escape(name)}</h2>")
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
                    lines.append(f"<tr{cls}><td>{html.escape(str(row.get('flag','')))})"
                                 f"<td>{html.escape(str(row.get('baseline','')))})"
                                 f"<td>{html.escape(str(row.get('head','')))})"
                                 f"<td>{sev}</td></tr>")
                lines.append("</table>")
        else:
            lines.append(f"<pre>{html.escape(str(data))}</pre>")

    lines.append(f"<hr><p>Generated by ggml regression audit</p></body></html>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="GGML regression audit against a known-good baseline")
    parser.add_argument("baseline", help="Baseline commit hash (known-good)")
    parser.add_argument("--head", default="HEAD", help="Head revision to compare (default: HEAD)")
    parser.add_argument("--files", nargs="*", help="Limit audit to specific files (default: all hot-path files)")
    parser.add_argument("--runtime", action="store_true", help="Run runtime microbenchmarks (requires builds)")
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

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tools/regression/tests/test_audit.py -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/regression/
git commit -m "feat(audit): scaffold CLI, HTML renderer, and test harness"
```

---

### Task 2: Pass 1 — Build flags, config.h propagation, NUMA check

**Files:**
- Create: `tools/regression/pass_flags.py`
- Test: `tools/regression/tests/test_pass_flags.py`
- Create: `tools/regression/tests/fixtures/CMakeLists.txt.flags`
- Create: `tools/regression/tests/fixtures/ggml-config.h.in.avx512`
- Create: `tools/regression/tests/fixtures/ggml-config.h.in.nonuma`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from tools.regression.pass_flags import (
    parse_cmake_options,
    check_config_h_propagation,
    check_numa,
    run_flags_pass,
)

# --- parse_cmake_options ---

def test_parse_cmake_options_finds_on():
    cmake_text = """
option(GGML_AVX512 "enable AVX-512" ON)
option(GGML_FMA "enable FMA" ON)
"""
    result = parse_cmake_options(cmake_text)
    assert result["GGML_AVX512"] == "ON"
    assert result["GGML_FMA"] == "ON"

def test_parse_cmake_options_finds_off():
    cmake_text = """
option(GGML_NUMA "enable NUMA" OFF)
"""
    result = parse_cmake_options(cmake_text)
    assert result["GGML_NUMA"] == "OFF"

def test_parse_cmake_options_missing_option():
    cmake_text = "# no options here"
    result = parse_cmake_options(cmake_text)
    assert result == {}

# --- check_config_h_propagation ---

def test_config_h_avx512_present():
    config_h = "#define GGML_AVX512 1\n/* #undef GGML_NUMA */"
    result = check_config_h_propagation(config_h, ["GGML_AVX512", "GGML_NUMA"])
    assert result["GGML_AVX512"] == "found"
    assert result["GGML_NUMA"] == "not found"

def test_config_h_template_fallback():
    """No generated config.h — use template."""
    template = "// AVX\n#cmakedefine GGML_AVX512\n#cmakedefine GGML_FMA\n"
    result = check_config_h_propagation(template, ["GGML_AVX512"])
    # From template we can only verify the #cmakedefine line exists
    assert "GGML_AVX512" in result  # line found in template

# --- check_numa ---

def test_numa_off_critical():
    """NUMA OFF on what looks like a multi-socket system should be critical."""
    warnings, critical = [], []
    check_numa("OFF", # cmake default
               "#define GGML_NUMA 0",  # config.h
               False, # ggml_numa_init called?
               warnings, critical)
    assert len(critical) > 0
    assert "NUMA" in critical[0]

def test_numa_on_ok():
    warnings, critical = [], []
    check_numa("ON",  # cmake
               "#define GGML_NUMA 1",  # config.h
               True,  # ggml_numa_init called
               warnings, critical)
    assert len(critical) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tools/regression/tests/test_pass_flags.py -v
```

Expected: ModuleNotFoundError / ImportError

- [ ] **Step 3: Write minimal implementation**

```python
"""Pass 1: Build flags, config.h propagation, NUMA check."""

import re
import subprocess
import tempfile
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


# Main entry
def run_flags_pass(baseline: str, head: str,
                   build_dir_baseline: str = None,
                   build_dir_head: str = None,
                   critical: list = None, warnings: list = None) -> list:
    if critical is None:
        critical = []
    if warnings is None:
        warnings = []
    results = []

    # 1. Parse CMake options from both revisions
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
                target.append(f"{flag}: {bv} → {hv} in {cmake_file}")
        # If flag exists in both but unchanged, still report it
        for flag in sorted(base_flags.keys() & head_flags.keys()):
            if base_flags[flag] == head_flags[flag]:
                results.append({"file": cmake_file, "flag": flag, "baseline": base_flags[flag],
                                "head": head_flags[flag], "severity": "info"})

    # 2. config.h propagation check (if build dirs available)
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
                # Fall back to template
                template_path = Path("ggml/include/ggml-config.h.in")
                if template_path.exists():
                    tmpl_text = template_path.read_text()
                    prop = check_config_h_propagation(tmpl_text, list(CRITICAL_FLAGS))
                    for flag, status in prop.items():
                        if status not in ("template_line_present",):
                            warnings.append(f"{flag}: #cmakedefine missing from ggml-config.h.in")

    # 3. NUMA check
    numa_cmake = parse_cmake_options(_git_show(head, "ggml/CMakeLists.txt")).get("GGML_NUMA", "NOT FOUND")
    numa_config_h = {"GGML_NUMA": "not found"}
    if build_dir_head and Path(build_dir_head).exists():
        config_h = Path(build_dir_head) / "ggml-config.h"
        if config_h.exists():
            numa_config_h = check_config_h_propagation(config_h.read_text(), ["GGML_NUMA"])
    # Check ggml_numa_init call site in source
    ggml_cpu_c = _git_show(head, "ggml/src/ggml-cpu/ggml-cpu.c")
    numa_init_found = "ggml_numa_init" in ggml_cpu_c
    check_numa(numa_cmake, numa_config_h, numa_init_found, warnings, critical)
    results.append({"file": "ggml/CMakeLists.txt", "flag": "GGML_NUMA",
                    "baseline": "", "head": numa_cmake,
                    "severity": "critical" if numa_cmake != "ON" else "info"})

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tools/regression/tests/test_pass_flags.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tools/regression/pass_flags.py tools/regression/tests/test_pass_flags.py tools/regression/tests/fixtures/
git commit -m "feat(audit): Pass 1 — build flags, config.h, NUMA check"
```

---

### Task 3: Pass 2 — Compiler environment diagnostic

**Files:**
- Create: `tools/regression/pass_compiler.py`
- Test: `tools/regression/tests/test_pass_compiler.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from tools.regression.pass_compiler import (
    get_compiler_version,
    get_effective_feature_macros,
    run_compiler_pass,
)

def test_get_compiler_version_found():
    result = get_compiler_version()
    assert "cc" in result or "clang" in result or "gcc" in result or "error" not in result.lower()

def test_get_effective_macros_avx512():
    """Simulate a cc that supports AVX-512."""
    # This is a smoke test — on a real system it will vary
    result = get_effective_feature_macros()
    # At minimum expect __AVX512F__ key if system supports it
    assert isinstance(result, dict)

def test_run_compiler_pass_critical_on_switch():
    """If compiler family changed, mark critical."""
    warnings, critical = [], []
    run_compiler_pass(None, None, critical, warnings,
                      fake_baseline="AOCC 5.2.0", fake_head="GCC 14.2.1")
    assert len(critical) > 0
    assert "COMPILER CHANGED" in critical[0]

def test_run_compiler_pass_info_on_same():
    warnings, critical = [], []
    run_compiler_pass(None, None, critical, warnings,
                      fake_baseline="AOCC 5.2.0", fake_head="AOCC 5.2.0")
    assert len(critical) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tools/regression/tests/test_pass_compiler.py -v
```

Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

```python
"""Pass: Compiler environment diagnostic."""

import re
import subprocess
import json
from pathlib import Path


def get_compiler_version(compiler: str = "cc") -> str:
    """Return first 3 lines of `compiler --version`."""
    try:
        r = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            lines = r.stdout.strip().split("\n")[:3]
            return "\n".join(lines)
        return f"error: {r.stderr.strip()}"
    except FileNotFoundError:
        return f"not found: {compiler}"
    except subprocess.TimeoutExpired:
        return "timeout"


def get_effective_feature_macros(compiler: str = "cc") -> dict:
    """Compile an empty file with -mavx512f -E -dM and extract AVX512/FMA macros."""
    macros = {}
    try:
        r = subprocess.run(
            [compiler, "-mavx512f", "-E", "-dM", "-"],
            input="#include <cstdint>\n", capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            m = re.match(r'#define\s+(\w+)\s+(\S+)', line)
            if m and m.group(1) in ("__AVX512F__", "__AVX512CD__", "__AVX512BW__",
                                     "__AVX512DQ__", "__AVX512VL__", "__AVX512VNNI__",
                                     "__AVX512BF16__", "__FMA__", "__AVX2__", "__AVX__",
                                     "__SSE4_2__", "__SSE__"):
                macros[m.group(1)] = m.group(2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return macros


def _extract_compiler_family(version_str: str) -> str:
    v = version_str.lower()
    if "aocc" in v or "amd" in v and "clang" in v:
        return "AOCC"
    if "clang" in v:
        return "Clang"
    if "gcc" in v or "g++" in v or "gnu" in v:
        return "GCC"
    if "msvc" in v:
        return "MSVC"
    return "unknown"


def run_compiler_pass(build_dir_baseline: str = None,
                       build_dir_head: str = None,
                       critical: list = None, warnings: list = None,
                       fake_baseline: str = None, fake_head: str = None) -> str:
    """Check compiler at both revisions."""
    if critical is None:
        critical = []
    if warnings is None:
        warnings = []

    def _resolve_compiler(build_dir: str) -> str:
        """Try compile_commands.json first, then PATH."""
        if build_dir and Path(build_dir).exists():
            cc_json = Path(build_dir) / "compile_commands.json"
            if cc_json.exists():
                try:
                    cmds = json.loads(cc_json.read_text())
                    if cmds:
                        cmd = cmds[0].get("command", "")
                        m = re.search(r'(/[\w/]+(?:cc|c\+\+|gcc|g\+\+|clang)\S*)', cmd)
                        if m:
                            return m.group(1)
                except (json.JSONDecodeError, IndexError):
                    pass
        return "cc"  # fallback to PATH

    if fake_baseline:
        base_ver = fake_baseline
        base_comp = fake_baseline.split()[0]
    else:
        cc_path = _resolve_compiler(build_dir_baseline) if build_dir_baseline else "cc"
        base_ver = get_compiler_version(cc_path)
        base_comp = _extract_compiler_family(base_ver)

    if fake_head:
        head_ver = fake_head
        head_comp = fake_head.split()[0]
    else:
        cc_path = _resolve_compiler(build_dir_head) if build_dir_head else "cc"
        head_ver = get_compiler_version(cc_path)
        head_comp = _extract_compiler_family(head_ver)

    lines = [f"Baseline compiler ({base_comp}):", base_ver,
             f"", f"HEAD compiler ({head_comp}):", head_ver]

    if base_comp != head_comp:
        msg = f"COMPILER CHANGED: {base_comp} → {head_comp}"
        lines.append(f"\n⚠ {msg}")
        critical.append(msg)
    elif base_ver.split('\n')[0] != head_ver.split('\n')[0]:
        msg = f"Compiler version changed within {base_comp} family"
        lines.append(f"\n⚠ {msg}")
        warnings.append(msg)

    # Effective feature macros
    macros = get_effective_feature_macros()
    if macros:
        lines.append(f"\nEffective AVX512 feature macros (HEAD):")
        for k, v in sorted(macros.items()):
            lines.append(f"  {k} = {v}")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tools/regression/tests/test_pass_compiler.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tools/regression/pass_compiler.py tools/regression/tests/test_pass_compiler.py
git commit -m "feat(audit): Pass 2 — compiler environment diagnostic"
```

---

### Task 4: Pass 3 — Structured code diff classifier

**Files:**
- Create: `tools/regression/pass_code.py`
- Test: `tools/regression/tests/test_pass_code.py`
- Create: `tools/regression/tests/fixtures/diff.q4_dot.txt`
- Create: `tools/regression/tests/fixtures/diff.empty.txt`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from tools.regression.pass_code import (
    classify_diff,
    CRITICAL_FUNCTIONS,
    run_code_pass,
)

@pytest.fixture
def diff_with_critical_change():
    return """--- a/ops.cpp
+++ b/ops.cpp
@@ -100,5 +100,7 @@
 static void ggml_compute_forward_mul_mat(...) {
     const int64_t ne00 = src0->ne[0];
-    for (int64_t i = 0; i < ne00; i++) { vec_dot(...) }
+    for (int64_t i = 0; i < ne01; i++) { vec_dot(...) }
 }
"""

@pytest.fixture
def diff_empty():
    return ""

@pytest.fixture
def diff_only_whitespace():
    return """--- a/ops.cpp
+++ b/ops.cpp
@@ -1,3 +1,3 @@
-// old comment
+// new comment
"""

def test_classify_diff_critical(diff_with_critical_change):
    classified = classify_diff("ops.cpp", diff_with_critical_change)
    critical = [c for c in classified if c["severity"] == "critical"]
    assert len(critical) >= 1
    assert "ggml_compute_forward_mul_mat" in critical[0]["function"]

def test_classify_diff_empty(diff_empty):
    assert classify_diff("ops.cpp", diff_empty) == []

def test_classify_diff_whitespace_only(diff_only_whitespace):
    assert classify_diff("ops.cpp", diff_only_whitespace) == []

def test_classify_diff_unknown_file():
    classified = classify_diff("unknown.cpp", "some diff")
    # Unknown file → warning severity
    assert len(classified) >= 1
    assert classified[0]["severity"] == "warning"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tools/regression/tests/test_pass_code.py -v
```

Expected: ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
"""Pass 3: Structured code diff with function-level classification."""

import re
import subprocess


# Files to scan with their function-name patterns
FILES_CONFIG = {
    "ops.cpp": {
        "CRITICAL": [
            "ggml_compute_forward_mul_mat",
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

DEFAULT_FILES = list(FILES_CONFIG.keys()) + [
    "ggml/src/ggml-cuda/ggml-cuda-epyc-pipeline.cu",
    "ggml/src/ggml-cuda/CMakeLists.txt",
    "ggml/src/ggml-cpu/CMakeLists.txt",
    "ggml/CMakeLists.txt",
    "ggml/src/ggml-cpu/vec.cpp",
    "ggml/src/ggml-cpu/ggml-cpu-impl.h",
]


def _get_git_diff(hash_a: str, hash_b: str, filepath: str) -> str:
    """Return unified diff of `filepath` between two revisions."""
    try:
        r = subprocess.run(
            ["git", "diff", hash_a, hash_b, "--", filepath],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return r.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def classify_diff(filename: str, diff_text: str) -> list:
    """Classify each changed hunk in diff_text by its enclosing function + severity.

    Returns list of dicts: {"function": str, "severity": str, "hunk": str}
    """
    if not diff_text.strip():
        return []

    results = []
    # Determine severity from file config
    config = FILES_CONFIG.get(filename, {})
    critical_funcs = config.get("CRITICAL", [])
    warn_funcs = config.get("WARN", [])
    is_cuda = "cuda" in filename

    # Parse hunks: each hunk has @@ -a,b +c,d @@ and may have a function context
    hunks = re.split(r'(?=^@@)', diff_text, flags=re.MULTILINE)
    for hunk in hunks:
        if not hunk.strip():
            continue
        # Try to extract function name from hunk header
        fn_match = re.search(r'@@[^@]*@@\s*(\w+)', hunk)
        func_name = fn_match.group(1) if fn_match else "(unknown)"
        # Count non-whitespace changed lines
        changed_lines = [l for l in hunk.splitlines() if l.startswith('-') or l.startswith('+')]
        non_ws_changes = [l for l in changed_lines if not re.match(r'^[+-]\s*$', l)]
        if not non_ws_changes:
            continue  # whitespace-only hunk → skip

        # Check severity
        sev = "info"
        for cf in critical_funcs:
            if cf in func_name:
                sev = "critical"
                break
        if sev == "info":
            for wf in warn_funcs:
                if wf in func_name:
                    sev = "warning"
                    break
        if sev == "info" and is_cuda:
            for pattern in CUDA_CRITICAL_PATTERNS:
                if pattern.lower() in hunk.lower():
                    sev = "critical"
                    break

        if sev != "info" or True:  # always report
            results.append({
                "function": func_name,
                "severity": sev,
                "hunk": hunk.strip(),
            })

    # If file not in config at all, flag as warning
    if not config and not is_cuda and diff_text.strip():
        results.append({
            "function": "(whole file)",
            "severity": "warning",
            "hunk": f"Unclassified file: {filename} — no function-level rules defined",
        })

    return results


def run_code_pass(baseline: str, head: str,
                  files_filter: list = None,
                  critical: list = None, warnings: list = None) -> list:
    """Run structured code diff across all hot-path files."""
    if critical is None:
        critical = []
    if warnings is None:
        warnings = []

    files_to_check = files_filter if files_filter else DEFAULT_FILES
    all_classified = []

    for fpath in files_to_check:
        diff = _get_git_diff(baseline, head, fpath)
        if not diff.strip():
            continue
        classified = classify_diff(fpath, diff)
        for item in classified:
            sev = item["severity"]
            entry = {"file": fpath, "function": item["function"], "severity": sev, "hunk": item["hunk"]}
            all_classified.append(entry)
            if sev == "critical":
                critical.append(f"{fpath}:{item['function']} — critical change")
            elif sev == "warning":
                warnings.append(f"{fpath}:{item['function']} — warning change")

    return all_classified
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tools/regression/tests/test_pass_code.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tools/regression/pass_code.py tools/regression/tests/test_pass_code.py
git commit -m "feat(audit): Pass 3 — structured code diff classifier"
```

---

### Task 5: Pass 4 — Runtime microbenchmark runner (optional)

**Files:**
- Create: `tools/regression/pass_runtime.py`
- Test: `tools/regression/tests/test_pass_runtime.py`

- [ ] **Step 1: Write failing tests

```python
import pytest
from tools.regression.pass_runtime import (
    build_revision,
    run_bench,
)

@pytest.fixture
def fake_bench_output():
    return """op=mul_mat,avg=1.234ms,n=100
op=get_rows,avg=0.567ms,n=100
"""

def test_run_bench_parse(fake_bench_output, tmp_path):
    bench = tmp_path / "bench"
    bench.write_text("#!/bin/sh\necho 'op=mul_mat,avg=1.234ms'")
    bench.chmod(0o755)
    result = run_bench(str(bench), repeat=10)
    assert isinstance(result, dict)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tools/regression/tests/test_pass_runtime.py -v
```

Expected: ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
"""Pass 4: Optional runtime microbenchmark.

Builds both revisions, runs bench, compares latency per op.
"""

import subprocess
import tempfile
import re
from pathlib import Path


def build_revision(commit: str, cmake_flags: list[str] = None, build_dir: str = None) -> str:
    """Checkout commit, cmake + build bench target, return path to bench binary."""
    if not build_dir:
        build_dir = tempfile.mkdtemp(prefix=f"audit-{commit[:8]}-")
    build_dir = Path(build_dir)
    src_dir = build_dir / "src"

    # Shallow checkout
    subprocess.run(["git", "worktree", "add", str(src_dir), commit], check=True, capture_output=True)
    try:
        # Configure
        cmake_cmd = ["cmake", "-B", str(build_dir / "b"), "-S", str(src_dir),
                     "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF",
                     "-DLLAMA_BUILD_BENCH=ON"]
        if cmake_flags:
            cmake_cmd.extend(cmake_flags)
        subprocess.run(cmake_cmd, check=True, capture_output=True)

        # Build bench
        subprocess.run(["cmake", "--build", str(build_dir / "b"), "--target", "bench"],
                       check=True, capture_output=True)
        bench_path = build_dir / "b" / "bin" / "bench"
        if bench_path.exists():
            return str(bench_path)
        return ""
    finally:
        subprocess.run(["git", "worktree", "remove", str(src_dir)], capture_output=True)


def run_bench(bench_path: str, repeat: int = 100, ops: str = "mul_mat,get_rows,rms_norm") -> dict:
    """Run bench binary, parse output, return {op: avg_ms}."""
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
    """Build both, bench, compare."""
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
            marker = " ⚠" if abs(diff_pct) > 5 else ""
            lines.append(f"{op:20s} baseline={base_ms:8.3f}ms  head={head_ms:8.3f}ms  "
                         f"Δ={diff_pct:+7.2f}%{marker}")
            if abs(diff_pct) > 10:
                critical.append(f"{op}: {diff_pct:+.1f}% regression (baseline={base_ms:.3f}ms, head={head_ms:.3f}ms)")
            elif abs(diff_pct) > 5:
                warnings.append(f"{op}: {diff_pct:+.1f}% change")
        elif base_ms:
            lines.append(f"{op:20s} baseline={base_ms:8.3f}ms  head=NOT FOUND")
        elif head_ms:
            lines.append(f"{op:20s} baseline=NOT FOUND  head={head_ms:8.3f}ms")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tools/regression/tests/test_pass_runtime.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tools/regression/pass_runtime.py tools/regression/tests/test_pass_runtime.py
git commit -m "feat(audit): Pass 4 — runtime microbenchmark runner (optional)"
```
