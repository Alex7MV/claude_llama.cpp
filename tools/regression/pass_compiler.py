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


TARGET_MACROS = {"__AVX512F__", "__AVX512CD__", "__AVX512BW__",
                 "__AVX512DQ__", "__AVX512VL__", "__AVX512VNNI__",
                 "__AVX512BF16__", "__FMA__", "__AVX2__", "__AVX__",
                 "__SSE4_2__", "__SSE__"}

FEATURE_FLAGS = {
    "generic": ["-mavx512f"],
    "znver4":  ["-mavx512f", "-mavx512vnni", "-mavx512bf16", "-mfma", "-mavx2"],
}


def get_effective_feature_macros(compiler: str = "cc",
                                  march: str = "generic") -> dict:
    """Compile an empty file with -E -dM and extract AVX512/FMA macros."""
    macros = {}
    flags = FEATURE_FLAGS.get(march, FEATURE_FLAGS["generic"])
    try:
        r = subprocess.run(
            [compiler] + flags + ["-E", "-dM", "-"],
            input="#include <cstdint>\n", capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            m = re.match(r'#define\s+(\w+)\s+(\S+)', line)
            if m and m.group(1) in TARGET_MACROS:
                macros[m.group(1)] = m.group(2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return macros


def _extract_compiler_family(version_str: str) -> str:
    v = version_str.lower()
    if "aocc" in v or ("amd" in v and "clang" in v):
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
    if critical is None:
        critical = []
    if warnings is None:
        warnings = []

    def _resolve_compiler(build_dir: str) -> str:
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
        return "cc"

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
             "", f"HEAD compiler ({head_comp}):", head_ver]

    if base_comp != head_comp:
        msg = f"COMPILER CHANGED: {base_comp} -> {head_comp}"
        lines.append(f"\n  {msg}")
        critical.append(msg)
    elif base_ver.split('\n')[0] != head_ver.split('\n')[0]:
        msg = f"Compiler version changed within {base_comp} family"
        lines.append(f"\n  {msg}")
        warnings.append(msg)

    head_march = "znver4" if head_comp == "AOCC" else "generic"
    head_cc = _resolve_compiler(build_dir_head) if build_dir_head else "cc"
    macros = get_effective_feature_macros(head_cc, head_march)
    if macros:
        lines.append(f"\nEffective feature macros ({head_comp} {head_march}):")
        for k, v in sorted(macros.items()):
            lines.append(f"  {k} = {v}")
        if "__AVX512VNNI__" not in macros:
            warnings.append("__AVX512VNNI__ not defined — Q8_0 VNNI kernel won't compile")

    return "\n".join(lines)
