import pytest
from tools.regression.pass_compiler import (
    get_compiler_version,
    get_effective_feature_macros,
    run_compiler_pass,
    _extract_compiler_family,
)


def test_extract_compiler_family():
    assert _extract_compiler_family("AOCC 5.2.0") == "AOCC"
    assert _extract_compiler_family("AMD clang version") == "AOCC"
    assert _extract_compiler_family("clang version 18") == "Clang"
    assert _extract_compiler_family("gcc (GCC) 14.2.1") == "GCC"
    assert _extract_compiler_family("g++ (GCC) 14.2.1") == "GCC"
    assert _extract_compiler_family("MSVC 19.40") == "MSVC"
    assert _extract_compiler_family("unknown compiler") == "unknown"


def test_run_compiler_pass_critical_on_switch():
    warnings, critical = [], []
    run_compiler_pass(None, None, critical, warnings,
                      fake_baseline="AOCC 5.2.0", fake_head="GCC 14.2.1")
    assert len(critical) > 0
    assert "COMPILER CHANGED" in critical[0]


def test_run_compiler_pass_warning_on_version_change():
    warnings, critical = [], []
    run_compiler_pass(None, None, critical, warnings,
                      fake_baseline="AOCC 5.2.0", fake_head="AOCC 5.3.0")
    assert len(critical) == 0
    assert len(warnings) > 0


def test_run_compiler_pass_info_on_same():
    warnings, critical = [], []
    run_compiler_pass(None, None, critical, warnings,
                      fake_baseline="AOCC 5.2.0", fake_head="AOCC 5.2.0")
    assert len(critical) == 0
    assert len(warnings) == 0
