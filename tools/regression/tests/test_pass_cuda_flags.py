import pytest
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


def test_check_cuda_architectures_missing_latest():
    archs = "75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real"
    problems = check_cuda_architectures(archs)
    # sm_121a, sm_100a, sm_101a expected but missing
    assert len(problems) == 3
    for p in problems:
        assert p["severity"] == "warning"


def test_cmp_cuda_flags_all_match():
    full = open(f"{FIXTURE_DIR}/CMakeLists.txt.cuda.full").read()
    findings = cmp_cuda_flags(full, full)
    mismatch_flags = [f for f in findings if f.get("status", "") == "mismatch"]
    # Also check no critical/warnings from the other fields
    critical_or_warning = [
        f for f in findings if f.get("severity") in ("critical", "warning")
    ]
    assert len(critical_or_warning) == 0


def test_cmp_cuda_flags_detects_stripped():
    full = open(f"{FIXTURE_DIR}/CMakeLists.txt.cuda.full").read()
    stripped = open(f"{FIXTURE_DIR}/CMakeLists.txt.cuda.stripped").read()
    findings = cmp_cuda_flags(full, stripped)
    # Should detect: GGML_CUDA_FA (ON→OFF), missing sm_120a/sm_121a/sm_100a/sm_101a,
    # missing -use_fast_math, missing -compress-mode
    # Use severity field (not status)
    all_findings = [f for f in findings]
    assert len(all_findings) >= 6
    # Check that critical findings exist for missing critical flags
    critical_flags = [f for f in findings if f.get("severity") == "critical"]
    assert len(critical_flags) >= 1
    # Check that warning findings exist for missing archs
    warning_flags = [f for f in findings if f.get("severity") == "warning"]
    assert len(warning_flags) >= 4
