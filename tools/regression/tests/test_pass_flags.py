import pytest
from pathlib import Path
from tools.regression.pass_flags import (
    parse_cmake_options,
    check_config_h_propagation,
    check_numa,
)


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


def test_parse_cmake_options_empty():
    cmake_text = "# no options here"
    result = parse_cmake_options(cmake_text)
    assert result == {}


def test_config_h_avx512_present():
    config_h = "#define GGML_AVX512 1\n/* #undef GGML_NUMA */"
    result = check_config_h_propagation(config_h, ["GGML_AVX512", "GGML_NUMA"])
    assert result["GGML_AVX512"] == "found"
    assert result["GGML_NUMA"] == "not found"


def test_config_h_template_present():
    template = "#cmakedefine GGML_AVX512\n#cmakedefine GGML_FMA\n"
    result = check_config_h_propagation(template, ["GGML_AVX512", "GGML_FMA"])
    assert result["GGML_AVX512"] == "template_line_present"
    assert result["GGML_FMA"] == "template_line_present"


def test_numa_off_critical():
    warnings, critical = [], []
    check_numa("OFF", {"GGML_NUMA": "found"}, True, warnings, critical)
    assert len(critical) > 0
    assert "NUMA" in critical[0]


def test_numa_in_config_h_missing_critical():
    warnings, critical = [], []
    check_numa("ON", {"GGML_NUMA": "not found"}, True, warnings, critical)
    assert len(critical) > 0
    assert "NUMA" in critical[0]


def test_numa_init_missing_critical():
    warnings, critical = [], []
    check_numa("ON", {"GGML_NUMA": "found"}, False, warnings, critical)
    assert len(critical) > 0
    assert "NUMA" in critical[0]


def test_numa_all_ok():
    warnings, critical = [], []
    check_numa("ON", {"GGML_NUMA": "found"}, True, warnings, critical)
    assert len(critical) == 0
