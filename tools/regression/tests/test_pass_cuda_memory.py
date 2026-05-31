import json
import os
from tools.regression.pass_cuda_memory import (
    load_cuda_bench_json,
    check_h2d_bandwidth,
    compute_pcie_limit,
    run_cuda_memory_pass,
)

FIXTURE_DIR = "tools/regression/tests/fixtures"


def test_load_good_json():
    data = json.load(open(f"{FIXTURE_DIR}/cuda_bench.good.json"))
    assert data["h2d_bandwidth_gbs"] == 58.2
    assert data["num_devices"] == 1


def test_load_slow_json():
    data = json.load(open(f"{FIXTURE_DIR}/cuda_bench.slow.json"))
    assert data["h2d_bandwidth_gbs"] == 12.3


def test_compute_pcie_gen5():
    limit = compute_pcie_limit("12.1")
    assert limit == 63.0


def test_compute_pcie_gen4():
    limit = compute_pcie_limit("8.9")
    assert limit == 31.5


def test_compute_pcie_gen3():
    limit = compute_pcie_limit("7.0")
    assert limit == 15.8


def test_compute_pcie_unknown():
    limit = compute_pcie_limit("0.0")
    assert limit is None


def test_check_h2d_good():
    data = {"h2d_bandwidth_gbs": 58.2, "compute_capability": "12.1"}
    findings = check_h2d_bandwidth(data)
    num_critical = sum(1 for f in findings if f.get("severity") == "critical")
    num_warning = sum(1 for f in findings if f.get("severity") == "warning")
    assert num_critical == 0
    assert num_warning == 0


def test_check_h2d_slow():
    data = {"h2d_bandwidth_gbs": 12.3, "compute_capability": "12.1"}
    findings = check_h2d_bandwidth(data)
    num_critical = sum(1 for f in findings if f.get("severity") == "critical")
    assert num_critical >= 1


def test_check_h2d_mid():
    data = {"h2d_bandwidth_gbs": 45.0, "compute_capability": "12.1"}
    findings = check_h2d_bandwidth(data)
    num_critical = sum(1 for f in findings if f.get("severity") == "critical")
    num_warning = sum(1 for f in findings if f.get("severity") == "warning")
    assert num_critical == 0
    assert num_warning >= 1


def test_run_pass_no_binary():
    critical, warnings = [], []
    findings = run_cuda_memory_pass(
        "HEAD", "HEAD", critical, warnings, binary_path="nonexistent_binary"
    )
    assert len(findings) > 0
    assert any("not found" in str(f.get("detail", "")) for f in findings)
