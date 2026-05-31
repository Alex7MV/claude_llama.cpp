"""Pass: CUDA H2D bandwidth verification via cuda_bench binary."""

import json
import os
import subprocess


def compute_pcie_limit(compute_capability: str) -> float | None:
    """Return expected PCIe bandwidth limit (GB/s) based on compute capability.

    Gen5 (Blackwell 12.x): 63 GB/s theoretical
    Gen4 (Ampere 8.x, Ada 8.9): 31.5 GB/s
    Gen3 (Volta 7.x, Turing 7.5): 15.8 GB/s
    """
    try:
        major, minor = compute_capability.split(".")
        sm = int(major) * 10 + int(minor)
    except (ValueError, AttributeError):
        return None
    if sm >= 120:
        return 63.0  # Blackwell: PCIe Gen5 x16
    elif sm >= 80:
        return 31.5  # Ampere/Ada: PCIe Gen4 x16
    elif sm >= 70:
        return 15.8  # Volta/Turing: PCIe Gen3 x16
    return None


def load_cuda_bench_json(path: str) -> dict | None:
    """Run cuda_bench binary and parse JSON output."""
    if not os.path.isfile(path):
        return None
    try:
        result = subprocess.run([path], capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        FileNotFoundError,
    ):
        return None


def check_h2d_bandwidth(data: dict) -> list:
    """Check H2D bandwidth against expected PCIe limit."""
    findings = []
    bw = data.get("h2d_bandwidth_gbs", 0.0)
    cc = data.get("compute_capability", "0.0")
    limit = compute_pcie_limit(cc)

    if limit is None:
        findings.append(
            {
                "flag": "h2d_bandwidth_pcie_limit",
                "baseline": "unknown",
                "head": f"{bw:.1f} GB/s",
                "severity": "info",
                "detail": f"Unknown PCIe generation for CC {cc}",
            }
        )
        return findings

    pct = bw / limit * 100.0
    if bw < limit * 0.5:
        findings.append(
            {
                "flag": "h2d_bandwidth",
                "baseline": f">={limit * 0.5:.0f} GB/s",
                "head": f"{bw:.1f} GB/s ({pct:.0f}% of {limit:.0f} GB/s limit)",
                "severity": "critical",
                "detail": f"H2D bandwidth {bw:.1f} GB/s is below 50% of PCIe {limit:.0f} GB/s limit",
            }
        )
    elif bw < limit * 0.8:
        findings.append(
            {
                "flag": "h2d_bandwidth",
                "baseline": f">={limit * 0.8:.0f} GB/s",
                "head": f"{bw:.1f} GB/s ({pct:.0f}% of {limit:.0f} GB/s limit)",
                "severity": "warning",
                "detail": f"H2D bandwidth {bw:.1f} GB/s is below 80% of PCIe {limit:.0f} GB/s limit",
            }
        )
    else:
        findings.append(
            {
                "flag": "h2d_bandwidth",
                "baseline": f">={limit * 0.5:.0f} GB/s",
                "head": f"{bw:.1f} GB/s ({pct:.0f}% of {limit:.0f} GB/s limit)",
                "severity": "pass",
                "detail": f"H2D bandwidth {bw:.1f} GB/s meets PCIe {limit:.0f} GB/s limit",
            }
        )

    return findings


def run_cuda_memory_pass(
    baseline: str,
    head: str,
    critical: list,
    warnings: list,
    binary_path: str | None = None,
) -> list:
    """Run the CUDA memory pass: execute cuda_bench and check H2D bandwidth.

    Args:
        baseline: baseline commit hash (unused, for interface compatibility)
        head: head commit hash (unused, for interface compatibility)
        critical: list to append critical findings
        warnings: list to append warnings
        binary_path: path to cuda_bench binary (default: cuda_bench in same directory)
    """
    findings = []
    if binary_path is None:
        binary_path = os.path.join(os.path.dirname(__file__), "cuda_bench")
    data = load_cuda_bench_json(binary_path)
    if data is None:
        findings.append(
            {
                "flag": "cuda_bench",
                "severity": "info",
                "detail": f"cuda_bench binary not found at {binary_path} or failed to execute",
            }
        )
        return findings

    # Record device info
    findings.append(
        {
            "flag": "gpu_info",
            "baseline": f"{data.get('gpu_name', '?')} CC {data.get('compute_capability', '?')}",
            "head": f"{data.get('gpu_name', '?')} CC {data.get('compute_capability', '?')}",
            "severity": "info",
        }
    )

    # Check H2D bandwidth
    bw_findings = check_h2d_bandwidth(data)
    findings.extend(bw_findings)

    for f in bw_findings:
        if f.get("severity") == "critical":
            critical.append(f"CUDA H2D bandwidth: {f['detail']}")
        elif f.get("severity") == "warning":
            warnings.append(f"CUDA H2D bandwidth: {f['detail']}")

    return findings
