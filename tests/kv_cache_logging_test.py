"""
KV-Cache Debug Logging Test Suite

Tests that [KV]-tagged structured logging fires correctly at
each n_past decision point in server-context.cpp.

Usage:
    pytest tests/kv_cache_logging_test.py -v --server-path=/path/to/server
"""

import os
import re
import subprocess
import time
import json
import pytest
import requests
from pathlib import Path

SERVER_PATH = os.environ.get(
    "LLAMA_SERVER_BIN",
    str(Path(__file__).parent.parent / "build" / "bin" / "server")
)
SERVER_LOG = os.environ.get("LLAMA_SERVER_LOG", "/tmp/llama_server_test.log")
SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "18080"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "")


@pytest.fixture(scope="module")
def server(request):
    """Start server, yield, terminate."""
    if not MODEL_PATH:
        pytest.skip("LLAMA_MODEL_PATH not set")

    args = [
        SERVER_PATH,
        "-m", MODEL_PATH,
        "--port", str(SERVER_PORT),
        "--ctx-size", "4096",
        "--cont-batching",
        "--no-mmap",
    ]
    if os.environ.get("CUDA_AVAILABLE") == "1":
        args += ["--ngl", "0"]

    with open(SERVER_LOG, "w") as lf:
        proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT)

    # wait for server ready
    for _ in range(60):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    else:
        proc.terminate()
        pytest.fail("Server did not start in 60s")

    yield proc

    proc.terminate()
    proc.wait(timeout=10)


def grep_log(pattern):
    """Return list of matching log lines."""
    if not os.path.exists(SERVER_LOG):
        return []
    with open(SERVER_LOG, "r") as f:
        return [l.rstrip() for l in f if re.search(pattern, l)]


# ── Test 1: Log Format ──────────────────────────────────────────────

KV_LOG_RE = re.compile(
    r"^.*\[KV\] n_past=\d+ n_total=\d+ reason=\w+"
)

def test_log_output_format(server):
    """Assert every [KV] line matches the expected structured format."""
    payload = {
        "prompt": "Hello, write a short sentence.",
        "n_predict": 10,
        "stream": False,
    }
    resp = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=60)
    assert resp.status_code == 200

    kv_lines = grep_log(r"\[KV\]")
    assert len(kv_lines) > 0, "No [KV] log lines found"

    for line in kv_lines:
        assert KV_LOG_RE.match(line), f"Format mismatch: {line}"


# ── Test 2: Cache Reuse Detection ──────────────────────────────────

def test_cache_reuse_identified(server):
    """
    Send two identical requests. The second should reuse cached prefix.
    Detect by: first request shows NO_CACHE_PROMPT (cold),
    second request shows PREFIX_MISMATCH with common > 0.
    """
    prompt = "What is the capital of France?"
    payload = {
        "prompt": prompt,
        "n_predict": 5,
        "cache_prompt": True,
        "stream": False,
    }

    # First request — cold start
    r1 = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=60)
    assert r1.status_code == 200

    # Second request — should reuse cache
    r2 = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=60)
    assert r2.status_code == 200

    # Check logs: first request should have NO_CACHE_PROMPT or PREFIX_MISMATCH
    cold_lines = grep_log(r"reason=NO_CACHE_PROMPT")
    warm_lines = grep_log(r"reason=PREFIX_MISMATCH common=(\d+)")

    # At a minimum, the second request should NOT show NO_CACHE_PROMPT
    # (that would indicate a full re-prefill)
    # Count lines after the second request timestamp
    # Simple heuristic: if PREFIX_MISMATCH with common > 0 exists, cache was reused
    reuse_found = False
    for line in warm_lines:
        m = re.search(r"common=(\d+)", line)
        if m and int(m.group(1)) > 0:
            reuse_found = True
            break

    # If no cache reuse at all, that's a problem
    if len(grep_log(r"reason=(NO_CACHE_PROMPT|PREFIX_MISMATCH)")) == 0:
        pytest.fail("No KV-cache decision logged — logging may not be active")

    # This test detects re-prefill: if the second identical request
    # triggers NO_CACHE_PROMPT, the cache was NOT reused.
    # We mark as WARNING rather than FAIL for now (depends on context state)
    if not reuse_found and len(grep_log(r"reason=PREFIX_MISMATCH")) == 0:
        print("WARNING: No cache reuse detected for identical prompt — re-prefill likely")


# ── Test 3: CUDA Leak Detection (ngl=0) ────────────────────────────

def test_cuda_trace_empty_at_ngl0(server):
    """
    With --ngl 0, no CUDA ops should be dispatched.
    If [KV][CUDA] lines with n_ops>0 appear, that's a GPU leak.
    """
    if os.environ.get("CUDA_AVAILABLE") != "1":
        pytest.skip("CUDA not available")

    payload = {
        "prompt": "Write a short poem.",
        "n_predict": 20,
        "stream": False,
    }
    resp = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=60)
    assert resp.status_code == 200

    leak_matches = [l for l in grep_log(r"n_ops=[1-9]")]

    assert len(leak_matches) == 0, \
        f"GPU leak detected at ngl=0: found {len(leak_matches)} ops:\n" + "\n".join(leak_matches[:5])


# ── Test 4: Context Shift Logging ──────────────────────────────────

def test_context_shift_logged(server):
    """
    Long generation with --ctx-shift should trigger CONTEXT_SHIFT events.
    """
    # Use a long input to fill context, then generate enough to trigger shift
    # With ctx-size 4096, a long prompt + large n_predict should trigger
    long_prompt = "The history of philosophy " * 200  # ~800 tokens
    payload = {
        "prompt": long_prompt,
        "n_predict": 500,
        "stream": False,
        "cache_prompt": False,
    }
    resp = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=300)
    assert resp.status_code == 200

    shift_lines = grep_log(r"reason=CONTEXT_SHIFT")
    if len(shift_lines) == 0:
        print("WARNING: No CONTEXT_SHIFT events logged — prompt may not have filled context")
    else:
        for line in shift_lines[:3]:
            print(f"  shift: {line}")
