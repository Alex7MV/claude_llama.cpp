"""Pass 3: Structured code diff with function-level classification."""

import re
import subprocess


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
    try:
        r = subprocess.run(
            ["git", "diff", hash_a, hash_b, "--", filepath],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return r.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def classify_diff(filename: str, diff_text: str) -> list:
    if not diff_text.strip():
        return []

    results = []
    config = FILES_CONFIG.get(filename, {})
    critical_funcs = config.get("CRITICAL", [])
    warn_funcs = config.get("WARN", [])
    is_cuda = "cuda" in filename

    hunks = re.split(r'(?=^@@)', diff_text, flags=re.MULTILINE)
    for hunk in hunks:
        if not hunk.strip():
            continue
        fn_match = re.search(r'@@[^@]*@@\s*(\w+)', hunk)
        func_name = fn_match.group(1) if fn_match else "(unknown)"
        changed_lines = [l for l in hunk.splitlines() if l.startswith('-') or l.startswith('+')]
        non_ws_changes = [l for l in changed_lines if not re.match(r'^[+-]\s*$', l)]
        if not non_ws_changes:
            continue

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

        results.append({
            "function": func_name,
            "severity": sev,
            "hunk": hunk.strip(),
        })

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
