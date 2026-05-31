import pytest
from tools.regression.pass_code import (
    classify_diff,
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
    classified = classify_diff("unknown.cpp", "some diff content @@ -1 +1 @@ func()")
    assert len(classified) >= 1
    assert classified[0]["severity"] == "warning"


def test_classify_diff_cuda():
    cuda_diff = """--- a/ggml-cuda-epyc-pipeline.cu
+++ b/ggml-cuda-epyc-pipeline.cu
@@ -50,5 +50,7 @@
     cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
-    cudaStreamSynchronize(stream);
+    // sync removed
 }
"""
    classified = classify_diff("ggml-cuda-epyc-pipeline.cu", cuda_diff)
    critical = [c for c in classified if c["severity"] == "critical"]
    assert len(critical) >= 1
