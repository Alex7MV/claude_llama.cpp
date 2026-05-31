from tools.regression.pass_cuda_mmvq import (
    classify_mmvq_hunks,
    run_cuda_mmvq_pass,
)

FIXTURE_DIR = "tools/regression/tests/fixtures"


def test_classify_prefetch_removal_critical():
    with open(f"{FIXTURE_DIR}/diff.mmvq.prefetch.txt") as f:
        diff_text = f.read()
    findings = classify_mmvq_hunks(diff_text, "diff.mmvq.prefetch.txt")
    critical = [f for f in findings if f.get("severity") == "critical"]
    assert len(critical) >= 1
    assert any("unroll" in c.get("detail", "").lower() for c in critical)


def test_classify_prefetch_removal_v4_load():
    with open(f"{FIXTURE_DIR}/diff.mmvq.prefetch.txt") as f:
        diff_text = f.read()
    findings = classify_mmvq_hunks(diff_text, "diff.mmvq.prefetch.txt")
    warning = [f for f in findings if f.get("severity") == "warning"]
    v4_findings = [f for f in warning if "v4_load" in f.get("detail", "")]
    assert len(v4_findings) >= 1


def test_classify_clean_diff_no_issues():
    with open(f"{FIXTURE_DIR}/diff.mmvq.clean.txt") as f:
        diff_text = f.read()
    findings = classify_mmvq_hunks(diff_text, "diff.mmvq.clean.txt")
    critical = [f for f in findings if f.get("severity") == "critical"]
    assert len(critical) == 0


def test_classify_empty_diff():
    findings = classify_mmvq_hunks("", "empty.diff")
    assert len(findings) == 0


def test_run_pass_no_diff():
    critical, warnings = [], []
    findings = run_cuda_mmvq_pass("HEAD", "HEAD", critical, warnings)
    assert len(findings) == 0
