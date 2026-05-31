import subprocess
import sys


def test_cli_help():
    result = subprocess.run([sys.executable, "tools/regression/audit.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_cli_missing_baseline():
    result = subprocess.run([sys.executable, "tools/regression/audit.py"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "baseline" in result.stderr.lower()


def test_report_html_structure():
    from tools.regression.audit import render_html
    report = {
        "warnings": [],
        "critical": [],
        "passes": {
            "flags": [{"flag": "GGML_AVX512", "baseline": "ON", "head": "OFF", "severity": "critical"}],
            "compiler": "info: AOCC 5.2.0 at both revisions",
            "code": "0 critical diffs, 2 warnings",
            "runtime": None,
        },
    }
    html = render_html(report)
    assert "<html>" in html
    assert "GGML_AVX512" in html
    assert "critical" in html.lower()
