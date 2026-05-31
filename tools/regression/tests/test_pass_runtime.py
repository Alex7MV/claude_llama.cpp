import pytest
from tools.regression.pass_runtime import run_bench


@pytest.fixture
def fake_bench_output(tmp_path):
    bench = tmp_path / "bench"
    bench.write_text("#!/bin/sh\necho 'op=mul_mat,avg=1.234ms,n=100'\necho 'op=get_rows,avg=0.567ms,n=100'")
    bench.chmod(0o755)
    return str(bench)


def test_run_bench_parse(fake_bench_output):
    result = run_bench(fake_bench_output, repeat=10)
    assert isinstance(result, dict)
    assert "mul_mat" in result
    assert result["mul_mat"] == 1.234
    assert "get_rows" in result
    assert result["get_rows"] == 0.567
