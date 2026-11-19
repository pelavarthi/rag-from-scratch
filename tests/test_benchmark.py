"""Tests for the pipeline benchmark script."""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "benchmark", SCRIPTS / "benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark()


def test_percentile():
    values = [float(i) for i in range(1, 11)]
    assert benchmark.percentile(values, 50) == 5.0
    assert benchmark.percentile(values, 95) == 10.0
    assert benchmark.percentile(values, 0) == 1.0
    assert benchmark.percentile([], 50) == 0.0


def test_format_bytes():
    assert benchmark.format_bytes(512) == "512 B"
    assert benchmark.format_bytes(2048) == "2.0 KB"
    assert benchmark.format_bytes(3 * 1024 * 1024) == "3.0 MB"


def test_run_benchmark_on_sample_data(tmp_path):
    result = benchmark.run_benchmark(
        "data/sample", "data/sample/eval.json", k=5, dim=256
    )
    assert result.chunks > 0
    assert result.dim == 256
    assert len(result.query_seconds) == 6  # six golden questions
    assert result.p95_ms >= result.p50_ms >= 0
    assert 0 <= result.mean_recall_at_k <= 1
    assert result.index_bytes > 0
    text = result.format()
    assert "query p50" in text and "index size" in text


@pytest.mark.parametrize("path", ["benchmark.py"])
def test_script_has_no_syntax_errors(path):
    source = (SCRIPTS / path).read_text(encoding="utf-8")
    compile(source, path, "exec")
