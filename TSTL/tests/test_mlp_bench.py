"""Smoke test for MLP benchmark pipeline."""

import math

from mlp_bench import BenchConfig, run_benchmark, save_benchmark_results


def test_mlp_bench_smoke(tmp_path):
    cfg = BenchConfig(
        num_hidden_layers=3,
        steps=50,
        n_train=64,
        n_test=32,
        seed=0,
    )
    result = run_benchmark(cfg)
    assert len(result.contributions) == 3
    assert "full" in result.strategies
    assert all(math.isfinite(v) for v in result.contributions.values())

    out = save_benchmark_results(cfg, result, tmp_path / "run")
    assert (out / "contributions.json").is_file()
    assert (out / "contribution.png").is_file()
    assert (out / "summary.md").is_file()
