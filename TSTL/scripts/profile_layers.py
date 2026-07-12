"""Run TSTL layer-contribution profile on MLP regression benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlp_bench import BenchConfig, default_out_dir, run_benchmark, save_benchmark_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="TSTL layer contribution profile (MLP)")
    parser.add_argument("--layers", type=int, default=5, help="Number of hidden Linear layers")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mid-k", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Short run for smoke (steps=100)")
    args = parser.parse_args()

    steps = 100 if args.quick else args.steps
    cfg = BenchConfig(
        num_hidden_layers=args.layers,
        steps=steps,
        seed=args.seed,
        lr=args.lr,
        mid_k=args.mid_k,
        top_k=args.top_k,
    )

    print(f"=== TSTL profile (hidden={args.layers}, steps={steps}, seed={args.seed}) ===")
    result = run_benchmark(cfg)
    out_dir = save_benchmark_results(cfg, result, default_out_dir())

    print(f"S_base={result.s_base:.4f} S_full={result.s_full:.4f}")
    print(f"best layer k={result.best_layer} C={result.contributions[result.best_layer]:.3f}")
    for name, score in result.strategies.items():
        print(f"  {name}: R2={score:.4f}")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
