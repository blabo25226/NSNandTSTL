"""
Depth-sweep study: verify the paper's observation that EML-tree training success
drops sharply for depth D >= 5.

For each (target, depth, seed) it trains a DNN-EML head of that depth and records
the soft / snapped holdout MSE, whether training stayed finite, and whether the
snapped closed form met the success threshold. Aggregated success rate per depth
quantifies the D >= 5 collapse.

Usage:
  python scripts/depth_sweep_eval.py --targets sin product --depths 2 3 4 5 6 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import DNNEML  # noqa: E402
from pipeline import OdrzywolekPipelineConfig, run_odrzywolek_pipeline  # noqa: E402
from snap import evaluate_snapped  # noqa: E402
from targets import DEFAULT_NOISE_STD_REL, TARGETS  # noqa: E402
from utils import set_seed  # noqa: E402

SUCCESS_THRESHOLD = 5e-2  # relaxed elementary-function threshold (matches sr_eval)


def run_one(target, depth: int, seed: int, steps: int, noise: float) -> dict:
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x, y = target.sample(256, gen, noise)

    finite = True
    soft_mse = float("nan")
    snapped_mse = float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # depth>4 recommendation warning
            model = DNNEML.build(
                input_dim=target.input_dim,
                feature_dim=6,
                head_depth=depth,
                hidden_dim=128,
                num_layers=3,
            )
        pipe_cfg = OdrzywolekPipelineConfig(total_steps=steps, lr=2e-3, seed=seed)
        run_odrzywolek_pipeline(model, x, y, pipe_cfg)

        gen_h = torch.Generator().manual_seed(seed + 1)
        x_h, y_h = target.sample(128, gen_h, noise)
        with torch.no_grad():
            pred = model(x_h)
            soft_mse = torch.mean((pred - y_h) ** 2).item()
        snapped_mse, _ = evaluate_snapped(model, x_h, y_h)
        finite = bool(torch.isfinite(pred).all().item()) and (soft_mse == soft_mse)
    except (RuntimeError, ValueError) as exc:
        finite = False
        soft_mse = float("inf")
        snapped_mse = float("inf")
        print(f"    depth={depth} seed={seed}: training failed ({exc})")

    success = finite and snapped_mse <= SUCCESS_THRESHOLD
    return {
        "target_id": target.id,
        "depth": depth,
        "seed": seed,
        "finite": finite,
        "soft_holdout_mse": soft_mse,
        "snapped_holdout_mse": snapped_mse,
        "success": success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth-sweep collapse study")
    parser.add_argument("--targets", nargs="+", default=["sin", "product"])
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--noise-std-rel", type=float, default=DEFAULT_NOISE_STD_REL)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"depth_sweep_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for tid in args.targets:
        target = TARGETS[tid]
        print(f"\n=== target={tid} ({target.formula}) ===")
        for depth in args.depths:
            for seed in args.seeds:
                row = run_one(target, depth, seed, args.steps, args.noise_std_rel)
                rows.append(row)
            depth_rows = [r for r in rows if r["target_id"] == tid and r["depth"] == depth]
            succ = sum(1 for r in depth_rows if r["success"])
            fin = sum(1 for r in depth_rows if r["finite"])
            print(f"  depth={depth}: success {succ}/{len(depth_rows)}  finite {fin}/{len(depth_rows)}")

    # Aggregate success rate per depth (across targets & seeds).
    per_depth: dict[int, dict] = {}
    for depth in args.depths:
        drows = [r for r in rows if r["depth"] == depth]
        n = len(drows)
        succ = sum(1 for r in drows if r["success"])
        fin = sum(1 for r in drows if r["finite"])
        finite_snapped = [r["snapped_holdout_mse"] for r in drows if r["finite"]]
        per_depth[depth] = {
            "runs": n,
            "success_rate": succ / n if n else float("nan"),
            "finite_rate": fin / n if n else float("nan"),
            "median_snapped_mse": (statistics.median(finite_snapped) if finite_snapped else float("nan")),
        }

    (out_dir / "summary.json").write_text(
        json.dumps({
            "depths": args.depths, "seeds": args.seeds, "steps": args.steps,
            "success_threshold": SUCCESS_THRESHOLD,
            "per_depth": {str(k): v for k, v in per_depth.items()},
            "results": rows,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# Depth-Sweep Study (D >= 5 collapse)",
        "",
        f"Success = snapped holdout MSE <= {SUCCESS_THRESHOLD:g}. Runs per depth: "
        f"{len(args.targets)} targets x {len(args.seeds)} seeds.",
        "",
        "| depth | success rate | finite rate | median snapped MSE |",
        "|-------|--------------|-------------|--------------------|",
    ]
    for depth in args.depths:
        d = per_depth[depth]
        md.append(
            f"| {depth} | {d['success_rate']:.2f} | {d['finite_rate']:.2f} | "
            f"{d['median_snapped_mse']:.2e} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n=== success rate by depth ===")
    for depth in args.depths:
        print(f"  D={depth}: {per_depth[depth]['success_rate']:.2f}")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
