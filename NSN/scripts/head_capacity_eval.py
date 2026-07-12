"""
Head-capacity study: does the EML head recover symbols, or does the MLP trunk
do all the fitting?

Compares configurations that progressively force the head to carry the load and
reports, for each, the soft holdout MSE vs the *snapped* (exported closed-form)
holdout MSE. A faithful symbolic recovery shows a small snapped/soft gap.

Configs:
  full        - baseline (large trunk, trunk trains throughout)
  small       - tiny trunk (feature_dim=2, hidden=16), so z has little capacity
  freeze      - trunk frozen after SEARCH; head must adapt during HARDEN/POLISH
  small+parent- tiny trunk with f_prev="parent" head recurrence

Usage:
  python scripts/head_capacity_eval.py --targets square sin --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


CONFIGS: dict[str, dict] = {
    "full": dict(feature_dim=6, hidden_dim=128, freeze=False, f_prev_mode="zero"),
    "small": dict(feature_dim=2, hidden_dim=16, freeze=False, f_prev_mode="zero"),
    "freeze": dict(feature_dim=6, hidden_dim=128, freeze=True, f_prev_mode="zero"),
    "small+parent": dict(feature_dim=2, hidden_dim=16, freeze=False, f_prev_mode="parent"),
}


def run_one(target, cfg_name: str, cfg: dict, seed: int, steps: int, noise: float) -> dict:
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x, y = target.sample(256, gen, noise)

    model = DNNEML.build(
        input_dim=target.input_dim,
        feature_dim=cfg["feature_dim"],
        head_depth=2,
        hidden_dim=cfg["hidden_dim"],
        num_layers=3,
        f_prev_mode=cfg["f_prev_mode"],
    )
    pipe_cfg = OdrzywolekPipelineConfig(
        total_steps=steps,
        lr=2e-3,
        seed=seed,
        freeze_trunk_after_search=cfg["freeze"],
    )
    t0 = time.perf_counter()
    result = run_odrzywolek_pipeline(model, x, y, pipe_cfg)
    elapsed = time.perf_counter() - t0

    gen_h = torch.Generator().manual_seed(seed + 1)
    x_h, y_h = target.sample(128, gen_h, noise)
    with torch.no_grad():
        soft_mse = torch.mean((model(x_h) - y_h) ** 2).item()
    snapped_mse, _ = evaluate_snapped(model, x_h, y_h)

    return {
        "config": cfg_name,
        "soft_holdout_mse": soft_mse,
        "snapped_holdout_mse": snapped_mse,
        "snap_degrade_ratio": snapped_mse / soft_mse if soft_mse > 0 else float("nan"),
        "train_seconds": elapsed,
        "expression_eml": result.expression_eml,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-capacity study")
    parser.add_argument("--targets", nargs="+", default=["square", "sin"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--noise-std-rel", type=float, default=DEFAULT_NOISE_STD_REL)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"head_capacity_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for tid in args.targets:
        target = TARGETS[tid]
        print(f"\n=== target={tid} ({target.formula}) ===")
        for cfg_name, cfg in CONFIGS.items():
            row = run_one(target, cfg_name, cfg, args.seed, args.steps, args.noise_std_rel)
            row["target_id"] = tid
            all_rows.append(row)
            print(f"  [{cfg_name:12s}] soft={row['soft_holdout_mse']:.3e} "
                  f"snapped={row['snapped_holdout_mse']:.3e} "
                  f"(x{row['snap_degrade_ratio']:.1f}) {row['train_seconds']:.1f}s")

    (out_dir / "summary.json").write_text(
        json.dumps({"seed": args.seed, "steps": args.steps, "results": all_rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# Head-Capacity Study",
        "",
        "Small snapped/soft gap ⇒ the exported closed-form head is faithful "
        "(genuine symbolic recovery). Large gap ⇒ the trunk MLP is doing the work.",
        "",
        "| target | config | soft MSE | snapped MSE | degrade |",
        "|--------|--------|----------|-------------|---------|",
    ]
    for r in all_rows:
        md.append(
            f"| {r['target_id']} | {r['config']} | {r['soft_holdout_mse']:.2e} | "
            f"{r['snapped_holdout_mse']:.2e} | x{r['snap_degrade_ratio']:.1f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nresults -> {out_dir}")


if __name__ == "__main__":
    main()
