"""
Trunk interpretability study: recover the lower half of the two-layer
explanation (x -> z) and compose the full closed form yhat(x).

For each target it trains a DNN-EML, snaps the head to its closed form
yhat = f(z), then:
  - fits a linear readout z_j ~= w_j^T x + b_j (exact for a linear trunk, else
    least-squares distillation with per-component R^2),
  - composes yhat(x) as a single EML expression in the raw inputs x,
  - reports per-z-component input-feature attributions.

Usage:
  python scripts/trunk_interpret_eval.py --targets product sin --trunk mlp
  python scripts/trunk_interpret_eval.py --targets product sin --trunk linear
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import DNNEML  # noqa: E402
from pipeline import OdrzywolekPipelineConfig, run_odrzywolek_pipeline  # noqa: E402
from snap import apply_snap_to_logits, evaluate_snapped, export_symbolic_expression  # noqa: E402
from targets import DEFAULT_NOISE_STD_REL, TARGETS  # noqa: E402
from trunk_interpret import compose_symbolic, linear_readout, trunk_attribution  # noqa: E402
from utils import set_seed  # noqa: E402


def run_one(target, trunk: str, seed: int, steps: int, noise: float) -> dict:
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x, y = target.sample(256, gen, noise)

    num_layers = 1 if trunk == "linear" else 3
    hidden_dim = 16 if trunk == "linear" else 128
    feature_dim = 4
    model = DNNEML.build(
        input_dim=target.input_dim,
        feature_dim=feature_dim,
        head_depth=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    pipe_cfg = OdrzywolekPipelineConfig(total_steps=steps, lr=2e-3, seed=seed)
    run_odrzywolek_pipeline(model, x, y, pipe_cfg)

    # Holdout data for faithfulness / R^2 reporting.
    gen_h = torch.Generator().manual_seed(seed + 1)
    x_h, y_h = target.sample(128, gen_h, noise)

    # Snap the head so exports reflect the closed form.
    apply_snap_to_logits(model.head)
    model.head.set_temperature(1e-3)
    model.eval()

    x_names = [f"x{i}" for i in range(target.input_dim)]
    head_expr = export_symbolic_expression(model.head)
    ro = linear_readout(model, x_h)
    composed = compose_symbolic(model, ro, x_names)
    attr = trunk_attribution(model, x_h, top_k=min(3, target.input_dim))

    snapped_mse, _ = evaluate_snapped(model, x_h, y_h)
    with torch.no_grad():
        soft_mse = torch.mean((model(x_h) - y_h) ** 2).item()

    return {
        "target_id": target.id,
        "true_formula": target.formula,
        "trunk": trunk,
        "trunk_linear_exact": ro.exact,
        "readout_r2": [round(v, 4) for v in ro.r2.tolist()],
        "soft_holdout_mse": soft_mse,
        "snapped_holdout_mse": snapped_mse,
        "head_expression": head_expr,
        "composed_expression_x": composed,
        "attribution": [
            [{"feature": f"x{i}", "score": round(s, 4)} for (i, s) in comp]
            for comp in attr
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trunk interpretability study")
    parser.add_argument("--targets", nargs="+", default=["product", "sin"])
    parser.add_argument("--trunk", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--noise-std-rel", type=float, default=DEFAULT_NOISE_STD_REL)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"trunk_interpret_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for tid in args.targets:
        target = TARGETS[tid]
        print(f"\n=== target={tid} ({target.formula}) trunk={args.trunk} ===")
        row = run_one(target, args.trunk, args.seed, args.steps, args.noise_std_rel)
        rows.append(row)
        print(f"  readout R^2 = {row['readout_r2']} (exact={row['trunk_linear_exact']})")
        print(f"  soft={row['soft_holdout_mse']:.3e} snapped={row['snapped_holdout_mse']:.3e}")
        print(f"  head:     yhat = {row['head_expression']}")
        print(f"  composed: yhat = {row['composed_expression_x']}")
        for j, comp in enumerate(row["attribution"]):
            top = ", ".join(f"{c['feature']}({c['score']})" for c in comp)
            print(f"    z{j} <- {top}")

    (out_dir / "summary.json").write_text(
        json.dumps({"seed": args.seed, "trunk": args.trunk, "results": rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# Trunk Interpretability Study",
        "",
        "Two-layer white-box explanation: the linear readout recovers `z_j = w_j^T x + b_j`",
        "(exact for a linear trunk, else distilled with per-component R^2), and the head",
        "export is composed into a single closed form `yhat(x)`.",
        "",
    ]
    for r in rows:
        md.append(f"## {r['target_id']} — `{r['true_formula']}` (trunk={r['trunk']})")
        md.append("")
        md.append(f"- readout R^2 (per z): {r['readout_r2']} (exact={r['trunk_linear_exact']})")
        md.append(f"- soft MSE: {r['soft_holdout_mse']:.3e} | snapped MSE: {r['snapped_holdout_mse']:.3e}")
        md.append(f"- head:     `yhat = {r['head_expression']}`")
        md.append(f"- composed: `yhat = {r['composed_expression_x']}`")
        for j, comp in enumerate(r["attribution"]):
            top = ", ".join(f"{c['feature']} ({c['score']})" for c in comp)
            md.append(f"  - z{j} <- {top}")
        md.append("")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nresults -> {out_dir}")


if __name__ == "__main__":
    main()
