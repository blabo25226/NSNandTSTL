"""
Real Feynman benchmark: NSN (DNN-EML head, depths {2,3,4}) vs baselines
(MLP, minimal EQL, optional KAN) across multiple seeds.

Reports per-(equation, method) holdout R^2 mean±std, MSE, complexity proxy, and
training time, plus an aggregate success rate (fraction of runs with R^2 >= thr).
This addresses roadmap item ② (Feynman DB × EQL/KAN) at a curated subset, and the
multi-seed aggregation gives the paper-style statistical evaluation (item D).

Usage:
  python scripts/feynman_benchmark.py --equations I.12.1 I.14.4 I.25.13 \
      --depths 2 3 4 --seeds 0 1 --steps 2000
  python scripts/feynman_benchmark.py --all --seeds 0 1 2   # full curated set
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

from baselines import EQLBaseline, KANBaseline, MLPBaseline  # noqa: E402
from feynman import FEYNMAN_TARGETS  # noqa: E402
from model import DNNEML  # noqa: E402
from pipeline import OdrzywolekPipelineConfig, run_odrzywolek_pipeline  # noqa: E402
from snap import evaluate_snapped  # noqa: E402
from targets import DEFAULT_NOISE_STD_REL  # noqa: E402
from utils import set_seed  # noqa: E402

SUCCESS_R2 = 0.99


def r2_score(pred: torch.Tensor, y: torch.Tensor) -> float:
    ss_res = torch.sum((y - pred) ** 2).item()
    ss_tot = torch.sum((y - y.mean()) ** 2).item()
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-9 else 0.0
    return 1.0 - ss_res / ss_tot


def run_nsn(target, depth, x_tr, y_tr, x_te, y_te, steps, seed, feature_dim=4) -> dict:
    # feature_dim=4 is the stable choice: larger d inflates beta^T z, which
    # overflows exp/ln and diverges (R^2 << 0), while d=4 reaches R^2 ~ 0.999.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DNNEML.build(
            input_dim=target.input_dim, feature_dim=feature_dim, head_depth=depth,
            hidden_dim=128, num_layers=3,
        )
    finite = True
    try:
        run_odrzywolek_pipeline(
            model, x_tr, y_tr, OdrzywolekPipelineConfig(total_steps=steps, lr=2e-3, seed=seed)
        )
        with torch.no_grad():
            pred = model(x_te)
        snapped_mse, snapped_pred = evaluate_snapped(model, x_te, y_te)
        soft_r2 = r2_score(pred, y_te)
        snap_r2 = r2_score(snapped_pred, y_te)
        finite = bool(torch.isfinite(pred).all())
    except (RuntimeError, ValueError):
        finite = False
        soft_r2 = snap_r2 = float("-inf")
        snapped_mse = float("inf")
    return {
        "method": f"nsn_d{depth}", "soft_r2": soft_r2, "snapped_r2": snap_r2,
        "snapped_mse": snapped_mse, "finite": finite, "size": 2**depth - 1,
    }


def run_baseline(bl, x_tr, y_tr, x_te, y_te, steps) -> dict:
    res = bl.fit_predict(x_tr, y_tr, x_te, steps=steps)
    return {
        "method": bl.name, "soft_r2": r2_score(res.pred, y_te), "snapped_r2": None,
        "train_seconds": res.train_seconds, "size": res.size, "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Feynman benchmark: NSN vs baselines")
    parser.add_argument("--equations", nargs="+", default=["I.12.1", "I.14.4", "I.25.13"])
    parser.add_argument("--all", action="store_true", help="Use the full curated set")
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--noise-std-rel", type=float, default=DEFAULT_NOISE_STD_REL)
    args = parser.parse_args()

    eq_ids = list(FEYNMAN_TARGETS) if args.all else args.equations
    kan = KANBaseline()
    baselines = [MLPBaseline(), EQLBaseline()]
    if kan.available:
        baselines.append(kan)
    print(f"KAN available: {kan.available}")

    rows: list[dict] = []
    for eid in eq_ids:
        target = FEYNMAN_TARGETS[eid]
        print(f"\n=== {eid} ({target.formula}) dim={target.input_dim} ===")
        for seed in args.seeds:
            set_seed(seed)
            gtr = torch.Generator().manual_seed(seed)
            gte = torch.Generator().manual_seed(seed + 1000)
            x_tr, y_tr = target.sample(args.n_train, gtr, args.noise_std_rel)
            x_te, y_te = target.sample(128, gte, args.noise_std_rel)

            for depth in args.depths:
                r = run_nsn(target, depth, x_tr, y_tr, x_te, y_te, args.steps, seed)
                r.update(equation=eid, seed=seed)
                rows.append(r)
                print(f"  [seed {seed}] {r['method']:8s} soft_R2={r['soft_r2']:.3f} "
                      f"snap_R2={r['snapped_r2']:.3f}")
            for bl in baselines:
                r = run_baseline(bl, x_tr, y_tr, x_te, y_te, args.steps)
                r.update(equation=eid, seed=seed)
                rows.append(r)
                print(f"  [seed {seed}] {r['method']:8s} soft_R2={r['soft_r2']:.3f} "
                      f"({r['train_seconds']:.1f}s)")

    # Aggregate per method: mean±std soft R2, success rate (R2 >= SUCCESS_R2).
    methods = sorted({r["method"] for r in rows})
    agg: dict[str, dict] = {}
    for m in methods:
        mrows = [r for r in rows if r["method"] == m]
        r2s = [r["soft_r2"] for r in mrows if r["soft_r2"] > float("-inf")]
        succ = sum(1 for r in mrows if r["soft_r2"] >= SUCCESS_R2)
        agg[m] = {
            "runs": len(mrows),
            "success_rate": succ / len(mrows) if mrows else float("nan"),
            "mean_r2": statistics.mean(r2s) if r2s else float("nan"),
            "std_r2": statistics.pstdev(r2s) if len(r2s) > 1 else 0.0,
        }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"feynman_bench_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({
            "equations": eq_ids, "depths": args.depths, "seeds": args.seeds,
            "steps": args.steps, "success_r2": SUCCESS_R2,
            "kan_available": kan.available, "aggregate": agg, "results": rows,
        }, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    md = [
        "# Feynman Benchmark: NSN vs baselines",
        "",
        f"Equations: {len(eq_ids)} | seeds: {args.seeds} | steps: {args.steps} | "
        f"success = holdout R² ≥ {SUCCESS_R2} | KAN: {kan.available}",
        "",
        "| method | runs | success rate | mean R² | std R² |",
        "|--------|------|--------------|---------|--------|",
    ]
    for m in methods:
        a = agg[m]
        md.append(f"| {m} | {a['runs']} | {a['success_rate']:.2f} | "
                  f"{a['mean_r2']:.3f} | {a['std_r2']:.3f} |")
    md += ["", "## Per-equation soft R² (mean over seeds)", "",
           "| equation | " + " | ".join(methods) + " |",
           "|" + "----|" * (len(methods) + 1)]
    for eid in eq_ids:
        cells = []
        for m in methods:
            vals = [r["soft_r2"] for r in rows
                    if r["equation"] == eid and r["method"] == m and r["soft_r2"] > float("-inf")]
            cells.append(f"{statistics.mean(vals):.3f}" if vals else "—")
        md.append(f"| {eid} | " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n=== aggregate success rate ===")
    for m in methods:
        print(f"  {m:8s}: {agg[m]['success_rate']:.2f} (mean R²={agg[m]['mean_r2']:.3f})")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
