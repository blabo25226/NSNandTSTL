"""Symbolic-regression evaluation: phases 1–3 (elementary / Feynman-style)."""

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

from simplify import evaluate_symbolic  # noqa: E402
from targets import DEFAULT_NOISE_STD_REL, TARGETS, targets_for_phase  # noqa: E402
from trainer import TrainConfig, train_target  # noqa: E402

MSE_OK_STRICT_DEFAULT = 1e-2
MSE_OK_RELAXED_DEFAULT = 5e-2


def phase_targets(phase: int) -> list[str]:
    if phase == 1:
        return ["sin_plus"]
    if phase == 2:
        return ["feynman_I29_product", "feynman_I9_inv_square"]
    if phase == 3:
        return ["square", "product", "sum", "exp", "sin", "sin_plus"]
    raise ValueError("phase must be 1, 2, or 3")


def mse_threshold(
    target_id: str,
    monotonic: bool,
    strict: float,
    relaxed: float,
    y_ref: torch.Tensor | None = None,
    noise_std_rel: float = 0.0,
) -> float:
    base = relaxed if not monotonic else strict
    if y_ref is not None and noise_std_rel > 0:
        noise_floor = (noise_std_rel * y_ref.std()).item() ** 2
        base = max(base, noise_floor * 2.0)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="NSN SR evaluation (phases 1–3)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mse-strict", type=float, default=MSE_OK_STRICT_DEFAULT)
    parser.add_argument("--mse-relaxed", type=float, default=MSE_OK_RELAXED_DEFAULT)
    parser.add_argument(
        "--noise-std-rel",
        type=float,
        default=DEFAULT_NOISE_STD_REL,
        help="Label noise: y += N(0, (noise_std_rel * std(y))^2). 0 = noiseless.",
    )
    args = parser.parse_args()

    mse_strict = args.mse_strict
    mse_relaxed = args.mse_relaxed
    noise_std_rel = args.noise_std_rel

    ids = phase_targets(args.phase)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"sr_phase{args.phase}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total_train_sec = 0.0
    phase_t0 = time.perf_counter()
    print(f"=== SR Phase {args.phase} (noise_std_rel={noise_std_rel}) ===")

    for tid in ids:
        target = TARGETS[tid]
        gen_ref = torch.Generator().manual_seed(args.seed)
        _, y_ref = target.sample(256, gen_ref, noise_std_rel)
        thr = mse_threshold(
            target.id, target.monotonic, mse_strict, mse_relaxed, y_ref, noise_std_rel
        )
        print(f"\n[{tid}] true={target.formula} monotonic={target.monotonic} thr={thr:.2e}")

        model, tr = train_target(
            target, TrainConfig(seed=args.seed, noise_std_rel=noise_std_rel)
        )
        total_train_sec += tr.train_seconds
        gen = torch.Generator().manual_seed(args.seed + 1)
        x_hold, y_hold = target.sample(128, gen, noise_std_rel)

        with torch.no_grad():
            hold_mse = torch.mean((model(x_hold) - y_hold) ** 2).item()

        numeric_ok = hold_mse <= thr
        sym = evaluate_symbolic(
            model,
            tr.expression_eml,
            x_hold,
            y_hold,
            true_formula=target.formula,
            mse_threshold=thr,
            numeric_ok=numeric_ok,
        )
        row = {
            "target_id": tid,
            "true_formula": target.formula,
            "monotonic": target.monotonic,
            "final_train_mse": tr.final_mse,
            "holdout_mse": hold_mse,
            "mse_threshold": thr,
            "numeric_ok": numeric_ok,
            "symbolic_ok": sym.symbolic_ok,
            "template_guess": sym.template_guess,
            "template_mse": sym.template_mse,
            "eml_expression": sym.eml_expression,
            "simplified": sym.simplified,
            "steps": tr.steps,
            "train_seconds": tr.train_seconds,
            "best_val_mse": tr.best_val_mse,
            "stopped_step": tr.stopped_step,
        }
        rows.append(row)

        status = []
        status.append("NUM OK" if numeric_ok else "NUM FAIL")
        status.append("SYM OK" if sym.symbolic_ok else "SYM FAIL")
        print(f"  holdout_mse={hold_mse:.4e} train_sec={tr.train_seconds:.1f}s ({', '.join(status)})")
        print(f"  template_guess={sym.template_guess} template_mse={sym.template_mse}")
        if sym.simplified:
            print(f"  simplified={sym.simplified}")

    phase_elapsed = time.perf_counter() - phase_t0
    summary = {
        "phase": args.phase,
        "seed": args.seed,
        "noise_std_rel": noise_std_rel,
        "total_train_seconds": total_train_sec,
        "total_wall_seconds": phase_elapsed,
        "numeric_success": sum(1 for r in rows if r["numeric_ok"]),
        "symbolic_success": sum(1 for r in rows if r["symbolic_ok"]),
        "total": len(rows),
        "results": rows,
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# SR Phase {args.phase} Results",
        "",
        f"- numeric OK: {summary['numeric_success']}/{summary['total']}",
        f"- symbolic OK: {summary['symbolic_success']}/{summary['total']}",
        "",
        "| target | true | mono | holdout MSE | numeric | symbolic | guess |",
        "|--------|------|------|-------------|---------|----------|-------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['target_id']} | `{r['true_formula']}` | {r['monotonic']} | "
            f"{r['holdout_mse']:.2e} | {r['numeric_ok']} | {r['symbolic_ok']} | "
            f"`{r['template_guess']}` |"
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\n=== Phase {args.phase} summary: numeric {summary['numeric_success']}/{summary['total']}, "
          f"symbolic {summary['symbolic_success']}/{summary['total']} ===")
    print(f"train time: {total_train_sec:.1f}s | wall time: {phase_elapsed:.1f}s")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
