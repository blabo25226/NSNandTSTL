"""
Multi-seed baseline gate evaluation (TSTL pre-integration).

Runs Phase-3 targets across seeds with a fixed protocol, then evaluates
Gate A (stable core), Gate B (sin_plus), and Gate C (snap faithfulness).

Usage:
  python scripts/baseline_gate_eval.py
  python scripts/baseline_gate_eval.py --protocol freeze --seeds 0 42
  python scripts/baseline_gate_eval.py --targets square product --quick-steps 800
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

from baseline_gate import (  # noqa: E402
    DEFAULT_SEEDS,
    GATE_A_TARGETS,
    GATE_B_TARGETS,
    MSE_OK_RELAXED_DEFAULT,
    MSE_OK_STRICT_DEFAULT,
    evaluate_gates,
    mse_threshold,
    protocol_train_flags,
)
from simplify import evaluate_symbolic  # noqa: E402
from targets import DEFAULT_NOISE_STD_REL, TARGETS  # noqa: E402
from trainer import TrainConfig, train_target  # noqa: E402

ALL_GATE_TARGETS = list(GATE_A_TARGETS) + list(GATE_B_TARGETS)


def run_one(
    target_id: str,
    seed: int,
    protocol: str,
    noise_std_rel: float,
    mse_strict: float,
    mse_relaxed: float,
    quick_steps: int | None,
) -> dict:
    target = TARGETS[target_id]
    flags = protocol_train_flags(protocol)
    gen_ref = torch.Generator().manual_seed(seed)
    _, y_ref = target.sample(256, gen_ref, noise_std_rel)
    thr = mse_threshold(
        target.id, target.monotonic, mse_strict, mse_relaxed, y_ref, noise_std_rel
    )

    cfg = TrainConfig(
        seed=seed,
        noise_std_rel=noise_std_rel,
        **flags,
    )
    if quick_steps is not None:
        cfg.steps = quick_steps

    row: dict = {
        "target_id": target_id,
        "seed": seed,
        "protocol": protocol,
        "true_formula": target.formula,
        "monotonic": target.monotonic,
        "mse_threshold": thr,
        "train_config": {
            "feature_dim": None,
            "head_depth": None,
            "steps": cfg.steps,
            **flags,
        },
    }

    try:
        model, tr = train_target(target, cfg)
        row["train_config"]["feature_dim"] = model.head.feature_dim
        row["train_config"]["head_depth"] = model.head.depth
        row["train_seconds"] = tr.train_seconds

        gen = torch.Generator().manual_seed(seed + 1)
        x_hold, y_hold = target.sample(128, gen, noise_std_rel)
        with torch.no_grad():
            pred = model(x_hold)
            finite_pred = bool(torch.isfinite(pred).all())
            hold_mse = torch.mean((pred - y_hold) ** 2).item()

        sym = evaluate_symbolic(
            model,
            tr.expression_eml,
            x_hold,
            y_hold,
            true_formula=target.formula,
            mse_threshold=thr,
            numeric_ok=hold_mse <= thr,
        )
        snapped_mse = sym.snapped_mse if sym.snapped_mse is not None else float("nan")
        snap_degrade = snapped_mse / hold_mse if hold_mse > 0 else float("nan")

        row.update(
            holdout_mse=hold_mse,
            snapped_holdout_mse=snapped_mse,
            snap_degrade_ratio=snap_degrade,
            numeric_ok=hold_mse <= thr,
            symbolic_ok=sym.symbolic_ok,
            finite=finite_pred and hold_mse == hold_mse,
            template_guess=sym.template_guess,
            eml_expression=tr.expression_eml,
        )
    except (RuntimeError, ValueError) as exc:
        row.update(
            holdout_mse=float("inf"),
            snapped_holdout_mse=float("inf"),
            snap_degrade_ratio=float("nan"),
            numeric_ok=False,
            symbolic_ok=False,
            finite=False,
            error=str(exc),
        )

    return row


def _format_verdict_md(verdict, protocol: str, seeds: list[int]) -> str:
    lines = [
        "# Baseline Gate Evaluation",
        "",
        f"- protocol: `{protocol}`",
        f"- seeds: {seeds}",
        f"- Gate A (core 80%): **{'PASS' if verdict.gate_a_pass else 'FAIL'}**",
        f"- Gate B (sin_plus 60%): **{'PASS' if verdict.gate_b_pass else 'FAIL'}** (optional)",
        f"- Gate C (snap degrade ≤ {1.5}): **{'PASS' if verdict.gate_c_pass else 'FAIL'}**",
    ]
    if verdict.gate_c_median_degrade is not None:
        lines.append(f"- Gate C median degrade (Gate A symbolic OK): {verdict.gate_c_median_degrade:.3f}")
    lines.extend(["", "## Gate A per target", ""])
    lines.append("| target | symbolic OK | finite | rate | degrade median |")
    lines.append("|--------|-------------|--------|------|----------------|")
    for tid, s in verdict.gate_a_targets.items():
        deg = f"{s.snap_degrade_median:.2f}" if s.snap_degrade_median is not None else "—"
        lines.append(
            f"| {tid} | {s.n_symbolic_ok}/{s.n_runs} | {s.n_finite}/{s.n_runs} | "
            f"{s.success_rate:.0%} | {deg} |"
        )
    if verdict.gate_b_targets:
        lines.extend(["", "## Gate B", ""])
        lines.append("| target | symbolic OK | rate |")
        lines.append("|--------|-------------|------|")
        for tid, s in verdict.gate_b_targets.items():
            lines.append(f"| {tid} | {s.n_symbolic_ok}/{s.n_runs} | {s.success_rate:.0%} |")
    if verdict.notes:
        lines.extend(["", "## Notes", ""])
        for note in verdict.notes:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="NSN baseline gate (TSTL pre-integration)")
    parser.add_argument("--protocol", choices=["full", "freeze", "trunk_first"], default="full")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--targets", nargs="+", default=ALL_GATE_TARGETS)
    parser.add_argument("--noise-std-rel", type=float, default=DEFAULT_NOISE_STD_REL)
    parser.add_argument("--mse-strict", type=float, default=MSE_OK_STRICT_DEFAULT)
    parser.add_argument("--mse-relaxed", type=float, default=MSE_OK_RELAXED_DEFAULT)
    parser.add_argument(
        "--quick-steps",
        type=int,
        default=None,
        help="Override training steps (smoke tests only; not for official gate runs).",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"baseline_gate_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t0 = time.perf_counter()
    print(f"=== Baseline gate | protocol={args.protocol} | seeds={args.seeds} ===")

    for tid in args.targets:
        for seed in args.seeds:
            print(f"  [{tid}] seed={seed} ...", flush=True)
            row = run_one(
                tid,
                seed,
                args.protocol,
                args.noise_std_rel,
                args.mse_strict,
                args.mse_relaxed,
                args.quick_steps,
            )
            results.append(row)
            status = "SYM OK" if row.get("symbolic_ok") else "FAIL"
            if row.get("error"):
                status = f"ERR: {row['error'][:40]}"
            print(f"    -> {status}")

    verdict = evaluate_gates(results)
    elapsed = time.perf_counter() - t0

    summary = {
        "protocol": args.protocol,
        "seeds": args.seeds,
        "targets": args.targets,
        "noise_std_rel": args.noise_std_rel,
        "wall_seconds": elapsed,
        "gate_a_pass": verdict.gate_a_pass,
        "gate_b_pass": verdict.gate_b_pass,
        "gate_c_pass": verdict.gate_c_pass,
        "gate_c_median_degrade": verdict.gate_c_median_degrade,
        "results": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(
        _format_verdict_md(verdict, args.protocol, args.seeds), encoding="utf-8"
    )

    print(f"\nGate A: {'PASS' if verdict.gate_a_pass else 'FAIL'}")
    print(f"Gate B: {'PASS' if verdict.gate_b_pass else 'FAIL'} (optional)")
    print(f"Gate C: {'PASS' if verdict.gate_c_pass else 'FAIL'}")
    print(f"wall time: {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
