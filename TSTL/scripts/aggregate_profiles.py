"""Aggregate multiple layer-contribution runs (seed-average C(k)).

Reads ``contributions.json`` from several result dirs (same layer count) and
writes a seed-averaged C(k) profile + bar chart + summary. Used to show the
mid-layer tendency across seeds rather than from a single run.

Example:
    python scripts/aggregate_profiles.py results/tstl_profile_* --label mlp5
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layer_contribution import plot_contribution_heatmap  # noqa: E402


def _load_contributions(run_dir: Path) -> dict[int, float]:
    data = json.loads((run_dir / "contributions.json").read_text(encoding="utf-8"))
    return {int(k): float(v) for k, v in data["contributions"].items()}


def aggregate(run_dirs: list[Path]) -> tuple[dict[int, float], dict[int, list[float]], list[int]]:
    per_k: dict[int, list[float]] = {}
    argmax_per_run: list[int] = []
    for d in run_dirs:
        contrib = _load_contributions(d)
        finite = {k: v for k, v in contrib.items() if v == v}  # drop NaN
        if not finite:
            continue
        argmax_per_run.append(max(finite, key=lambda k: finite[k]))
        for k, v in finite.items():
            per_k.setdefault(k, []).append(v)
    mean_c = {k: sum(vs) / len(vs) for k, vs in per_k.items()}
    return mean_c, per_k, argmax_per_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate C(k) across seeds/runs")
    parser.add_argument("globs", nargs="+", help="Result dirs or globs with contributions.json")
    parser.add_argument("--label", default="aggregate")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dirs: list[Path] = []
    for g in args.globs:
        for p in sorted(glob.glob(g)):
            pp = Path(p)
            if (pp / "contributions.json").is_file():
                run_dirs.append(pp)
    if not run_dirs:
        raise SystemExit("No result dirs with contributions.json matched")

    mean_c, per_k, argmaxes = aggregate(run_dirs)
    if not mean_c:
        raise SystemExit("No finite contributions to aggregate")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (ROOT / "results" / f"{args.label}_aggregate_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_contribution_heatmap(
        mean_c, out_dir / "contribution_mean.png", title=f"Mean C(k) over {len(run_dirs)} runs"
    )
    (out_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "runs": [str(d) for d in run_dirs],
                "mean_contribution": {str(k): v for k, v in mean_c.items()},
                "per_layer_values": {str(k): v for k, v in per_k.items()},
                "argmax_per_run": argmaxes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_k = max(mean_c, key=lambda k: mean_c[k])
    n_layers = max(mean_c) + 1
    endpoints = {0, n_layers - 1}
    mid_argmax = sum(1 for a in argmaxes if a not in endpoints)
    lines = [
        f"# Aggregated C(k) — {args.label}",
        "",
        f"- runs: {len(run_dirs)}",
        f"- mean-C best layer: k={best_k} (C={mean_c[best_k]:.3f})",
        f"- argmax off the endpoints: {mid_argmax}/{len(argmaxes)} runs",
        "",
        "| k | mean C(k) | n |",
        "|---|-----------|---|",
    ]
    for k in sorted(mean_c):
        lines.append(f"| {k} | {mean_c[k]:.4f} | {len(per_k[k])} |")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"aggregated {len(run_dirs)} runs -> {out_dir}")
    print(f"mean-C best k={best_k} C={mean_c[best_k]:.3f}; "
          f"mid-argmax {mid_argmax}/{len(argmaxes)}")


if __name__ == "__main__":
    main()
