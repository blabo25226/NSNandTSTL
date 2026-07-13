"""
FLOPs cost analysis for the EML head (paper's hardware-efficiency claim).

Reports the per-eml-node FLOP breakdown (~111 transcendental FLOPs, reproducing
the paper), the total head cost by depth, and a comparison against an MLP of
similar size. This quantifies the software-side of the "hardware-efficient" thesis
(the paper argues that this per-node cost motivates a dedicated EML cell / FPGA);
FPGA/analog synthesis itself is out of scope for a software reproduction.

Usage:
  python scripts/flops_analysis.py --feature-dim 6 --depths 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cost import DEFAULT_WEIGHTS, eml_node_flops, head_flops, mlp_flops  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="EML FLOPs cost analysis")
    parser.add_argument("--feature-dim", type=int, default=6)
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--input-dim", type=int, default=2)
    args = parser.parse_args()

    node = eml_node_flops()

    # A representative MLP trunk of the size used in experiments.
    mlp_dims = [args.input_dim] + [args.mlp_hidden] * (args.mlp_layers - 1) + [args.feature_dim]
    trunk_cost = mlp_flops(mlp_dims)

    rows = []
    for depth in args.depths:
        h = head_flops(depth, args.feature_dim)
        h["trunk_mlp_flops"] = trunk_cost
        h["total_dnn_eml_flops"] = h["head_flops"] + trunk_cost
        rows.append(h)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"flops_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "weights": DEFAULT_WEIGHTS,
        "eml_node": {
            "transcendental_flops": node.transcendental_flops,
            "arithmetic_flops": node.arithmetic_flops,
            "total_flops": node.total_flops,
            "breakdown": node.breakdown,
        },
        "mlp_trunk_dims": mlp_dims,
        "mlp_trunk_flops": trunk_cost,
        "by_depth": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# EML FLOPs Cost Analysis",
        "",
        f"Per-eml-node transcendental FLOPs = **{node.transcendental_flops:.0f}** "
        f"(paper: ~111), + {node.arithmetic_flops:.0f} arithmetic "
        f"= **{node.total_flops:.0f}** total.",
        "",
        "Breakdown (weighted FLOP-equivalents): "
        + ", ".join(f"{k}={v:.0f}" for k, v in node.breakdown.items()),
        "",
        f"MLP trunk {mlp_dims}: {trunk_cost:.0f} FLOPs.",
        "",
        "| depth | eml nodes | node FLOPs | head FLOPs | +trunk = total |",
        "|-------|-----------|------------|------------|----------------|",
    ]
    for h in rows:
        md.append(
            f"| {h['depth']} | {h['num_internal_nodes']} | {h['eml_node_flops']:.0f} | "
            f"{h['head_flops']:.0f} | {h['total_dnn_eml_flops']:.0f} |"
        )
    md += [
        "",
        "The per-node transcendental cost (~111 FLOPs) is the paper's motivation for "
        "a dedicated EML cell (FPGA/analog); on CPU/GPU each node is expensive relative "
        "to a plain MAC. FPGA/analog synthesis is out of scope here.",
    ]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"eml node: transcendental={node.transcendental_flops:.0f} "
          f"total={node.total_flops:.0f} FLOPs")
    for h in rows:
        print(f"  depth={h['depth']}: head={h['head_flops']:.0f} "
              f"total(+trunk)={h['total_dnn_eml_flops']:.0f}")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
