"""Train DNN-EML on simple synthetic targets (Phase 0 demo)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import DNNEML  # noqa: E402
from snap import (  # noqa: E402
    apply_hardening,
    apply_snap_to_logits,
    export_leaf_summary,
    export_symbolic_expression,
    export_tree_structure,
)
from targets import TARGETS  # noqa: E402
from utils import set_seed  # noqa: E402

FEYNMAN_DEMO_ID = "feynman_I29_product"  # Feynman I.29 style: y = x0 * x1


def make_data(demo: str, n: int = 128, seed: int = 0):
    if demo == "square":
        x = torch.linspace(-2, 2, n).unsqueeze(1)
        y = x.squeeze() ** 2
    elif demo == "sin_plus":
        x = torch.randn(n, 2)
        y = torch.sin(x[:, 0]) + x[:, 1]
    elif demo == "feynman":
        gen = torch.Generator().manual_seed(seed)
        target = TARGETS[FEYNMAN_DEMO_ID]
        x, y = target.sample(n if n >= 256 else 256, gen, noise_std_rel=0.0)
    else:
        raise ValueError(f"unknown demo: {demo}")
    return x, y


def _demo_config(demo: str, head_depth: int) -> dict:
    if demo == "sin_plus":
        return dict(feature_dim=6, head_depth=head_depth, hidden_dim=64, lr=3e-3, steps_mul=2, optimizer="adam")
    if demo == "feynman":
        return dict(feature_dim=6, head_depth=2, hidden_dim=128, lr=2e-3, steps_mul=5, optimizer="adam")
    return dict(feature_dim=4, head_depth=head_depth, hidden_dim=64, lr=1e-3, steps_mul=1, optimizer="adam")


def main() -> None:
    parser = argparse.ArgumentParser(description="DNN-EML training demo")
    parser.add_argument("--demo", choices=["square", "sin_plus", "feynman"], default="square")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-depth", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    x, y = make_data(args.demo, seed=args.seed)
    input_dim = x.shape[1]
    cfg = _demo_config(args.demo, args.head_depth)
    steps = args.steps * cfg["steps_mul"]

    model = DNNEML.build(
        input_dim=input_dim,
        feature_dim=cfg["feature_dim"],
        head_depth=cfg["head_depth"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=3,
    )
    opt_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=cfg["lr"])

    losses: list[float] = []
    for step in range(1, steps + 1):
        apply_hardening(model.head, step, steps)
        opt.zero_grad()
        loss = model.mse_loss(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    apply_snap_to_logits(model.head)
    z_names = [f"z{i}" for i in range(model.head.feature_dim)]
    expression = export_symbolic_expression(model.head, z_names)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"{args.demo}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "train_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"demo={args.demo}\n")
        if args.demo == "feynman":
            f.write(f"target={FEYNMAN_DEMO_ID}\n")
            f.write(f"formula={TARGETS[FEYNMAN_DEMO_ID].formula}\n")
        f.write(f"steps={steps}\n")
        f.write(f"final_mse={losses[-1]:.6e}\n")
        f.write(f"initial_mse={losses[0]:.6e}\n")
        if args.demo == "feynman":
            f.write("eval_mode=overfit (train == test)\n")
        f.write("\n--- tree ---\n")
        f.write(export_tree_structure(cfg["head_depth"]) + "\n")
        f.write("\n--- leaf summary ---\n")
        f.write("\n".join(export_leaf_summary(model.head)) + "\n")
        f.write("\n--- snapped expression ---\n")
        f.write(expression + "\n")

    expr_path = out_dir / "expression.txt"
    expr_path.write_text(expression + "\n", encoding="utf-8")

    print(f"demo={args.demo} final_mse={losses[-1]:.6e}")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
