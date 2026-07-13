"""E8 (toy): per-layer weight change ‖Δθ_k‖ vs contribution C(k) on the tiny TF.

TSTL §5 observes that full-training weight-change magnitude is roughly uniform
across layers and does NOT track C(k) — i.e. contribution comes from subspace
usefulness, not from how much a layer moves. This reproduces that *analysis* on
the CPU tiny Transformer: it full-trains from the same init as the layer scan,
measures ‖Δθ_k‖ per block, and correlates it with the scan's C(k).

Not a paper reproduction (toy model, supervised task). Reuses the scan config so
the full run matches the one behind contributions.json.

    python scripts/weight_change_vs_contribution.py --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from layer_contribution import plot_contribution_heatmap  # noqa: E402
from llm_freeze import transformer_block_modules, unfreeze_all  # noqa: E402
from tiny_transformer import TinyCausalLM, TinyConfig, make_dataset, train_model  # noqa: E402
from utils import pearson_corr, set_seed, spearman_corr  # noqa: E402


def block_delta_norms(model: TinyCausalLM, init_state: dict) -> dict[int, float]:
    """L2 norm of (trained - init) params per transformer block."""
    norms: dict[int, float] = {}
    for k, block in enumerate(transformer_block_modules(model)):
        sq = 0.0
        for name, p in block.named_parameters():
            full_name = f"layers.{k}.{name}"
            init_p = init_state[full_name]
            sq += float(torch.sum((p.detach() - init_p) ** 2).item())
        norms[k] = sq ** 0.5
    return norms


def load_contributions(scan_dir: Path) -> dict[int, float]:
    data = json.loads((scan_dir / "contributions.json").read_text(encoding="utf-8"))
    return {int(k): float(v) for k, v in data["contributions"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="E8: ‖Δθ_k‖ vs C(k) (tiny TF, CPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--scan-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    scan_dir = args.scan_dir or (ROOT / "results" / f"tstl_tiny_tf_seed{args.seed}_L{args.layers}")
    contributions = load_contributions(scan_dir)

    cfg = TinyConfig(
        vocab_size=args.vocab, seq_len=args.seq_len, d_model=args.d_model, n_layers=args.layers
    )
    x_train, y_train = make_dataset(cfg, args.n_train, seed=args.seed)

    set_seed(args.seed)
    model = TinyCausalLM(cfg)
    init_state = copy.deepcopy(model.state_dict())

    unfreeze_all(model)
    train_model(
        model, x_train, y_train,
        steps=args.steps, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
    )
    deltas = block_delta_norms(model, init_state)

    ks = sorted(contributions)
    c_vals = [contributions[k] for k in ks]
    d_vals = [deltas[k] for k in ks]
    pear = pearson_corr(d_vals, c_vals)
    spear = spearman_corr(d_vals, c_vals)
    dmean = sum(d_vals) / len(d_vals)
    dspread = (max(d_vals) - min(d_vals)) / dmean if dmean else float("nan")

    out_dir = args.out_dir or (ROOT / "results" / f"tstl_e8_seed{args.seed}_L{args.layers}")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_contribution_heatmap(deltas, out_dir / "delta_norm_bar.png", title="‖Δθ_k‖ per layer")
    (out_dir / "e8.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "delta_norms": {str(k): deltas[k] for k in ks},
                "contributions": {str(k): contributions[k] for k in ks},
                "pearson_delta_vs_C": pear,
                "spearman_delta_vs_C": spear,
                "delta_relative_spread": dspread,
                "scan_dir": str(scan_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        f"# E8 (toy): ‖Δθ_k‖ vs C(k) — seed {args.seed}",
        "",
        f"- Pearson(‖Δθ‖, C): {pear:.3f}",
        f"- Spearman(‖Δθ‖, C): {spear:.3f}",
        f"- ‖Δθ‖ relative spread (max-min)/mean: {dspread:.3f}",
        "",
        "| k | ‖Δθ_k‖ | C(k) |",
        "|---|--------|------|",
    ]
    for k in ks:
        lines.append(f"| {k} | {deltas[k]:.3f} | {contributions[k]:.3f} |")
    lines.append("")
    lines.append(
        "Interpretation: TSTL §5 expects weak |Δθ|–C(k) correlation (contribution "
        "is not explained by how much a layer moves). Toy-scale, single seed."
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"seed {args.seed}: pearson={pear:.3f} spearman={spear:.3f} "
          f"Δspread={dspread:.3f} -> {out_dir}")


if __name__ == "__main__":
    main()
