"""CPU-only TSTL layer-contribution scan on a from-scratch tiny Transformer.

No GPU and no HuggingFace download. Reproduces the TSTL pipeline (base ->
full -> per-layer training -> C(k)) on an actual decoder Transformer using the
existing ``llm_freeze`` / ``llm_profile`` utilities.

Example:
    python scripts/run_tiny_transformer_scan.py --layers 6 --steps 400 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from llm_freeze import freeze_all_except_layers, unfreeze_all  # noqa: E402
from llm_profile import profile_layers_from_scores  # noqa: E402
from tiny_transformer import (  # noqa: E402
    TinyCausalLM,
    TinyConfig,
    make_dataset,
    token_accuracy,
)
from utils import set_seed  # noqa: E402


def train(
    model: TinyCausalLM,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> None:
    """Cross-entropy training over trainable params only (freeze applied upstream)."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=gen)
        xb, yb = x[idx], y[idx]
        opt.zero_grad()
        logits = model(xb)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), yb.reshape(-1)
        )
        if not torch.isfinite(loss):
            raise RuntimeError("NaN loss during training")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()


def fresh_model(cfg: TinyConfig, init_state: dict, seed: int) -> TinyCausalLM:
    set_seed(seed)
    m = TinyCausalLM(cfg)
    m.load_state_dict(copy.deepcopy(init_state))
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="TSTL tiny-Transformer layer scan (CPU)")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = TinyConfig(
        vocab_size=args.vocab,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.layers,
    )

    x_train, y_train = make_dataset(cfg, args.n_train, seed=args.seed)
    x_test, y_test = make_dataset(cfg, args.n_test, seed=args.seed + 1)

    set_seed(args.seed)
    init_model = TinyCausalLM(cfg)
    init_state = copy.deepcopy(init_model.state_dict())

    def train_from_init(layer_indices: list[int] | None) -> float:
        m = fresh_model(cfg, init_state, args.seed)
        if layer_indices is None:
            unfreeze_all(m)
        else:
            freeze_all_except_layers(m, layer_indices)
        train(
            m,
            x_train,
            y_train,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        return token_accuracy(m, x_test, y_test)

    s_base = token_accuracy(init_model, x_test, y_test)
    print(f"S_base (accuracy) = {s_base:.4f}  (chance ≈ {1/cfg.vocab_size:.4f})")

    s_full = train_from_init(None)
    print(f"S_full (accuracy) = {s_full:.4f}")

    s_per_layer: dict[int, float] = {}
    for k in range(cfg.n_layers):
        s_per_layer[k] = train_from_init([k])
        print(f"  layer {k}: S_k = {s_per_layer[k]:.4f}")

    out_dir = args.out_dir or (
        ROOT / "results" / f"tstl_tiny_tf_seed{args.seed}_L{args.layers}"
    )
    result = profile_layers_from_scores(
        s_base,
        s_full,
        s_per_layer,
        out_dir,
        config={
            "model": "tiny_transformer",
            "task": "modular_running_sum",
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len,
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
        },
    )
    best_k = max(result.contributions, key=lambda k: result.contributions[k])
    print(f"best layer k={best_k}  C={result.contributions[best_k]:.3f}")
    print(f"results -> {result.out_dir}")


if __name__ == "__main__":
    main()
