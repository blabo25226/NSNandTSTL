"""Training loop for DNN-EML symbolic-regression experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from model import DNNEML
from snap import apply_hardening, apply_snap_to_logits, export_symbolic_expression
from targets import DEFAULT_NOISE_STD_REL, SRTarget
from utils import set_seed


@dataclass
class TrainConfig:
    feature_dim: int = 6
    head_depth: int = 2
    hidden_dim: int = 128
    num_layers: int = 3
    steps: int = 5000
    lr: float = 2e-3
    seed: int = 42
    n_train: int = 256
    noise_std_rel: float = DEFAULT_NOISE_STD_REL
    weight_decay: float = 1e-4


@dataclass
class TrainResult:
    target_id: str
    final_mse: float
    initial_mse: float
    best_val_mse: float
    expression_eml: str
    steps: int
    train_seconds: float
    stopped_step: int


def _config_for_target(target: SRTarget, base: TrainConfig) -> TrainConfig:
    cfg = TrainConfig(**vars(base))
    if target.id == "sin_plus":
        cfg.feature_dim = 8
        cfg.head_depth = 2
        cfg.steps = 6000
        cfg.lr = 3e-3
        cfg.hidden_dim = 128
    elif target.id == "square":
        cfg.feature_dim = 4
        cfg.head_depth = 2
        cfg.steps = 5000
        cfg.lr = 1e-3
    elif target.id == "exp":
        cfg.feature_dim = 4
        cfg.head_depth = 1
        cfg.steps = 4000
        cfg.lr = 2e-3
    elif target.id == "sin":
        cfg.feature_dim = 6
        cfg.head_depth = 2
        cfg.steps = 6000
        cfg.lr = 2e-3
    elif target.id == "feynman_I9_inv_square":
        cfg.feature_dim = 4
        cfg.head_depth = 2
        cfg.steps = 5000
        cfg.lr = 2e-3
    elif target.id.startswith("feynman"):
        cfg.feature_dim = 6
        cfg.head_depth = 2
        cfg.steps = 5000
        cfg.lr = 2e-3
    elif target.input_dim == 1:
        cfg.feature_dim = 4
        cfg.head_depth = 2
        cfg.steps = 5000
        cfg.lr = 2e-3
    else:
        cfg.feature_dim = 6
        cfg.head_depth = 2
        cfg.steps = 5000
        cfg.lr = 2e-3
    return cfg


def train_target(target: SRTarget, base_config: TrainConfig | None = None) -> tuple[DNNEML, TrainResult]:
    import time

    t0 = time.perf_counter()
    cfg = _config_for_target(target, base_config or TrainConfig())
    set_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    x, y = target.sample(cfg.n_train, gen, cfg.noise_std_rel)

    model = DNNEML.build(
        input_dim=target.input_dim,
        feature_dim=cfg.feature_dim,
        head_depth=cfg.head_depth,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    losses: list[float] = []
    for step in range(1, cfg.steps + 1):
        apply_hardening(model.head, step, cfg.steps)
        opt.zero_grad()
        loss = model.mse_loss(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN loss for {target.id} at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    soft_state = copy.deepcopy(model.state_dict())
    apply_snap_to_logits(model.head)
    z_names = [f"z{i}" for i in range(model.head.feature_dim)]
    expr = export_symbolic_expression(model.head, z_names)
    # Restore soft weights: hard snap helps symbolic export but hurts holdout accuracy.
    model.load_state_dict(soft_state)

    elapsed = time.perf_counter() - t0
    result = TrainResult(
        target_id=target.id,
        final_mse=losses[-1],
        initial_mse=losses[0],
        best_val_mse=losses[-1],
        expression_eml=expr,
        steps=cfg.steps,
        train_seconds=elapsed,
        stopped_step=cfg.steps,
    )
    return model, result
