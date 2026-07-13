"""Layer-selective training strategies (TSTL §4)."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from freeze import freeze_all_except_hidden, unfreeze_all
from mlp_model import RegressionMLP


def select_mid_layers(num_hidden: int, k: int) -> list[int]:
    """
    Heuristic Mid-k (paper §4.3): layers in [floor(L/2 - k/2), floor(L/2 + k/2)).

    L here is num_hidden (hidden layer count).
    """
    if k < 1 or k > num_hidden:
        raise ValueError("k must be in [1, num_hidden]")
    start = math.floor(num_hidden / 2 - k / 2)
    end = math.floor(num_hidden / 2 + k / 2)
    start = max(0, start)
    end = min(num_hidden, max(start + 1, end))
    return list(range(start, end))


def select_top_layers(contributions: dict[int, float], k: int) -> list[int]:
    """Only Bk: k hidden layers with highest C(k)."""
    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    return [idx for idx, _ in ranked[:k]]


def select_worst_layers(contributions: dict[int, float], k: int) -> list[int]:
    """Only Wk: k hidden layers with lowest C(k) (control)."""
    ranked = sorted(contributions.items(), key=lambda kv: kv[1])
    return [idx for idx, _ in ranked[:k]]


def configure_trainable_hidden(
    model: RegressionMLP,
    hidden_indices: Iterable[int] | None,
) -> None:
    """None = full training; otherwise only listed hidden layers (+ boundaries)."""
    if hidden_indices is None:
        unfreeze_all(model)
        return
    freeze_all_except_hidden(model, hidden_indices)


def train_regression(
    model: RegressionMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    steps: int = 2000,
    lr: float = 1e-3,
    hidden_indices: Iterable[int] | None = None,
    param_groups: list[dict] | None = None,
) -> list[float]:
    """
    Train with MSE loss. Returns loss history.

    param_groups: optional Adam param groups (for Boost Bk). If set, hidden_indices
    is ignored for optimizer setup but freeze should already be applied.
    """
    configure_trainable_hidden(model, hidden_indices)
    if param_groups is not None:
        opt = torch.optim.Adam(param_groups, lr=lr)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)

    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        pred = model(x_train)
        loss = nn.functional.mse_loss(pred, y_train)
        if not torch.isfinite(loss):
            raise RuntimeError("NaN loss during training")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def build_boost_param_groups(
    model: RegressionMLP,
    boost_indices: Iterable[int],
    base_lr: float,
    boost_lr_mult: float = 2.0,
) -> list[dict]:
    """Boost Bk: higher LR on selected hidden layers (paper §4.1)."""
    boost_set = set(boost_indices)
    boost_params: list[nn.Parameter] = []
    base_params: list[nn.Parameter] = []

    for i, layer in enumerate(model.hidden):
        for p in layer.parameters():
            if not p.requires_grad:
                continue
            if i in boost_set:
                boost_params.append(p)
            else:
                base_params.append(p)

    for m in (model.input_fc, model.output_fc):
        for p in m.parameters():
            if p.requires_grad:
                base_params.append(p)

    groups: list[dict] = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    if boost_params:
        groups.append({"params": boost_params, "lr": base_lr * boost_lr_mult})
    return groups


def train_full(
    model: RegressionMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    **kwargs,
) -> list[float]:
    return train_regression(model, x_train, y_train, hidden_indices=None, **kwargs)


def train_only_layers(
    model: RegressionMLP,
    layer_indices: Iterable[int],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    **kwargs,
) -> list[float]:
    return train_regression(model, x_train, y_train, hidden_indices=layer_indices, **kwargs)


def train_boost_layers(
    model: RegressionMLP,
    boost_indices: Iterable[int],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    boost_lr_mult: float = 2.0,
    lr: float = 1e-3,
    **kwargs,
) -> list[float]:
    unfreeze_all(model)
    groups = build_boost_param_groups(model, boost_indices, lr, boost_lr_mult)
    return train_regression(
        model, x_train, y_train, param_groups=groups, lr=lr, **kwargs
    )
