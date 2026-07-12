"""Layer freezing utilities for MLP layer-selective training."""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn

from mlp_model import RegressionMLP


def hidden_layer_modules(model: RegressionMLP) -> list[nn.Linear]:
    """Return hidden Linear layers (TSTL 'layers' k = 0 .. L_h-1)."""
    return list(model.hidden)


def boundary_modules(model: RegressionMLP) -> list[nn.Module]:
    """Input and output layers (always trainable in single-layer runs)."""
    return [model.input_fc, model.output_fc]


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def freeze_all_except_hidden(
    model: RegressionMLP,
    hidden_indices: Iterable[int],
) -> None:
    """
    Freeze all hidden layers except those in hidden_indices.

    Input and output Linear layers remain trainable (analogous to θ_emb, θ_head).
    """
    idx_set = set(hidden_indices)
    freeze_all(model)
    for m in boundary_modules(model):
        _set_requires_grad(m, True)
    for i, layer in enumerate(hidden_layer_modules(model)):
        if i in idx_set:
            _set_requires_grad(layer, True)


def trainable_hidden_indices(model: RegressionMLP) -> list[int]:
    """Indices of hidden layers with any trainable parameter."""
    out: list[int] = []
    for i, layer in enumerate(hidden_layer_modules(model)):
        if any(p.requires_grad for p in layer.parameters()):
            out.append(i)
    return out
