"""Utilities: reproducibility and metrics."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Coefficient of determination R² (higher is better)."""
    y_true = y_true.detach().flatten()
    y_pred = y_pred.detach().flatten()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    if ss_tot.item() < 1e-12:
        return 1.0 if ss_res.item() < 1e-12 else 0.0
    return (1.0 - ss_res / ss_tot).item()
