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


def pearson_corr(a: list[float], b: list[float]) -> float:
    """Pearson correlation; NaN if either series has zero variance."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks; no scipy needed)."""

    def _rank(v: np.ndarray) -> np.ndarray:
        order = v.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)
        # Average ranks for ties.
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        return (sums / counts)[inv]

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2:
        return float("nan")
    return pearson_corr(list(_rank(x)), list(_rank(y)))
