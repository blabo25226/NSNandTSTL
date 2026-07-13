"""Tests for layer contribution C(k)."""

import math
from pathlib import Path

from layer_contribution import (
    compute_contributions,
    layer_contribution,
    plot_contribution_heatmap,
    score_from_r2,
)
import torch


def test_layer_contribution_values():
    assert layer_contribution(0.0, 1.0, 1.0) == 1.0
    assert layer_contribution(0.0, 1.0, 0.5) == 0.5
    assert layer_contribution(0.0, 1.0, 1.2) == 1.2
    assert math.isnan(layer_contribution(1.0, 1.0, 1.5))


def test_compute_contributions():
    s = {0: 0.5, 1: 1.0, 2: 0.2}
    c = compute_contributions(0.0, 1.0, s)
    assert c[1] == 1.0
    assert c[0] == 0.5


def test_score_from_r2_perfect():
    y = torch.tensor([1.0, 2.0, 3.0])
    assert abs(score_from_r2(y, y) - 1.0) < 1e-6


def test_plot_contribution_heatmap(tmp_path):
    plot_contribution_heatmap({0: 0.2, 1: 1.1, 2: 0.4}, tmp_path / "c.png")
    assert (tmp_path / "c.png").is_file()
