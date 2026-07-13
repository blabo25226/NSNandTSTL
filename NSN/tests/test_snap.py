"""Tests for symbolic snap and expression export."""

import torch

from eml_tree import EMLTreeHead
from snap import (
    apply_snap_to_logits,
    export_symbolic_expression,
    snap_leaf_weights,
    temperature_schedule,
)
from utils import set_seed


def test_temperature_schedule_endpoints():
    assert temperature_schedule(0, 100) == 1.0
    assert abs(temperature_schedule(100, 100) - 0.1) < 1e-9


def test_snap_leaf_weights_one_hot():
    head = EMLTreeHead(feature_dim=2, depth=2)
    head.leaf_logits.data[0] = torch.tensor([5.0, -1.0, -1.0])
    hard = snap_leaf_weights(head)
    assert hard[0].tolist() == [1.0, 0.0, 0.0]


def test_export_symbolic_eml_depth2():
    head = EMLTreeHead(feature_dim=2, depth=2)
    head.set_leaf_constant(0, 0.0)
    head.set_leaf_constant(1, 1.0)
    head.set_leaf_linear(2, [0.0, 1.0])
    head.set_leaf_linear(3, [1.0, 0.0])
    apply_snap_to_logits(head)

    expr = export_symbolic_expression(head, ["x", "y"])
    assert expr.startswith("Re[eml(")
    assert "eml(" in expr
    assert "x" in expr and "y" in expr


def test_snap_after_training_produces_expression():
    set_seed(0)
    head = EMLTreeHead(feature_dim=1, depth=1)
    head.set_leaf_linear(0, [1.0])
    head.set_leaf_constant(1, 1.0)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)

    z = torch.linspace(-1, 1, 32).unsqueeze(1)
    target = torch.exp(z.squeeze())

    for _ in range(200):
        opt.zero_grad()
        pred = head(z)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()

    apply_snap_to_logits(head)
    expr = export_symbolic_expression(head, ["x"])
    assert len(expr) > 10
    assert "eml(" in expr
