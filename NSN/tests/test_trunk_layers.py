"""Tests for MLPTrunk layer-selective training API."""

import torch

from trunk import MLPTrunk


def test_linear_layers_count_for_depth_three():
    trunk = MLPTrunk(input_dim=2, feature_dim=4, hidden_dim=8, num_layers=3)
    assert len(trunk.linear_layers()) == 3


def test_set_trainable_linear_layers_only_middle():
    trunk = MLPTrunk(input_dim=2, feature_dim=4, hidden_dim=8, num_layers=3)
    trunk.set_trainable_linear_layers({1})
    layers = trunk.linear_layers()
    assert layers[0].weight.requires_grad is False
    assert layers[1].weight.requires_grad is True
    assert layers[2].weight.requires_grad is False


def test_freeze_unfreeze_all():
    trunk = MLPTrunk(input_dim=1, feature_dim=2, num_layers=2)
    trunk.freeze_all()
    assert not any(p.requires_grad for p in trunk.parameters())
    trunk.unfreeze_all()
    assert all(p.requires_grad for p in trunk.parameters())


def test_forward_unchanged_after_freeze():
    trunk = MLPTrunk(input_dim=2, feature_dim=3, hidden_dim=4, num_layers=2)
    x = torch.randn(5, 2)
    y1 = trunk(x)
    trunk.freeze_all()
    y2 = trunk(x)
    assert torch.allclose(y1, y2)
