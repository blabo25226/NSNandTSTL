"""Tests for layer freezing."""

import torch

from freeze import freeze_all_except_hidden, hidden_layer_modules
from mlp_model import RegressionMLP


def test_freeze_all_except_single_hidden_layer():
    model = RegressionMLP(input_dim=8, hidden_dim=16, num_hidden_layers=3)
    freeze_all_except_hidden(model, [1])

    assert model.input_fc.weight.requires_grad
    assert model.output_fc.weight.requires_grad
    assert not model.hidden[0].weight.requires_grad
    assert model.hidden[1].weight.requires_grad
    assert not model.hidden[2].weight.requires_grad


def test_gradient_only_on_trainable_hidden():
    model = RegressionMLP(input_dim=4, hidden_dim=8, num_hidden_layers=3)
    freeze_all_except_hidden(model, [1])
    x = torch.randn(5, 4)
    y = torch.randn(5)
    loss = torch.mean((model(x) - y) ** 2)
    loss.backward()

    assert model.hidden[0].weight.grad is None
    assert model.hidden[1].weight.grad is not None
    assert model.hidden[2].weight.grad is None
    assert model.input_fc.weight.grad is not None
    assert model.output_fc.weight.grad is not None


def test_hidden_layer_count():
    model = RegressionMLP(num_hidden_layers=5)
    assert len(hidden_layer_modules(model)) == 5
