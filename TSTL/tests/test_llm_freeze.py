"""Tests for llm_freeze (CPU, no HuggingFace download)."""

from __future__ import annotations

import torch.nn as nn

from llm_freeze import (
    freeze_all_except_layers,
    num_transformer_layers,
    trainable_layer_indices,
    transformer_block_modules,
)


class _Inner(nn.Module):
    def __init__(self, n: int, dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n)])


class MockCausalLM(nn.Module):
    """Minimal Qwen/Llama-style layout for freeze tests."""

    def __init__(self, n_layers: int = 4, dim: int = 8, vocab: int = 32) -> None:
        super().__init__()
        self.model = _Inner(n_layers, dim)
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        h = self.embed_tokens(x)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(h)


def test_transformer_block_count():
    m = MockCausalLM(n_layers=6)
    assert num_transformer_layers(m) == 6
    assert len(transformer_block_modules(m)) == 6


def test_single_layer_only_k_trainable():
    m = MockCausalLM(n_layers=4)
    freeze_all_except_layers(m, [2])
    assert trainable_layer_indices(m) == [2]
    assert m.embed_tokens.weight.requires_grad
    assert m.lm_head.weight.requires_grad
    assert not m.model.layers[0].weight.requires_grad
    assert not m.model.layers[1].weight.requires_grad
    assert m.model.layers[2].weight.requires_grad
    assert not m.model.layers[3].weight.requires_grad


def test_full_run_trains_all_blocks():
    m = MockCausalLM(n_layers=3)
    freeze_all_except_layers(m, None)
    assert trainable_layer_indices(m) == [0, 1, 2]
