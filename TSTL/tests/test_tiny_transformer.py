"""Tests for the CPU tiny Transformer and its llm_freeze integration."""

from __future__ import annotations

import torch

from llm_freeze import freeze_all_except_layers, num_transformer_layers
from tiny_transformer import TinyCausalLM, TinyConfig, make_dataset, token_accuracy


def _cfg() -> TinyConfig:
    return TinyConfig(vocab_size=7, seq_len=8, d_model=16, n_heads=2, n_layers=4, d_ff=32)


def test_forward_shape():
    cfg = _cfg()
    m = TinyCausalLM(cfg)
    x, y = make_dataset(cfg, 5, seed=0)
    logits = m(x)
    assert logits.shape == (5, cfg.seq_len, cfg.vocab_size)
    assert y.shape == (5, cfg.seq_len)


def test_dataset_is_modular_running_sum():
    cfg = _cfg()
    x, y = make_dataset(cfg, 3, seed=1)
    expected = torch.cumsum(x, dim=1) % cfg.vocab_size
    assert torch.equal(y, expected)


def test_num_layers_matches_config():
    cfg = _cfg()
    m = TinyCausalLM(cfg)
    assert num_transformer_layers(m) == cfg.n_layers


def test_freeze_single_layer_marks_only_block_embed_head():
    cfg = _cfg()
    m = TinyCausalLM(cfg)
    freeze_all_except_layers(m, [2])

    for i, block in enumerate(m.layers):
        want = i == 2
        assert all(p.requires_grad == want for p in block.parameters())

    assert all(p.requires_grad for p in m.embed_tokens.parameters())
    assert all(p.requires_grad for p in m.lm_head.parameters())
    # Final norm is neither a block nor emb/head -> frozen in single-layer runs.
    assert all(not p.requires_grad for p in m.norm.parameters())


def test_token_accuracy_range():
    cfg = _cfg()
    m = TinyCausalLM(cfg)
    x, y = make_dataset(cfg, 32, seed=2)
    acc = token_accuracy(m, x, y)
    assert 0.0 <= acc <= 1.0
