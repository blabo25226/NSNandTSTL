"""Freeze test on a real (tiny) HF model built from config — no download, no GPU.

Validates freeze_all_except_layers on the actual Qwen2 layout the R1 pipeline
targets. Skips cleanly where transformers is not installed (e.g. bare CPU dev).
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from llm_freeze import (  # noqa: E402
    freeze_all_except_layers,
    num_transformer_layers,
    trainable_layer_indices,
)


def _tiny_qwen2():
    from transformers import AutoModelForCausalLM, Qwen2Config

    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return AutoModelForCausalLM.from_config(cfg)


def test_qwen2_layer_count():
    model = _tiny_qwen2()
    assert num_transformer_layers(model) == 4


def test_qwen2_single_layer_freeze():
    model = _tiny_qwen2()
    freeze_all_except_layers(model, [1])
    assert trainable_layer_indices(model) == [1]

    # Embedding + LM head stay trainable (paper θ_emb / θ_head convention).
    assert all(p.requires_grad for p in model.model.embed_tokens.parameters())
    assert all(p.requires_grad for p in model.lm_head.parameters())


def test_qwen2_full_unfreezes_all_blocks():
    model = _tiny_qwen2()
    freeze_all_except_layers(model, None)
    assert trainable_layer_indices(model) == [0, 1, 2, 3]
