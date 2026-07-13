"""Transformer layer freezing for TSTL LLM reproduction (Phase R1+)."""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def transformer_block_modules(model: nn.Module) -> list[nn.Module]:
    """
    Return one module per transformer block (TSTL layer k).

    Supports Llama/Qwen-style (`model.model.layers`) and GPT-2 (`transformer.h`).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError(
        "Unsupported model layout: expected model.model.layers, transformer.h, or layers"
    )


def always_trainable_modules(model: nn.Module) -> list[nn.Module]:
    """Embedding and LM head stay trainable in single-layer runs (paper §2.2)."""
    modules: list[nn.Module] = []
    for name in ("embed_tokens", "wte", "embeddings"):
        if hasattr(model, name):
            modules.append(getattr(model, name))
            break
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        modules.append(model.model.embed_tokens)
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        modules.append(model.transformer.wte)

    for name in ("lm_head",):
        if hasattr(model, name):
            modules.append(getattr(model, name))
    return modules


def freeze_all_except_layers(
    model: nn.Module,
    layer_indices: Iterable[int] | None,
) -> None:
    """
    Freeze all parameters except selected transformer blocks.

    ``layer_indices is None`` → train all blocks (full-parameter run).
    Input embedding and LM head remain trainable in all modes.
    """
    blocks = transformer_block_modules(model)
    idx_set = None if layer_indices is None else set(layer_indices)

    freeze_all(model)
    for m in always_trainable_modules(model):
        _set_requires_grad(m, True)

    if idx_set is None:
        for block in blocks:
            _set_requires_grad(block, True)
        return

    for i, block in enumerate(blocks):
        if i in idx_set:
            _set_requires_grad(block, True)


def trainable_layer_indices(model: nn.Module) -> list[int]:
    """Indices of transformer blocks with any trainable parameter."""
    out: list[int] = []
    for i, block in enumerate(transformer_block_modules(model)):
        if any(p.requires_grad for p in block.parameters()):
            out.append(i)
    return out


def num_transformer_layers(model: nn.Module) -> int:
    return len(transformer_block_modules(model))
