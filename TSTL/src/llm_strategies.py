"""Layer-selective GRPO strategies for the LLM reproduction (paper §4).

Thin wrappers that map a strategy to a set of transformer-layer indices and run
GRPO training on only those layers (embedding + LM head stay trainable, see
``llm_freeze``). Selection helpers are pure and unit-testable on CPU; the
training call requires GPU + transformers + trl (Colab / GPU PC).

Implemented: **Only Bk** (top-k by C(k)) and **Mid-k** (positional middle).
**Boost Bk** needs per-layer-group learning rates, which the single-optimizer
``trl.GRPOTrainer`` does not expose directly; it is deferred to R2 with a custom
optimizer/callback. See ``boost_layer_lr_groups`` for the intended grouping.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def select_top_layers(contributions: dict[int, float], k: int) -> list[int]:
    """Only Bk: the k layers with the highest (finite) C(k)."""
    finite = {i: c for i, c in contributions.items() if c == c}  # drop NaN
    ranked = sorted(finite.items(), key=lambda kv: kv[1], reverse=True)
    return sorted(idx for idx, _ in ranked[:k])


def select_mid_layers(num_layers: int, k: int) -> list[int]:
    """
    Mid-k (paper §4.3): the k contiguous layers centered on the network middle,
    ``[floor(L/2 - k/2), floor(L/2 + k/2))``.
    """
    if k < 1 or k > num_layers:
        raise ValueError("k must be in [1, num_layers]")
    start = max(0, math.floor(num_layers / 2 - k / 2))
    end = min(num_layers, max(start + 1, math.floor(num_layers / 2 + k / 2)))
    start = max(0, end - k)
    return list(range(start, end))


def strategy_layer_indices(
    strategy: str,
    num_layers: int,
    k: int,
    *,
    contributions: dict[int, float] | None = None,
) -> list[int] | None:
    """
    Resolve a strategy name to trainable layer indices.

    - ``"full"`` -> None (train every layer)
    - ``"only_bk"`` -> top-k by contributions (requires ``contributions``)
    - ``"mid_k"`` -> positional middle k
    """
    if strategy == "full":
        return None
    if strategy == "only_bk":
        if contributions is None:
            raise ValueError("only_bk needs contributions from a layer scan")
        return select_top_layers(contributions, k)
    if strategy == "mid_k":
        return select_mid_layers(num_layers, k)
    raise ValueError(f"unknown strategy: {strategy!r}")


def boost_layer_lr_groups(
    model: Any,
    boost_indices: list[int],
    base_lr: float,
    boost_lr_mult: float = 2.0,
) -> list[dict]:
    """
    Optimizer param groups for Boost Bk (top layers at ``base_lr * mult``).

    Provided for a future custom-optimizer path; ``trl.GRPOTrainer`` builds its
    own optimizer, so this is not wired into ``run_grpo_train`` yet.
    """
    from llm_freeze import transformer_block_modules

    boost_set = set(boost_indices)
    boost_ids: set[int] = set()
    boost_params, base_params = [], []
    for i, block in enumerate(transformer_block_modules(model)):
        if i not in boost_set:
            continue
        for p in block.parameters():
            if p.requires_grad:
                boost_params.append(p)
                boost_ids.add(id(p))
    # Everything else trainable (other blocks + embedding/head) at the base LR.
    for p in model.parameters():
        if p.requires_grad and id(p) not in boost_ids:
            base_params.append(p)

    groups: list[dict] = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    if boost_params:
        groups.append({"params": boost_params, "lr": base_lr * boost_lr_mult})
    return groups


def run_layer_strategy(
    model: Any,
    dataset: Any,
    config: Any,
    layer_indices: list[int] | None,
    *,
    output_dir: Path | None = None,
) -> Path:
    """
    Train ``model`` under a strategy by setting ``config.train_layer_indices``.

    ``config`` is a ``GrpoRunConfig``. Requires GPU + trl at call time.
    """
    from llm_grpo import run_grpo_train

    config.train_layer_indices = layer_indices
    if output_dir is not None:
        config.output_dir = output_dir
    return run_grpo_train(model, dataset, config)
