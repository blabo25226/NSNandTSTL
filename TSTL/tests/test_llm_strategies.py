"""Tests for llm_strategies selection helpers (no GPU / no trl)."""

import pytest

from llm_strategies import select_mid_layers, select_top_layers, strategy_layer_indices


def test_select_top_layers_by_contribution():
    contrib = {0: 0.2, 1: 0.9, 2: 0.5, 3: 1.1, 4: 0.1}
    assert select_top_layers(contrib, 2) == [1, 3]


def test_select_top_layers_drops_nan():
    contrib = {0: float("nan"), 1: 0.9, 2: 0.5}
    assert select_top_layers(contrib, 2) == [1, 2]


def test_select_mid_layers_even_and_odd():
    # L=28, k=5 -> middle 5 layers 11..15 (paper §4.3 example).
    assert select_mid_layers(28, 5) == [11, 12, 13, 14, 15]
    assert select_mid_layers(6, 2) == [2, 3]


def test_select_mid_layers_bounds():
    with pytest.raises(ValueError):
        select_mid_layers(4, 5)


def test_strategy_layer_indices():
    contrib = {0: 0.2, 1: 0.9, 2: 0.5, 3: 1.1}
    assert strategy_layer_indices("full", 4, 2) is None
    assert strategy_layer_indices("only_bk", 4, 2, contributions=contrib) == [1, 3]
    assert strategy_layer_indices("mid_k", 4, 2) == [1, 2]


def test_only_bk_requires_contributions():
    with pytest.raises(ValueError):
        strategy_layer_indices("only_bk", 4, 2)
