"""Tests for training strategies."""

import math

from strategies import select_mid_layers, select_top_layers, select_worst_layers


def test_select_mid_layers_five_hidden():
    # L=5, k=3 -> floor(2.5-1.5)=1, floor(2.5+1.5)=4 -> [1,2,3]
    assert select_mid_layers(5, 3) == [1, 2, 3]


def test_select_mid_layers_three_hidden():
    assert select_mid_layers(3, 1) == [1]


def test_select_top_and_worst():
    c = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.2}
    assert select_top_layers(c, 2) == [1, 2]
    assert select_worst_layers(c, 2) == [0, 3]
