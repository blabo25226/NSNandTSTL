"""Tests for R1 presets."""

from r1_presets import QUICK, estimate_grpo_forward_passes, layer_indices_to_scan


def test_layer_stride():
    assert layer_indices_to_scan(24, 4) == [0, 4, 8, 12, 16, 20]


def test_estimate_quick_smaller_than_full():
    quick = estimate_grpo_forward_passes(QUICK, num_layers_scanned=6)
    full = estimate_grpo_forward_passes(QUICK, num_layers_scanned=24)
    assert quick < full
