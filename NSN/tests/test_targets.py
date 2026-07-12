"""Tests for noisy SR target sampling."""

import torch

from targets import DEFAULT_NOISE_STD_REL, TARGETS, _apply_label_noise


def test_apply_label_noise_zero():
    gen = torch.Generator().manual_seed(0)
    y = torch.tensor([1.0, 2.0, 3.0])
    out = _apply_label_noise(y, gen, 0.0)
    assert torch.allclose(out, y)


def test_apply_label_noise_nonzero():
    gen = torch.Generator().manual_seed(0)
    y = torch.tensor([1.0, 2.0, 3.0])
    out = _apply_label_noise(y, gen, 0.05)
    assert not torch.allclose(out, y)
    assert out.shape == y.shape


def test_sample_differs_with_and_without_noise():
    gen_clean = torch.Generator().manual_seed(1)
    gen_noisy = torch.Generator().manual_seed(1)
    target = TARGETS["square"]
    _, y_clean = target.sample(64, gen_clean, noise_std_rel=0.0)
    _, y_noisy = target.sample(64, gen_noisy, noise_std_rel=0.05)
    assert not torch.allclose(y_clean, y_noisy)
