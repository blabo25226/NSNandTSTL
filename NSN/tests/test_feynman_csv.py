"""Tests for Feynman CSV loader and benchmark helpers."""

import math

import torch

from feynman_csv import (
    RMSE_THRESHOLD_FIXED,
    classify_rmse,
    default_csv_path,
    generate_feynman_samples,
    load_feynman_csv,
    noise_aware_rmse_ok_threshold,
    rmse,
    to_sr_target,
)


def test_load_csv_contains_i12_1():
    eqs = load_feynman_csv(default_csv_path())
    ids = {e.filename for e in eqs}
    assert "I.12.1" in ids
    assert len(eqs) >= 90


def test_generate_samples_shape_and_finite():
    eqs = load_feynman_csv(default_csv_path())
    eq = next(e for e in eqs if e.filename == "I.12.1")
    out = generate_feynman_samples(eq, 64, seed=0, noise_std_rel=0.0)
    assert out is not None
    x, y_noisy, y_clean = out
    assert x.shape == (64, 2)
    assert y_noisy.shape == (64,)
    assert y_clean.shape == (64,)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y_clean).all()


def test_gaussian_noise_changes_labels():
    eqs = load_feynman_csv(default_csv_path())
    eq = next(e for e in eqs if e.filename == "I.12.1")
    out0 = generate_feynman_samples(eq, 128, seed=1, noise_std_rel=0.0)
    out1 = generate_feynman_samples(eq, 128, seed=1, noise_std_rel=0.01)
    assert out0 is not None and out1 is not None
    _, _, y_clean0 = out0
    _, y_noisy1, y_clean1 = out1
    assert torch.allclose(y_clean0, y_clean1)
    assert not torch.allclose(y_noisy1, y_clean1)


def test_to_sr_target_sample():
    eqs = load_feynman_csv(default_csv_path())
    eq = next(e for e in eqs if e.filename == "I.12.1")
    target = to_sr_target(eq, data_seed=7)
    assert target.input_dim == 2
    assert target.phase == 5
    gen = torch.Generator().manual_seed(7)
    x, y = target.sample(32, gen, noise_std_rel=0.01)
    assert x.shape == (32, 2)
    assert y.shape == (32,)


def test_rmse_and_thresholds():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = y.clone()
    assert rmse(y, pred) < 1e-12

    thr = noise_aware_rmse_ok_threshold(y, noise_std_rel=0.01)
    assert thr >= RMSE_THRESHOLD_FIXED
    assert thr > 0.01

    assert classify_rmse(1e-5, ok_thr=1e-4) == "ok"
    assert classify_rmse(5e-3, ok_thr=1e-4, partial_thr=1e-2) == "partial"
    assert classify_rmse(0.5, ok_thr=1e-4) == "failed"
    assert classify_rmse(float("inf"), ok_thr=1e-4) == "failed"
