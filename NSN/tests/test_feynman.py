"""Smoke tests for the Feynman equations and SR baselines."""

import torch

from baselines import EQLBaseline, MLPBaseline
from feynman import FEYNMAN_TARGETS


def test_feynman_targets_sample_shapes():
    assert len(FEYNMAN_TARGETS) >= 10
    gen = torch.Generator().manual_seed(0)
    for tid, target in FEYNMAN_TARGETS.items():
        x, y = target.sample(16, gen, noise_std_rel=0.0)
        assert x.shape == (16, target.input_dim), tid
        assert y.shape == (16,), tid
        assert torch.isfinite(y).all(), tid


def test_baselines_fit_predict_runs():
    torch.manual_seed(0)
    x_tr = torch.rand(64, 2) * 4 + 1
    y_tr = x_tr[:, 0] * x_tr[:, 1]
    x_te = torch.rand(32, 2) * 4 + 1
    for bl in (MLPBaseline(hidden=16, layers=2), EQLBaseline()):
        res = bl.fit_predict(x_tr, y_tr, x_te, steps=50)
        assert res.pred.shape == (32,)
        assert torch.isfinite(res.pred).all()
        assert res.size > 0
