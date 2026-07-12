"""Tests for leaf softmax modes."""

import torch

from eml_tree import EMLTreeHead
from leaf_softmax import LeafSoftmaxMode, leaf_weights, sample_gumbel


def test_softmax_sums_to_one():
    logits = torch.randn(4, 3)
    w = leaf_weights(logits, 1.0, LeafSoftmaxMode.SOFTMAX, training=True)
    assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_gumbel_differs_from_plain_in_training():
    torch.manual_seed(0)
    logits = torch.zeros(2, 3)
    w_soft = leaf_weights(logits, 1.0, LeafSoftmaxMode.SOFTMAX, training=True)
    w_gum = leaf_weights(logits, 1.0, LeafSoftmaxMode.GUMBEL, training=True, generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(w_soft, w_gum)


def test_gumbel_eval_uses_softmax():
    logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    w_train = leaf_weights(logits, 1.0, LeafSoftmaxMode.GUMBEL, training=True, generator=torch.Generator().manual_seed(0))
    w_eval = leaf_weights(logits, 1.0, LeafSoftmaxMode.GUMBEL, training=False)
    w_soft = leaf_weights(logits, 1.0, LeafSoftmaxMode.SOFTMAX, training=False)
    assert torch.allclose(w_eval, w_soft)
    assert w_train.shape == w_eval.shape


def test_head_respects_leaf_softmax_mode():
    head = EMLTreeHead(feature_dim=2, depth=1, leaf_softmax_mode=LeafSoftmaxMode.GUMBEL)
    assert head.leaf_softmax_mode == LeafSoftmaxMode.GUMBEL
    z = torch.randn(3, 2)
    out = head(z)
    assert out.shape == (3,)


def test_sample_gumbel_finite():
    g = sample_gumbel((10, 3), device=torch.device("cpu"), dtype=torch.float32)
    assert torch.isfinite(g).all()
