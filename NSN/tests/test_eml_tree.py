"""Tests for EML tree head."""

import torch

from eml import eml
from eml_tree import EMLTreeHead
from utils import as_complex


def test_eml_depth2_manual():
    """
    Reproduce Odrzywołek eml_depth2: e - ln(exp(y) - ln(x)).

    Tree (D=2): eml(eml(0,1), eml(y,x)) with z = [x, y].
    """
    head = EMLTreeHead(feature_dim=2, depth=2)
    head.set_leaf_constant(0, 0.0)
    head.set_leaf_constant(1, 1.0)
    head.set_leaf_linear(2, [0.0, 1.0])
    head.set_leaf_linear(3, [1.0, 0.0])

    x = torch.linspace(0.5, 2.0, 8)
    y = torch.linspace(0.3, 1.2, 8)
    z = torch.stack([x, y], dim=1)

    out = head(z)

    xc, yc = as_complex(x), as_complex(y)
    one = torch.ones_like(xc)
    n1 = eml(yc, xc)
    expected = eml(one, n1).real

    assert torch.allclose(out, expected, atol=1e-4)


def test_eml_depth1_exp_manual():
    """D=1 tree: eml(z0, 1) = exp(z0) with single feature z0."""
    head = EMLTreeHead(feature_dim=1, depth=1)
    head.set_leaf_linear(0, [1.0])
    head.set_leaf_constant(1, 1.0)

    z = torch.linspace(-1.5, 1.5, 6).unsqueeze(1)
    out = head(z)
    expected = torch.exp(z.squeeze())
    assert torch.allclose(out, expected, atol=1e-4)


def test_tree_shape_counts():
    for depth in (1, 2, 3, 4):
        head = EMLTreeHead(feature_dim=3, depth=depth)
        assert head.num_leaves == 2**depth
        assert head.num_internal == 2**depth - 1


def test_forward_scalar_batch():
    head = EMLTreeHead(feature_dim=2, depth=2)
    z = torch.randn(5, 2)
    out = head(z)
    assert out.shape == (5,)


def test_forward_single_vector():
    head = EMLTreeHead(feature_dim=2, depth=2)
    z = torch.randn(2)
    out = head(z)
    assert out.shape == tuple()


def test_forward_no_nan():
    torch.manual_seed(1)
    head = EMLTreeHead(feature_dim=4, depth=3)
    z = torch.randn(8, 4)
    out = head(z)
    assert torch.isfinite(out).all()


def test_f_prev_parent_mode_changes_output():
    """Parent-feedback mode injects gamma*f_prev, differing from zero mode."""
    torch.manual_seed(3)
    zero = EMLTreeHead(feature_dim=3, depth=2, f_prev_mode="zero")
    parent = EMLTreeHead(feature_dim=3, depth=2, f_prev_mode="parent")
    parent.load_state_dict(zero.state_dict())  # identical weights
    # Force non-trivial gamma weight so the f_prev term matters.
    with torch.no_grad():
        zero.leaf_logits.fill_(0.0)
        parent.leaf_logits.fill_(0.0)
    z = torch.randn(6, 3)
    out_zero = zero(z)
    out_parent = parent(z)
    assert out_zero.shape == out_parent.shape
    assert torch.isfinite(out_parent).all()
    assert not torch.allclose(out_zero, out_parent, atol=1e-4)


def test_f_prev_mode_validation():
    import pytest

    with pytest.raises(ValueError):
        EMLTreeHead(feature_dim=2, depth=2, f_prev_mode="bogus")
