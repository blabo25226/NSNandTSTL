"""Tests for Odrzywołek canonical EML theory."""

import torch

from odrzywolek import (
    build_depth1_exp_tree,
    build_depth2_ln_chain_tree,
    canonical_mul_positive,
    verify_canonical_identities,
)


def test_canonical_identities():
    errs = verify_canonical_identities()
    assert errs["ok"] is True
    assert errs["exp"] < 1e-4
    assert errs["ln"] < 1e-3
    assert errs["mul"] < 1e-3


def test_depth1_exp_tree_matches_exp():
    head = build_depth1_exp_tree(feature_dim=1)
    z = torch.linspace(-1.0, 1.0, 8).unsqueeze(1)
    out = head(z)
    assert torch.allclose(out, torch.exp(z.squeeze()), atol=1e-4)


def test_depth2_ln_chain_matches_reference():
    head = build_depth2_ln_chain_tree(feature_dim=2)
    x = torch.linspace(0.5, 2.0, 8)
    y = torch.linspace(0.3, 1.2, 8)
    z = torch.stack([x, y], dim=1)
    out = head(z)
    from eml import eml
    from utils import as_complex

    xc, yc = as_complex(x), as_complex(y)
    one = torch.ones_like(xc)
    expected = eml(one, eml(yc, xc)).real
    assert torch.allclose(out, expected, atol=1e-4)


def test_canonical_mul_positive():
    a = torch.tensor([2.0, 3.0])
    b = torch.tensor([4.0, 5.0])
    out = canonical_mul_positive(a, b)
    assert torch.allclose(out, a * b, atol=1e-3)
