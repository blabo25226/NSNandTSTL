"""Tests for EML primitive."""

import math

import torch

from eml import EMLNode, eml, stable_exp, stable_log
from utils import as_complex


def test_eml_exp_identity():
    x = torch.linspace(-2.0, 2.0, 7)
    z = as_complex(x)
    one = torch.ones_like(z)
    out = eml(z, one)
    expected = stable_exp(z)
    assert torch.allclose(out.real, expected.real, atol=1e-5)
    assert torch.allclose(out.imag, expected.imag, atol=1e-5)


def test_eml_log_identity():
    # Positive real z only (principal branch).
    z = torch.linspace(0.5, 3.0, 6)
    zc = as_complex(z)
    one = torch.ones_like(zc)
    inner_a = eml(one, zc)
    inner_b = eml(inner_a, one)
    out = eml(one, inner_b)
    assert torch.allclose(out.real, torch.log(z), atol=1e-4)


def test_eml_node_forward():
    node = EMLNode()
    x = as_complex(torch.tensor([0.0, 1.0]))
    y = as_complex(torch.tensor([1.0, math.e]))
    out = node(x, y)
    assert out.shape == x.shape


def test_eml_gradients_exist():
    x = torch.tensor([0.5], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)
    xc, yc = as_complex(x), as_complex(y)
    out = eml(xc, yc).real.sum()
    out.backward()
    assert x.grad is not None and y.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()


def test_random_eml_no_nan():
    torch.manual_seed(0)
    for _ in range(100):
        x = as_complex(torch.randn(4) * 0.5)
        y = as_complex(torch.randn(4).abs() + 0.5)
        out = eml(x, y)
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()


def test_eml_analytic_jacobian_real():
    """Match paper Eq. (15): d/dx eml = exp(x), d/dy eml = -1/y (real axis)."""
    a = torch.tensor([0.5], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)
    out = eml(as_complex(a), as_complex(b)).real.sum()
    out.backward()
    assert torch.allclose(a.grad, torch.exp(a.detach()), atol=1e-4)
    assert torch.allclose(b.grad, -1.0 / b.detach(), atol=1e-4)
