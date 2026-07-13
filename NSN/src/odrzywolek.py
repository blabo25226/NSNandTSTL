"""Odrzywołek EML theory: canonical constructions and identity checks."""

from __future__ import annotations

import math

import torch

from eml import eml
from eml_tree import EMLTreeHead
from utils import as_complex


def canonical_exp(x: torch.Tensor) -> torch.Tensor:
    """exp(x) = eml(x, 1)."""
    z = as_complex(x)
    one = torch.ones_like(z)
    return eml(z, one)


def canonical_ln(z: torch.Tensor) -> torch.Tensor:
    """ln(z) = eml(1, eml(eml(1, z), 1)) on the principal branch."""
    zc = as_complex(z)
    one = torch.ones_like(zc)
    inner_a = eml(one, zc)
    inner_b = eml(inner_a, one)
    return eml(one, inner_b)


def canonical_euler() -> torch.Tensor:
    """e = eml(1, 1)."""
    one = torch.complex(torch.tensor(1.0), torch.tensor(0.0))
    return eml(one, one)


def eml_one_minus_ln(x: torch.Tensor) -> torch.Tensor:
    """eml(0, x) = 1 - ln(x) for positive real x."""
    zero = torch.zeros_like(as_complex(x))
    return eml(zero, as_complex(x))


def log_from_eml(x: torch.Tensor) -> torch.Tensor:
    """Recover ln(x) from eml(0, x) on positive reals: ln(x) = 1 - Re[eml(0,x)]."""
    return (1.0 - eml_one_minus_ln(x).real).to(dtype=x.dtype)


def canonical_mul_positive(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x * y = exp(ln(x) + ln(y)) via canonical EML identities (x, y > 0)."""
    lnx = log_from_eml(x)
    lny = log_from_eml(y)
    return canonical_exp(lnx + lny).real.to(dtype=x.dtype)


def build_depth1_exp_tree(feature_dim: int = 1) -> EMLTreeHead:
    """Reference tree: eml(z0, 1) = exp(z0) with z0 = 1 * z."""
    head = EMLTreeHead(feature_dim=feature_dim, depth=1)
    head.set_leaf_linear(0, [1.0] + [0.0] * (feature_dim - 1))
    head.set_leaf_constant(1, 1.0)
    return head


def build_depth2_ln_chain_tree(feature_dim: int = 2) -> EMLTreeHead:
    """
    Reference tree reproducing eml_depth2 pattern:
    Re[eml(1, eml(y, x))] with leaves wired to z = [x, y].
    """
    head = EMLTreeHead(feature_dim=feature_dim, depth=2)
    head.set_leaf_constant(0, 0.0)
    head.set_leaf_constant(1, 1.0)
    head.set_leaf_linear(2, [0.0, 1.0] + [0.0] * (feature_dim - 2))
    head.set_leaf_linear(3, [1.0, 0.0] + [0.0] * (feature_dim - 2))
    return head


def verify_canonical_identities(
    *,
    atol_exp: float = 1e-4,
    atol_ln: float = 1e-3,
    atol_mul: float = 1e-3,
) -> dict[str, float]:
    """Numeric checks for core Odrzywołek identities. Returns max errors."""
    x = torch.linspace(-1.5, 1.5, 7)
    z = torch.linspace(0.5, 2.5, 7)
    pos_a = torch.linspace(0.5, 2.0, 6)
    pos_b = torch.linspace(0.4, 1.8, 6)

    exp_via = canonical_exp(x).real
    err_exp = (exp_via - torch.exp(x)).abs().max().item()

    ln_via = canonical_ln(z).real
    err_ln = (ln_via - torch.log(z)).abs().max().item()

    e_val = canonical_euler().real.item()
    err_e = abs(e_val - math.e)

    mul_via = canonical_mul_positive(pos_a, pos_b)
    err_mul = (mul_via - pos_a * pos_b).abs().max().item()

    return {
        "exp": err_exp,
        "ln": err_ln,
        "euler": err_e,
        "mul": err_mul,
        "ok": err_exp < atol_exp and err_ln < atol_ln and err_mul < atol_mul,
    }
