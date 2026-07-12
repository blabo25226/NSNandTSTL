"""EML operator: eml(x, y) = exp(x) - ln(y)."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils import as_complex


DEFAULT_CLAMP = 20.0
DEFAULT_LOG_EPS = 1e-12


def stable_exp(z: torch.Tensor, clamp: float = DEFAULT_CLAMP) -> torch.Tensor:
    """Complex exp with clamped real/imag parts to limit overflow."""
    z = as_complex(z)
    real = torch.clamp(z.real, -clamp, clamp)
    imag = torch.clamp(z.imag, -clamp, clamp)
    return torch.exp(torch.complex(real, imag))


def stable_log(z: torch.Tensor, eps: float = DEFAULT_LOG_EPS) -> torch.Tensor:
    """Principal-branch complex log; epsilon avoids log(0)."""
    z = as_complex(z)
    # Shift away from exact zero while preserving differentiability.
    z_safe = z + torch.complex(
        torch.full_like(z.real, eps),
        torch.zeros_like(z.imag),
    )
    return torch.log(z_safe)


def eml(x: torch.Tensor, y: torch.Tensor, clamp: float = DEFAULT_CLAMP) -> torch.Tensor:
    """Exp-Minus-Log: eml(x, y) = exp(x) - ln(y)."""
    return stable_exp(x, clamp=clamp) - stable_log(y)


class EMLNode(nn.Module):
    """Differentiable EML binary node."""

    def __init__(self, clamp: float = DEFAULT_CLAMP) -> None:
        super().__init__()
        self.clamp = clamp

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return eml(x, y, clamp=self.clamp)
