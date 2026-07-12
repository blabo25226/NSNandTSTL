"""Symbolic-regression benchmark targets (elementary & Feynman-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

# Label noise: y_noisy = y_true + noise_std_rel * std(y_true) * N(0,1)
DEFAULT_NOISE_STD_REL = 0.01


@dataclass(frozen=True)
class SRTarget:
    id: str
    formula: str
    input_dim: int
    monotonic: bool
    phase: int
    sample: Callable[..., tuple[torch.Tensor, torch.Tensor]]


def _apply_label_noise(
    y: torch.Tensor,
    gen: torch.Generator,
    noise_std_rel: float,
) -> torch.Tensor:
    """Additive Gaussian noise proportional to the target's standard deviation."""
    if noise_std_rel <= 0:
        return y
    scale = y.std()
    if scale.item() < 1e-12:
        scale = torch.tensor(1.0, dtype=y.dtype)
    noise = torch.randn(y.shape, generator=gen, dtype=y.dtype) * noise_std_rel * scale
    return y + noise


def _linspace_input(n: int, dim: int, lo: float, hi: float, gen: torch.Generator) -> torch.Tensor:
    if dim == 1:
        return torch.linspace(lo, hi, n).unsqueeze(1)
    return torch.rand(n, dim, generator=gen) * (hi - lo) + lo


def _make_square(n: int, gen: torch.Generator, noise_std_rel: float = DEFAULT_NOISE_STD_REL) -> tuple[torch.Tensor, torch.Tensor]:
    x = _linspace_input(n, 1, -2.0, 2.0, gen)
    y = _apply_label_noise(x.squeeze() ** 2, gen, noise_std_rel)
    return x, y


def _make_product(n: int, gen: torch.Generator, noise_std_rel: float = DEFAULT_NOISE_STD_REL) -> tuple[torch.Tensor, torch.Tensor]:
    x = _linspace_input(n, 2, 0.2, 2.0, gen)
    y = _apply_label_noise(x[:, 0] * x[:, 1], gen, noise_std_rel)
    return x, y


def _make_sum(n: int, gen: torch.Generator, noise_std_rel: float = DEFAULT_NOISE_STD_REL) -> tuple[torch.Tensor, torch.Tensor]:
    x = _linspace_input(n, 2, -1.5, 1.5, gen)
    y = _apply_label_noise(x[:, 0] + x[:, 1], gen, noise_std_rel)
    return x, y


def _make_exp(n: int, gen: torch.Generator, noise_std_rel: float = DEFAULT_NOISE_STD_REL) -> tuple[torch.Tensor, torch.Tensor]:
    x = _linspace_input(n, 1, -1.2, 1.2, gen)
    y = _apply_label_noise(torch.exp(x.squeeze()), gen, noise_std_rel)
    return x, y


def _make_sin(n: int, gen: torch.Generator, noise_std_rel: float = DEFAULT_NOISE_STD_REL) -> tuple[torch.Tensor, torch.Tensor]:
    x = _linspace_input(n, 1, -1.5, 1.5, gen)
    y = _apply_label_noise(torch.sin(x.squeeze()), gen, noise_std_rel)
    return x, y


def _make_sin_plus(n: int, gen: torch.Generator, noise_std_rel: float = DEFAULT_NOISE_STD_REL) -> tuple[torch.Tensor, torch.Tensor]:
    x = _linspace_input(n, 2, -1.5, 1.5, gen)
    y = _apply_label_noise(torch.sin(x[:, 0]) + x[:, 1], gen, noise_std_rel)
    return x, y


def _make_inv_square(n: int, gen: torch.Generator, noise_std_rel: float = DEFAULT_NOISE_STD_REL) -> tuple[torch.Tensor, torch.Tensor]:
    x = _linspace_input(n, 1, 0.5, 2.5, gen)
    y = _apply_label_noise(1.0 / (x.squeeze() ** 2), gen, noise_std_rel)
    return x, y


TARGETS: dict[str, SRTarget] = {
    # Phase 1
    "sin_plus": SRTarget(
        id="sin_plus",
        formula="sin(x0) + x1",
        input_dim=2,
        monotonic=False,
        phase=1,
        sample=_make_sin_plus,
    ),
    # Phase 2 — Feynman-style (2-var forms common in Feynman DB)
    "feynman_I29_product": SRTarget(
        id="feynman_I29_product",
        formula="x0 * x1",
        input_dim=2,
        monotonic=False,
        phase=2,
        sample=_make_product,
    ),
    "feynman_I9_inv_square": SRTarget(
        id="feynman_I9_inv_square",
        formula="1 / x0^2",
        input_dim=1,
        monotonic=True,
        phase=2,
        sample=_make_inv_square,
    ),
    # Phase 3 — general elementary suite
    "square": SRTarget("square", "x0^2", 1, False, 3, _make_square),
    "product": SRTarget("product", "x0 * x1", 2, False, 3, _make_product),
    "sum": SRTarget("sum", "x0 + x1", 2, True, 3, _make_sum),
    "exp": SRTarget("exp", "exp(x0)", 1, True, 3, _make_exp),
    "sin": SRTarget("sin", "sin(x0)", 1, False, 3, _make_sin),
}


def targets_for_phase(phase: int) -> list[SRTarget]:
    return [t for t in TARGETS.values() if t.phase == phase or (phase == 3 and t.phase == 3)]
