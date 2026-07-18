"""
Curated subset of the real Feynman Symbolic Regression equations
(Udrescu & Tegmark, AI Feynman). Low-to-mid dimensional, elementary-function
forms, with the dataset's standard uniform sampling ranges.

Reuses `targets.SRTarget` (and label-noise helper) so the benchmark harness can
treat these identically to the synthetic elementary targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from targets import DEFAULT_NOISE_STD_REL, SRTarget, _apply_label_noise


@dataclass(frozen=True)
class FeynmanEq:
    id: str
    formula: str
    ranges: list[tuple[float, float]]  # per-variable uniform sampling range
    fn: Callable[[torch.Tensor], torch.Tensor]  # y from x (n, input_dim)

    @property
    def input_dim(self) -> int:
        return len(self.ranges)


_SQRT_2PI = math.sqrt(2.0 * math.pi)

# (id, formula, per-variable ranges, y(x)). Ranges follow the AI Feynman dataset
# conventions (mostly [1,5], gaussians use [1,3]).
_EQUATIONS: list[FeynmanEq] = [
    FeynmanEq("I.6.20a", "exp(-x0^2/2)/sqrt(2*pi)", [(1.0, 3.0)],
              lambda x: torch.exp(-x[:, 0] ** 2 / 2) / _SQRT_2PI),
    FeynmanEq("I.6.20", "exp(-(x0/x1)^2/2)/(sqrt(2*pi)*x1)", [(1.0, 3.0), (1.0, 3.0)],
              lambda x: torch.exp(-(x[:, 0] / x[:, 1]) ** 2 / 2) / (_SQRT_2PI * x[:, 1])),
    FeynmanEq("I.12.1", "x0*x1", [(1.0, 5.0), (1.0, 5.0)],
              lambda x: x[:, 0] * x[:, 1]),
    FeynmanEq("I.14.3", "x0*x1*x2", [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
              lambda x: x[:, 0] * x[:, 1] * x[:, 2]),
    FeynmanEq("I.14.4", "0.5*x0*x1^2", [(1.0, 5.0), (1.0, 5.0)],
              lambda x: 0.5 * x[:, 0] * x[:, 1] ** 2),
    FeynmanEq("I.25.13", "x0/x1", [(1.0, 5.0), (1.0, 5.0)],
              lambda x: x[:, 0] / x[:, 1]),
    FeynmanEq("I.27.6", "1/(1/x0 + x1/x2)", [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
              lambda x: 1.0 / (1.0 / x[:, 0] + x[:, 1] / x[:, 2])),
    FeynmanEq("I.29.16", "sqrt(x0^2+x1^2-2*x0*x1*cos(x2-x3))",
              [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
              lambda x: torch.sqrt(
                  x[:, 0] ** 2 + x[:, 1] ** 2
                  - 2 * x[:, 0] * x[:, 1] * torch.cos(x[:, 2] - x[:, 3])
              )),
    FeynmanEq("I.34.8", "x0*x1*x2/x3", [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
              lambda x: x[:, 0] * x[:, 1] * x[:, 2] / x[:, 3]),
    FeynmanEq("I.39.1", "1.5*x0*x1", [(1.0, 5.0), (1.0, 5.0)],
              lambda x: 1.5 * x[:, 0] * x[:, 1]),
    FeynmanEq("II.3.24", "x0/(4*pi*x1^2)", [(1.0, 5.0), (1.0, 5.0)],
              lambda x: x[:, 0] / (4 * math.pi * x[:, 1] ** 2)),
    FeynmanEq("II.8.31", "0.5*x0*x1^2", [(1.0, 5.0), (1.0, 5.0)],
              lambda x: 0.5 * x[:, 0] * x[:, 1] ** 2),
]


def _make_sampler(eq: FeynmanEq):
    def sample(n: int, gen: torch.Generator, noise_std_rel: float = DEFAULT_NOISE_STD_REL):
        cols = []
        for (lo, hi) in eq.ranges:
            cols.append(torch.rand(n, generator=gen) * (hi - lo) + lo)
        x = torch.stack(cols, dim=1)
        y = _apply_label_noise(eq.fn(x), gen, noise_std_rel)
        return x, y
    return sample


def feynman_targets() -> dict[str, SRTarget]:
    """Return the curated Feynman equations as SRTarget objects (phase 4)."""
    out: dict[str, SRTarget] = {}
    for eq in _EQUATIONS:
        out[eq.id] = SRTarget(
            id=eq.id,
            formula=eq.formula,
            input_dim=eq.input_dim,
            monotonic=False,
            phase=4,
            sample=_make_sampler(eq),
        )
    return out


FEYNMAN_TARGETS: dict[str, SRTarget] = feynman_targets()
