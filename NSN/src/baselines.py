"""
Symbolic-regression baselines for the Feynman benchmark.

- MLPBaseline : plain MLP regressor (black-box reference).
- EQLBaseline : minimal Equation Learner — a linear layer feeding symbolic units
                {identity, sin, cos, multiplication} plus L1 sparsity, then a
                linear readout. A compact in-repo stand-in for EQL (not the full
                published network).
- KANBaseline : Kolmogorov-Arnold Network via `pykan`, used only if importable;
                otherwise `available=False` and it is skipped in the benchmark.

Each baseline exposes fit_predict(x_tr, y_tr, x_te) -> (pred_te, info).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BaselineResult:
    pred: torch.Tensor
    train_seconds: float
    size: int  # method-specific complexity proxy (param / active-unit count)
    name: str


def _train_regressor(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     steps: int, lr: float, l1: float = 0.0) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y = y.view(-1, 1) if y.dim() == 1 else y
    for _ in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        if l1 > 0:
            loss = loss + l1 * sum(p.abs().sum() for p in model.parameters())
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


class MLPBaseline:
    name = "mlp"

    def __init__(self, hidden: int = 64, layers: int = 3):
        self.hidden = hidden
        self.layers = layers

    def fit_predict(self, x_tr, y_tr, x_te, steps: int = 3000, lr: float = 3e-3) -> BaselineResult:
        d = x_tr.shape[1]
        mods: list[nn.Module] = []
        in_d = d
        for _ in range(self.layers - 1):
            mods += [nn.Linear(in_d, self.hidden), nn.ReLU()]
            in_d = self.hidden
        mods.append(nn.Linear(in_d, 1))
        model = nn.Sequential(*mods)
        t0 = time.perf_counter()
        _train_regressor(model, x_tr, y_tr, steps, lr)
        dt = time.perf_counter() - t0
        with torch.no_grad():
            pred = model(x_te).squeeze(-1)
        size = sum(p.numel() for p in model.parameters())
        return BaselineResult(pred=pred, train_seconds=dt, size=size, name=self.name)


class _EQLNet(nn.Module):
    """Single symbolic layer: linear -> {id, sin, cos, mul} -> linear."""

    def __init__(self, d: int, n_each: int = 2, n_mul: int = 2):
        super().__init__()
        self.n_id, self.n_sin, self.n_cos, self.n_mul = n_each, n_each, n_each, n_mul
        # linear outputs feeding the units (mul consumes 2 inputs per product)
        self.lin_units = self.n_id + self.n_sin + self.n_cos + 2 * self.n_mul
        self.pre = nn.Linear(d, self.lin_units)
        self.act_dim = self.n_id + self.n_sin + self.n_cos + self.n_mul
        self.readout = nn.Linear(self.act_dim, 1)

    def forward(self, x):
        h = self.pre(x)
        i = 0
        parts = []
        parts.append(h[:, i:i + self.n_id]); i += self.n_id
        parts.append(torch.sin(h[:, i:i + self.n_sin])); i += self.n_sin
        parts.append(torch.cos(h[:, i:i + self.n_cos])); i += self.n_cos
        mul_in = h[:, i:i + 2 * self.n_mul]
        prod = mul_in[:, 0::2] * mul_in[:, 1::2]
        parts.append(prod)
        a = torch.cat(parts, dim=1)
        return self.readout(a)


class EQLBaseline:
    name = "eql"

    def fit_predict(self, x_tr, y_tr, x_te, steps: int = 3000, lr: float = 3e-3,
                   l1: float = 1e-3) -> BaselineResult:
        d = x_tr.shape[1]
        model = _EQLNet(d)
        t0 = time.perf_counter()
        _train_regressor(model, x_tr, y_tr, steps, lr, l1=l1)
        dt = time.perf_counter() - t0
        with torch.no_grad():
            pred = model(x_te).squeeze(-1)
        # complexity proxy: number of readout weights above a sparsity threshold
        active = int((model.readout.weight.abs() > 1e-2).sum().item())
        return BaselineResult(pred=pred, train_seconds=dt, size=active, name=self.name)


class KANBaseline:
    name = "kan"

    def __init__(self):
        try:
            import kan  # noqa: F401
            self.available = True
        except Exception:
            self.available = False

    def fit_predict(self, x_tr, y_tr, x_te, steps: int = 100, lr: float = 1e-2) -> BaselineResult:
        if not self.available:
            raise RuntimeError("pykan not available")
        from kan import KAN

        d = x_tr.shape[1]
        model = KAN(width=[d, 5, 1], grid=5, k=3)
        dataset = {
            "train_input": x_tr, "train_label": y_tr.view(-1, 1),
            "test_input": x_te, "test_label": torch.zeros(x_te.shape[0], 1),
        }
        t0 = time.perf_counter()
        model.fit(dataset, opt="Adam", steps=steps, lr=lr)
        dt = time.perf_counter() - t0
        with torch.no_grad():
            pred = model(x_te).squeeze(-1)
        size = sum(p.numel() for p in model.parameters())
        return BaselineResult(pred=pred, train_seconds=dt, size=size, name=self.name)
