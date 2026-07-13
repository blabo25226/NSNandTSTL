"""MLP benchmark pipeline for TSTL Phase 0-B."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from layer_contribution import (
    compute_contributions,
    plot_contribution_heatmap,
    save_contributions_json,
)
from mlp_model import RegressionMLP
from strategies import (
    select_mid_layers,
    select_top_layers,
    train_boost_layers,
    train_full,
    train_only_layers,
)
from utils import r2_score, set_seed


@dataclass
class BenchConfig:
    input_dim: int = 32
    hidden_dim: int = 64
    num_hidden_layers: int = 3
    n_train: int = 512
    n_test: int = 256
    steps: int = 2000
    lr: float = 1e-3
    seed: int = 42
    noise_std: float = 0.05
    mid_k: int = 3
    top_k: int = 3
    boost_lr_mult: float = 2.0


@dataclass
class BenchResult:
    s_base: float
    s_full: float
    s_per_layer: dict[int, float]
    contributions: dict[int, float]
    strategies: dict[str, float] = field(default_factory=dict)
    best_layer: int = 0
    mid_layer_indices: list[int] = field(default_factory=list)


def make_synthetic_data(
    cfg: BenchConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nonlinear regression: y depends on products and sin of input dims."""
    gen = torch.Generator().manual_seed(cfg.seed)
    x_train = torch.randn(cfg.n_train, cfg.input_dim, generator=gen)
    x_test = torch.randn(cfg.n_test, cfg.input_dim, generator=gen)

    def target(x: torch.Tensor) -> torch.Tensor:
        t = (
            0.4 * (x[:, 0] * x[:, 1])
            + 0.3 * torch.sin(x[:, 2])
            + 0.2 * (x[:, 3] ** 2)
            + 0.1 * x[:, 4]
        )
        return t

    y_train = target(x_train)
    y_test = target(x_test)
    if cfg.noise_std > 0:
        y_train = y_train + cfg.noise_std * torch.randn(y_train.shape, generator=gen)
        y_test = y_test + cfg.noise_std * torch.randn(y_test.shape, generator=gen)
    return x_train, y_train, x_test, y_test


def _eval_r2(model: RegressionMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(x)
    return r2_score(y, pred)


def _model_factory(cfg: BenchConfig) -> RegressionMLP:
    set_seed(cfg.seed)
    return RegressionMLP(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
    )


def _clone_init(model: RegressionMLP) -> RegressionMLP:
    """Return a new model with identical weights."""
    other = RegressionMLP(
        model.input_dim,
        model.hidden_dim,
        model.num_hidden_layers,
    )
    other.load_state_dict(copy.deepcopy(model.state_dict()))
    return other


def run_benchmark(cfg: BenchConfig) -> BenchResult:
    set_seed(cfg.seed)
    x_train, y_train, x_test, y_test = make_synthetic_data(cfg)

    init_model = _model_factory(cfg)
    init_state = copy.deepcopy(init_model.state_dict())
    s_base = _eval_r2(init_model, x_test, y_test)

    # Full training
    full_model = RegressionMLP(cfg.input_dim, cfg.hidden_dim, cfg.num_hidden_layers)
    full_model.load_state_dict(copy.deepcopy(init_state))
    train_full(full_model, x_train, y_train, steps=cfg.steps, lr=cfg.lr)
    s_full = _eval_r2(full_model, x_test, y_test)

    # Per-layer training
    s_per_layer: dict[int, float] = {}
    for k in range(cfg.num_hidden_layers):
        m = RegressionMLP(cfg.input_dim, cfg.hidden_dim, cfg.num_hidden_layers)
        m.load_state_dict(copy.deepcopy(init_state))
        train_only_layers(m, [k], x_train, y_train, steps=cfg.steps, lr=cfg.lr)
        s_per_layer[k] = _eval_r2(m, x_test, y_test)

    contributions = compute_contributions(s_base, s_full, s_per_layer)
    best_layer = max(contributions, key=lambda k: contributions[k])
    mid_indices = select_mid_layers(cfg.num_hidden_layers, cfg.mid_k)
    top_indices = select_top_layers(contributions, min(cfg.top_k, cfg.num_hidden_layers))

    strategies: dict[str, float] = {"full": s_full}

    m_only_b = RegressionMLP(cfg.input_dim, cfg.hidden_dim, cfg.num_hidden_layers)
    m_only_b.load_state_dict(copy.deepcopy(init_state))
    train_only_layers(m_only_b, top_indices, x_train, y_train, steps=cfg.steps, lr=cfg.lr)
    strategies[f"only_b{cfg.top_k}"] = _eval_r2(m_only_b, x_test, y_test)

    m_mid = RegressionMLP(cfg.input_dim, cfg.hidden_dim, cfg.num_hidden_layers)
    m_mid.load_state_dict(copy.deepcopy(init_state))
    train_only_layers(m_mid, mid_indices, x_train, y_train, steps=cfg.steps, lr=cfg.lr)
    strategies[f"mid_{cfg.mid_k}"] = _eval_r2(m_mid, x_test, y_test)

    m_boost = RegressionMLP(cfg.input_dim, cfg.hidden_dim, cfg.num_hidden_layers)
    m_boost.load_state_dict(copy.deepcopy(init_state))
    train_boost_layers(
        m_boost,
        top_indices,
        x_train,
        y_train,
        steps=cfg.steps,
        lr=cfg.lr,
        boost_lr_mult=cfg.boost_lr_mult,
    )
    strategies[f"boost_b{cfg.top_k}"] = _eval_r2(m_boost, x_test, y_test)

    return BenchResult(
        s_base=s_base,
        s_full=s_full,
        s_per_layer=s_per_layer,
        contributions=contributions,
        strategies=strategies,
        best_layer=best_layer,
        mid_layer_indices=mid_indices,
    )


def save_benchmark_results(
    cfg: BenchConfig,
    result: BenchResult,
    out_dir: Path | str,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_contributions_json(
        out_dir / "contributions.json",
        s_base=result.s_base,
        s_full=result.s_full,
        s_per_layer=result.s_per_layer,
        contributions=result.contributions,
    )
    plot_contribution_heatmap(result.contributions, out_dir / "contribution.png")

    (out_dir / "config.json").write_text(
        json.dumps(cfg.__dict__, indent=2), encoding="utf-8"
    )

    lines = [
        "# TSTL MLP layer profile",
        "",
        f"- seed: {cfg.seed}",
        f"- hidden layers: {cfg.num_hidden_layers}",
        f"- S_base (R²): {result.s_base:.4f}",
        f"- S_full (R²): {result.s_full:.4f}",
        f"- best layer k: {result.best_layer} (C={result.contributions[result.best_layer]:.3f})",
        f"- mid-k indices: {result.mid_layer_indices}",
        "",
        "## C(k)",
        "",
        "| k | S_k | C(k) |",
        "|---|-----|------|",
    ]
    for k in sorted(result.contributions):
        lines.append(
            f"| {k} | {result.s_per_layer[k]:.4f} | {result.contributions[k]:.4f} |"
        )
    lines.extend(["", "## Strategies (R²)", "", "| strategy | R² |", "|----------|-----|"])
    for name, score in result.strategies.items():
        lines.append(f"| {name} | {score:.4f} |")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir


def default_out_dir(prefix: str = "tstl_profile") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parents[1] / "results" / f"{prefix}_{stamp}"
