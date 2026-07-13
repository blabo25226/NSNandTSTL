"""
Full Feynman equation set from AI Feynman CSV (Udrescu & Tegmark).

Loads ``data/FeynmanEquations.csv``, samples inputs from per-variable ranges,
evaluates formulas, and optionally adds relative Gaussian label noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from targets import DEFAULT_NOISE_STD_REL, SRTarget, _apply_label_noise

RMSE_THRESHOLD_FIXED = 1e-4
RMSE_PARTIAL_FIXED = 1e-2
FEYNMAN_CSV_PHASE = 5
MIN_VALID_SAMPLES = 10


@dataclass(frozen=True)
class FeynmanCsvEq:
    filename: str
    formula: str
    output_var: str
    n_vars: int
    var_names: list[str]
    var_ranges: list[tuple[float, float]]


def _build_eval_env() -> dict:
    return {
        "exp": np.exp,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "log": np.log,
        "ln": np.log,
        "abs": np.abs,
        "tanh": np.tanh,
        "pi": np.pi,
        "e": np.e,
        "__builtins__": None,
    }


def default_csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "FeynmanEquations.csv"


def load_feynman_csv(csv_path: str | Path | None = None) -> list[FeynmanCsvEq]:
    path = Path(csv_path) if csv_path is not None else default_csv_path()
    df = pd.read_csv(path)
    df = df.dropna(subset=["Filename", "Formula"])
    df = df[df["Filename"].str.strip() != ""]

    equations: list[FeynmanCsvEq] = []
    for _, row in df.iterrows():
        filename = str(row["Filename"]).strip()
        formula = str(row["Formula"]).strip()
        output_var = str(row["Output"]).strip()
        n_vars = int(row["# variables"])

        var_names: list[str] = []
        var_ranges: list[tuple[float, float]] = []
        for i in range(1, n_vars + 1):
            nc, lc, hc = f"v{i}_name", f"v{i}_low", f"v{i}_high"
            if nc in row and pd.notna(row[nc]):
                var_names.append(str(row[nc]).strip())
                var_ranges.append((float(row[lc]), float(row[hc])))

        if len(var_names) != n_vars:
            continue

        equations.append(
            FeynmanCsvEq(
                filename=filename,
                formula=formula,
                output_var=output_var,
                n_vars=n_vars,
                var_names=var_names,
                var_ranges=var_ranges,
            )
        )
    return equations


def generate_feynman_samples(
    eq: FeynmanCsvEq,
    n: int,
    seed: int = 42,
    noise_std_rel: float = DEFAULT_NOISE_STD_REL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Sample (X, y_noisy, y_clean). Returns None if fewer than MIN_VALID_SAMPLES.
    """
    rng = np.random.default_rng(seed)
    base_env = _build_eval_env()

    x_list: list[list[float]] = []
    y_clean_list: list[float] = []
    max_attempts = n * 6
    attempt = 0

    while len(y_clean_list) < n and attempt < max_attempts:
        attempt += 1
        point: list[float] = []
        local_env = base_env.copy()
        for name, (low, high) in zip(eq.var_names, eq.var_ranges):
            val = float(rng.uniform(low, high))
            point.append(val)
            local_env[name] = val
        try:
            y = float(eval(eq.formula, {"__builtins__": None}, local_env))
            if np.isfinite(y):
                x_list.append(point)
                y_clean_list.append(y)
        except Exception:
            continue

    if len(y_clean_list) < MIN_VALID_SAMPLES:
        return None

    x_np = np.array(x_list, dtype=np.float64)
    y_np = np.array(y_clean_list, dtype=np.float64)

    x = torch.tensor(x_np, dtype=torch.float32)
    y_clean = torch.tensor(y_np, dtype=torch.float32)

    gen = torch.Generator().manual_seed(seed)
    y_noisy = _apply_label_noise(y_clean, gen, noise_std_rel)
    return x, y_noisy, y_clean


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    diff = y_true - y_pred
    if not torch.isfinite(diff).all():
        return float("inf")
    return float(torch.sqrt(torch.mean(diff ** 2)).item())


def noise_aware_rmse_ok_threshold(
    y_clean: torch.Tensor,
    noise_std_rel: float,
    fixed_ok: float = RMSE_THRESHOLD_FIXED,
) -> float:
    """RMSE ok threshold: max(fixed_ok, sqrt(2) * noise_std_rel * std(y))."""
    scale = y_clean.std().item()
    if scale < 1e-12:
        scale = 1.0
    noise_floor = math.sqrt(2.0) * noise_std_rel * scale
    return max(fixed_ok, noise_floor)


def classify_rmse(
    rmse_val: float,
    ok_thr: float,
    partial_thr: float = RMSE_PARTIAL_FIXED,
) -> str:
    if not math.isfinite(rmse_val):
        return "failed"
    if rmse_val < ok_thr:
        return "ok"
    if rmse_val < partial_thr:
        return "partial"
    return "failed"


def _make_sampler(eq: FeynmanCsvEq, data_seed: int) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    def sample(
        n: int,
        gen: torch.Generator,
        noise_std_rel: float = DEFAULT_NOISE_STD_REL,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seed = int(gen.initial_seed()) if gen.initial_seed() != 0 else data_seed
        out = generate_feynman_samples(eq, n, seed=seed, noise_std_rel=noise_std_rel)
        if out is None:
            raise RuntimeError(f"Too few valid samples for {eq.filename}")
        x, y_noisy, _ = out
        return x, y_noisy

    return sample


def to_sr_target(eq: FeynmanCsvEq, data_seed: int = 42) -> SRTarget:
    return SRTarget(
        id=eq.filename,
        formula=eq.formula,
        input_dim=eq.n_vars,
        monotonic=False,
        phase=FEYNMAN_CSV_PHASE,
        sample=_make_sampler(eq, data_seed),
    )
