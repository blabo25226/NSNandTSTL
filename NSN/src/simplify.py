"""Elementary simplification and template matching for SR evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

import torch

from model import DNNEML


@dataclass
class SimplifyResult:
    eml_expression: str
    simplified: str | None
    template_guess: str | None
    template_mse: float | None
    symbolic_ok: bool


# Canonical elementary templates (ground-truth families for evaluation).
ELEMENTARY_TEMPLATES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "x0^2": lambda x: x[:, 0] ** 2,
    "x0 * x1": lambda x: x[:, 0] * x[:, 1],
    "x0 + x1": lambda x: x[:, 0] + x[:, 1],
    "exp(x0)": lambda x: torch.exp(x[:, 0]),
    "sin(x0)": lambda x: torch.sin(x[:, 0]),
    "sin(x0) + x1": lambda x: torch.sin(x[:, 0]) + x[:, 1],
    "1 / x0^2": lambda x: 1.0 / (x[:, 0] ** 2),
}


def _try_simplify_eml_once(expr: str) -> str | None:
    """Apply single-step Odrzywołek identities on eml(...) strings."""
    # eml(x, 1) -> exp(x)
    m = re.fullmatch(r"eml\((.+), 1\)", expr.strip())
    if m:
        return f"exp({m.group(1)})"

    # eml(1, eml(eml(1, z), 1)) -> ln(z)  (exact nested pattern)
    m = re.fullmatch(r"eml\(1, eml\(eml\(1, (.+)\), 1\)\)", expr.strip())
    if m:
        return f"ln({m.group(1)})"

    return None


def simplify_eml_expression(expr: str, max_passes: int = 12) -> str:
    """Bottom-up style string rewriting (best-effort, not complete)."""
    current = expr
    if current.startswith("Re[") and current.endswith("]"):
        current = current[3:-1]

    for _ in range(max_passes):
        changed = False

        def repl(match: re.Match[str]) -> str:
            nonlocal changed
            inner = match.group(0)
            sub = _try_simplify_eml_once(inner)
            if sub is not None:
                changed = True
                return sub
            return inner

        new = re.sub(r"eml\([^()]+(?:\([^()]*\)[^()]*)*\)", repl, current)
        if new == current and not changed:
            break
        current = new
    return current


def match_elementary_template(
    model: DNNEML,
    x: torch.Tensor,
    y_true: torch.Tensor,
    true_formula: str,
    mse_threshold: float = 1e-2,
) -> tuple[str | None, float, float]:
    """
    Pick template that best matches model predictions.

    Returns (best_template_name, mse_pred_vs_template, mse_template_vs_truth).
  """
    with torch.no_grad():
        pred = model(x)
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)

    best_name: str | None = None
    best_mse = float("inf")
    best_truth_mse = float("inf")

    for name, fn in ELEMENTARY_TEMPLATES.items():
        try:
            cand = fn(x)
            mse_pred = torch.mean((pred - cand) ** 2).item()
            mse_truth = torch.mean((y_true - cand) ** 2).item()
        except Exception:
            continue
        if mse_pred < best_mse:
            best_mse = mse_pred
            best_truth_mse = mse_truth
            best_name = name

    return best_name, best_mse, best_truth_mse


def evaluate_symbolic(
    model: DNNEML,
    eml_expr: str,
    x: torch.Tensor,
    y_true: torch.Tensor,
    true_formula: str,
    mse_threshold: float = 1e-2,
    numeric_ok: bool = False,
) -> SimplifyResult:
    simplified = simplify_eml_expression(eml_expr)
    guess, tmpl_mse, _ = match_elementary_template(
        model, x, y_true, true_formula, mse_threshold
    )

    symbolic_ok = numeric_ok and (guess == true_formula)

    simp_out = simplified if simplified != eml_expr.replace("Re[", "").replace("]", "") else None
    return SimplifyResult(
        eml_expression=eml_expr,
        simplified=simp_out,
        template_guess=guess,
        template_mse=tmpl_mse,
        symbolic_ok=symbolic_ok,
    )
