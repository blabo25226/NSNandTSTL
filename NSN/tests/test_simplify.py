"""Tests for elementary simplification and template matching."""

import torch

from eml_tree import EMLTreeHead
from model import DNNEML
from simplify import ELEMENTARY_TEMPLATES, evaluate_symbolic, simplify_eml_expression
from trainer import TrainConfig, train_target
from targets import TARGETS


def test_simplify_eml_exp_pattern():
    out = simplify_eml_expression("Re[eml(z0, 1)]")
    assert "exp" in out


def test_template_library_covers_targets():
    for key in ("square", "product", "sin_plus"):
        t = TARGETS[key]
        assert t.formula in ELEMENTARY_TEMPLATES


def test_evaluate_symbolic_on_trained_square():
    target = TARGETS["square"]
    model, tr = train_target(target, TrainConfig(noise_std_rel=0.0))
    gen = torch.Generator().manual_seed(1)
    x, y = target.sample(64, gen, noise_std_rel=0.0)
    sym = evaluate_symbolic(
        model, tr.expression_eml, x, y, target.formula, mse_threshold=0.1, numeric_ok=True
    )
    assert sym.template_mse is not None
