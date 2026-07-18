"""Tests for trunk interpretability (linear readout, composition, attribution)."""

import torch

from model import DNNEML
from snap import apply_snap_to_logits
from trunk_interpret import compose_symbolic, linear_readout, trunk_attribution


def _eval_x_string(expr: str, x: torch.Tensor) -> torch.Tensor:
    """Evaluate a Re[eml(...)] export string expressed in x0..x_{p-1}."""
    from eml import eml as _eml
    from utils import as_complex as _as_complex

    batch = x.shape[0]

    def _promote(v):
        if isinstance(v, (int, float)):
            return _as_complex(torch.full((batch,), float(v)))
        return v

    def eml_wrap(a, b):
        return _eml(_promote(a), _promote(b))

    inner = expr
    if inner.startswith("Re[") and inner.endswith("]"):
        inner = inner[3:-1]
    scope = {"eml": eml_wrap}
    for j in range(x.shape[-1]):
        scope[f"x{j}"] = _as_complex(x[:, j])
    return eval(inner, {"__builtins__": {}}, scope).real  # noqa: S307


def test_linear_readout_exact_and_composition_faithful():
    """Single-Linear trunk: composed yhat(x) equals the snapped model output."""
    torch.manual_seed(0)
    model = DNNEML.build(
        input_dim=2, feature_dim=3, head_depth=2,
        num_layers=1,  # linear trunk
    )
    with torch.no_grad():
        model.head.alpha.copy_(torch.randn(model.head.num_leaves))
        model.head.beta.copy_(torch.randn(model.head.num_leaves, 3) * 0.5)
        model.head.leaf_logits.copy_(
            torch.tensor([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0],
                          [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])
        )
    apply_snap_to_logits(model.head)
    model.head.set_temperature(1e-3)
    model.eval()

    x = torch.randn(16, 2)
    ro = linear_readout(model, x)
    assert ro.exact
    assert torch.all(ro.r2 > 0.999)

    expr = compose_symbolic(model, ro, ["x0", "x1"])
    with torch.no_grad():
        model_out = model(x)
    composed = _eval_x_string(expr, x)
    assert torch.allclose(model_out, composed, atol=1e-3), expr


def test_nonlinear_distillation_r2_and_attribution():
    torch.manual_seed(1)
    model = DNNEML.build(
        input_dim=4, feature_dim=3, head_depth=2,
        hidden_dim=16, num_layers=3,  # nonlinear trunk
    )
    x = torch.randn(64, 4)
    ro = linear_readout(model, x)
    assert not ro.exact
    assert ro.W.shape == (3, 4)
    assert torch.all(ro.r2 <= 1.0 + 1e-6)

    attr = trunk_attribution(model, x, top_k=2)
    assert len(attr) == 3  # one list per z-component
    for comp in attr:
        assert len(comp) == 2
        # descending by score
        assert comp[0][1] >= comp[1][1]
