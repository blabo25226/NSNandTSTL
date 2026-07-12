"""Tests for EML tree head."""

import torch

from eml import eml
from eml_tree import EMLTreeHead
from utils import as_complex


def test_eml_depth2_manual():
    """
    Reproduce Odrzywołek eml_depth2: e - ln(exp(y) - ln(x)).

    Tree (D=2): eml(eml(0,1), eml(y,x)) with z = [x, y].
    """
    head = EMLTreeHead(feature_dim=2, depth=2)
    head.set_leaf_constant(0, 0.0)
    head.set_leaf_constant(1, 1.0)
    head.set_leaf_linear(2, [0.0, 1.0])
    head.set_leaf_linear(3, [1.0, 0.0])

    x = torch.linspace(0.5, 2.0, 8)
    y = torch.linspace(0.3, 1.2, 8)
    z = torch.stack([x, y], dim=1)

    out = head(z)

    xc, yc = as_complex(x), as_complex(y)
    one = torch.ones_like(xc)
    n1 = eml(yc, xc)
    expected = eml(one, n1).real

    assert torch.allclose(out, expected, atol=1e-4)


def test_eml_depth1_exp_manual():
    """D=1 tree: eml(z0, 1) = exp(z0) with single feature z0."""
    head = EMLTreeHead(feature_dim=1, depth=1)
    head.set_leaf_linear(0, [1.0])
    head.set_leaf_constant(1, 1.0)

    z = torch.linspace(-1.5, 1.5, 6).unsqueeze(1)
    out = head(z)
    expected = torch.exp(z.squeeze())
    assert torch.allclose(out, expected, atol=1e-4)


def test_tree_shape_counts():
    for depth in (1, 2, 3, 4):
        head = EMLTreeHead(feature_dim=3, depth=depth)
        assert head.num_leaves == 2**depth
        assert head.num_internal == 2**depth - 1


def test_forward_scalar_batch():
    head = EMLTreeHead(feature_dim=2, depth=2)
    z = torch.randn(5, 2)
    out = head(z)
    assert out.shape == (5,)


def test_forward_single_vector():
    head = EMLTreeHead(feature_dim=2, depth=2)
    z = torch.randn(2)
    out = head(z)
    assert out.shape == tuple()


def test_forward_no_nan():
    torch.manual_seed(1)
    head = EMLTreeHead(feature_dim=4, depth=3)
    z = torch.randn(8, 4)
    out = head(z)
    assert torch.isfinite(out).all()


def test_f_prev_parent_mode_changes_output():
    """Parent-feedback mode injects gamma*f_prev, differing from zero mode."""
    torch.manual_seed(3)
    zero = EMLTreeHead(feature_dim=3, depth=2, f_prev_mode="zero")
    parent = EMLTreeHead(feature_dim=3, depth=2, f_prev_mode="parent")
    parent.load_state_dict(zero.state_dict())  # identical weights
    # Force non-trivial gamma weight so the f_prev term matters.
    with torch.no_grad():
        zero.leaf_logits.fill_(0.0)
        parent.leaf_logits.fill_(0.0)
    z = torch.randn(6, 3)
    out_zero = zero(z)
    out_parent = parent(z)
    assert out_zero.shape == out_parent.shape
    assert torch.isfinite(out_parent).all()
    assert not torch.allclose(out_zero, out_parent, atol=1e-4)


def test_f_prev_mode_validation():
    import pytest

    with pytest.raises(ValueError):
        EMLTreeHead(feature_dim=2, depth=2, f_prev_mode="bogus")


def test_zero_mode_masks_gamma_branch():
    """In zero mode the gamma (f_prev) branch must never win the argmax snap."""
    from snap import snap_leaf_weights

    head = EMLTreeHead(feature_dim=3, depth=2, f_prev_mode="zero")
    with torch.no_grad():
        head.leaf_logits.fill_(0.0)
        head.leaf_logits[:, 2] = 5.0  # try to make gamma dominant
    eff = head.effective_logits()
    assert torch.all(eff[:, 2] < -1e8)
    hard = snap_leaf_weights(head)
    assert torch.all(hard[:, 2] == 0.0)  # gamma never selected


def test_parent_mode_keeps_gamma_branch():
    head = EMLTreeHead(feature_dim=3, depth=2, f_prev_mode="parent")
    with torch.no_grad():
        head.leaf_logits.fill_(0.0)
    eff = head.effective_logits()
    assert torch.allclose(eff, head.leaf_logits)


def _eval_exported_string(expr: str, z: torch.Tensor) -> torch.Tensor:
    """Numerically evaluate a Re[eml(...)] export string for faithfulness tests."""
    from eml import eml as _eml
    from utils import as_complex as _as_complex

    batch = z.shape[0]

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
    for j in range(z.shape[-1]):
        scope[f"z{j}"] = _as_complex(z[:, j])
    return eval(inner, {"__builtins__": {}}, scope).real  # noqa: S307 (trusted string)


def _snap_head_onehot(head: EMLTreeHead) -> None:
    from snap import apply_snap_to_logits

    apply_snap_to_logits(head)


def test_parent_export_faithful_k1_and_k2():
    """Exported closed form must reproduce the snapped forward pass for K in {1,2}."""
    for passes in (1, 2):
        torch.manual_seed(10 + passes)
        head = EMLTreeHead(
            feature_dim=3, depth=2, f_prev_mode="parent", f_prev_passes=passes
        )
        with torch.no_grad():
            head.alpha.copy_(torch.randn(head.num_leaves))
            head.beta.copy_(torch.randn(head.num_leaves, 3) * 0.5)
            # Mix of branches incl. gamma so f_prev recurrence is exercised.
            head.leaf_logits.copy_(
                torch.tensor([[0.0, 0.0, 5.0], [5.0, 0.0, 0.0],
                              [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
            )
        _snap_head_onehot(head)
        head.set_temperature(1e-3)
        head.eval()

        z = torch.randn(6, 3)
        from snap import export_symbolic_expression

        expr = export_symbolic_expression(head, [f"z{i}" for i in range(3)])
        forward = head(z)
        exported = _eval_exported_string(expr, z)
        assert torch.allclose(forward, exported, atol=1e-3), (
            f"K={passes}: forward vs exported mismatch\n{expr}"
        )


def test_parent_fixed_point_residual_decreases():
    """More f_prev passes should not increase the Jacobi fixed-point residual."""
    torch.manual_seed(7)
    z = torch.randn(8, 3)

    def residual(passes: int) -> float:
        head = EMLTreeHead(
            feature_dim=3, depth=2, f_prev_mode="parent", f_prev_passes=passes
        )
        with torch.no_grad():
            head.alpha.copy_(torch.randn(head.num_leaves) * 0.3)
            head.beta.copy_(torch.randn(head.num_leaves, 3) * 0.1)
            head.leaf_logits.fill_(0.0)  # soft mix, gamma active
        # leaves^(K) vs one extra Jacobi update.
        leaves_k = head.leaf_values(z)
        head_next = EMLTreeHead(
            feature_dim=3, depth=2, f_prev_mode="parent", f_prev_passes=passes + 1
        )
        head_next.load_state_dict(head.state_dict())
        leaves_k1 = head_next.leaf_values(z)
        return (leaves_k1 - leaves_k).abs().max().item()

    r1, r3 = residual(1), residual(3)
    assert r3 <= r1 + 1e-6


def test_depth_five_allowed_with_warning():
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        head = EMLTreeHead(feature_dim=3, depth=5)
        assert any("recommended" in str(c.message) for c in caught)
    z = torch.randn(4, 3)
    out = head(z)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
