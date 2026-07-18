"""Tests for the EML compute-cost model."""

from cost import eml_node_flops, head_flops, leaf_affine_flops, mlp_flops


def test_eml_node_transcendental_matches_paper():
    """Transcendental FLOPs per eml node reproduce the paper's ~111 figure."""
    cost = eml_node_flops()
    assert abs(cost.transcendental_flops - 111.0) < 1e-6
    # total is transcendental + a small arithmetic overhead
    assert 111.0 <= cost.total_flops <= 160.0


def test_eml_node_breakdown_positive():
    cost = eml_node_flops()
    assert all(v > 0 for v in cost.breakdown.values())
    assert set(cost.breakdown) >= {"exp", "sin", "cos", "log", "atan2", "sqrt"}


def test_head_flops_monotonic_in_depth():
    prev = 0.0
    for depth in (1, 2, 3, 4, 5):
        h = head_flops(depth, feature_dim=6)
        assert h["head_flops"] > prev
        assert h["num_internal_nodes"] == 2**depth - 1
        prev = h["head_flops"]


def test_leaf_and_mlp_flops_positive():
    assert leaf_affine_flops(6) > 0
    assert mlp_flops([4, 64, 64, 4]) > 0
    # deeper/wider MLP costs more
    assert mlp_flops([4, 128, 128, 4]) > mlp_flops([4, 16, 16, 4])
