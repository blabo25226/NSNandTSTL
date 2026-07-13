"""Tests for llm_profile (no GPU)."""

from pathlib import Path

from llm_profile import depth_normalized_positions, profile_layers_from_scores


def test_depth_normalized_endpoints():
    pos = depth_normalized_positions(5)
    assert pos[0] == 0.0
    assert pos[4] == 1.0


def test_profile_layers_from_scores(tmp_path: Path):
    s_base, s_full = 0.1, 0.5
    s_per = {0: 0.2, 1: 0.6, 2: 0.3}
    result = profile_layers_from_scores(
        s_base, s_full, s_per, tmp_path, config={"seed": 42}
    )
    assert result.contributions[1] > result.contributions[0]
    assert (tmp_path / "contributions.json").exists()
    assert (tmp_path / "contributions_depth.png").exists()
