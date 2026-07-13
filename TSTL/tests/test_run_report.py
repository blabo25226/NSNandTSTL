"""Tests for the consolidated R1 run report (no GPU)."""

import json
from pathlib import Path

from llm_profile import save_run_report


def test_save_run_report_writes_json_and_md(tmp_path: Path):
    out = save_run_report(
        tmp_path,
        config={"model": "tiny", "preset": "quick"},
        s_base=0.10,
        s_full=0.50,
        s_per_layer={0: 0.2, 1: 0.6, 2: 0.3},
        contributions={0: 0.25, 1: 1.25, 2: 0.5},
        strategies={"only_bk": {"layers": [1], "score": 0.55}},
    )
    assert out == tmp_path
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert data["best_layer"] == 1
    assert data["s_full"] == 0.50
    assert data["strategies"]["only_bk"]["score"] == 0.55

    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "S_full: 0.5000" in md
    assert "Strategies vs Full" in md
    assert "only_bk" in md


def test_save_run_report_handles_nan_contributions(tmp_path: Path):
    # Denominator ~0 -> NaN contribution; best_layer should skip NaN.
    save_run_report(
        tmp_path,
        config={"model": "tiny", "preset": "quick"},
        s_base=0.3,
        s_full=0.3,
        s_per_layer={0: 0.3, 1: 0.3},
        contributions={0: float("nan"), 1: float("nan")},
        strategies=None,
    )
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert data["best_layer"] is None
