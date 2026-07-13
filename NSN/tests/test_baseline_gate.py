"""Tests for baseline gate verdict logic."""

from baseline_gate import evaluate_gates, protocol_train_flags


def _row(tid: str, seed: int, sym: bool, degrade: float = 1.0) -> dict:
    return {
        "target_id": tid,
        "seed": seed,
        "holdout_mse": 1e-4,
        "snapped_holdout_mse": 1e-4,
        "snap_degrade_ratio": degrade,
        "symbolic_ok": sym,
    }


def test_gate_a_passes_at_eighty_percent():
    rows = []
    for tid in ("square", "product", "sum", "exp", "sin"):
        for seed in range(5):
            ok = seed != 0 if tid == "square" else True
            rows.append(_row(tid, seed, ok))
    verdict = evaluate_gates(rows)
    assert verdict.gate_a_targets["square"].success_rate == 0.8
    assert verdict.gate_a_pass is True


def test_gate_a_fails_below_eighty_percent():
    rows = [_row("square", s, s < 4) for s in range(5)]
    verdict = evaluate_gates(rows)
    assert verdict.gate_a_pass is False


def test_gate_c_median_degrade():
    rows = [
        _row("square", 0, True, 1.0),
        _row("product", 0, True, 1.2),
        _row("sum", 0, True, 1.1),
        _row("exp", 0, True, 1.0),
        _row("sin", 0, True, 1.3),
        _row("sin_plus", 0, False, 10.0),
    ]
    verdict = evaluate_gates(rows)
    assert verdict.gate_c_median_degrade == 1.1
    assert verdict.gate_c_pass is True


def test_protocol_train_flags():
    assert protocol_train_flags("freeze")["freeze_trunk_after_search"] is True
    assert protocol_train_flags("trunk_first")["trunk_only_search"] is True
