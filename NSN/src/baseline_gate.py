"""TSTL pre-integration baseline gate definitions and verdict logic."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import torch

GATE_A_TARGETS: tuple[str, ...] = ("square", "product", "sum", "exp", "sin")
GATE_B_TARGETS: tuple[str, ...] = ("sin_plus",)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 7, 42, 123)

GATE_A_MIN_SUCCESS_RATE = 0.8
GATE_B_MIN_SUCCESS_RATE = 0.6
GATE_C_MAX_SNAP_DEGRADE = 1.5

MSE_OK_STRICT_DEFAULT = 1e-2
MSE_OK_RELAXED_DEFAULT = 5e-2


def mse_threshold(
    target_id: str,
    monotonic: bool,
    strict: float,
    relaxed: float,
    y_ref: torch.Tensor | None = None,
    noise_std_rel: float = 0.0,
) -> float:
    base = relaxed if not monotonic else strict
    if y_ref is not None and noise_std_rel > 0:
        noise_floor = (noise_std_rel * y_ref.std()).item() ** 2
        base = max(base, noise_floor * 2.0)
    return base


@dataclass
class TargetGateSummary:
    target_id: str
    n_runs: int
    n_finite: int
    n_symbolic_ok: int
    success_rate: float
    finite_rate: float
    snap_degrade_median: float | None


@dataclass
class GateVerdict:
    gate_a_pass: bool
    gate_b_pass: bool
    gate_c_pass: bool
    gate_a_targets: dict[str, TargetGateSummary] = field(default_factory=dict)
    gate_b_targets: dict[str, TargetGateSummary] = field(default_factory=dict)
    gate_c_median_degrade: float | None = None
    notes: list[str] = field(default_factory=list)


def _is_finite_run(row: dict[str, Any]) -> bool:
    if row.get("error"):
        return False
    hold = row.get("holdout_mse")
    snapped = row.get("snapped_holdout_mse")
    if hold is None or snapped is None:
        return False
    for v in (hold, snapped):
        if not isinstance(v, (int, float)) or v != v or v == float("inf"):
            return False
    return True


def _summarize_target(rows: list[dict[str, Any]]) -> TargetGateSummary:
    tid = rows[0]["target_id"]
    n = len(rows)
    finite_rows = [r for r in rows if _is_finite_run(r)]
    sym_ok = [r for r in finite_rows if r.get("symbolic_ok")]
    degrades = [
        r["snap_degrade_ratio"]
        for r in sym_ok
        if isinstance(r.get("snap_degrade_ratio"), (int, float)) and r["snap_degrade_ratio"] == r["snap_degrade_ratio"]
    ]
    return TargetGateSummary(
        target_id=tid,
        n_runs=n,
        n_finite=len(finite_rows),
        n_symbolic_ok=len(sym_ok),
        success_rate=len(sym_ok) / n if n else 0.0,
        finite_rate=len(finite_rows) / n if n else 0.0,
        snap_degrade_median=statistics.median(degrades) if degrades else None,
    )


def evaluate_gates(results: list[dict[str, Any]]) -> GateVerdict:
    """Evaluate Gate A/B/C from per-(target, seed) result rows."""
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_target.setdefault(row["target_id"], []).append(row)

    gate_a: dict[str, TargetGateSummary] = {}
    gate_a_pass = True
    for tid in GATE_A_TARGETS:
        rows = by_target.get(tid, [])
        if not rows:
            gate_a_pass = False
            continue
        summary = _summarize_target(rows)
        gate_a[tid] = summary
        if summary.success_rate < GATE_A_MIN_SUCCESS_RATE:
            gate_a_pass = False

    gate_b: dict[str, TargetGateSummary] = {}
    gate_b_pass = True
    for tid in GATE_B_TARGETS:
        rows = by_target.get(tid, [])
        if not rows:
            gate_b_pass = False
            continue
        summary = _summarize_target(rows)
        gate_b[tid] = summary
        if summary.success_rate < GATE_B_MIN_SUCCESS_RATE:
            gate_b_pass = False

    degrade_samples: list[float] = []
    for tid in GATE_A_TARGETS:
        for row in by_target.get(tid, []):
            if row.get("symbolic_ok") and _is_finite_run(row):
                d = row.get("snap_degrade_ratio")
                if isinstance(d, (int, float)) and d == d:
                    degrade_samples.append(float(d))

    gate_c_median = statistics.median(degrade_samples) if degrade_samples else None
    gate_c_pass = gate_c_median is not None and gate_c_median <= GATE_C_MAX_SNAP_DEGRADE

    notes: list[str] = []
    if not gate_b_pass:
        notes.append(
            "Gate B optional: sin_plus below 60% does not block TSTL if Gate A+C pass."
        )

    return GateVerdict(
        gate_a_pass=gate_a_pass,
        gate_b_pass=gate_b_pass,
        gate_c_pass=gate_c_pass,
        gate_a_targets=gate_a,
        gate_b_targets=gate_b,
        gate_c_median_degrade=gate_c_median,
        notes=notes,
    )


def protocol_train_flags(protocol: str) -> dict[str, bool]:
    """Map CLI protocol name to TrainConfig pipeline flags."""
    if protocol == "full":
        return dict(freeze_trunk_after_search=False, trunk_only_search=False)
    if protocol == "freeze":
        return dict(freeze_trunk_after_search=True, trunk_only_search=False)
    if protocol == "trunk_first":
        return dict(freeze_trunk_after_search=False, trunk_only_search=True)
    raise ValueError(f"unknown protocol: {protocol}")
