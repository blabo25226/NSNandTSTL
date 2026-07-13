"""Tests for correlation helpers used by the E8 (‖Δθ‖ vs C(k)) analysis."""

import math

from utils import pearson_corr, spearman_corr


def test_pearson_perfect_positive():
    assert math.isclose(pearson_corr([1, 2, 3, 4], [2, 4, 6, 8]), 1.0, abs_tol=1e-9)


def test_pearson_perfect_negative():
    assert math.isclose(pearson_corr([1, 2, 3, 4], [4, 3, 2, 1]), -1.0, abs_tol=1e-9)


def test_spearman_monotonic_nonlinear():
    # Monotonic but nonlinear -> Spearman 1.0 even though Pearson < 1.
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    assert math.isclose(spearman_corr(x, y), 1.0, abs_tol=1e-9)
    assert pearson_corr(x, y) < 1.0


def test_zero_variance_is_nan():
    assert math.isnan(pearson_corr([1, 1, 1], [1, 2, 3]))


def test_spearman_handles_ties():
    val = spearman_corr([1, 1, 2, 3], [1, 2, 2, 4])
    assert -1.0 <= val <= 1.0
