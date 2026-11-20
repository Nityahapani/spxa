"""Statistical expectations: calibrated under the null, powerful under alternatives."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from spxa.laws import LawViolation
from spxa.laws.stats import (
    LEVEL,
    ecf_test,
    expect_not_rejected,
    independence_test,
    mean_test,
    variance_test,
)


def _normal_cf(mu: float, s: float):
    return lambda u: np.exp(1j * mu * u - 0.5 * s**2 * u**2)


def test_ecf_pvalues_are_uniform_under_the_null() -> None:
    rng = np.random.default_rng(0)
    cf = _normal_cf(0.3, 1.2)
    p = [ecf_test(rng.normal(0.3, 1.2, 2000), cf, [0.7]) for _ in range(400)]
    # Bonferroni over Re and Im makes the test conservative: p >= its uniform counterpart
    assert np.mean(np.array(p) < 0.05) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 400)
    assert np.mean(np.array(p) < 0.5) > 0.3  # but not degenerate


def test_mean_and_variance_pvalues_are_uniform_under_the_null() -> None:
    rng = np.random.default_rng(1)
    pm = [mean_test(rng.exponential(2.0, 3000), 2.0, 4.0) for _ in range(300)]
    pv = [variance_test(rng.exponential(2.0, 3000), 4.0, 6 * 2.0**4) for _ in range(300)]
    assert stats.kstest(pm, "uniform").pvalue > 1e-3
    assert stats.kstest(pv, "uniform").pvalue > 1e-3


def test_tests_have_power() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 20_000)
    assert ecf_test(x, _normal_cf(0.0, 1.1), [0.5, 1.5]) < LEVEL  # 10% scale error
    assert mean_test(x + 0.05, 0.0, 1.0) < LEVEL
    assert variance_test(1.1 * x, 1.0, 0.0) < LEVEL


def test_independence_test_detects_dependence() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=20_000)
    b_ind = rng.normal(size=20_000)
    b_dep = 0.3 * a + np.sqrt(1 - 0.09) * rng.normal(size=20_000)
    cf = _normal_cf(0.0, 1.0)
    pairs = [(0.8, 0.8), (0.8, -0.8)]
    assert independence_test(a, b_ind, cf, cf, pairs) > 1e-3
    assert independence_test(a, b_dep, cf, cf, pairs) < LEVEL


def test_degenerate_laws_are_compared_exactly() -> None:
    x = np.full(100, 0.25)
    assert mean_test(x, 0.25, 0.0) == 1.0
    assert mean_test(x, 0.26, 0.0) == 0.0
    assert ecf_test(x, lambda u: np.exp(0.25j * u), [1.0, 3.0]) == 1.0


def test_expect_not_rejected() -> None:
    expect_not_rejected(0.5)
    with pytest.raises(LawViolation, match="rejected"):
        expect_not_rejected(1e-9, what="cf")
