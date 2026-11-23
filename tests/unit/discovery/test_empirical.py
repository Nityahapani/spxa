"""Empirical summaries: k-statistics, their covariance and the Lévy diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from spxa.discovery.empirical import (
    acf,
    cumulant_covariance,
    from_statistics,
    k_statistics,
    ljung_box,
    rank_ljung_box,
    summarize,
)


def test_k_statistics_match_scipy() -> None:
    x = np.random.default_rng(0).gamma(2.0, 1.5, size=257)
    k = k_statistics(x)
    np.testing.assert_allclose(k[1:], [stats.kstat(x, n) for n in (2, 3, 4)], rtol=1e-12)
    assert k[0] == pytest.approx(x.mean())
    with pytest.raises(ValueError):
        k_statistics([1.0, 2.0, 3.0])


def test_k_statistics_are_unbiased() -> None:
    # Gamma(shape 2, scale 1.5): κ_n = 2 (n-1)! 1.5^n
    rng = np.random.default_rng(1)
    ks = np.array([k_statistics(rng.gamma(2.0, 1.5, size=30)) for _ in range(20_000)])
    exact = np.array([2 * 1.5, 2 * 1.5**2, 2 * 2 * 1.5**3, 2 * 6 * 1.5**4])
    se = ks.std(axis=0) / np.sqrt(len(ks))
    assert np.all(np.abs(ks.mean(axis=0) - exact) < 4 * se)


def test_influence_function_covariance_matches_the_exact_asymptotics() -> None:
    # Laplace(0, 1): κ_2 = 2, κ_4 = 12, κ_6 = 240, κ_8 = 10080, odd cumulants zero.
    # Kendall & Stuart (1977, §12.15) give n·Var(k_j) for symmetric laws:
    #   k_1: κ_2 = 2
    #   k_2: κ_4 + 2κ_2² = 20
    #   k_3: κ_6 + 9κ_2κ_4 + 6κ_2³ = 504
    #   k_4: κ_8 + 16κ_2κ_6 + 34κ_4² + 72κ_2²κ_4 + 24κ_2⁴ = 26496
    x = np.random.default_rng(2).laplace(size=400_000)
    est = np.diag(cumulant_covariance(x)) * x.size
    np.testing.assert_allclose(est[:3], [2, 20, 504], rtol=0.05)
    # the eighth sample moment of a Laplace law converges slowly
    assert est[3] == pytest.approx(26496, rel=0.15)
    # Gaussian reference: Var(k_j) ≈ j! σ^{2j} / n
    g = cumulant_covariance(np.random.default_rng(3).normal(size=400_000)) * 400_000
    np.testing.assert_allclose(np.diag(g), [1, 2, 6, 24], rtol=0.05)


def test_hac_covariance_widens_under_dependence() -> None:
    rng = np.random.default_rng(3)
    e = rng.normal(size=20_001)
    ar = e[1:] + 0.6 * e[:-1]  # MA(1): long-run variance of the mean is (1.6)² = 2.56
    plain = cumulant_covariance(ar)[0, 0] * ar.size
    hac = cumulant_covariance(ar, hac_lags=20)[0, 0] * ar.size
    assert plain == pytest.approx(1.36, rel=0.05)
    assert hac == pytest.approx(2.56, rel=0.1)


def test_acf_and_ljung_box() -> None:
    rng = np.random.default_rng(4)
    iid = rng.normal(size=5_000)
    r = acf(iid, 5)
    assert np.all(np.abs(r) < 4 / np.sqrt(5_000))
    q, p = ljung_box(iid, 10)
    assert p > 0.001 and q > 0
    e = rng.normal(size=5_001)
    q_ma, p_ma = ljung_box(e[1:] + 0.5 * e[:-1], 10)
    assert p_ma < 1e-10
    assert np.all(acf(np.ones(10), 3) == 0)


def test_summary_per_unit_time_and_horizon_scaling() -> None:
    rng = np.random.default_rng(5)
    dt = 1 / 252
    x = rng.normal(0.08 * dt, 0.2 * np.sqrt(dt), size=10_000)
    s = summarize(x, dt=dt)
    assert s.n == 10_000 and s.dt == dt
    assert s.variance == pytest.approx(0.04, rel=0.05)
    assert s.mean == pytest.approx(0.08, abs=4 * s.standard_errors[0])
    assert s.is_levy_consistent()
    # excess kurtosis of a Lévy increment decays like 1/horizon
    s2 = summarize(rng.laplace(size=50_000), dt=1.0)
    assert s2.excess_kurtosis() == pytest.approx(3.0, rel=0.1)
    assert s2.excess_kurtosis(horizon=10.0) == pytest.approx(s2.excess_kurtosis() / 10)
    assert s2.skewness(horizon=4.0) == pytest.approx(s2.skewness() / 2)
    with pytest.raises(ValueError):
        summarize(x[:5])
    with pytest.raises(ValueError):
        summarize(x, dt=0.0)


def test_volatility_clustering_is_flagged() -> None:
    rng = np.random.default_rng(6)
    n = 4_000
    sig = np.empty(n)
    r = np.empty(n)
    sig[0], r[0] = 0.01, 0.0
    for t in range(1, n):  # GARCH(1,1)
        sig[t] = np.sqrt(1e-6 + 0.1 * r[t - 1] ** 2 + 0.88 * sig[t - 1] ** 2)
        r[t] = sig[t] * rng.normal()
    s = summarize(r)
    assert s.ljung_box_squared[1] < 1e-6
    assert s.rank_ljung_box_volatility[1] < 1e-6
    assert not s.is_levy_consistent()


def test_heavy_tailed_iid_increments_are_not_flagged() -> None:
    # daily Gamma(4, 25) increments have excess kurtosis 6/(4/252) = 378;
    # the classical tests reject independence far above their level, the
    # rank tests do not
    classical = ranks = 0
    for seed in range(40):
        x = np.random.default_rng(seed).gamma(4 / 252, 1 / 25, size=5_000)
        s = summarize(x, dt=1 / 252)
        classical += min(s.ljung_box_returns[1], s.ljung_box_squared[1]) < 0.01
        ranks += not s.is_levy_consistent()
    assert classical >= 3
    assert ranks <= 2


def test_rank_ljung_box_detects_dependence() -> None:
    e = np.random.default_rng(3).standard_t(3, size=3_001)
    _, p_iid = rank_ljung_box(e, 10)
    _, p_ma = rank_ljung_box(e[1:] + 0.5 * e[:-1], 10)
    assert p_iid > 0.01
    assert p_ma < 1e-10


def test_from_statistics() -> None:
    s = from_statistics(
        mean=0.001, variance=0.0002, skewness=-0.5, excess_kurtosis=6.0, dt=1 / 252, n=2520
    )
    k = s.cumulants
    assert k[1] == pytest.approx(0.0002 * 252)
    assert s.skewness() == pytest.approx(-0.5) and s.excess_kurtosis() == pytest.approx(6.0)
    assert s.excess_kurtosis(horizon=1.0) == pytest.approx(6.0 / 252)
    assert s.standard_errors is not None and s.is_levy_consistent()
    assert from_statistics(0.0, 1.0).covariance is None
    with pytest.raises(ValueError):
        from_statistics(0.0, -1.0)
