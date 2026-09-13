"""
Exactness tests for GammaProcess.

Each test verifies a closed-form result against a published formula.
These tests must never be skipped or marked xfail.
"""

import numpy as np
import pytest

from spxa.zoo.levy import GammaProcess


@pytest.mark.exactness
def test_gamma_is_subordinator() -> None:
    """
    GammaProcess is a subordinator (non-decreasing paths).
    Cont & Tankov (2004), Table 4.1: Gamma process has non-negative jumps only.
    """
    g = GammaProcess(a=1.0, b=1.0)
    assert g.properties.is_subordinator is True


@pytest.mark.exactness
def test_gamma_no_gaussian_component() -> None:
    """
    GammaProcess has σ² = 0 — pure jump, no Brownian component.
    Cont & Tankov (2004), Table 4.1: Gamma process Lévy triplet has σ=0.
    """
    g = GammaProcess(a=2.0, b=3.0)
    assert g.triplet.sigma_sq == pytest.approx(0.0)


@pytest.mark.exactness
def test_gamma_infinite_activity() -> None:
    """
    GammaProcess has infinite activity: ν(ℝ\{0}) = ∞.
    The Lévy density ∫_0^∞ a·x^{-1}·e^{-bx} dx diverges at 0.
    Cont & Tankov (2004), Table 4.1.
    """
    import numpy as np
    g = GammaProcess(a=1.0, b=1.0)
    assert np.isinf(g.triplet.nu.total_mass())


@pytest.mark.exactness
def test_gamma_cumulant_1() -> None:
    """
    κ_1(X_1) = a/b for GammaProcess.
    Cont & Tankov (2004), Table 4.2: κ_n = a·(n-1)!/b^n, so κ_1 = a/b.
    """
    g = GammaProcess(a=2.0, b=4.0)
    assert g.cumulant_exact(1) == pytest.approx(0.5, rel=1e-10)


@pytest.mark.exactness
def test_gamma_cumulant_2() -> None:
    """
    κ_2(X_1) = a/b² for GammaProcess.
    Cont & Tankov (2004), Table 4.2: κ_2 = a·1!/b² = a/b².
    """
    g = GammaProcess(a=2.0, b=4.0)
    assert g.cumulant_exact(2) == pytest.approx(2.0 / 16.0, rel=1e-10)


@pytest.mark.exactness
def test_gamma_cumulant_3() -> None:
    """
    κ_3(X_1) = 2a/b³ for GammaProcess.
    Cont & Tankov (2004), Table 4.2: κ_3 = a·2!/b³ = 2a/b³.
    """
    g = GammaProcess(a=3.0, b=2.0)
    assert g.cumulant_exact(3) == pytest.approx(2 * 3.0 / 8.0, rel=1e-10)


@pytest.mark.exactness
def test_gamma_cumulant_time_scaling() -> None:
    """
    κ_n(X_t) = t · κ_n(X_1) for any Lévy process.
    Sato (1999), Theorem 25.3: cumulants scale linearly with t.
    """
    g = GammaProcess(a=1.0, b=2.0)
    t = 3.5
    kappa1_t1 = g.cumulant_exact(1, t=1.0)
    kappa1_t = g.cumulant_exact(1, t=t)
    assert kappa1_t == pytest.approx(t * kappa1_t1, rel=1e-10)


@pytest.mark.exactness
def test_gamma_simulate_mean() -> None:
    """
    Empirical mean of GammaProcess(a,b)_1 converges to a/b.
    E[X_1] = a/b (mean of Gamma(a,b) distribution).
    """
    rng = np.random.default_rng(0)
    a, b = 2.0, 3.0
    g = GammaProcess(a=a, b=b)
    paths = g.simulate(n_steps=50, n_paths=20_000, T=1.0, rng=rng)
    empirical_mean = paths[:, -1].mean()
    assert empirical_mean == pytest.approx(a / b, rel=0.02)


@pytest.mark.exactness
def test_gamma_simulate_variance() -> None:
    """
    Empirical variance of GammaProcess(a,b)_1 converges to a/b².
    Var(X_1) = a/b² (variance of Gamma distribution).
    """
    rng = np.random.default_rng(1)
    a, b = 2.0, 3.0
    g = GammaProcess(a=a, b=b)
    paths = g.simulate(n_steps=50, n_paths=20_000, T=1.0, rng=rng)
    empirical_var = paths[:, -1].var()
    assert empirical_var == pytest.approx(a / b**2, rel=0.03)


@pytest.mark.exactness
def test_gamma_simulate_paths_nondecreasing() -> None:
    """
    GammaProcess paths are non-decreasing (subordinator property).
    All increments must be non-negative.
    """
    rng = np.random.default_rng(7)
    g = GammaProcess(a=1.0, b=1.0)
    paths = g.simulate(n_steps=200, n_paths=100, T=1.0, rng=rng)
    increments = np.diff(paths, axis=1)
    assert np.all(increments >= -1e-14)


@pytest.mark.exactness
def test_gamma_addition_triplet_sigma_sq() -> None:
    """
    Sum of two Gamma processes has σ²=0 (both are pure-jump).
    Sato (1999), Proposition 11.10: σ²_Z = σ²_X + σ²_Y = 0.
    """
    g1 = GammaProcess(a=1.0, b=2.0)
    g2 = GammaProcess(a=2.0, b=1.0)
    z = g1 + g2
    assert z.triplet.sigma_sq == pytest.approx(0.0)
