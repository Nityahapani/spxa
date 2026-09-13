"""
Exactness tests for VarianceGamma.

Each test verifies a closed-form result against a published formula.
These tests must never be skipped or marked xfail.
"""

import numpy as np
import pytest

from spxa.zoo.levy import VarianceGamma


@pytest.mark.exactness
def test_vg_no_gaussian_component() -> None:
    """
    VG process has σ²=0 in its Lévy triplet — it is a pure-jump process.
    Madan, Carr & Chang (1998): VG has infinite activity and finite variation,
    no Brownian component.
    """
    vg = VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)
    assert vg.triplet.sigma_sq == pytest.approx(0.0, abs=1e-10)


@pytest.mark.exactness
def test_vg_infinite_activity() -> None:
    """
    VG Lévy measure has infinite total mass.
    Madan, Carr & Chang (1998): VG has infinitely many small jumps per unit time.
    """
    vg = VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)
    assert np.isinf(vg.triplet.nu.total_mass())


@pytest.mark.exactness
def test_vg_char_func_at_zero() -> None:
    """
    φ_VG(0; t) = 1 for any valid characteristic function.
    E[e^{i·0·X_t}] = E[1] = 1. Madan, Carr & Chang (1998), eq. (4).
    """
    vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
    cf = vg.char_func_exact(u=0.0, t=1.0)
    assert abs(cf - 1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.exactness
def test_vg_char_func_modulus_leq_1() -> None:
    """
    |φ(u)| ≤ 1 for all u — characteristic functions are bounded by 1.
    This holds for any probability distribution.
    """
    vg = VarianceGamma(sigma=0.3, nu=0.2, theta=-0.05)
    u_vals = np.linspace(-5.0, 5.0, 50)
    cf_vals = vg.char_func_exact(u=u_vals, t=1.0)
    assert np.all(np.abs(cf_vals) <= 1.0 + 1e-12)


@pytest.mark.exactness
def test_vg_cumulant_1_mean() -> None:
    """
    κ_1 = θ for VG(σ, ν, θ).
    Madan, Carr & Chang (1998), Section 2: mean per unit time equals θ.
    """
    vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.15)
    kappas = vg.cumulants(order=2)
    assert kappas[1] == pytest.approx(-0.15, rel=1e-4)


@pytest.mark.exactness
def test_vg_cumulant_2_variance() -> None:
    """
    κ_2 = σ² + θ²ν for VG(σ, ν, θ).
    Madan, Carr & Chang (1998), Section 2.
    """
    sigma, nu, theta = 0.2, 0.1, -0.15
    vg = VarianceGamma(sigma=sigma, nu=nu, theta=theta)
    expected = sigma**2 + theta**2 * nu
    kappas = vg.cumulants(order=2)
    assert kappas[2] == pytest.approx(expected, rel=1e-3)


@pytest.mark.exactness
def test_vg_symmetric_when_theta_zero() -> None:
    """
    VG with θ=0 is symmetric: characteristic function is real-valued for real u.
    φ(u) = (1 + σ²νu²/2)^{-t/ν} is real when θ=0.
    Madan, Carr & Chang (1998), eq. (4).
    """
    vg = VarianceGamma(sigma=0.3, nu=0.2, theta=0.0)
    u_vals = np.array([0.5, 1.0, 2.0])
    cf_vals = vg.char_func_exact(u=u_vals, t=1.0)
    assert np.all(np.abs(cf_vals.imag) < 1e-12)


@pytest.mark.exactness
def test_vg_simulate_mean() -> None:
    """
    Empirical mean of VG_1 converges to θ.
    E[X_1] = θ (Madan, Carr & Chang 1998, Section 2).
    """
    rng = np.random.default_rng(10)
    theta = -0.1
    vg = VarianceGamma(sigma=0.2, nu=0.1, theta=theta)
    paths = vg.simulate(n_steps=50, n_paths=30_000, T=1.0, rng=rng)
    empirical_mean = paths[:, -1].mean()
    assert empirical_mean == pytest.approx(theta, abs=0.01)


@pytest.mark.exactness
def test_vg_simulate_variance() -> None:
    """
    Empirical variance of VG_1 converges to σ² + θ²ν.
    Var(X_1) = σ² + θ²ν (Madan, Carr & Chang 1998, Section 2).
    """
    rng = np.random.default_rng(11)
    sigma, nu, theta = 0.2, 0.1, -0.1
    vg = VarianceGamma(sigma=sigma, nu=nu, theta=theta)
    paths = vg.simulate(n_steps=50, n_paths=30_000, T=1.0, rng=rng)
    empirical_var = paths[:, -1].var()
    expected = sigma**2 + theta**2 * nu
    assert empirical_var == pytest.approx(expected, rel=0.05)


@pytest.mark.exactness
def test_vg_char_func_time_additivity() -> None:
    """
    For a Lévy process, φ(u; s+t) = φ(u; s) · φ(u; t).
    This is equivalent to the characteristic exponent being linear in t:
    log φ(u; t) = t · ψ(u). Verified for VG exact formula.
    Sato (1999), Definition 1.6.
    """
    vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.05)
    u = 1.5
    s, t = 0.4, 0.7
    cf_s = complex(vg.char_func_exact(u=u, t=s))
    cf_t = complex(vg.char_func_exact(u=u, t=t))
    cf_st = complex(vg.char_func_exact(u=u, t=s + t))
    assert abs(cf_s * cf_t - cf_st) == pytest.approx(0.0, abs=1e-12)
