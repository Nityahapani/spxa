"""
Tests for rough volatility, fractional kernels, rBergomi model, and fractional Riccati solver.
"""

import numpy as np
import pytest

from spxa.rough.kernel import FractionalKernel
from spxa.rough.rbergomi import RoughBergomi
from spxa.rough.riccati import FractionalRiccatiSolver


def test_fractional_kernel_and_singular_power_law():
    kernel = FractionalKernel(hurst=0.1)
    assert kernel.alpha == pytest.approx(-0.4)

    # Singular power law: K_H(t) -> infty as t -> 0
    k_small = kernel.kernel_value(0.001)
    k_mid = kernel.kernel_value(0.1)
    k_large = kernel.kernel_value(1.0)

    assert k_small > k_mid > k_large > 0.0

    # Hybrid convolution
    rng = np.random.default_rng(42)
    n_steps = 100
    dt = 1.0 / n_steps
    dw = rng.normal(0.0, np.sqrt(dt), size=n_steps)
    volterra_path = kernel.hybrid_convolution(dw, dt=dt, b_cutoff=3)
    assert len(volterra_path) == n_steps + 1
    assert volterra_path[0] == 0.0
    assert not np.isnan(volterra_path).any()


def test_fbm_cholesky_covariance_structure():
    hurst = 0.2
    kernel = FractionalKernel(hurst=hurst)
    # 2000 paths on [0, 1]
    res = kernel.sample_fbm_cholesky(t_mat=1.0, n_steps=20, n_paths=2000, seed=42)
    paths = res["paths"]

    # Theoretical variance: Var(B_t^H) = t^{2H}
    emp_var_half = np.var(paths[:, 10])  # at t = 0.5
    theo_var_half = 0.5 ** (2.0 * hurst)

    emp_var_one = np.var(paths[:, -1])  # at t = 1.0
    theo_var_one = 1.0 ** (2.0 * hurst)

    assert np.isclose(emp_var_half, theo_var_half, rtol=0.12)
    assert np.isclose(emp_var_one, theo_var_one, rtol=0.10)


def test_rbergomi_simulation_and_short_time_skew():
    model = RoughBergomi(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, s0=100.0)

    # Test short-time skew blow-up as tau -> 0
    skew_001 = model.theoretical_short_time_skew(tau=0.01)
    skew_01 = model.theoretical_short_time_skew(tau=0.1)
    skew_1 = model.theoretical_short_time_skew(tau=1.0)

    # In rough volatility (H < 0.5), skew blows up for small maturities
    assert skew_001 > skew_01 > skew_1
    # Check log-ratio slope matches H - 0.5 = -0.4
    log_slope = (np.log(skew_001) - np.log(skew_1)) / (np.log(0.01) - np.log(1.0))
    assert np.isclose(log_slope, -0.4, atol=1e-5)

    # Test path simulation
    sim_res = model.simulate_paths(t_mat=1.0, n_steps=30, n_paths=50, seed=42)
    spot = sim_res["spot_paths"]
    var = sim_res["variance_paths"]

    assert spot.shape == (50, 31)
    assert var.shape == (50, 31)
    # Variance must be strictly positive
    assert (var > 0.0).all()
    # Initial spot must match s0
    assert np.allclose(spot[:, 0], 100.0)


def test_fractional_riccati_abm_solver_and_char_func():
    solver = FractionalRiccatiSolver(hurst=0.1, lambda_param=1.5, rho=-0.7, nu=0.3)

    # At u = 0, char function of log price must be identically 1.0
    cf_0 = solver.rough_heston_char_func(u=0.0, t_mat=1.0, v0=0.04, theta=0.04, n_steps=50)
    assert np.isclose(cf_0.real, 1.0, atol=1e-5)
    assert np.isclose(cf_0.imag, 0.0, atol=1e-5)

    # At u = 1.0, modulus |phi(u)| <= 1
    cf_1 = solver.rough_heston_char_func(u=1.0, t_mat=1.0, v0=0.04, theta=0.04, n_steps=50)
    modulus = abs(cf_1)
    assert modulus <= 1.0 + 1e-4


def test_volterra_joint_covariance_matches_quadrature():
    from scipy.integrate import quad

    kernel = FractionalKernel(hurst=0.1)
    a = kernel.alpha
    t = np.array([0.2, 0.5, 1.0])
    cov = kernel.volterra_joint_covariance(t)
    n = len(t)

    for i in range(n):
        for j in range(n):
            s_, l_ = min(t[i], t[j]), max(t[i], t[j])
            ref_ww = quad(lambda u: (s_ - u) ** a * (l_ - u) ** a, 0.0, s_, limit=200)[0]
            assert cov[i, j] == pytest.approx(ref_ww, rel=1e-6)
            ref_wz = quad(lambda u: (t[i] - u) ** a, 0.0, min(t[i], t[j]), limit=200)[0]
            assert cov[i, n + j] == pytest.approx(ref_wz, rel=1e-6)
            assert cov[n + i, n + j] == pytest.approx(min(t[i], t[j]))

    assert np.all(np.linalg.eigvalsh(cov) > -1e-12)


def test_rbergomi_forward_variance_and_leverage():
    model = RoughBergomi(hurst=0.1, eta=1.9, rho=-0.9, xi0=0.04, s0=100.0)
    res = model.simulate_paths(t_mat=1.0, n_steps=50, n_paths=20000, seed=7)
    v = res["variance_paths"]
    s = res["spot_paths"]

    # Flat forward variance curve: E[v_t] = xi0
    assert np.mean(v[:, 25]) == pytest.approx(0.04, rel=0.05)
    assert np.mean(v[:, -1]) == pytest.approx(0.04, rel=0.08)
    # Discounted spot is a martingale
    assert np.mean(s[:, -1]) == pytest.approx(100.0, rel=0.01)
    # Leverage: returns negatively correlated with variance when rho < 0
    corr = np.corrcoef(np.log(s[:, 10] / 100.0), v[:, 10])[0, 1]
    assert corr < -0.2


def test_rbergomi_short_time_skew_constant_matches_fukasawa():
    model = RoughBergomi(hurst=0.1, eta=1.5, rho=-0.7)
    h = 0.1
    expected = 0.7 * 1.5 * np.sqrt(2 * h) / (2 * (h + 0.5) * (h + 1.5))
    assert model.theoretical_short_time_skew(1.0) == pytest.approx(expected, rel=1e-12)


def test_rbergomi_monte_carlo_skew_matches_asymptotic():
    from scipy.optimize import brentq
    from scipy.stats import norm

    model = RoughBergomi(hurst=0.1, eta=0.5, rho=-0.7, xi0=0.04, s0=1.0)
    t_mat = 0.1
    s_t = model.simulate_paths(t_mat=t_mat, n_steps=64, n_paths=60000, seed=3)["spot_paths"][:, -1]

    def implied_vol(k: float) -> float:
        strike = np.exp(k)
        price = np.maximum(s_t - strike, 0.0).mean()
        sq = np.sqrt(t_mat)

        def bs(sig: float) -> float:
            d1 = -k / (sig * sq) + 0.5 * sig * sq
            return norm.cdf(d1) - strike * norm.cdf(d1 - sig * sq) - price

        return brentq(bs, 1e-3, 3.0)

    h = 0.02
    mc_skew = (implied_vol(h) - implied_vol(-h)) / (2 * h)
    assert mc_skew < 0.0
    assert mc_skew == pytest.approx(-model.theoretical_short_time_skew(t_mat), rel=0.15)


def test_fractional_riccati_reduces_to_classical_heston():
    from spxa.zoo.beyond.heston import HestonProcess

    lam, nu, rho, v0, theta = 1.5, 0.3, -0.7, 0.04, 0.05
    heston = HestonProcess(v0=v0, kappa=lam, theta=theta, sigma_v=nu, rho=rho)
    solver = FractionalRiccatiSolver(hurst=0.5, lambda_param=lam, rho=rho, nu=nu)
    for u in [0.5, 1.0, 3.0, -2.0]:
        cf_rough = solver.rough_heston_char_func(u, t_mat=1.0, v0=v0, theta=theta, n_steps=400)
        cf_heston = heston.char_func(u, t=1.0)
        assert abs(cf_rough - cf_heston) < 1e-6


def test_rough_heston_martingale_and_convergence():
    solver = FractionalRiccatiSolver(hurst=0.1, lambda_param=1.5, rho=-0.7, nu=0.3)
    # E[S_T / S_0] = phi(-i) = 1
    cf_mart = solver.rough_heston_char_func(-1j, t_mat=1.0, v0=0.04, theta=0.05, n_steps=100)
    assert abs(cf_mart - 1.0) < 1e-12
    # Grid refinement converges
    cf_200 = solver.rough_heston_char_func(1.0, t_mat=1.0, v0=0.04, theta=0.05, n_steps=200)
    cf_400 = solver.rough_heston_char_func(1.0, t_mat=1.0, v0=0.04, theta=0.05, n_steps=400)
    assert abs(cf_200 - cf_400) < 1e-5
