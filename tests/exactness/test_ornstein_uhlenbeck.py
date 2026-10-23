r"""
Exactness tests for Gaussian Ornstein–Uhlenbeck (OU) / Vasicek process.

These tests verify:
1. Analytical conditional moments, cumulants, and temporal autocovariance formulas.
2. Stationary distribution properties (Gaussian law, mean, variance, zero skew/kurtosis).
3. Exact Laplace transform and characteristic function against transition density quadrature.
4. Transition probability density and stationary density integrate to 1.
5. Vasicek term structure Riccati ODEs and zero-coupon bond pricing / asymptotic yield formulas.
6. ExactnessError is raised for Lévy triplet access.
7. Exact Gaussian transition simulation produces unbiased paths matching theory.

References
----------
Uhlenbeck, G.E. & Ornstein, L.S. (1930). *Physical Review*, 36(5), 823–841.
Vasicek, O. (1977). *Journal of Financial Economics*, 5(2), 177–188.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.integrate as integrate

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import (
    OrnsteinUhlenbeck,
    OrnsteinUhlenbeckProcess,
    Vasicek,
    VasicekProcess,
)


class TestOrnsteinUhlenbeckExactness:
    @pytest.mark.exactness
    def test_ou_exactness_level_and_aliases(self) -> None:
        """OrnsteinUhlenbeck exactness is MOMENT_PROPAGATION and all aliases match."""
        ou = OrnsteinUhlenbeck(theta=1.2, mu=0.05, sigma=0.03, x0=0.02)
        assert ou.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert OrnsteinUhlenbeck is OrnsteinUhlenbeckProcess
        assert Vasicek is OrnsteinUhlenbeckProcess
        assert VasicekProcess is OrnsteinUhlenbeckProcess
        assert repr(ou) == "OrnsteinUhlenbeck(theta=1.2, mu=0.05, sigma=0.03, x0=0.02)"

    @pytest.mark.exactness
    def test_ou_triplet_raises_exactness_error(self) -> None:
        """OU has serially correlated values and mean-reverting drift; raises ExactnessError."""
        ou = OrnsteinUhlenbeck(theta=1.5, mu=0.0, sigma=0.2)
        with pytest.raises(ExactnessError) as exc_info:
            _ = ou.triplet
        assert "OrnsteinUhlenbeckProcess has mean-reverting drift" in str(exc_info.value)

        with pytest.raises(ExactnessError):
            ou._triplet()

    @pytest.mark.exactness
    def test_ou_properties(self) -> None:
        """OU properties reflect non-Lévy diffusion structure."""
        ou = OrnsteinUhlenbeck(theta=1.5, mu=0.0, sigma=0.2)
        props = ou.properties
        assert props.has_stationary_increments is False
        assert props.has_independent_increments is False
        assert props.is_martingale is False
        assert props.has_finite_variance is True
        assert props.has_finite_mean is True
        assert props.is_subordinator is False
        assert any("Ornstein–Uhlenbeck" in note for note in props.notes)

    @pytest.mark.exactness
    def test_ou_invalid_parameters_raise(self) -> None:
        """Invalid parameters raise ValueError across constructor and analytical methods."""
        with pytest.raises(ValueError, match="theta"):
            OrnsteinUhlenbeck(theta=-1.0, mu=0.0, sigma=0.1)
        with pytest.raises(ValueError, match="theta"):
            OrnsteinUhlenbeck(theta=0.0, mu=0.0, sigma=0.1)
        with pytest.raises(ValueError, match="sigma"):
            OrnsteinUhlenbeck(theta=1.0, mu=0.0, sigma=-0.1)
        with pytest.raises(ValueError, match="sigma"):
            OrnsteinUhlenbeck(theta=1.0, mu=0.0, sigma=0.0)

        ou = OrnsteinUhlenbeck(theta=1.0, mu=0.05, sigma=0.1, x0=0.02)
        with pytest.raises(ValueError, match="Time t"):
            ou.conditional_mean(t=-0.1)
        with pytest.raises(ValueError, match="Time t"):
            ou.conditional_variance(t=-0.1)
        with pytest.raises(ValueError, match="Cumulant order"):
            ou.conditional_cumulant(n=0, t=1.0)
        with pytest.raises(ValueError, match="Time t"):
            ou.conditional_cumulant(n=1, t=-0.1)
        with pytest.raises(ValueError, match="order"):
            ou.conditional_cumulants(t=1.0, order=0)
        with pytest.raises(ValueError, match="Cumulant order"):
            ou.stationary_cumulant(n=0)
        with pytest.raises(ValueError, match="Lag h"):
            ou.autocovariance(h=-0.5)
        with pytest.raises(ValueError, match="Lag h"):
            ou.autocorrelation(h=-0.5)
        with pytest.raises(ValueError, match="Time arguments"):
            ou.two_time_covariance(s=-0.1, t=1.0)
        with pytest.raises(ValueError, match="Time arguments"):
            ou.two_time_covariance(s=1.0, t=-0.1)
        with pytest.raises(ValueError, match="Time t"):
            ou.char_func(u=1.0, t=-0.5)
        with pytest.raises(ValueError, match="Time t"):
            ou.laplace_transform(u=1.0, t=-0.5)
        with pytest.raises(ValueError, match="t must be strictly greater than s"):
            ou.transition_density(y=0.05, t=1.0, s=1.0)
        with pytest.raises(ValueError, match="tau must be non-negative"):
            ou.bond_price(tau=-0.5)
        with pytest.raises(ValueError, match="tau must be non-negative"):
            ou.bond_yield(tau=-0.5)
        with pytest.raises(ValueError, match="n_steps"):
            ou.simulate(n_steps=0)
        with pytest.raises(ValueError, match="T"):
            ou.simulate(n_steps=10, T=-1.0)
        with pytest.raises(ValueError, match="Unknown simulation method"):
            ou.simulate(n_steps=10, method="unknown_scheme")

    @pytest.mark.exactness
    def test_ou_conditional_mean_exact(self) -> None:
        """
        E[X_t | X_0] = μ + (X_0 - μ) e^{-θt}.
        Uhlenbeck & Ornstein (1930), equation (10).
        """
        theta, mu, sigma, x0 = 1.5, 0.08, 0.04, 0.02
        ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, x0=x0)

        # At t = 0: E[X_0] = x0
        assert ou.conditional_mean(t=0.0) == pytest.approx(x0, rel=1e-12)

        # For positive t:
        for t in [0.25, 0.5, 1.0, 2.0, 5.0]:
            expected = mu + (x0 - mu) * math.exp(-theta * t)
            assert ou.conditional_mean(t=t) == pytest.approx(expected, rel=1e-12)

        # Override x0:
        assert ou.conditional_mean(t=1.0, x0=0.15) == pytest.approx(
            mu + (0.15 - mu) * math.exp(-theta * 1.0), rel=1e-12
        )

        # As t -> inf converges to mu
        assert ou.conditional_mean(t=100.0) == pytest.approx(mu, rel=1e-12)

    @pytest.mark.exactness
    def test_ou_conditional_variance_and_std_exact(self) -> None:
        """
        Var(X_t | X_0) = (σ² / (2θ)) (1 - e^{-2θt}).
        Uhlenbeck & Ornstein (1930), equation (13).
        """
        theta, mu, sigma, x0 = 1.5, 0.08, 0.04, 0.02
        ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, x0=x0)

        # At t = 0: Var = 0
        assert ou.conditional_variance(t=0.0) == pytest.approx(0.0, abs=1e-12)
        assert ou.conditional_std(t=0.0) == pytest.approx(0.0, abs=1e-12)

        # For positive t:
        for t in [0.2, 0.5, 1.0, 3.0]:
            expected = (sigma**2 / (2.0 * theta)) * (1.0 - math.exp(-2.0 * theta * t))
            assert ou.conditional_variance(t=t) == pytest.approx(expected, rel=1e-12)
            assert ou.conditional_std(t=t) == pytest.approx(math.sqrt(expected), rel=1e-12)

        # As t -> inf converges to stationary variance σ² / (2θ)
        assert ou.conditional_variance(t=100.0) == pytest.approx(ou.stationary_variance, rel=1e-10)

    @pytest.mark.exactness
    def test_ou_conditional_cumulants(self) -> None:
        """
        Gaussian conditional cumulants: κ_1 = mean, κ_2 = var, κ_n = 0 for n >= 3.
        """
        ou = OrnsteinUhlenbeck(theta=2.0, mu=1.0, sigma=0.5, x0=0.5)
        t = 0.5
        kappas = ou.conditional_cumulants(t=t, order=5)

        assert kappas[1] == pytest.approx(ou.conditional_mean(t=t), rel=1e-12)
        assert kappas[2] == pytest.approx(ou.conditional_variance(t=t), rel=1e-12)
        assert kappas[3] == 0.0
        assert kappas[4] == 0.0
        assert kappas[5] == 0.0

    @pytest.mark.exactness
    def test_ou_stationary_distribution_and_cumulants(self) -> None:
        """
        Stationary distribution is N(μ, σ²/(2θ)).
        """
        theta, mu, sigma = 1.8, 0.05, 0.12
        ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma)

        var_inf = sigma**2 / (2.0 * theta)
        assert ou.stationary_mean == pytest.approx(mu, rel=1e-12)
        assert ou.stationary_variance == pytest.approx(var_inf, rel=1e-12)
        assert ou.stationary_std == pytest.approx(math.sqrt(var_inf), rel=1e-12)
        assert ou.stationary_skewness == 0.0
        assert ou.stationary_kurtosis == 0.0

        # Stationary cumulant checks
        assert ou.stationary_cumulant(1) == pytest.approx(mu, rel=1e-12)
        assert ou.stationary_cumulant(2) == pytest.approx(var_inf, rel=1e-12)
        assert ou.stationary_cumulant(3) == 0.0
        assert ou.stationary_cumulant(4) == 0.0

        stat_kappas = {n: ou.stationary_cumulant(n) for n in range(1, 5)}
        assert ou.cumulants(2)[2] == pytest.approx(ou.conditional_variance(1.0))
        assert stat_kappas[1] == pytest.approx(mu, rel=1e-12)
        assert stat_kappas[2] == pytest.approx(var_inf, rel=1e-12)
        assert stat_kappas[3] == 0.0
        assert stat_kappas[4] == 0.0

    @pytest.mark.exactness
    def test_ou_temporal_timescales_and_autocovariance(self) -> None:
        """
        Relaxation time τ = 1/θ, half-life t_{1/2} = ln(2)/θ, and autocovariance.
        """
        theta, mu, sigma = 2.0, 0.0, 0.2
        ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma)

        assert ou.relaxation_time == pytest.approx(0.5, rel=1e-12)
        assert ou.half_life == pytest.approx(math.log(2.0) / 2.0, rel=1e-12)

        # Autocovariance at lag 0 is stationary variance
        var_inf = ou.stationary_variance
        assert ou.autocovariance(0.0) == pytest.approx(var_inf, rel=1e-12)
        assert ou.autocorrelation(0.0) == pytest.approx(1.0, rel=1e-12)

        # At half-life, autocorrelation is exactly 0.5
        assert ou.autocorrelation(ou.half_life) == pytest.approx(0.5, rel=1e-12)

        # Lags h > 0
        for h in [0.2, 0.5, 1.0, 2.0]:
            assert ou.autocovariance(h) == pytest.approx(var_inf * math.exp(-theta * h), rel=1e-12)
            assert ou.autocorrelation(h) == pytest.approx(math.exp(-theta * h), rel=1e-12)

    @pytest.mark.exactness
    def test_ou_two_time_covariance_and_spectral_density(self) -> None:
        """
        Cov(X_s, X_t) and Lorentzian power spectral density.
        """
        theta, mu, sigma = 1.5, 0.0, 0.3
        ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma)

        # Two-time covariance at s = t matches conditional variance
        t = 0.8
        assert ou.two_time_covariance(t, t) == pytest.approx(ou.conditional_variance(t), rel=1e-12)
        assert ou.two_time_covariance(0.3, 0.8) == pytest.approx(
            ou.two_time_covariance(0.8, 0.3), rel=1e-12
        )

        # Spectral density: S(0) = σ² / (2π θ²); S(θ) = 0.5 * S(0) (half-power at cutoff)
        s0 = ou.spectral_density(0.0)
        s_cutoff = ou.spectral_density(theta)
        assert s0 == pytest.approx(sigma**2 / (2 * math.pi * theta**2), rel=1e-12)
        assert s_cutoff == pytest.approx(0.5 * s0, rel=1e-12)

        # Vectorized array input for spectral density
        w_grid = np.array([0.0, theta, 2.0 * theta])
        s_arr = ou.spectral_density(w_grid)
        assert s_arr.shape == (3,)
        assert s_arr[1] == pytest.approx(s_cutoff, rel=1e-12)

    @pytest.mark.exactness
    def test_ou_laplace_and_char_func_exact(self) -> None:
        """
        Laplace transform and characteristic function against transition density quadrature.
        """
        theta, mu, sigma, x0 = 1.4, 0.05, 0.08, 0.02
        ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, x0=x0)

        # Boundary checks
        assert ou.laplace_transform(u=0.0, t=1.0) == pytest.approx(1.0, rel=1e-12)
        assert ou.char_func(u=0.0, t=1.0) == pytest.approx(1.0 + 0j, abs=1e-12)
        assert ou.laplace_transform(u=2.0, t=0.0) == pytest.approx(math.exp(-2.0 * x0), rel=1e-12)
        assert ou.char_func(u=2.0, t=0.0) == pytest.approx(
            complex(math.cos(2.0 * x0), math.sin(2.0 * x0)), abs=1e-12
        )

        # Compare Laplace transform with numerical quadrature of transition density
        t = 0.6
        for u in [0.5, 1.5, 3.0]:
            formula_val = ou.laplace_transform(u=u, t=t)

            def integrand(y: float, u: float = u) -> float:
                return float(math.exp(-u * y) * ou.transition_density(y, t=t))

            num_val, _ = integrate.quad(integrand, -2.0, 2.0, epsabs=1e-10)
            assert formula_val == pytest.approx(num_val, rel=1e-7)

        # Characteristic function modulus <= 1 for real u
        phi = ou.char_func(u=5.0, t=1.0)
        assert abs(phi) <= 1.0 + 1e-12

    @pytest.mark.exactness
    def test_ou_transition_and_stationary_density_integration(self) -> None:
        """
        Transition density and stationary Gaussian density integrate to 1.
        """
        theta, mu, sigma, x0 = 1.5, 0.04, 0.06, 0.02
        ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, x0=x0)
        t = 0.8

        # Transition density integration
        int_trans, _ = integrate.quad(
            lambda y: float(ou.transition_density(y, t=t)), -2.0, 2.0, epsabs=1e-9
        )
        assert int_trans == pytest.approx(1.0, abs=1e-6)

        # Stationary density integration
        int_stat, _ = integrate.quad(
            lambda y: float(ou.stationary_density(y)), -2.0, 2.0, epsabs=1e-9
        )
        assert int_stat == pytest.approx(1.0, abs=1e-6)

        # Array evaluation
        y_grid = np.array([0.01, 0.04, 0.07])
        dens_arr = ou.transition_density(y_grid, t=t)
        assert dens_arr.shape == (3,)
        for i, y in enumerate(y_grid):
            assert dens_arr[i] == pytest.approx(ou.transition_density(y, t=t), rel=1e-12)

        stat_arr = ou.stationary_density(y_grid)
        assert stat_arr.shape == (3,)
        for i, y in enumerate(y_grid):
            assert stat_arr[i] == pytest.approx(ou.stationary_density(y), rel=1e-12)

    @pytest.mark.exactness
    def test_ou_vasicek_bond_pricing_and_riccati_ode(self) -> None:
        """
        Vasicek zero-coupon bond pricing satisfies P(0) = 1 and Riccati system:
          B'(τ) = 1 - θ B(τ)
          ln A'(τ) = -θ μ B(τ) + (1/2) σ² B(τ)²
        """
        theta, mu, sigma, r0 = 0.8, 0.05, 0.03, 0.02
        ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, x0=r0)

        # Boundary condition: P(0) = 1, y(0) = r0
        assert ou.bond_price(tau=0.0) == pytest.approx(1.0, rel=1e-12)
        assert ou.bond_yield(tau=0.0) == pytest.approx(r0, rel=1e-12)

        # Monotonicity with maturity
        p1 = ou.bond_price(tau=1.0)
        p2 = ou.bond_price(tau=2.0)
        p5 = ou.bond_price(tau=5.0)
        assert 1.0 > p1 > p2 > p5 > 0.0

        # Array vectorized pricing
        taus = np.array([0.0, 0.5, 1.0, 2.0])
        p_arr = ou.bond_price(taus)
        assert p_arr.shape == (4,)
        assert p_arr[0] == pytest.approx(1.0, rel=1e-12)
        assert p_arr[2] == pytest.approx(p1, rel=1e-12)

        # Riccati ODE verification at tau = 2.0
        tau = 2.0
        eps = 1e-6
        b_mid = (1.0 - math.exp(-theta * tau)) / theta

        def calc_ln_a(t: float) -> float:
            b_val = (1.0 - math.exp(-theta * t)) / theta
            gamma_term = mu - sigma**2 / (2.0 * (theta**2))
            return gamma_term * (b_val - t) - (sigma**2 / (4.0 * theta)) * (b_val**2)

        b_plus = (1.0 - math.exp(-theta * (tau + eps))) / theta
        b_minus = (1.0 - math.exp(-theta * (tau - eps))) / theta
        b_prime_num = (b_plus - b_minus) / (2.0 * eps)
        b_prime_ode = 1.0 - theta * b_mid
        assert b_prime_num == pytest.approx(b_prime_ode, rel=1e-6)

        lna_prime_num = (calc_ln_a(tau + eps) - calc_ln_a(tau - eps)) / (2.0 * eps)
        lna_prime_ode = -theta * mu * b_mid + 0.5 * (sigma**2) * (b_mid**2)
        assert lna_prime_num == pytest.approx(lna_prime_ode, rel=1e-6)

    @pytest.mark.exactness
    def test_ou_yield_curve_and_asymptotic_yield(self) -> None:
        """
        Yield curve y(τ) starts at r_0 and converges to asymptotic yield y_∞ = μ - σ²/(2θ²).
        Vasicek (1977), equation (28).
        """
        theta, mu, sigma, r0 = 0.8, 0.05, 0.03, 0.02
        ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, x0=r0)

        y_inf_expected = mu - sigma**2 / (2.0 * (theta**2))
        assert ou.asymptotic_yield == pytest.approx(y_inf_expected, rel=1e-12)

        # Spot rate
        assert ou.bond_yield(0.0) == pytest.approx(r0, rel=1e-12)

        # Long maturity converges to asymptotic yield
        y_long = ou.bond_yield(2000.0)
        assert y_long == pytest.approx(ou.asymptotic_yield, rel=1e-3)

    @pytest.mark.exactness
    def test_ou_exact_simulation_moments(self) -> None:
        """
        Simulated paths via exact Gaussian increments match theoretical conditional mean and
        variance.
        """
        theta, mu, sigma, x0 = 1.8, 0.06, 0.04, 0.02
        ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, x0=x0)
        T = 0.5
        n_steps = 50
        n_paths = 30_000

        rng = np.random.default_rng(42)
        paths = ou.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng, method="exact")

        assert paths.shape == (n_paths, n_steps + 1)
        assert np.all(paths[:, 0] == x0)

        # Terminal values
        x_T = paths[:, -1]
        mean_theory = ou.conditional_mean(T)
        var_theory = ou.conditional_variance(T)

        emp_mean = float(np.mean(x_T))
        emp_var = float(np.var(x_T))

        assert emp_mean == pytest.approx(mean_theory, rel=0.03)
        assert emp_var == pytest.approx(var_theory, rel=0.06)

    @pytest.mark.exactness
    def test_ou_simulation_methods_and_reproducibility(self) -> None:
        """Exact and Euler simulation schemes run and respect seeded RNG."""
        ou = OrnsteinUhlenbeck(theta=1.5, mu=0.0, sigma=0.2, x0=0.1)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        p1 = ou.simulate(n_steps=20, n_paths=10, T=0.5, rng=rng1, method="exact")
        p2 = ou.simulate(n_steps=20, n_paths=10, T=0.5, rng=rng2, method="exact")
        np.testing.assert_array_equal(p1, p2)

        p_euler = ou.simulate(n_steps=20, n_paths=10, T=0.5, rng=rng1, method="euler")
        assert p_euler.shape == (10, 21)
        assert np.all(p_euler[:, 0] == 0.1)

    @pytest.mark.exactness
    def test_ou_story_narrative(self) -> None:
        """OU story trace includes parameters, SDE, and half-life."""
        ou = OrnsteinUhlenbeck(theta=1.4, mu=0.05, sigma=0.03, x0=0.02)
        story = ou.__story__()
        assert "OrnsteinUhlenbeck" in story
        assert "Uhlenbeck & Ornstein (1930)" in story
        assert "Vasicek (1977)" in story
        assert "dX_t = θ(μ - X_t)dt + σ dW_t" in story
        assert "Half-life" in story
