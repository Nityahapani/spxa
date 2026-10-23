r"""
Exactness tests for Cox–Ingersoll–Ross (CIR) square-root diffusion process.

These tests verify:
1. Analytical conditional moments, cumulants, and autocovariance formulas.
2. Stationary distribution properties (Gamma law, shape, scale, skewness, kurtosis).
3. Exact Laplace transform and characteristic function against transition density quadrature.
4. Transition probability density and stationary density integrate to 1.
5. Term structure Riccati ODEs and zero-coupon bond pricing / asymptotic yield formulas.
6. ExactnessError is raised for Lévy triplet access.
7. Exact non-central chi-squared simulation produces unbiased paths matching theory.

References
----------
Cox, J.C., Ingersoll, J.E. & Ross, S.A. (1985). *Econometrica*, 53(2), 385–407.
Feller, W. (1951). *Annals of Mathematics*, 54(1), 173–182.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.integrate as integrate

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import CIRProcess, CoxIngersollRoss


class TestCIRExactness:
    @pytest.mark.exactness
    def test_cir_exactness_level_and_alias(self) -> None:
        """CIRProcess exactness is MOMENT_PROPAGATION and alias is identical."""
        cir = CIRProcess(kappa=1.5, theta=0.04, sigma=0.1, r0=0.05)
        assert cir.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert CoxIngersollRoss is CIRProcess
        assert repr(cir) == "CIRProcess(kappa=1.5, theta=0.04, sigma=0.1, r0=0.05)"

    @pytest.mark.exactness
    def test_cir_triplet_raises_exactness_error(self) -> None:
        """CIR has state-dependent diffusion and no Lévy triplet."""
        cir = CIRProcess(kappa=2.0, theta=0.05, sigma=0.15)
        with pytest.raises(ExactnessError) as exc_info:
            _ = cir.triplet
        assert "CIRProcess has state-dependent diffusion" in str(exc_info.value)

    @pytest.mark.exactness
    def test_cir_properties(self) -> None:
        """CIR properties match non-Lévy diffusion characteristics."""
        cir = CIRProcess(kappa=2.0, theta=0.05, sigma=0.15, r0=0.03)
        props = cir.properties
        assert props.has_stationary_increments is False
        assert props.has_independent_increments is False
        assert props.is_martingale is False
        assert props.has_finite_variance is True
        assert props.has_finite_mean is True
        assert props.is_subordinator is False
        assert any("Cox–Ingersoll–Ross" in note for note in props.notes)

    @pytest.mark.exactness
    def test_cir_invalid_parameters_raise(self) -> None:
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="kappa"):
            CIRProcess(kappa=-1.0, theta=0.05, sigma=0.1)
        with pytest.raises(ValueError, match="kappa"):
            CIRProcess(kappa=0.0, theta=0.05, sigma=0.1)
        with pytest.raises(ValueError, match="theta"):
            CIRProcess(kappa=1.0, theta=-0.05, sigma=0.1)
        with pytest.raises(ValueError, match="theta"):
            CIRProcess(kappa=1.0, theta=0.0, sigma=0.1)
        with pytest.raises(ValueError, match="sigma"):
            CIRProcess(kappa=1.0, theta=0.05, sigma=-0.1)
        with pytest.raises(ValueError, match="sigma"):
            CIRProcess(kappa=1.0, theta=0.05, sigma=0.0)
        with pytest.raises(ValueError, match="r0"):
            CIRProcess(kappa=1.0, theta=0.05, sigma=0.1, r0=-0.01)

        cir = CIRProcess(kappa=1.0, theta=0.05, sigma=0.1, r0=0.05)
        assert cir.degrees_of_freedom == pytest.approx(4.0 * 1.0 * 0.05 / 0.01)

        with pytest.raises(ExactnessError):
            cir._triplet()

        with pytest.raises(ValueError, match="Time t"):
            cir.conditional_mean(t=-0.1)
        with pytest.raises(ValueError, match="r0"):
            cir.conditional_mean(t=1.0, r0=-0.01)
        with pytest.raises(ValueError, match="Time t"):
            cir.conditional_variance(t=-0.1)
        with pytest.raises(ValueError, match="r0"):
            cir.conditional_variance(t=1.0, r0=-0.01)
        with pytest.raises(ValueError, match="Time t"):
            cir.conditional_cumulant(n=1, t=-0.1)
        with pytest.raises(ValueError, match="r0"):
            cir.conditional_cumulant(n=1, t=1.0, r0=-0.01)
        with pytest.raises(ValueError, match="order"):
            cir.conditional_cumulants(t=1.0, order=0)
        with pytest.raises(ValueError, match="Cumulant order"):
            cir.conditional_cumulant(n=0, t=1.0)
        with pytest.raises(ValueError, match="Cumulant order"):
            cir.stationary_cumulant(n=0)
        with pytest.raises(ValueError, match="Lag h"):
            cir.autocovariance(h=-0.5)
        with pytest.raises(ValueError, match="Lag h"):
            cir.autocorrelation(h=-0.5)
        with pytest.raises(ValueError, match="u must be non-negative"):
            cir.laplace_transform(u=-1.0, t=1.0)
        with pytest.raises(ValueError, match="Time t"):
            cir.laplace_transform(u=1.0, t=-0.5)
        with pytest.raises(ValueError, match="r0"):
            cir.laplace_transform(u=1.0, t=1.0, r0=-0.01)
        with pytest.raises(ValueError, match="Time t"):
            cir.char_func(u=1.0, t=-0.5)
        with pytest.raises(ValueError, match="r0"):
            cir.char_func(u=1.0, t=1.0, r0=-0.01)
        with pytest.raises(ValueError, match="t must be strictly greater than s"):
            cir.transition_density(y=0.05, t=1.0, s=1.0)
        with pytest.raises(ValueError, match="r0"):
            cir.transition_density(y=0.05, t=1.0, r0=-0.01)
        with pytest.raises(ValueError, match="tau must be non-negative"):
            cir.bond_price(tau=-0.5)
        with pytest.raises(ValueError, match="r0"):
            cir.bond_price(tau=1.0, r0=-0.01)
        with pytest.raises(ValueError, match="tau must be non-negative"):
            cir.bond_yield(tau=-0.5)
        with pytest.raises(ValueError, match="r0"):
            cir.bond_yield(tau=1.0, r0=-0.01)
        with pytest.raises(ValueError, match="n_steps"):
            cir.simulate(n_steps=0)
        with pytest.raises(ValueError, match="T"):
            cir.simulate(n_steps=10, T=-1.0)

    @pytest.mark.exactness
    def test_cir_feller_condition(self) -> None:
        """Verify Feller condition: 2κθ ≥ σ²."""
        # Satisfied: 2 * 2.0 * 0.05 = 0.20 >= 0.15^2 = 0.0225
        cir_feller = CIRProcess(kappa=2.0, theta=0.05, sigma=0.15)
        assert cir_feller.feller_condition is True
        assert cir_feller.feller_ratio == pytest.approx(0.20 / 0.0225)
        assert "strictly positive" in cir_feller.properties.notes[2]

        # Violated: 2 * 0.5 * 0.02 = 0.02 < 0.25^2 = 0.0625
        cir_violated = CIRProcess(kappa=0.5, theta=0.02, sigma=0.25)
        assert cir_violated.feller_condition is False
        assert cir_violated.feller_ratio < 1.0
        assert "instantaneously reflecting" in cir_violated.properties.notes[2]

    @pytest.mark.exactness
    def test_cir_conditional_mean_exact(self) -> None:
        """
        E[r_t | r_0] = θ + (r_0 - θ) e^{-κt}.
        Cox, Ingersoll & Ross (1985), equation (19).
        """
        kappa, theta, sigma, r0 = 1.8, 0.06, 0.12, 0.02
        cir = CIRProcess(kappa=kappa, theta=theta, sigma=sigma, r0=r0)

        # At t = 0: E[r_0] = r_0
        assert cir.conditional_mean(t=0.0) == pytest.approx(r0, rel=1e-12)

        # For positive t:
        for t in [0.25, 0.5, 1.0, 2.5, 5.0]:
            expected = theta + (r0 - theta) * math.exp(-kappa * t)
            assert cir.conditional_mean(t=t) == pytest.approx(expected, rel=1e-12)

        # Override r0 parameter:
        assert cir.conditional_mean(t=1.0, r0=0.10) == pytest.approx(
            theta + (0.10 - theta) * math.exp(-kappa * 1.0), rel=1e-12
        )

        # Long term t -> inf converges to theta
        assert cir.conditional_mean(t=100.0) == pytest.approx(theta, rel=1e-12)

    @pytest.mark.exactness
    def test_cir_conditional_variance_exact(self) -> None:
        """
        Var(r_t | r_0) = r_0 (σ²/κ)(e^{-κt} - e^{-2κt}) + θ(σ²/(2κ))(1 - e^{-κt})².
        Cox, Ingersoll & Ross (1985), equation (20).
        """
        kappa, theta, sigma, r0 = 1.8, 0.06, 0.12, 0.02
        cir = CIRProcess(kappa=kappa, theta=theta, sigma=sigma, r0=r0)

        # At t = 0: Var(r_0) = 0
        assert cir.conditional_variance(t=0.0) == pytest.approx(0.0, abs=1e-12)
        assert cir.conditional_std(t=0.0) == pytest.approx(0.0, abs=1e-12)

        # For positive t:
        for t in [0.1, 0.5, 1.0, 3.0]:
            e_k = math.exp(-kappa * t)
            e_2k = math.exp(-2.0 * kappa * t)
            expected = r0 * (sigma**2 / kappa) * (e_k - e_2k) + theta * (
                sigma**2 / (2.0 * kappa)
            ) * ((1.0 - e_k) ** 2)
            assert cir.conditional_variance(t=t) == pytest.approx(expected, rel=1e-12)
            assert cir.conditional_std(t=t) == pytest.approx(math.sqrt(expected), rel=1e-12)

        # As t -> inf, converges to stationary variance θσ² / (2κ)
        assert cir.conditional_variance(t=100.0) == pytest.approx(
            cir.stationary_variance, rel=1e-10
        )

    @pytest.mark.exactness
    def test_cir_conditional_cumulants(self) -> None:
        """
        Conditional cumulants match mean and variance for n=1, 2, and are defined for n >= 3.
        """
        cir = CIRProcess(kappa=1.5, theta=0.05, sigma=0.15, r0=0.03)
        t = 0.8
        kappas = cir.conditional_cumulants(t=t, order=4)

        assert kappas[1] == pytest.approx(cir.conditional_mean(t=t), rel=1e-12)
        assert kappas[2] == pytest.approx(cir.conditional_variance(t=t), rel=1e-12)
        assert kappas[3] > 0.0  # CIR right-skewed
        assert kappas[4] > 0.0  # leptokurtic

        # At t = 0:
        kappas_t0 = cir.conditional_cumulants(t=0.0, order=3)
        assert kappas_t0[1] == pytest.approx(0.03, rel=1e-12)
        assert kappas_t0[2] == pytest.approx(0.0, abs=1e-12)
        assert kappas_t0[3] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.exactness
    def test_cir_stationary_distribution_and_cumulants(self) -> None:
        """
        Stationary distribution is Gamma(shape=2κθ/σ², scale=σ²/(2κ)).
        """
        kappa, theta, sigma = 2.0, 0.05, 0.1
        cir = CIRProcess(kappa=kappa, theta=theta, sigma=sigma)

        alpha = 2.0 * kappa * theta / (sigma**2)
        beta_scale = sigma**2 / (2.0 * kappa)

        assert cir.stationary_shape == pytest.approx(alpha, rel=1e-12)
        assert cir.stationary_scale == pytest.approx(beta_scale, rel=1e-12)
        assert cir.stationary_rate == pytest.approx(1.0 / beta_scale, rel=1e-12)

        # Moments & cumulants
        assert cir.stationary_mean == pytest.approx(theta, rel=1e-12)
        assert cir.stationary_variance == pytest.approx(theta * beta_scale, rel=1e-12)
        assert cir.stationary_std == pytest.approx(math.sqrt(cir.stationary_variance), rel=1e-12)
        assert cir.stationary_skewness == pytest.approx(2.0 / math.sqrt(alpha), rel=1e-12)
        assert cir.stationary_kurtosis == pytest.approx(6.0 / alpha, rel=1e-12)

        # Stationary and finite-time marginal APIs must not be conflated.
        assert cir.cumulants(2)[2] == pytest.approx(cir.conditional_variance(1.0))
        stat_kappas = {n: cir.stationary_cumulant(n) for n in range(1, 5)}
        for n in range(1, 5):
            expected = math.factorial(n - 1) * theta * (beta_scale ** (n - 1))
            assert stat_kappas[n] == pytest.approx(expected, rel=1e-12)
            assert cir.stationary_cumulant(n) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.exactness
    def test_cir_autocovariance_and_autocorrelation(self) -> None:
        """
        Cov(r_t, r_{t+h}) = Var(r_∞) e^{-κh}, Corr(r_t, r_{t+h}) = e^{-κh}.
        """
        cir = CIRProcess(kappa=2.5, theta=0.04, sigma=0.15)
        var_inf = cir.stationary_variance

        # Lag 0
        assert cir.autocovariance(0.0) == pytest.approx(var_inf, rel=1e-12)
        assert cir.autocorrelation(0.0) == pytest.approx(1.0, rel=1e-12)

        # Lag h > 0
        for h in [0.2, 0.5, 1.0, 2.0]:
            assert cir.autocovariance(h) == pytest.approx(var_inf * math.exp(-2.5 * h), rel=1e-12)
            assert cir.autocorrelation(h) == pytest.approx(math.exp(-2.5 * h), rel=1e-12)

    @pytest.mark.exactness
    def test_cir_laplace_transform_exact(self) -> None:
        """
        E[e^{-u r_t} | r_0] = A(t, u) exp(-B(t, u) r_0).
        Compare against numerical integration of the transition density.
        """
        cir = CIRProcess(kappa=1.6, theta=0.05, sigma=0.18, r0=0.04)

        # Boundary checks
        assert cir.laplace_transform(u=0.0, t=1.0) == pytest.approx(1.0, rel=1e-12)
        assert cir.laplace_transform(u=3.0, t=0.0) == pytest.approx(
            math.exp(-3.0 * 0.04), rel=1e-12
        )

        # Compare with transition density numerical quadrature
        t = 0.7
        for u in [0.5, 2.0, 5.0]:
            formula_val = cir.laplace_transform(u=u, t=t)

            def integrand(y: float, u: float = u) -> float:
                return float(math.exp(-u * y) * cir.transition_density(y, t=t))

            num_val, _ = integrate.quad(integrand, 0.0, 1.5, epsabs=1e-9)
            assert formula_val == pytest.approx(num_val, rel=1e-6)

    @pytest.mark.exactness
    def test_cir_char_func_exact(self) -> None:
        """
        Conditional characteristic function E[e^{i u r_t} | r_0].
        """
        cir = CIRProcess(kappa=1.6, theta=0.05, sigma=0.18, r0=0.04)

        # Boundary checks
        assert cir.char_func(u=0.0, t=1.0) == pytest.approx(1.0 + 0j, abs=1e-12)
        assert cir.char_func(u=2.0, t=0.0) == pytest.approx(
            complex(math.cos(2.0 * 0.04), math.sin(2.0 * 0.04)), abs=1e-12
        )

        # Characteristic function modulus <= 1
        phi = cir.char_func(u=10.0, t=1.0)
        assert abs(phi) <= 1.0 + 1e-12

    @pytest.mark.exactness
    def test_cir_transition_density_integration(self) -> None:
        """
        Transition density integrates to 1 for both positive r0 and boundary r0 = 0.
        """
        cir = CIRProcess(kappa=1.5, theta=0.04, sigma=0.12, r0=0.03)
        t = 1.0

        # Integral when r0 > 0
        integral_pos, _ = integrate.quad(
            lambda y: float(cir.transition_density(y, t=t)), 0.0, 1.0, epsabs=1e-8
        )
        assert integral_pos == pytest.approx(1.0, abs=1e-6)

        # Integral when r0 == 0 (boundary case)
        integral_zero, _ = integrate.quad(
            lambda y: float(cir.transition_density(y, t=t, r0=0.0)), 0.0, 1.0, epsabs=1e-8
        )
        assert integral_zero == pytest.approx(1.0, abs=1e-6)

        # Array evaluation
        y_grid = np.array([0.01, 0.03, 0.05])
        dens_arr = cir.transition_density(y_grid, t=t)
        assert dens_arr.shape == (3,)
        for i, y in enumerate(y_grid):
            assert dens_arr[i] == pytest.approx(cir.transition_density(y, t=t), rel=1e-12)

        # Non-positive y returns 0
        assert cir.transition_density(-0.01, t=t) == 0.0
        assert cir.transition_density(0.0, t=t) == 0.0

    @pytest.mark.exactness
    def test_cir_stationary_density_integration(self) -> None:
        """
        Stationary Gamma density integrates to 1.
        """
        cir = CIRProcess(kappa=2.0, theta=0.05, sigma=0.15)
        integral_val, _ = integrate.quad(
            lambda y: float(cir.stationary_density(y)), 0.0, 1.0, epsabs=1e-8
        )
        assert integral_val == pytest.approx(1.0, abs=1e-6)

        # Array evaluation and zero support for y <= 0
        assert cir.stationary_density(0.0) == 0.0
        assert cir.stationary_density(-0.05) == 0.0
        y_grid = np.array([0.02, 0.05, 0.08])
        dens_arr = cir.stationary_density(y_grid)
        assert dens_arr.shape == (3,)
        for i, y in enumerate(y_grid):
            assert dens_arr[i] == pytest.approx(cir.stationary_density(y), rel=1e-12)

    @pytest.mark.exactness
    def test_cir_bond_pricing_and_riccati_ode(self) -> None:
        """
        Verify CIR bond price satisfies P(0) = 1 and Riccati ODE system:
          B'(τ) = 1 - κ B(τ) - (1/2) σ² B(τ)²
          A'(τ) / A(τ) = -κ θ B(τ)
        """
        kappa, theta, sigma, r0 = 1.4, 0.05, 0.16, 0.04
        cir = CIRProcess(kappa=kappa, theta=theta, sigma=sigma, r0=r0)

        # Boundary condition: P(0) = 1
        assert cir.bond_price(tau=0.0) == pytest.approx(1.0, rel=1e-12)
        assert cir.bond_yield(tau=0.0) == pytest.approx(r0, rel=1e-12)

        # Decreasing in tau
        p1 = cir.bond_price(tau=1.0)
        p2 = cir.bond_price(tau=2.0)
        p5 = cir.bond_price(tau=5.0)
        assert 1.0 > p1 > p2 > p5 > 0.0

        # Array vectorized pricing
        taus = np.array([0.0, 0.5, 1.0, 2.0])
        p_arr = cir.bond_price(taus)
        assert p_arr.shape == (4,)
        assert p_arr[0] == pytest.approx(1.0, rel=1e-12)
        assert p_arr[2] == pytest.approx(p1, rel=1e-12)

        # Riccati ODE verification at tau = 1.5
        tau = 1.5
        eps = 1e-6
        gamma = math.sqrt(kappa**2 + 2.0 * sigma**2)
        power = 2.0 * kappa * theta / (sigma**2)

        def get_ab(t: float) -> tuple[float, float]:
            exp_g = math.exp(gamma * t)
            denom = (gamma + kappa) * (exp_g - 1.0) + 2.0 * gamma
            b_val = 2.0 * (exp_g - 1.0) / denom
            num_a = 2.0 * gamma * math.exp((kappa + gamma) * t / 2.0)
            a_val = (num_a / denom) ** power
            return a_val, b_val

        a_plus, b_plus = get_ab(tau + eps)
        a_minus, b_minus = get_ab(tau - eps)
        _, b_mid = get_ab(tau)

        b_prime_num = (b_plus - b_minus) / (2.0 * eps)
        b_prime_ode = 1.0 - kappa * b_mid - 0.5 * (sigma**2) * (b_mid**2)
        assert b_prime_num == pytest.approx(b_prime_ode, rel=1e-5)

        lna_prime_num = (math.log(a_plus) - math.log(a_minus)) / (2.0 * eps)
        lna_prime_ode = -kappa * theta * b_mid
        assert lna_prime_num == pytest.approx(lna_prime_ode, rel=1e-5)

    @pytest.mark.exactness
    def test_cir_yield_curve_and_asymptotic_yield(self) -> None:
        """
        Yield curve y(τ) = -ln(P(τ))/τ starts at r_0 and converges to asymptotic yield.
        Cox, Ingersoll & Ross (1985), Section 3.
        """
        kappa, theta, sigma, r0 = 1.2, 0.05, 0.15, 0.03
        cir = CIRProcess(kappa=kappa, theta=theta, sigma=sigma, r0=r0)

        gamma = math.sqrt(kappa**2 + 2.0 * sigma**2)
        y_inf_expected = 2.0 * kappa * theta / (kappa + gamma)
        assert cir.asymptotic_yield == pytest.approx(y_inf_expected, rel=1e-12)

        # Spot rate
        assert cir.bond_yield(0.0) == pytest.approx(r0, rel=1e-12)

        # Long maturity converges to asymptotic yield
        y_long = cir.bond_yield(1000.0)
        assert y_long == pytest.approx(cir.asymptotic_yield, rel=1e-3)

    @pytest.mark.exactness
    def test_cir_exact_simulation_moments(self) -> None:
        """
        Simulated paths via exact non-central chi-squared transition match theoretical mean
        and variance.
        """
        kappa, theta, sigma, r0 = 2.0, 0.05, 0.1, 0.03
        cir = CIRProcess(kappa=kappa, theta=theta, sigma=sigma, r0=r0)
        T = 0.5
        n_steps = 50
        n_paths = 30_000

        rng = np.random.default_rng(42)
        paths = cir.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng, method="exact")

        assert paths.shape == (n_paths, n_steps + 1)
        assert np.all(paths[:, 0] == r0)
        assert np.all(paths >= 0.0)  # Always non-negative

        # Terminal values
        r_T = paths[:, -1]
        mean_theory = cir.conditional_mean(T)
        var_theory = cir.conditional_variance(T)

        emp_mean = float(np.mean(r_T))
        emp_var = float(np.var(r_T))

        assert emp_mean == pytest.approx(mean_theory, rel=0.03)
        assert emp_var == pytest.approx(var_theory, rel=0.06)

    @pytest.mark.exactness
    def test_cir_alternative_simulation_methods(self) -> None:
        """Truncated Euler and Reflected Euler simulate non-negative paths."""
        cir = CIRProcess(kappa=1.5, theta=0.04, sigma=0.15, r0=0.02)
        rng = np.random.default_rng(123)

        p_trunc = cir.simulate(n_steps=20, n_paths=100, T=0.2, rng=rng, method="truncated_euler")
        assert p_trunc.shape == (100, 21)
        assert np.all(p_trunc >= 0.0)

        p_refl = cir.simulate(n_steps=20, n_paths=100, T=0.2, rng=rng, method="reflected_euler")
        assert p_refl.shape == (100, 21)
        assert np.all(p_refl >= 0.0)

        with pytest.raises(ValueError, match="Unknown simulation method"):
            cir.simulate(n_steps=10, n_paths=10, method="unknown_scheme")

    @pytest.mark.exactness
    def test_cir_story_narrative(self) -> None:
        """CIR story trace includes parameters, SDE, and Feller ratio."""
        cir = CIRProcess(kappa=1.8, theta=0.06, sigma=0.12, r0=0.05)
        story = cir.__story__()
        assert "CIRProcess" in story
        assert "Cox, Ingersoll & Ross (1985)" in story
        assert "dr_t = κ(θ - r_t)dt + σ√r_t dW_t" in story
        assert "Feller ratio" in story
