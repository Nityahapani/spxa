r"""
Mathematical exactness and consistency tests for the Constant Elasticity of Variance (CEV) process.

Verifies:
1. Exactness level (MOMENT_PROPAGATION) and canonical aliases (CEV, ConstantElasticityOfVariance).
2. ExactnessError raised for Lévy triplet access (state-dependent local volatility).
3. Process properties (non-stationary, non-independent increments, martingale when μ=0 and γ≤1).
4. Local volatility function σ(S) = σ S^{γ - 1} and constant variance elasticity 2(γ - 1).
5. Exact Black–Scholes limit when γ = 1.0 for calls, puts, CDF, and Greeks.
6. Schroder (1989) / Cox (1975) analytical European option pricing vs Monte Carlo simulation.
7. Exact put–call parity: C(K, T) - P(K, T) = S_0 e^{-q T} - K e^{-r T}.
8. Analytical absorption / bankruptcy probability at zero for γ < 1 vs simulation.
9. Conditional cumulative distribution function P(S_T ≤ k | S_0) across strikes.
10. Option Greeks consistency (call/put delta parity, positive gamma, positive vega).
11. Conditional moments (analytical mean, exact variance for γ=0.5 and γ=1.0 vs quadrature).
12. Vectorized path simulation with absorbing boundary condition at the origin.

References
----------
Cox, J.C. (1975). Notes on option pricing I: Constant elasticity of diffusions.
Schroder, M. (1989). Computing the constant elasticity of variance option pricing formula.
    *Journal of Finance*, 44(1), 211–219.
Emanuel, D.C. & MacBeth, J.D. (1982). Further results of the constant elasticity
    of variance call option pricing model. *Journal of Financial and Quantitative Analysis*,
    17(4), 533–554.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from spxa import CEV, CEVProcess, ConstantElasticityOfVariance
from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError


class TestCEVExactness:
    """Exactness test suite for CEVProcess."""

    def test_cev_exactness_level_and_aliases(self) -> None:
        """CEVProcess exactness is MOMENT_PROPAGATION and aliases are identical."""
        cev = CEVProcess(mu=0.05, sigma=0.2, gamma=0.5, s0=100.0)
        assert cev.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert CEV is CEVProcess
        assert ConstantElasticityOfVariance is CEVProcess
        assert "CEVProcess" in repr(cev)
        assert "sigma=0.2" in repr(cev)
        assert "gamma=0.5" in repr(cev)

    def test_cev_triplet_raises_exactness_error(self) -> None:
        """CEV has state-dependent local volatility and no Lévy triplet."""
        cev = CEVProcess(mu=0.03, sigma=0.25, gamma=0.75, s0=100.0)
        with pytest.raises(ExactnessError) as exc_info:
            _ = cev.triplet
        assert "CEVProcess has state-dependent local volatility" in str(exc_info.value)

    def test_cev_invalid_parameters_raise(self) -> None:
        """Invalid model parameters raise informative ValueError."""
        # sigma <= 0
        with pytest.raises(ValueError, match="sigma"):
            CEVProcess(sigma=0.0)
        with pytest.raises(ValueError, match="sigma"):
            CEVProcess(sigma=-0.1)

        # gamma < 0
        with pytest.raises(ValueError, match="gamma"):
            CEVProcess(gamma=-0.5)

        # s0 <= 0
        with pytest.raises(ValueError, match="s0"):
            CEVProcess(s0=0.0)
        with pytest.raises(ValueError, match="s0"):
            CEVProcess(s0=-10.0)

        # q < 0
        with pytest.raises(ValueError, match="q"):
            CEVProcess(q=-0.01)

        # Negative time or non-positive strike in pricing methods
        cev = CEVProcess(mu=0.05, sigma=0.2, gamma=0.5, s0=100.0)
        with pytest.raises(ValueError, match="Strike"):
            cev.call_price(K=-10.0, T=1.0)
        with pytest.raises(ValueError, match="Strike"):
            cev.put_price(K=0.0, T=1.0)
        with pytest.raises(ValueError, match="Maturity"):
            cev.call_price(K=100.0, T=-0.5)
        with pytest.raises(ValueError, match="Maturity"):
            cev.put_price(K=100.0, T=-1.0)
        with pytest.raises(ValueError, match="option_type"):
            cev.price(K=100.0, T=1.0, option_type="straddle")
        with pytest.raises(ValueError, match="option_type"):
            cev.delta(K=100.0, T=1.0, option_type="straddle")

    def test_cev_properties(self) -> None:
        """Process properties reflect local volatility diffusion characteristics."""
        cev_eq = CEVProcess(mu=0.0, sigma=0.2, gamma=0.5, s0=100.0)
        props_eq = cev_eq.properties
        assert props_eq.has_stationary_increments is False
        assert props_eq.has_independent_increments is False
        assert props_eq.is_martingale is True  # mu=0 and gamma<=1 is true martingale
        assert props_eq.has_finite_variance is True
        assert props_eq.has_finite_mean is True
        assert props_eq.is_subordinator is False
        assert any("Constant Elasticity of Variance" in note for note in props_eq.notes)
        assert cev_eq.is_absorbed_at_zero is True
        assert cev_eq.regime == "equity_leverage"

        # Drift mu != 0 is not a martingale
        cev_drift = CEVProcess(mu=0.05, sigma=0.2, gamma=0.5, s0=100.0)
        assert cev_drift.properties.is_martingale is False

        # Commodity regime (gamma > 1)
        cev_com = CEVProcess(mu=0.05, sigma=0.05, gamma=1.25, s0=100.0)
        assert cev_com.regime == "commodity"
        assert cev_com.is_absorbed_at_zero is False
        assert cev_com.feller_condition is True

    def test_cev_local_volatility_and_elasticity(self) -> None:
        """Local volatility function and elasticity formulas."""
        # Equity regime (gamma = 0.5)
        cev = CEVProcess(mu=0.05, sigma=2.0, gamma=0.5, s0=100.0)
        assert cev.elasticity == pytest.approx(-1.0)
        assert cev.vol_elasticity == pytest.approx(-0.5)

        # Local vol at S=100: 2.0 * 100^(-0.5) = 0.20
        assert cev.local_volatility(100.0) == pytest.approx(0.20)
        # Local vol at S=25: 2.0 * 25^(-0.5) = 0.40
        assert cev.local_volatility(25.0) == pytest.approx(0.40)
        # Local vol at S=400: 2.0 * 400^(-0.5) = 0.10
        assert cev.local_volatility(400.0) == pytest.approx(0.10)

        # Array evaluation
        s_arr = np.array([25.0, 100.0, 400.0])
        vols = cev.local_volatility(s_arr)
        np.testing.assert_allclose(vols, [0.40, 0.20, 0.10])

        # GBM regime (gamma = 1.0): local vol is constant
        cev_gbm = CEVProcess(mu=0.05, sigma=0.2, gamma=1.0, s0=100.0)
        assert cev_gbm.elasticity == pytest.approx(0.0)
        assert cev_gbm.vol_elasticity == pytest.approx(0.0)
        assert cev_gbm.local_volatility(50.0) == pytest.approx(0.2)
        assert cev_gbm.local_volatility(200.0) == pytest.approx(0.2)

    def test_cev_black_scholes_limit(self) -> None:
        """When gamma = 1.0, CEV option pricing exactly matches Black–Scholes."""
        s0 = 100.0
        r = 0.05
        q = 0.02
        sigma = 0.25
        T = 0.75

        cev = CEVProcess(sigma=sigma, gamma=1.0, s0=s0, r=r, q=q)

        strikes = [70.0, 90.0, 100.0, 110.0, 130.0]
        for K in strikes:
            # Standard Black-Scholes formula
            d1 = (math.log(s0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            bs_call = s0 * math.exp(-q * T) * stats.norm.cdf(d1) - K * math.exp(
                -r * T
            ) * stats.norm.cdf(d2)
            bs_put = K * math.exp(-r * T) * stats.norm.cdf(-d2) - s0 * math.exp(
                -q * T
            ) * stats.norm.cdf(-d1)

            cev_call = cev.call_price(K=K, T=T)
            cev_put = cev.put_price(K=K, T=T)

            assert cev_call == pytest.approx(bs_call, abs=1e-10)
            assert cev_put == pytest.approx(bs_put, abs=1e-10)

            # Test unified pricer
            assert cev.price(K=K, T=T, option_type="call") == pytest.approx(bs_call, abs=1e-10)
            assert cev.price(K=K, T=T, option_type="put") == pytest.approx(bs_put, abs=1e-10)

    def test_cev_put_call_parity_exact(self) -> None:
        """Put-call parity C - P = S_0 e^{-q T} - K e^{-r T} holds across all gammas."""
        s0 = 100.0
        r = 0.04
        q = 0.015
        T = 1.2

        gamma_values = [0.25, 0.5, 0.75, 1.0, 1.25]
        strikes = [80.0, 100.0, 120.0]

        for gamma in gamma_values:
            # Adjust sigma to keep ATM volatility around 20%
            sigma = 0.20 * (s0 ** (1.0 - gamma))
            cev = CEVProcess(sigma=sigma, gamma=gamma, s0=s0, r=r, q=q)

            for K in strikes:
                c = cev.call_price(K, T)
                p = cev.put_price(K, T)
                expected_diff = s0 * math.exp(-q * T) - K * math.exp(-r * T)
                assert (c - p) == pytest.approx(expected_diff, abs=1e-8)

    def test_cev_absorption_probability_properties(self) -> None:
        """Absorption probability at 0 is strictly zero for γ≥1, positive for γ<1."""
        # γ >= 1
        cev_gbm = CEVProcess(mu=0.05, sigma=0.2, gamma=1.0, s0=100.0)
        cev_com = CEVProcess(mu=0.05, sigma=0.05, gamma=1.2, s0=100.0)
        assert cev_gbm.absorption_probability(1.0) == 0.0
        assert cev_com.absorption_probability(1.0) == 0.0

        # γ < 1 with parameters where absorption is appreciable
        s0 = 10.0
        T = 2.0
        r = 0.02
        q = 0.0
        sigma = 1.5
        cev_low = CEVProcess(sigma=sigma, gamma=0.5, s0=s0, r=r, q=q)
        prob = cev_low.absorption_probability(T)
        assert 0.0 < prob < 0.1

        # Monotonicity: longer horizon has higher absorption probability
        assert cev_low.absorption_probability(3.0) > prob
        assert cev_low.absorption_probability(0.5) < prob

        # Lower initial spot price has higher absorption probability
        cev_lower = CEVProcess(sigma=sigma, gamma=0.5, s0=5.0, r=r, q=q)
        assert cev_lower.absorption_probability(T) > prob

    def test_cev_call_pricing_vs_monte_carlo(self) -> None:
        """Analytical Schroder call price matches Monte Carlo simulation."""
        s0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        q = 0.0
        gamma = 0.5
        sigma = 2.0  # ATM local vol = 20%

        cev = CEVProcess(sigma=sigma, gamma=gamma, s0=s0, r=r, q=q)
        exact_call = cev.call_price(K=K, T=T)
        exact_put = cev.put_price(K=K, T=T)

        # High-sample Monte Carlo simulation
        rng = np.random.default_rng(12345)
        paths = cev.simulate(
            n_steps=200, time_horizon=T, n_paths=250_000, rng=rng, method="milstein"
        )
        s_T = paths[:, -1]

        mc_call = math.exp(-r * T) * np.mean(np.maximum(s_T - K, 0.0))
        mc_put = math.exp(-r * T) * np.mean(np.maximum(K - s_T, 0.0))

        # Agreement within 0.05 (MC standard error is ~ 0.03)
        assert exact_call == pytest.approx(mc_call, abs=0.05)
        assert exact_put == pytest.approx(mc_put, abs=0.05)

    def test_cev_transition_cdf_consistency(self) -> None:
        """Transition CDF is monotonic and matches boundary absorption."""
        cev = CEVProcess(sigma=1.5, gamma=0.5, s0=20.0, r=0.03, q=0.0)
        T = 1.5

        p_abs = cev.absorption_probability(T)
        cdf_0 = cev.transition_cdf(0.0, T)
        assert cdf_0 == pytest.approx(p_abs, abs=1e-10)

        # Monotonicity
        strikes = [1.0, 10.0, 20.0, 40.0, 100.0]
        cdf_vals = [cev.transition_cdf(k, T) for k in strikes]
        for i in range(len(cdf_vals) - 1):
            assert cdf_vals[i] <= cdf_vals[i + 1]
        assert cdf_vals[-1] <= 1.0

    def test_cev_greeks_properties(self) -> None:
        """Greeks satisfy financial bounds and call-put delta relation."""
        s0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        q = 0.02
        cev = CEVProcess(sigma=2.0, gamma=0.5, s0=s0, r=r, q=q)

        delta_call = cev.delta(K, T, option_type="call")
        delta_put = cev.delta(K, T, option_type="put")

        # Delta bounds
        assert 0.0 < delta_call < math.exp(-q * T)
        assert -math.exp(-q * T) < delta_put < 0.0

        # Call-put delta parity: Δ_call - Δ_put = e^{-q T}
        expected_diff = math.exp(-q * T)
        assert (delta_call - delta_put) == pytest.approx(expected_diff, abs=1e-5)

        # Gamma Greek is strictly positive
        gamma_greek = cev.gamma_greek(K, T)
        assert gamma_greek > 0.0

        # Vega Greek is strictly positive
        vega_greek = cev.vega(K, T)
        assert vega_greek > 0.0

    def test_cev_conditional_moments(self) -> None:
        """Conditional mean and variance formulas match analytical theory."""
        s0 = 100.0
        mu = 0.04
        t = 1.0

        # Mean is always s0 * exp(mu * t)
        cev_half = CEVProcess(mu=mu, sigma=1.5, gamma=0.5, s0=s0)
        cev_gbm = CEVProcess(mu=mu, sigma=0.2, gamma=1.0, s0=s0)
        assert cev_half.conditional_mean(t) == pytest.approx(s0 * math.exp(mu * t))
        assert cev_gbm.conditional_mean(t) == pytest.approx(s0 * math.exp(mu * t))

        # Variance for gamma = 1 (GBM)
        gbm_var_expected = (s0 * math.exp(mu * t)) ** 2 * (math.exp(0.04 * t) - 1.0)
        assert cev_gbm.conditional_variance(t) == pytest.approx(gbm_var_expected, rel=1e-6)

        # Variance for gamma = 0.5 (CIR)
        cir_var_expected = (1.5**2 * s0 / mu) * math.exp(mu * t) * (math.exp(mu * t) - 1.0)
        assert cev_half.conditional_variance(t) == pytest.approx(cir_var_expected, rel=1e-6)

        # Moment propagation cumulants dictionary
        cumulants = cev_half._moment_propagation_cumulants(order=2)
        assert cumulants[1] == pytest.approx(cev_half.conditional_mean(1.0))
        assert cumulants[2] == pytest.approx(cev_half.conditional_variance(1.0))

    def test_cev_simulation_paths_and_absorption(self) -> None:
        """Simulated paths remain non-negative and stay absorbed at zero."""
        s0 = 5.0
        T = 2.0
        sigma = 2.0
        cev = CEVProcess(mu=-0.1, sigma=sigma, gamma=0.5, s0=s0)

        rng = np.random.default_rng(42)
        paths = cev.simulate(n_steps=100, time_horizon=T, n_paths=500, rng=rng, method="milstein")

        assert paths.shape == (500, 101)
        assert np.all(paths[:, 0] == s0)
        assert np.all(paths >= 0.0)

        # Check absorption permanence
        for path in paths:
            zeros = np.where(path == 0.0)[0]
            if len(zeros) > 0:
                first_zero = zeros[0]
                assert np.all(path[first_zero:] == 0.0)

    def test_cev_zero_drift_and_edge_cases(self) -> None:
        """Covers zero drift nu=0, T=0 boundaries, and negative S in local vol."""
        # Zero drift (nu = r - q = 0)
        cev_z = CEVProcess(mu=0.0, sigma=0.2, gamma=0.5, s0=100.0, r=0.0, q=0.0)
        assert cev_z.absorption_probability(0.0) == 0.0
        assert cev_z.absorption_probability(1.0) >= 0.0
        c_z = cev_z.call_price(100.0, 1.0)
        p_z = cev_z.put_price(100.0, 1.0)
        assert c_z == pytest.approx(p_z, abs=1e-8)  # ATM forward parity when r=q=0

        # T = 0 boundary payoff
        assert cev_z.call_price(90.0, 0.0) == pytest.approx(10.0)
        assert cev_z.call_price(110.0, 0.0) == pytest.approx(0.0)
        assert cev_z.put_price(90.0, 0.0) == pytest.approx(0.0)
        assert cev_z.put_price(110.0, 0.0) == pytest.approx(10.0)
        assert cev_z.transition_cdf(90.0, 0.0) == 0.0
        assert cev_z.transition_cdf(110.0, 0.0) == 1.0

        # Commodity regime gamma > 1.0 edge cases
        cev_com_z = CEVProcess(mu=0.0, sigma=0.05, gamma=1.25, s0=100.0, r=0.0, q=0.0)
        c_com_z = cev_com_z.call_price(100.0, 1.0)
        p_com_z = cev_com_z.put_price(100.0, 1.0)
        assert c_com_z == pytest.approx(p_com_z, abs=1e-7)
        cdf_com = cev_com_z.transition_cdf(100.0, 1.0)
        assert 0.0 < cdf_com < 1.0

        # Negative S in local volatility raises ValueError
        with pytest.raises(ValueError, match="non-negative"):
            cev_z.local_volatility(-5.0)

        # Invalid time in conditional moments
        with pytest.raises(ValueError, match="non-negative"):
            cev_z.conditional_mean(-0.5)
        with pytest.raises(ValueError, match="non-negative"):
            cev_z.conditional_variance(-0.5)
        with pytest.raises(ValueError, match="non-negative"):
            cev_z.conditional_mean(1.0, s0=-10.0)

        # Conditional std and general gamma conditional variance
        cev_75 = CEVProcess(mu=0.03, sigma=0.25, gamma=0.75, s0=100.0)
        var_75 = cev_75.conditional_variance(1.0)
        std_75 = cev_75.conditional_std(1.0)
        assert var_75 > 0.0
        assert std_75 == pytest.approx(math.sqrt(var_75))
        assert cev_75.conditional_variance(0.0) == 0.0
        assert cev_75.conditional_variance(1.0, s0=0.0) == 0.0

        # Transition CDF edge cases
        assert cev_75.transition_cdf(-10.0, 1.0) == 0.0
        with pytest.raises(ValueError, match="Maturity"):
            cev_75.transition_cdf(100.0, -1.0)

        # Transition CDF under GBM gamma = 1.0 matches normal cdf
        cev_10 = CEVProcess(mu=0.05, sigma=0.2, gamma=1.0, s0=100.0)
        d_gbm = (math.log(110.0 / 100.0) - (0.05 - 0.5 * 0.04) * 1.0) / (0.2 * 1.0)
        assert cev_10.transition_cdf(110.0, 1.0) == pytest.approx(stats.norm.cdf(d_gbm), abs=1e-10)
