"""
Exactness and mathematical consistency tests for the Heston stochastic volatility model.

Verifies:
1. MOMENT_PROPAGATION exactness level and alias bindings.
2. Property lattice invariants and ExactnessError on triplet access.
3. Parameter constraints and Feller condition behavior.
4. Analytical CIR variance moments and integrated variance.
5. Continuous characteristic function (Albrecher et al. 2007) and branch-cut stability.
6. Martingale identity phi(-i, t) == exp(mu * t).
7. Analytical cumulants and leverage skewness.
8. Convergence to Black–Scholes option prices in the vanishing vol-of-vol limit.
9. Put-Call parity and arbitrage bounds.
10. Implied volatility smile and skew direction.
11. Monte Carlo simulation convergence against exact moments.
12. Joint simulation and non-negative variance paths.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from spxa import Heston, HestonProcess
from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError


class TestHestonExactness:
    """Mathematical validation suite for HestonProcess."""

    def test_heston_exactness_level_and_alias(self):
        h = HestonProcess(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.6)
        assert h.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert Heston is HestonProcess
        assert "Heston" in str(h)
        assert "v0=" in repr(h)

    def test_heston_triplet_raises_exactness_error(self):
        h = Heston(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.6)
        with pytest.raises(ExactnessError) as exc_info:
            _ = h.triplet
        assert "Heston" in str(exc_info.value.reason)
        assert exc_info.value.required == "EXACT"

    def test_heston_properties(self):
        h = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.25, rho=-0.5, mu=0.03, s0=100.0)
        p = h.properties
        assert not p.has_stationary_increments
        assert not p.has_independent_increments
        assert not p.is_martingale
        assert p.has_finite_mean
        assert p.has_finite_variance
        assert any("Feller" in note for note in p.notes)

    def test_heston_invalid_parameters_raise(self):
        # v0 <= 0
        with pytest.raises(ValueError, match="v0"):
            Heston(v0=0.0, kappa=1.0, theta=0.04, sigma_v=0.2, rho=0.0)
        with pytest.raises(ValueError, match="v0"):
            Heston(v0=-0.01, kappa=1.0, theta=0.04, sigma_v=0.2, rho=0.0)

        # kappa <= 0
        with pytest.raises(ValueError, match="kappa"):
            Heston(v0=0.04, kappa=0.0, theta=0.04, sigma_v=0.2, rho=0.0)

        # theta <= 0
        with pytest.raises(ValueError, match="theta"):
            Heston(v0=0.04, kappa=1.0, theta=-0.04, sigma_v=0.2, rho=0.0)

        # sigma_v <= 0
        with pytest.raises(ValueError, match="sigma_v"):
            Heston(v0=0.04, kappa=1.0, theta=0.04, sigma_v=0.0, rho=0.0)

        # rho outside [-1, 1]
        with pytest.raises(ValueError, match="rho"):
            Heston(v0=0.04, kappa=1.0, theta=0.04, sigma_v=0.2, rho=1.05)
        with pytest.raises(ValueError, match="rho"):
            Heston(v0=0.04, kappa=1.0, theta=0.04, sigma_v=0.2, rho=-1.5)

        # s0 <= 0
        with pytest.raises(ValueError, match="s0"):
            Heston(v0=0.04, kappa=1.0, theta=0.04, sigma_v=0.2, rho=0.0, s0=0.0)

        # missing sigma_v
        with pytest.raises(ValueError, match="Must provide 'sigma_v'"):
            Heston(v0=0.04, kappa=1.0, theta=0.04, rho=0.0)

        # unexpected keyword argument
        with pytest.raises(TypeError, match="Unexpected keyword"):
            Heston(v0=0.04, kappa=1.0, theta=0.04, sigma_v=0.2, rho=0.0, invalid_param=123)

        # alias keyword argument sigma and xi
        h1 = Heston(v0=0.04, kappa=1.0, theta=0.04, sigma=0.25, rho=0.0)
        assert h1.sigma_v == 0.25
        h2 = Heston(v0=0.04, kappa=1.0, theta=0.04, xi=0.35, rho=0.0)
        assert h2.sigma_v == 0.35

    def test_heston_feller_condition(self):
        # 2 * kappa * theta = 2 * 2.0 * 0.04 = 0.16
        # sigma_v^2 = 0.3^2 = 0.09 -> 0.16 / 0.09 = 1.777... > 1 (satisfied)
        h_sat = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        assert h_sat.is_feller_satisfied
        assert np.isclose(h_sat.feller_ratio, 0.16 / 0.09)

        # sigma_v^2 = 0.5^2 = 0.25 -> 0.16 / 0.25 = 0.64 <= 1 (violated)
        h_viol = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7)
        assert not h_viol.is_feller_satisfied
        assert np.isclose(h_viol.feller_ratio, 0.64)

    def test_heston_cir_variance_moments(self):
        h = Heston(v0=0.02, kappa=1.8, theta=0.05, sigma_v=0.25, rho=-0.4)

        # At t=0
        assert np.isclose(h.variance_mean(0.0), h.v0)
        assert np.isclose(h.variance_var(0.0), 0.0)
        assert np.isclose(h.expected_integrated_variance(0.0), 0.0)

        # Large horizon t=50 -> stationary limits
        assert np.isclose(h.variance_mean(50.0), h.stationary_variance_mean, atol=1e-6)
        assert np.isclose(h.variance_var(50.0), h.stationary_variance_var, atol=1e-6)

        # Integrated variance formula check
        t_eval = 2.5
        th_int_v = (
            h.theta * t_eval + (h.v0 - h.theta) * (1.0 - math.exp(-h.kappa * t_eval)) / h.kappa
        )
        assert np.isclose(h.expected_integrated_variance(t_eval), th_int_v)

        with pytest.raises(ValueError, match="Time t must be non-negative"):
            h.variance_mean(-0.1)
        with pytest.raises(ValueError, match="Time t must be non-negative"):
            h.variance_var(-0.1)
        with pytest.raises(ValueError, match="Time t must be non-negative"):
            h.expected_integrated_variance(-0.1)

    def test_heston_char_func_properties(self):
        h = Heston(v0=0.04, kappa=2.5, theta=0.04, sigma_v=0.35, rho=-0.65, mu=0.05, s0=100.0)

        # phi(0, t) == 1.0
        for t in [0.1, 1.0, 5.0]:
            cf_0 = h.char_func(0.0, t=t)
            assert np.isclose(complex(cf_0), 1.0 + 0.0j, atol=1e-12)

        # Modulus <= 1 for real frequencies
        u_vals = np.linspace(-10.0, 10.0, 50)
        cfs = h.char_func(u_vals, t=1.0)
        assert len(cfs) == 50
        assert np.all(np.abs(cfs) <= 1.0 + 1e-10)

        # t=0 returns 1.0
        assert np.isclose(h.char_func(2.5, t=0.0), 1.0)
        assert np.all(h.char_func(u_vals, t=0.0) == 1.0)

        # Negative t raises
        with pytest.raises(ValueError, match="Time t must be non-negative"):
            h.char_func(1.0, t=-0.5)

        # Asset char func matches phase shift
        acf = h.asset_char_func(1.5, t=1.0)
        expected_acf = np.exp(1j * 1.5 * np.log(100.0)) * h.char_func(1.5, t=1.0)
        assert np.isclose(acf, expected_acf)

    def test_heston_martingale_condition_phi_neg_i(self):
        # Under risk-neutral drift mu, E[exp(X_t)] = exp(mu * t)
        h = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, mu=0.05)
        for t in [0.25, 0.5, 1.0, 3.0]:
            cf_neg_i = h.char_func(-1j, t=t)
            expected = math.exp(h.mu * t)
            assert np.isclose(complex(cf_neg_i).real, expected, rtol=1e-8)
            assert np.isclose(complex(cf_neg_i).imag, 0.0, atol=1e-12)

    def test_heston_little_trap_branch_cut_stability(self):
        # Test long maturities and large frequencies where standard formulation
        # encounters branch-cut jumps
        h = Heston(v0=0.04, kappa=1.0, theta=0.04, sigma_v=0.4, rho=-0.8, mu=0.02)
        for t in [10.0, 25.0, 50.0]:
            for u in [5.0, 20.0, 50.0, 100.0]:
                cf_val = h.char_func(u, t=t)
                assert not np.isnan(complex(cf_val).real)
                assert not np.isinf(complex(cf_val).real)
                assert abs(cf_val) <= 1.0 + 1e-8

    def test_heston_conditional_mean_exact(self):
        h = Heston(v0=0.03, kappa=2.2, theta=0.04, sigma_v=0.3, rho=-0.5, mu=0.04)
        t = 1.5
        th_mean = h.conditional_mean(t)
        expected_int_v = h.expected_integrated_variance(t)
        assert np.isclose(th_mean, h.mu * t - 0.5 * expected_int_v)

        # Differentiate char_func numerically to check expectation
        step = 1e-5
        cf_p = h.char_func(step, t=t)
        cf_m = h.char_func(-step, t=t)
        num_mean = float(np.real((cf_p - cf_m) / (2.0 * 1j * step)))
        assert np.isclose(th_mean, num_mean, atol=1e-5)

        with pytest.raises(ValueError, match="Time t must be non-negative"):
            h.conditional_mean(-1.0)

    def test_heston_conditional_cumulants(self):
        h = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, mu=0.05)
        kappas = h.conditional_cumulants(t=1.0, order=4)
        assert np.isclose(kappas[1], h.conditional_mean(1.0), atol=1e-5)
        assert kappas[2] > 0.0
        # Negative correlation rho < 0 generates negative skewness (kappa_3 < 0)
        assert kappas[3] < 0.0

        # Excess kurtosis kappa_4 > 0 (fat tails from stochastic volatility)
        assert kappas[4] > 0.0

        # cumulants() method at t=1
        c_1 = h.cumulants(order=2)
        assert np.isclose(c_1[1], kappas[1])
        assert np.isclose(c_1[2], kappas[2])

        # Order validation
        with pytest.raises(ValueError, match="Cumulant order must be at least 1"):
            h.conditional_cumulants(order=0)

        # t=0 returns zeroes
        k_0 = h.conditional_cumulants(t=0.0, order=3)
        assert all(v == 0.0 for v in k_0.values())

    def test_heston_call_pricing_matches_black_scholes_limit(self):
        # As sigma_v -> 0 with v0 = theta = sig^2, Heston converges to Black–Scholes
        S0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        q = 0.0
        vol = 0.20

        d1 = (math.log(S0 / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
        d2 = d1 - vol * math.sqrt(T)
        bs_call = S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

        # Heston model with tiny sigma_v = 1e-4
        h_bs = Heston(
            v0=vol**2,
            kappa=2.0,
            theta=vol**2,
            sigma_v=1e-4,
            rho=0.0,
            mu=r,
            s0=S0,
        )
        h_call = h_bs.call_price(K, T=T, r=r, q=q)
        assert np.isclose(h_call, bs_call, atol=1e-5)

    def test_heston_call_price_bounds_and_strikes(self):
        h = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, mu=0.05, s0=100.0)

        # Deep in-the-money call price >= intrinsic value S0 - K*exp(-r*T)
        c_itm = h.call_price(K=30.0, T=1.0, r=0.05)
        intrinsic_itm = 100.0 - 30.0 * math.exp(-0.05 * 1.0)
        assert c_itm >= intrinsic_itm - 1e-10

        # Deep out-of-the-money call approaches 0
        c_otm = h.call_price(K=250.0, T=1.0, r=0.05)
        assert 0.0 <= c_otm < 0.1

        # Vector of strikes
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        calls = h.call_price(strikes, T=1.0)
        assert isinstance(calls, np.ndarray)
        assert len(calls) == 5
        # Call prices strictly decrease with strike
        assert np.all(np.diff(calls) < 0.0)

        # Invalid strikes and maturity
        with pytest.raises(ValueError, match="Strike K must be strictly positive"):
            h.call_price(K=-10.0)
        with pytest.raises(ValueError, match="Maturity T must be strictly positive"):
            h.call_price(K=100.0, T=0.0)

    def test_heston_put_price_and_put_call_parity(self):
        h = Heston(v0=0.04, kappa=1.8, theta=0.04, sigma_v=0.32, rho=-0.6, mu=0.04, s0=100.0)
        S0 = 100.0
        r = 0.04
        q = 0.01
        T = 1.0

        for K in [80.0, 95.0, 100.0, 105.0, 120.0]:
            call = h.call_price(K, T=T, r=r, q=q)
            put = h.put_price(K, T=T, r=r, q=q)
            # Put-Call Parity: C - P = S0 * exp(-q*T) - K * exp(-r*T)
            parity_diff = (call - put) - (S0 * math.exp(-q * T) - K * math.exp(-r * T))
            assert np.isclose(parity_diff, 0.0, atol=1e-8)

    def test_heston_implied_volatility_and_smile(self):
        S0 = 100.0
        h_neg = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, mu=0.05, s0=S0)

        # ATM implied vol should be close to sqrt(v0) = 0.20
        iv_atm = h_neg.implied_volatility(K=100.0, T=1.0)
        assert np.isclose(iv_atm, 0.20, atol=0.02)

        # Negative rho generates downward sloping skew
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        vols_neg = h_neg.implied_vol_smile(strikes, T=1.0)
        assert vols_neg[0] > vols_neg[2] > vols_neg[4]

        # Positive rho generates upward sloping skew
        h_pos = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=0.7, mu=0.05, s0=S0)
        vols_pos = h_pos.implied_vol_smile(strikes, T=1.0)
        assert vols_pos[0] < vols_pos[2] < vols_pos[4]

    def test_heston_simulate_terminal_moments(self):
        h = Heston(v0=0.04, kappa=2.5, theta=0.04, sigma_v=0.25, rho=-0.6, mu=0.05, s0=100.0)
        T = 0.5
        n_steps = 100
        n_paths = 30_000
        rng = np.random.default_rng(999)

        paths_x = h.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
        assert paths_x.shape == (n_paths, n_steps + 1)
        assert np.all(paths_x[:, 0] == 0.0)

        terminal_x = paths_x[:, -1]
        mc_mean = float(np.mean(terminal_x))
        th_mean = h.conditional_mean(T)
        assert np.isclose(mc_mean, th_mean, atol=5e-3)

        # Martingale check E[exp(X_T)] == exp(mu * T)
        mc_exp_x = float(np.mean(np.exp(terminal_x)))
        th_exp_x = math.exp(h.mu * T)
        assert np.isclose(mc_exp_x, th_exp_x, rtol=0.01)

    def test_heston_simulate_joint_and_return_price(self):
        h = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, mu=0.05, s0=100.0)
        rng = np.random.default_rng(42)

        # return_price=True returns paths of S_t starting at S0
        paths_s = h.simulate(n_steps=20, n_paths=100, T=0.2, rng=rng, return_price=True)
        assert np.all(np.isclose(paths_s[:, 0], 100.0))

        # simulate_joint returns (S_paths, V_paths)
        s_paths, v_paths = h.simulate_joint(n_steps=20, n_paths=100, T=0.2, rng=rng)
        assert s_paths.shape == (100, 21)
        assert v_paths.shape == (100, 21)
        assert np.all(np.isclose(s_paths[:, 0], 100.0))
        assert np.all(np.isclose(v_paths[:, 0], 0.04))
        # Variance stays strictly non-negative
        assert np.all(v_paths >= 0.0)

    def test_heston_properties_and_accessors(self):
        h = Heston(v0=0.03, kappa=1.5, theta=0.04, sigma_v=0.25, rho=-0.4, mu=0.02, s0=50.0)
        assert h.s0 == 50.0
        assert h.mu == 0.02
        assert h.rho == -0.4
        assert h.sigma_v == 0.25
        assert h.v0 == 0.03
        assert h.kappa == 1.5
        assert h.theta == 0.04

        # conditional_variance
        var_t = h.conditional_variance(1.0)
        assert var_t > 0.0

        # Higher orders are computed from differentiated affine Riccati equations.
        k_6 = h.conditional_cumulants(order=6)
        assert np.isfinite(k_6[5])
        assert np.isfinite(k_6[6])

        # implied_volatility with vector of strikes
        iv_vec = h.implied_volatility(np.array([45.0, 50.0, 55.0]), T=1.0)
        assert len(iv_vec) == 3
        assert np.all(iv_vec > 0.0)

    def test_heston_story_narrative(self):
        h = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        story = h.__story__()
        assert "Heston" in story
        assert "v0=0.04" in story
        assert "Albrecher" in story
        assert "MOMENT_PROPAGATION" in story
