"""
Mathematical exactness and consistency tests for the Merton (1976) Jump-Diffusion process.

Verifies:
1. Exactness level (EXACT) and canonical aliases (MertonProcess, Merton76).
2. Exact Lévy triplet (b, σ², ν) with analytical truncated Gaussian drift integral.
3. Property lattice invariants (martingale condition, finite activity, non-subordinator).
4. Characteristic exponent and time additivity.
5. Analytical cumulants of all orders matching raw moments of Gaussian jumps.
6. Algebraic composition with Brownian motion and scaling operators under triplet rules.
7. Merton (1976) analytical option pricing series vs Fourier inversion.
8. Black–Scholes limit when jump intensity λ -> 0.
9. Put-Call parity and arbitrage bounds.
10. Implied volatility smile and short-dated jump skew.
11. Exact Monte Carlo simulation convergence with zero discretization error.
"""

import math

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

import spxa
from spxa import BrownianMotion, Merton76, MertonJumpDiffusion, MertonProcess
from spxa.core.exactness import ExactnessLevel


class TestMertonExactness:
    """Exactness test suite for MertonJumpDiffusion."""

    def test_merton_exactness_level_and_aliases(self):
        m = MertonJumpDiffusion(drift=0.05, sigma=0.2, rate=1.5, jump_mean=-0.1, jump_std=0.15)
        assert m.exactness == ExactnessLevel.EXACT
        assert MertonProcess is MertonJumpDiffusion
        assert Merton76 is MertonJumpDiffusion
        assert "Merton" in str(m)
        assert "drift=" in repr(m)

    def test_merton_invalid_parameters_raise(self):
        # sigma <= 0
        with pytest.raises(ValueError, match="sigma"):
            MertonJumpDiffusion(sigma=0.0)
        with pytest.raises(ValueError, match="sigma"):
            MertonJumpDiffusion(sigma=-0.1)

        # rate <= 0
        with pytest.raises(ValueError, match="rate"):
            MertonJumpDiffusion(rate=0.0)
        with pytest.raises(ValueError, match="rate"):
            MertonJumpDiffusion(rate=-1.0)

        # jump_std <= 0
        with pytest.raises(ValueError, match="jump_std"):
            MertonJumpDiffusion(jump_std=0.0)
        with pytest.raises(ValueError, match="jump_std"):
            MertonJumpDiffusion(jump_std=-0.05)

        # Unexpected keyword argument
        with pytest.raises(TypeError, match="Unexpected keyword"):
            MertonJumpDiffusion(invalid_param=123)

        # Aliases support
        m_alias = MertonJumpDiffusion(mu=0.03, vol=0.25, lam=2.0, mu_j=-0.08, sigma_j=0.12)
        assert m_alias.drift == 0.03
        assert m_alias.sigma == 0.25
        assert m_alias.rate == 2.0
        assert m_alias.jump_mean == -0.08
        assert m_alias.jump_std == 0.12

    def test_merton_triplet_exact(self):
        drift = 0.02
        sigma = 0.18
        rate = 1.2
        mu_j = -0.05
        sigma_j = 0.14

        m = MertonJumpDiffusion(
            drift=drift, sigma=sigma, rate=rate, jump_mean=mu_j, jump_std=sigma_j
        )
        trip = m.triplet

        # Diffusion coefficient
        assert np.isclose(trip.sigma_sq, sigma**2)

        # Finite activity jump measure
        assert np.isclose(trip.nu.total_mass(), rate)
        assert trip.nu.is_finite_activity()

        # Truncated drift integral b = drift + rate * E[Y * 1_{|Y| <= 1}]
        # Compare with numerical quadrature
        quad_trunc = quad(lambda x: x * stats.norm.pdf(x, loc=mu_j, scale=sigma_j), -1.0, 1.0)[0]
        expected_b = drift + rate * quad_trunc
        assert np.isclose(trip.b, expected_b, atol=1e-12)

        # Density evaluation
        x_grid = np.array([-0.5, 0.1, 0.3])
        expected_dens = rate * stats.norm.pdf(x_grid, loc=mu_j, scale=sigma_j)
        assert np.allclose(trip.nu.density(x_grid), expected_dens)

    def test_merton_properties(self):
        # Martingale when drift + rate * jump_mean == 0
        m_mart = MertonJumpDiffusion(drift=0.10, sigma=0.2, rate=1.0, jump_mean=-0.10)
        assert m_mart.properties.is_martingale
        assert m_mart.properties.has_stationary_increments
        assert m_mart.properties.has_independent_increments
        assert m_mart.properties.has_finite_mean
        assert m_mart.properties.has_finite_variance
        assert not m_mart.properties.is_subordinator

        # Non-martingale
        m_non_mart = MertonJumpDiffusion(drift=0.05, sigma=0.2, rate=1.0, jump_mean=-0.10)
        assert not m_non_mart.properties.is_martingale

    def test_merton_characteristic_function(self):
        m = MertonJumpDiffusion(drift=0.04, sigma=0.22, rate=1.8, jump_mean=-0.08, jump_std=0.16)

        # phi(0, t) == 1.0
        assert np.isclose(complex(m.char_func(0.0, t=1.0)), 1.0 + 0j)
        assert np.isclose(complex(m.char_func(0.0, t=3.5)), 1.0 + 0j)

        # Modulus <= 1
        u_vals = np.linspace(-15.0, 15.0, 50)
        cfs = m.char_func(u_vals, t=1.0)
        assert np.all(np.abs(cfs) <= 1.0 + 1e-12)

        # Time additivity phi(u, t1 + t2) == phi(u, t1) * phi(u, t2)
        cf_t1 = m.char_func(u_vals, t=0.7)
        cf_t2 = m.char_func(u_vals, t=1.3)
        cf_both = m.char_func(u_vals, t=2.0)
        assert np.allclose(cf_t1 * cf_t2, cf_both, atol=1e-12)

        # Negative t raises
        with pytest.raises(ValueError, match="Time t must be non-negative"):
            m.char_func(1.0, t=-0.5)

    def test_merton_cumulants_exact(self):
        drift = 0.03
        sigma = 0.20
        rate = 2.0
        mu_j = 0.05
        sigma_j = 0.10

        m = MertonJumpDiffusion(
            drift=drift, sigma=sigma, rate=rate, jump_mean=mu_j, jump_std=sigma_j
        )
        k = m.cumulants(order=5)

        # Exact formulas
        assert np.isclose(k[1], drift + rate * mu_j)
        assert np.isclose(k[2], sigma**2 + rate * (mu_j**2 + sigma_j**2))
        assert np.isclose(k[3], rate * (mu_j**3 + 3.0 * mu_j * sigma_j**2))
        assert np.isclose(k[4], rate * (mu_j**4 + 6.0 * mu_j**2 * sigma_j**2 + 3.0 * sigma_j**4))

        # Check k[5] matches integration against raw normal 5th moment
        # E[Y^5] = mu_j^5 + 10*mu_j^3*sigma_j^2 + 15*mu_j*sigma_j^4
        th_m5 = mu_j**5 + 10.0 * mu_j**3 * sigma_j**2 + 15.0 * mu_j * sigma_j**4
        assert np.isclose(k[5], rate * th_m5)

        with pytest.raises(ValueError, match="order"):
            m.cumulants(order=0)

    def test_merton_triplet_addition_algebra(self):
        m = MertonJumpDiffusion(drift=0.02, sigma=0.15, rate=1.0, jump_mean=-0.05, jump_std=0.12)
        bm = BrownianMotion(mu=0.01, sigma=0.10)

        # Composition Z = m + bm
        z = m + bm
        assert z.exactness == ExactnessLevel.EXACT
        trip_z = z.triplet

        assert np.isclose(trip_z.sigma_sq, m.sigma**2 + bm.sigma**2)
        assert np.isclose(trip_z.nu.total_mass(), m.rate)

        # Scaled process 2 * m
        m_scaled = 2.0 * m
        assert m_scaled.exactness == ExactnessLevel.EXACT
        assert np.isclose(m_scaled.triplet.sigma_sq, (2.0 * m.sigma) ** 2)

    def test_merton_option_pricing_matches_fourier_inversion(self):
        S0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        q = 0.01

        m = MertonJumpDiffusion(drift=0.02, sigma=0.15, rate=1.2, jump_mean=-0.08, jump_std=0.14)
        merton_call = m.call_price(K, T=T, S0=S0, r=r, q=q)

        # Fourier inversion of characteristic function
        # Risk-neutral drift adjustment: mu_rn = (r - q) - 0.5*sigma^2 - rate*k_bar
        k_bar = m.expected_relative_jump
        mu_rn = (r - q) - 0.5 * m.sigma**2 - m.rate * k_bar
        m_rn = MertonJumpDiffusion(
            drift=mu_rn, sigma=m.sigma, rate=m.rate, jump_mean=m.jump_mean, jump_std=m.jump_std
        )

        def integrand1(u):
            cf1 = math.exp(-(r - q) * T) * m_rn.char_func(u - 1j, t=T)
            return float(np.real(np.exp(-1j * u * math.log(K / S0)) * cf1 / (1j * u)))

        def integrand2(u):
            cf2 = m_rn.char_func(u, t=T)
            return float(np.real(np.exp(-1j * u * math.log(K / S0)) * cf2 / (1j * u)))

        P1 = 0.5 + (1.0 / np.pi) * quad(integrand1, 1e-8, 100.0)[0]
        P2 = 0.5 + (1.0 / np.pi) * quad(integrand2, 1e-8, 100.0)[0]
        fourier_call = S0 * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2

        assert np.isclose(merton_call, fourier_call, atol=1e-5)

    def test_merton_option_pricing_black_scholes_limit(self):
        S0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        vol = 0.20

        d1 = (math.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
        d2 = d1 - vol * math.sqrt(T)
        bs_call = S0 * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)

        # Merton with rate -> 0
        m_zero_jump = MertonJumpDiffusion(sigma=vol, rate=1e-8, jump_mean=0.0, jump_std=0.01)
        m_call = m_zero_jump.call_price(K, T=T, S0=S0, r=r)
        assert np.isclose(m_call, bs_call, atol=1e-6)

    def test_merton_put_call_parity(self):
        m = MertonJumpDiffusion(drift=0.03, sigma=0.18, rate=1.5, jump_mean=-0.1, jump_std=0.15)
        S0 = 100.0
        r = 0.04
        q = 0.02
        T = 0.75

        for K in [80.0, 95.0, 100.0, 110.0, 125.0]:
            call = m.call_price(K, T=T, S0=S0, r=r, q=q)
            put = m.put_price(K, T=T, S0=S0, r=r, q=q)
            parity_diff = (call - put) - (S0 * math.exp(-q * T) - K * math.exp(-r * T))
            assert np.isclose(parity_diff, 0.0, atol=1e-10)

    def test_merton_implied_volatility_and_smile(self):
        S0 = 100.0
        m = MertonJumpDiffusion(sigma=0.15, rate=1.5, jump_mean=-0.08, jump_std=0.15)
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

        calls = m.call_price(strikes, T=0.5, S0=S0, r=0.05)
        assert isinstance(calls, np.ndarray)
        assert len(calls) == 5
        assert np.all(np.diff(calls) < 0.0)

        # Negative jump mean generates downward sloping skew
        vols = m.implied_vol_smile(strikes, T=0.5, S0=S0, r=0.05)
        assert vols[0] > vols[2] > vols[4]

        # Single strike evaluation
        iv_100 = m.implied_volatility(K=100.0, T=0.5, S0=S0, r=0.05)
        assert np.isclose(iv_100, vols[2])

    def test_merton_simulate_exact_moments(self):
        m = MertonJumpDiffusion(drift=0.04, sigma=0.18, rate=2.0, jump_mean=-0.05, jump_std=0.12)
        T = 1.0
        n_steps = 10
        n_paths = 40_000
        rng = np.random.default_rng(2026)

        paths = m.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
        assert paths.shape == (n_paths, n_steps + 1)
        assert np.all(paths[:, 0] == 0.0)

        terminal_x = paths[:, -1]
        mc_mean = float(np.mean(terminal_x))
        mc_var = float(np.var(terminal_x))

        th_k = m.cumulants(order=2)
        th_mean = th_k[1] * T
        th_var = th_k[2] * T

        assert np.isclose(mc_mean, th_mean, atol=0.005)
        assert np.isclose(mc_var, th_var, atol=0.005)

    def test_merton_properties_and_accessors(self):
        m = MertonJumpDiffusion(jump_rate=2.5, mean_jump=-0.04, std_jump=0.10)
        assert m.rate == 2.5
        assert m.jump_mean == -0.04
        assert m.jump_std == 0.10
        assert m.mu == m.drift
        assert isinstance(m.jump_distribution, spxa.zoo.levy.compound_poisson.NormalJump)

        # Invalid S0 or T in call_price
        with pytest.raises(ValueError, match="Maturity T"):
            m.call_price(K=100.0, T=0.0)
        with pytest.raises(ValueError, match="Spot price S0"):
            m.call_price(K=100.0, S0=0.0)
        with pytest.raises(ValueError, match="Strike K"):
            m.call_price(K=0.0)

        # implied_volatility with vector of strikes
        ivs = m.implied_volatility(np.array([90.0, 100.0, 110.0]), T=1.0)
        assert len(ivs) == 3
        assert np.all(ivs > 0.0)

    def test_merton_story_narrative(self):
        m = MertonJumpDiffusion(drift=0.05, sigma=0.2, rate=1.5, jump_mean=-0.1, jump_std=0.15)
        story = m.__story__()
        assert "Merton" in story
        assert "EXACT" in story
        assert "1976" in story
