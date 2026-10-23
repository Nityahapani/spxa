"""
Unit tests for the analytics module (Fourier pricing, divergence, cumulants).
"""

import numpy as np
import pytest

from spxa.analytics.cumulants import cumulant_table, excess_kurtosis, skewness
from spxa.analytics.divergence import (
    hellinger_distance,
    kl_divergence_mc,
    l2_char_func_distance,
    wasserstein2_distance,
)
from spxa.analytics.fourier import (
    european_call_price,
    european_put_price,
    implied_volatility,
)
from spxa.zoo.levy import BrownianMotion, VarianceGamma


class TestCumulantsAnalytics:
    def test_cumulant_table_keys_and_values(self) -> None:
        bm = BrownianMotion(mu=0.5, sigma=1.0)
        table = cumulant_table(bm, order=4, t=1.0)
        assert table["mean"] == pytest.approx(0.5)
        assert table["variance"] == pytest.approx(1.0)
        assert table["skewness"] == pytest.approx(0.0, abs=1e-10)
        assert table["excess_kurtosis"] == pytest.approx(0.0, abs=1e-10)

    def test_skewness_and_kurtosis(self) -> None:
        bm = BrownianMotion(mu=0.2, sigma=0.5)
        s = skewness(bm, t=2.0)
        k = excess_kurtosis(bm, t=2.0)
        assert s == pytest.approx(0.0, abs=1e-10)
        assert k == pytest.approx(0.0, abs=1e-10)


class TestFourierPricing:
    def test_call_price_bounds(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=0.2)
        s0, k, r, t = 100.0, 100.0, 0.05, 1.0
        price = european_call_price(bm, S0=s0, K=k, r=r, t=t)
        assert price > 0.0
        # Call price must be below S0
        assert price < s0

    def test_put_call_parity(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=0.15)
        s0, k, r, t = 100.0, 105.0, 0.03, 0.5
        c = european_call_price(bm, S0=s0, K=k, r=r, t=t)
        p = european_put_price(bm, S0=s0, K=k, r=r, t=t)
        # S0 - K * exp(-r*t)
        expected_diff = s0 - k * np.exp(-r * t)
        assert (c - p) == pytest.approx(expected_diff, rel=0.05)

    def test_implied_volatility_recovery(self) -> None:
        from scipy.stats import norm

        sigma_true = 0.25
        s0, k, r, t = 100.0, 100.0, 0.05, 1.0
        d1 = (np.log(s0 / k) + (r + 0.5 * sigma_true**2) * t) / (sigma_true * np.sqrt(t))
        d2 = d1 - sigma_true * np.sqrt(t)
        bs_call = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

        iv = implied_volatility(bs_call, S0=s0, K=k, r=r, t=t)
        assert iv == pytest.approx(sigma_true, abs=1e-4)


class TestDivergenceAnalytics:
    def test_l2_distance_positivity(self) -> None:
        bm1 = BrownianMotion(sigma=0.2)
        bm2 = BrownianMotion(sigma=0.4)
        dist = l2_char_func_distance(bm1, bm2, t=1.0)
        assert dist > 0.0

    def test_hellinger_distance_range(self) -> None:
        bm = BrownianMotion(sigma=0.3)
        vg = VarianceGamma(sigma=0.3, nu=0.2, theta=0.0)
        h = hellinger_distance(bm, vg, t=1.0)
        assert 0.0 <= h <= 1.0

    def test_wasserstein2_distance_identical_is_zero(self) -> None:
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=0.0, sigma=1.0)
        w2 = wasserstein2_distance(bm1, bm2, t=1.0)
        assert w2 == pytest.approx(0.0, abs=0.05)

    def test_kl_divergence_mc_same_is_small(self) -> None:
        rng = np.random.default_rng(42)
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=0.0, sigma=1.0)
        kl = kl_divergence_mc(bm1, bm2, t=1.0, n_samples=2000, rng=rng)
        assert abs(kl) < 0.2
