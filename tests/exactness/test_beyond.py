"""
Exactness tests for beyond-Lévy processes (fBM, Hawkes, OULevy).

These tests verify the analytical formulas that ARE available despite the
processes being outside the Lévy class. They also verify that ExactnessError
is raised correctly when users attempt to access unavailable quantities.

These tests must never be skipped or marked xfail.
"""

import numpy as np
import pytest

from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import FractionalBrownianMotion, HawkesProcess, OULevy
from spxa.zoo.levy import GammaProcess


class TestFractionalBrownianMotion:
    @pytest.mark.exactness
    def test_fbm_triplet_raises(self) -> None:
        """
        fBM with H≠0.5 has no Lévy triplet. Must raise ExactnessError.
        Mandelbrot & Van Ness (1968): fBM is not a semimartingale for H≠0.5.
        """
        fbm = FractionalBrownianMotion(H=0.7)
        with pytest.raises(ExactnessError):
            _ = fbm.triplet

    @pytest.mark.exactness
    def test_fbm_variance_formula(self) -> None:
        """
        Var(B^H_t) = σ²·t^{2H}.
        Mandelbrot & Van Ness (1968), equation (1.3).
        """
        fbm = FractionalBrownianMotion(H=0.7, sigma=2.0)
        t = 0.5
        assert fbm.variance(t) == pytest.approx(4.0 * t**1.4, rel=1e-10)

    @pytest.mark.exactness
    def test_fbm_covariance_formula(self) -> None:
        """
        Cov(B^H_s, B^H_t) = σ²/2·(s^{2H} + t^{2H} - |s-t|^{2H}).
        Mandelbrot & Van Ness (1968), equation (1.2).
        """
        H, sigma = 0.7, 1.0
        fbm = FractionalBrownianMotion(H=H, sigma=sigma)
        s, t = 0.3, 0.8
        expected = 0.5 * (s**(2*H) + t**(2*H) - abs(s - t)**(2*H))
        assert fbm.covariance(s, t) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_fbm_self_similarity_index(self) -> None:
        """
        fBM has self-similarity index H.
        B^H_{ct} ∼ c^H B^H_t. Mandelbrot & Van Ness (1968), Property 1.
        """
        for H in [0.3, 0.5, 0.7, 0.9]:
            fbm = FractionalBrownianMotion(H=H)
            assert fbm.properties.self_similarity_index == pytest.approx(H)
            assert fbm.properties.hurst_index == pytest.approx(H)

    @pytest.mark.exactness
    def test_fbm_half_is_martingale(self) -> None:
        """
        fBM with H=0.5 is standard BM and a martingale.
        fBM with H≠0.5 is not a martingale.
        """
        assert FractionalBrownianMotion(H=0.5).properties.is_martingale is True
        assert FractionalBrownianMotion(H=0.7).properties.is_martingale is False

    @pytest.mark.exactness
    def test_fbm_no_independent_increments_when_h_neq_half(self) -> None:
        """
        fBM with H≠0.5 has correlated increments — not independent.
        Mandelbrot & Van Ness (1968): only H=0.5 gives independent increments.
        """
        assert FractionalBrownianMotion(H=0.7).properties.has_independent_increments is False
        assert FractionalBrownianMotion(H=0.5).properties.has_independent_increments is True

    @pytest.mark.exactness
    def test_fbm_simulate_variance(self) -> None:
        """
        Empirical variance of B^H_t converges to σ²·t^{2H}.
        Mandelbrot & Van Ness (1968), equation (1.3).
        """
        rng = np.random.default_rng(50)
        H, sigma, t = 0.7, 1.0, 1.0
        fbm = FractionalBrownianMotion(H=H, sigma=sigma)
        paths = fbm.simulate(n_steps=200, n_paths=2000, T=t, rng=rng)
        empirical_var = paths[:, -1].var()
        expected_var = sigma**2 * t**(2 * H)
        assert empirical_var == pytest.approx(expected_var, rel=0.1)

    @pytest.mark.exactness
    def test_fbm_simulate_starts_at_zero(self) -> None:
        """All fBM paths start at 0: B^H_0 = 0 a.s."""
        fbm = FractionalBrownianMotion(H=0.7)
        paths = fbm.simulate(n_steps=50, n_paths=20)
        assert np.all(paths[:, 0] == 0.0)


class TestHawkesProcess:
    @pytest.mark.exactness
    def test_hawkes_supercritical_raises(self) -> None:
        """
        Hawkes process requires alpha < beta (subcritical). Hawkes (1971).
        """
        with pytest.raises(ValueError, match="supercritical"):
            HawkesProcess(mu=1.0, alpha=1.0, beta=0.5)

    @pytest.mark.exactness
    def test_hawkes_triplet_raises(self) -> None:
        """Hawkes process has no Lévy triplet. Must raise ExactnessError."""
        h = HawkesProcess(mu=1.0, alpha=0.5, beta=1.0)
        with pytest.raises(ExactnessError):
            _ = h.triplet

    @pytest.mark.exactness
    def test_hawkes_mean_intensity(self) -> None:
        """
        λ̄ = μ / (1 - α/β).
        Hawkes (1971), equation (3.5).
        """
        mu, alpha, beta = 1.0, 0.5, 1.0
        h = HawkesProcess(mu=mu, alpha=alpha, beta=beta)
        expected = mu / (1 - alpha / beta)
        assert h.mean_intensity == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_hawkes_fano_factor_geq_1(self) -> None:
        """
        Fano factor F ≥ 1 for all Hawkes processes (always overdispersed).
        Hawkes & Oakes (1974), Theorem 1.
        """
        h = HawkesProcess(mu=1.0, alpha=0.5, beta=2.0)
        assert h.fano_factor >= 1.0

    @pytest.mark.exactness
    def test_hawkes_simulate_mean_rate(self) -> None:
        """
        Empirical event rate N_T/T converges to λ̄ for large T.
        Hawkes (1971): stationary mean intensity is λ̄ = μ/(1-α/β).
        """
        rng = np.random.default_rng(60)
        mu, alpha, beta = 2.0, 0.5, 2.0
        h = HawkesProcess(mu=mu, alpha=alpha, beta=beta)
        T = 100.0
        paths = h.simulate(n_steps=1000, n_paths=50, T=T, rng=rng)
        empirical_rate = paths[:, -1].mean() / T
        assert empirical_rate == pytest.approx(h.mean_intensity, rel=0.1)

    @pytest.mark.exactness
    def test_hawkes_simulate_non_decreasing(self) -> None:
        """Counting process N_t is non-decreasing in t."""
        rng = np.random.default_rng(61)
        h = HawkesProcess(mu=1.0, alpha=0.3, beta=1.0)
        paths = h.simulate(n_steps=100, n_paths=20, T=10.0, rng=rng)
        assert np.all(np.diff(paths, axis=1) >= 0)


class TestOULevy:
    @pytest.mark.exactness
    def test_ou_triplet_raises(self) -> None:
        """OULevy has no Lévy triplet. Must raise ExactnessError."""
        ou = OULevy(lam=1.0)
        with pytest.raises(ExactnessError):
            _ = ou.triplet

    @pytest.mark.exactness
    def test_ou_stationary_variance_formula(self) -> None:
        """
        Var(V_∞) = κ_2(L_1) / (2λ).
        Barndorff-Nielsen & Shephard (2001), equation (16).
        """
        lam = 2.0
        g = GammaProcess(a=1.0, b=1.0)
        ou = OULevy(lam=lam, subordinator=g)
        kappa2_L = g.cumulants(order=2)[2]
        expected_var = kappa2_L / (2 * lam)
        assert ou.stationary_variance == pytest.approx(expected_var, rel=1e-10)

    @pytest.mark.exactness
    def test_ou_autocovariance_formula(self) -> None:
        """
        Cov(V_t, V_{t+h}) = e^{-λh} · Var(V_∞).
        Barndorff-Nielsen & Shephard (2001), equation (17).
        """
        lam = 1.5
        ou = OULevy(lam=lam)
        h = 0.5
        expected = np.exp(-lam * h) * ou.stationary_variance
        assert ou.autocovariance(h) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_ou_autocovariance_decays_to_zero(self) -> None:
        """
        Cov(V_t, V_{t+h}) → 0 as h → ∞ (mean reversion to stationarity).
        """
        ou = OULevy(lam=2.0)
        assert ou.autocovariance(100.0) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_ou_requires_subordinator(self) -> None:
        """
        OULevy must be driven by a subordinator (non-decreasing process).
        Barndorff-Nielsen & Shephard (2001): L must be a subordinator.
        """
        from spxa.zoo.levy import BrownianMotion
        with pytest.raises(ValueError, match="subordinator"):
            OULevy(lam=1.0, subordinator=BrownianMotion())

    @pytest.mark.exactness
    def test_ou_simulate_shape(self) -> None:
        """Simulation returns correct shape."""
        ou = OULevy(lam=1.0)
        paths = ou.simulate(n_steps=50, n_paths=10, T=1.0)
        assert paths.shape == (10, 51)

    @pytest.mark.exactness
    def test_ou_simulate_non_negative_with_gamma_driver(self) -> None:
        """
        OU driven by Gamma subordinator has non-negative paths (V_0=0, increments≥0).
        Gamma process has non-negative increments; OU preserves positivity.
        """
        rng = np.random.default_rng(70)
        ou = OULevy(lam=1.0, subordinator=GammaProcess(a=1.0, b=1.0), v0=0.0)
        paths = ou.simulate(n_steps=200, n_paths=50, T=5.0, rng=rng)
        assert np.all(paths >= -1e-12)
