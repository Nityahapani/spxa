"""
Tests for the sim module — Euler-Maruyama, exact samplers, bridge.
"""

import numpy as np
import pytest

from spxa.sim.bridge import (
    brownian_bridge,
    brownian_bridge_interpolate,
    first_passage_time_bm,
    first_passage_time_exact_bm,
    gamma_bridge,
)
from spxa.sim.euler import (
    euler_maruyama,
    euler_maruyama_autonomous,
    geometric_levy,
)
from spxa.sim.exact import (
    increments_to_paths,
    sample_alpha_stable,
    sample_brownian,
    sample_gamma,
    sample_nig,
    sample_vg,
)
from spxa.zoo.levy import BrownianMotion, GammaProcess, VarianceGamma


class TestEulerMaruyama:
    def test_autonomous_constant_drift(self) -> None:
        """dZ = a dt + 0 dX → Z_t = z0 + a·t exactly."""
        bm = BrownianMotion(sigma=0.0001)
        rng = np.random.default_rng(0)
        paths = euler_maruyama_autonomous(bm, drift=2.0, diffusion=0.0, z0=1.0,
                                          T=1.0, n_steps=100, n_paths=10, rng=rng)
        assert paths[:, -1] == pytest.approx(3.0, abs=0.01)

    def test_autonomous_shape(self) -> None:
        bm = BrownianMotion(sigma=1.0)
        paths = euler_maruyama_autonomous(bm, drift=0.0, diffusion=1.0,
                                          T=1.0, n_steps=50, n_paths=5)
        assert paths.shape == (5, 51)

    def test_em_zero_diffusion_is_ode(self) -> None:
        """With b=0, Euler-Maruyama solves the ODE Z_t = z0·exp(a·t)."""
        bm = BrownianMotion(sigma=0.0001)
        rng = np.random.default_rng(1)
        a = 1.0
        paths = euler_maruyama(
            bm,
            drift=lambda z, t: a * z,
            diffusion=lambda z, t: 0.0,
            z0=1.0, T=1.0, n_steps=5000, n_paths=1, rng=rng,
        )
        assert float(paths[0, -1]) == pytest.approx(np.exp(a), rel=0.01)

    def test_em_shape(self) -> None:
        bm = BrownianMotion(sigma=1.0)
        paths = euler_maruyama(
            bm,
            drift=lambda z, t: 0.0,
            diffusion=lambda z, t: 1.0,
            z0=0.0, T=1.0, n_steps=50, n_paths=5,
        )
        assert paths.shape == (5, 51)

    def test_em_starts_at_z0(self) -> None:
        bm = BrownianMotion(sigma=1.0)
        z0 = 3.5
        paths = euler_maruyama(
            bm,
            drift=lambda z, t: 0.0,
            diffusion=lambda z, t: 1.0,
            z0=z0, T=1.0, n_steps=20, n_paths=10,
        )
        np.testing.assert_array_equal(paths[:, 0], z0)

    def test_geometric_levy_shape(self) -> None:
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
        paths = geometric_levy(vg, mu=0.05, sigma=1.0, s0=100.0,
                               T=1.0, n_steps=100, n_paths=10)
        assert paths.shape == (10, 101)

    def test_geometric_levy_starts_at_s0(self) -> None:
        bm = BrownianMotion(sigma=1.0)
        s0 = 50.0
        paths = geometric_levy(bm, mu=0.0, sigma=1.0, s0=s0,
                               T=1.0, n_steps=100, n_paths=5)
        np.testing.assert_array_equal(paths[:, 0], s0)

    def test_geometric_levy_positive(self) -> None:
        """Geometric Lévy process is always positive."""
        bm = BrownianMotion(sigma=0.5)
        rng = np.random.default_rng(2)
        paths = geometric_levy(bm, mu=0.0, sigma=1.0, s0=1.0,
                               T=1.0, n_steps=200, n_paths=100, rng=rng)
        assert np.all(paths > 0)

    def test_geometric_bm_mean(self) -> None:
        """E[S_t] = S_0·exp(μt) for geometric BM."""
        rng = np.random.default_rng(3)
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        mu, s0, T = 0.1, 100.0, 1.0
        paths = geometric_levy(bm, mu=mu, sigma=1.0, s0=s0,
                               T=T, n_steps=252, n_paths=20000, rng=rng)
        empirical_mean = paths[:, -1].mean()
        assert empirical_mean == pytest.approx(s0 * np.exp(mu * T), rel=0.03)


class TestExactSamplers:
    def test_sample_brownian_shape(self) -> None:
        rng = np.random.default_rng(10)
        inc = sample_brownian(mu=0.0, sigma=1.0, dt=0.01,
                              n_steps=100, n_paths=5, rng=rng)
        assert inc.shape == (5, 100)

    def test_sample_brownian_mean(self) -> None:
        """E[ΔX] = μ·Δt."""
        rng = np.random.default_rng(11)
        mu, dt = 2.0, 0.01
        inc = sample_brownian(mu=mu, sigma=0.1, dt=dt,
                              n_steps=1000, n_paths=5000, rng=rng)
        assert inc.mean() == pytest.approx(mu * dt, abs=0.01)

    def test_sample_gamma_non_negative(self) -> None:
        """Gamma increments are always non-negative."""
        rng = np.random.default_rng(12)
        inc = sample_gamma(a=1.0, b=2.0, dt=0.01,
                           n_steps=200, n_paths=100, rng=rng)
        assert np.all(inc >= 0)

    def test_sample_gamma_mean(self) -> None:
        """E[ΔX] = a·Δt/b."""
        rng = np.random.default_rng(13)
        a, b, dt = 2.0, 3.0, 0.1
        inc = sample_gamma(a=a, b=b, dt=dt, n_steps=500, n_paths=5000, rng=rng)
        assert inc.mean() == pytest.approx(a * dt / b, rel=0.03)

    def test_sample_vg_shape(self) -> None:
        rng = np.random.default_rng(14)
        inc = sample_vg(sigma=0.2, nu=0.1, theta=-0.05, dt=0.01,
                        n_steps=100, n_paths=5, rng=rng)
        assert inc.shape == (5, 100)

    def test_sample_vg_mean(self) -> None:
        """E[ΔX] = θ·Δt for VG."""
        rng = np.random.default_rng(15)
        theta, dt = -0.1, 0.1
        inc = sample_vg(sigma=0.2, nu=0.5, theta=theta, dt=dt,
                        n_steps=500, n_paths=10000, rng=rng)
        assert inc.mean() == pytest.approx(theta * dt, abs=0.01)

    def test_sample_nig_shape(self) -> None:
        rng = np.random.default_rng(16)
        inc = sample_nig(alpha=3.0, beta=-0.5, delta=1.0, mu=0.0, dt=0.01,
                         n_steps=100, n_paths=5, rng=rng)
        assert inc.shape == (5, 100)

    def test_sample_alpha_stable_shape(self) -> None:
        rng = np.random.default_rng(17)
        inc = sample_alpha_stable(alpha=1.5, beta=0.0, sigma=1.0, dt=0.01,
                                  n_steps=100, n_paths=5, rng=rng)
        assert inc.shape == (5, 100)

    def test_sample_alpha_stable_symmetric(self) -> None:
        """Symmetric stable (β=0) increments have median ≈ 0."""
        rng = np.random.default_rng(18)
        inc = sample_alpha_stable(alpha=1.5, beta=0.0, sigma=1.0, dt=1.0,
                                  n_steps=1, n_paths=50000, rng=rng)
        assert np.median(inc.ravel()) == pytest.approx(0.0, abs=0.05)

    def test_increments_to_paths_shape(self) -> None:
        inc = np.ones((5, 10))
        paths = increments_to_paths(inc, x0=2.0)
        assert paths.shape == (5, 11)
        assert np.all(paths[:, 0] == 2.0)

    def test_increments_to_paths_cumsum(self) -> None:
        inc = np.ones((1, 4))
        paths = increments_to_paths(inc, x0=0.0)
        np.testing.assert_array_equal(paths[0], [0.0, 1.0, 2.0, 3.0, 4.0])


class TestBridge:
    def test_brownian_bridge_endpoints(self) -> None:
        """Bridge paths must start at x_start and end at x_end."""
        rng = np.random.default_rng(20)
        t_grid, paths = brownian_bridge(x_start=0.0, x_end=1.0,
                                        n_points=50, n_paths=10, rng=rng)
        np.testing.assert_array_equal(paths[:, 0], 0.0)
        np.testing.assert_array_equal(paths[:, -1], 1.0)

    def test_brownian_bridge_shape(self) -> None:
        t_grid, paths = brownian_bridge(n_points=50, n_paths=5)
        assert paths.shape == (5, 52)
        assert t_grid.shape == (52,)

    def test_brownian_bridge_midpoint_mean(self) -> None:
        """E[B(t) | B(T)=b] = b·t/T: check by running 5000 bridges and inspecting mean at t=0.5."""
        rng = np.random.default_rng(21)
        x_end = 2.0
        n_points = 98  # interior points so total = 100, midpoint at index 50
        _, paths = brownian_bridge(x_start=0.0, x_end=x_end,
                                   n_points=n_points, n_paths=5000, rng=rng)
        # time_grid is linspace(0, 1, 100); index 50 is t=50/99 ≈ 0.5051
        t_mid = 50 / (n_points + 1)
        expected_mean = x_end * t_mid
        assert paths[:, 50].mean() == pytest.approx(expected_mean, abs=0.1)

    def test_gamma_bridge_endpoints(self) -> None:
        """Gamma bridge paths must start at 0 and end at x_end."""
        rng = np.random.default_rng(22)
        x_end = 2.0
        t_grid, paths = gamma_bridge(a=1.0, b=1.0, T=1.0, x_end=x_end,
                                     n_points=50, n_paths=10, rng=rng)
        np.testing.assert_array_equal(paths[:, 0], 0.0)
        np.testing.assert_array_equal(paths[:, -1], x_end)

    def test_gamma_bridge_non_negative(self) -> None:
        """Gamma bridge values are all non-negative."""
        rng = np.random.default_rng(23)
        _, paths = gamma_bridge(a=1.0, b=1.0, T=1.0, x_end=3.0,
                                n_points=100, n_paths=20, rng=rng)
        assert np.all(paths >= -1e-12)

    def test_bridge_interpolate_at_coarse_times(self) -> None:
        """Interpolated paths agree with coarse paths at shared times."""
        rng = np.random.default_rng(24)
        bm = BrownianMotion(sigma=1.0)
        coarse_times = np.linspace(0, 1, 6)
        coarse_paths = bm.simulate(n_steps=5, n_paths=3, T=1.0, rng=rng)
        fine_times = np.linspace(0, 1, 21)
        fine_paths = brownian_bridge_interpolate(
            coarse_paths, coarse_times, fine_times, sigma=1.0, rng=rng
        )
        assert fine_paths.shape == (3, 21)
        for i, t in enumerate(coarse_times):
            j = np.argmin(np.abs(fine_times - t))
            np.testing.assert_allclose(
                fine_paths[:, j], coarse_paths[:, i], atol=1e-10
            )

    def test_first_passage_time_bm_positive(self) -> None:
        """First passage times are positive (or inf)."""
        rng = np.random.default_rng(25)
        tau = first_passage_time_bm(level=1.0, sigma=1.0, T_max=5.0,
                                    n_steps=1000, n_paths=100, rng=rng)
        assert np.all(tau > 0)

    def test_first_passage_time_exact_bm_mean(self) -> None:
        """E[τ] for BM to level a with σ=1 is infinite (Cauchy distribution).
        Verify samples are positive and roughly right-skewed."""
        rng = np.random.default_rng(26)
        tau = first_passage_time_exact_bm(level=1.0, sigma=1.0, n_samples=5000, rng=rng)
        assert np.all(tau > 0)
        assert np.median(tau) < np.mean(tau)

    def test_first_passage_time_exact_bm_negative_level_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            first_passage_time_exact_bm(level=-1.0)
