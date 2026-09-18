"""
Exactness tests for InverseGaussianProcess and MeixnerProcess.

Each test verifies a closed-form result against a published formula.
These tests must never be skipped or marked xfail.
"""

import numpy as np
import pytest

from spxa.zoo.levy import InverseGaussianProcess, MeixnerProcess
from spxa.ops.subordination import bernstein_function


class TestInverseGaussianProcess:
    @pytest.mark.exactness
    def test_ig_is_subordinator(self) -> None:
        """IG process is non-decreasing. Tweedie (1957): IG has support (0,∞)."""
        ig = InverseGaussianProcess(mu=1.0, lam=2.0)
        assert ig.properties.is_subordinator is True

    @pytest.mark.exactness
    def test_ig_no_gaussian_component(self) -> None:
        """IG process is pure-jump: σ²=0. Cont & Tankov (2004), Table 4.1."""
        ig = InverseGaussianProcess(mu=1.0, lam=2.0)
        assert ig.triplet.sigma_sq == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_ig_cumulant_1(self) -> None:
        """κ₁(X₁) = μ. Tweedie (1957), equation (2.5), n=1: (2·1-1)!!=1, μ^1/λ^0 = μ."""
        ig = InverseGaussianProcess(mu=1.5, lam=3.0)
        assert ig.cumulant_exact(1) == pytest.approx(1.5, rel=1e-10)

    @pytest.mark.exactness
    def test_ig_cumulant_2(self) -> None:
        """κ₂(X₁) = 3μ³/λ. Tweedie (1957), eq (2.5): (2·2-1)!!=3, μ^3/λ^1 → 3μ³/λ."""
        mu, lam = 1.5, 3.0
        ig = InverseGaussianProcess(mu=mu, lam=lam)
        assert ig.cumulant_exact(2) == pytest.approx(3 * mu**3 / lam, rel=1e-10)

    @pytest.mark.exactness
    def test_ig_cumulant_3(self) -> None:
        """κ₃(X₁) = 15μ⁵/λ². Tweedie (1957): (2·3-1)!!=15, μ^5/λ^2 → 15μ⁵/λ²."""
        mu, lam = 1.0, 2.0
        ig = InverseGaussianProcess(mu=mu, lam=lam)
        assert ig.cumulant_exact(3) == pytest.approx(15 * mu**5 / lam**2, rel=1e-10)

    @pytest.mark.exactness
    def test_ig_bernstein_function(self) -> None:
        """φ(s) = (λ/μ)(1 - sqrt(1-2μ²s/λ)). Schilling et al. (2012) Ex 3.12."""
        mu, lam = 1.0, 2.0
        ig = InverseGaussianProcess(mu=mu, lam=lam)
        s_vals = np.array([0.0, 0.5, 1.0])
        phi = ig.bernstein_function(s_vals)
        expected = (lam / mu) * (1 - np.sqrt(1 - 2 * mu**2 * s_vals / lam))
        np.testing.assert_allclose(phi, expected, rtol=1e-10)

    @pytest.mark.exactness
    def test_ig_bernstein_dispatched(self) -> None:
        """bernstein_function() from ops dispatches to IG closed-form."""
        ig = InverseGaussianProcess(mu=1.0, lam=2.0)
        phi_ops = bernstein_function(ig, lam=1.0)
        phi_exact = ig.bernstein_function(1.0)
        assert float(phi_ops) == pytest.approx(float(phi_exact), rel=1e-10)

    @pytest.mark.exactness
    def test_ig_char_func_at_zero(self) -> None:
        """φ(0;t) = 1. Barndorff-Nielsen & Shephard (2001), eq. (A.2)."""
        ig = InverseGaussianProcess(mu=1.0, lam=2.0)
        cf = ig.char_func_exact(u=0.0, t=1.0)
        assert abs(complex(cf) - 1.0) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_ig_char_func_time_additivity(self) -> None:
        """φ(u;s+t) = φ(u;s)·φ(u;t). Lévy process property."""
        ig = InverseGaussianProcess(mu=1.0, lam=2.0)
        u, s, t = 0.5, 0.3, 0.7
        cf_s = complex(ig.char_func_exact(u=u, t=s))
        cf_t = complex(ig.char_func_exact(u=u, t=t))
        cf_st = complex(ig.char_func_exact(u=u, t=s + t))
        assert abs(cf_s * cf_t - cf_st) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_ig_simulate_mean(self) -> None:
        """Empirical mean converges to μ. Tweedie (1957)."""
        rng = np.random.default_rng(80)
        mu = 1.5
        ig = InverseGaussianProcess(mu=mu, lam=3.0)
        paths = ig.simulate(n_steps=1, n_paths=20000, T=1.0, rng=rng)
        assert paths[:, -1].mean() == pytest.approx(mu, rel=0.03)

    @pytest.mark.exactness
    def test_ig_simulate_non_negative(self) -> None:
        """IG paths are non-negative (subordinator)."""
        rng = np.random.default_rng(81)
        ig = InverseGaussianProcess(mu=1.0, lam=2.0)
        paths = ig.simulate(n_steps=100, n_paths=50, T=1.0, rng=rng)
        assert np.all(paths >= 0.0)

    @pytest.mark.exactness
    def test_ig_simulate_non_decreasing(self) -> None:
        """IG process paths are non-decreasing."""
        rng = np.random.default_rng(82)
        ig = InverseGaussianProcess(mu=1.0, lam=2.0)
        paths = ig.simulate(n_steps=100, n_paths=50, T=1.0, rng=rng)
        assert np.all(np.diff(paths, axis=1) >= -1e-12)


class TestMeixnerProcess:
    @pytest.mark.exactness
    def test_meixner_invalid_beta_raises(self) -> None:
        """β must be in (-π, π). Schoutens & Teugels (1998)."""
        with pytest.raises(ValueError):
            MeixnerProcess(alpha=1.0, beta=np.pi, delta=1.0)
        with pytest.raises(ValueError):
            MeixnerProcess(alpha=1.0, beta=-np.pi, delta=1.0)

    @pytest.mark.exactness
    def test_meixner_no_gaussian_component(self) -> None:
        """Meixner is a pure-jump process: σ²=0. Schoutens & Teugels (1998)."""
        m = MeixnerProcess(alpha=0.4, beta=-1.0, delta=0.5)
        assert m.triplet.sigma_sq == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_meixner_char_func_at_zero(self) -> None:
        """φ(0;t)=1. Schoutens & Teugels (1998), equation (3)."""
        m = MeixnerProcess(alpha=0.4, beta=-1.0, delta=0.5)
        cf = m.char_func_exact(u=0.0, t=1.0)
        assert abs(complex(cf) - 1.0) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_meixner_char_func_modulus_leq_1(self) -> None:
        """|φ(u)| ≤ 1 for all real u."""
        m = MeixnerProcess(alpha=0.4, beta=-1.0, delta=0.5)
        u_vals = np.linspace(-5.0, 5.0, 50)
        cf_vals = m.char_func_exact(u=u_vals, t=1.0)
        assert np.all(np.abs(cf_vals) <= 1.0 + 1e-10)

    @pytest.mark.exactness
    def test_meixner_char_func_time_additivity(self) -> None:
        """φ(u;s+t) = φ(u;s)·φ(u;t). Lévy process property."""
        m = MeixnerProcess(alpha=0.4, beta=-0.5, delta=0.5)
        u, s, t = 1.0, 0.3, 0.7
        cf_s = complex(m.char_func_exact(u=u, t=s))
        cf_t = complex(m.char_func_exact(u=u, t=t))
        cf_st = complex(m.char_func_exact(u=u, t=s + t))
        assert abs(cf_s * cf_t - cf_st) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_meixner_symmetric_when_beta_zero(self) -> None:
        """Meixner with β=0 is symmetric: char func is real for real u.
        Schoutens & Teugels (1998): β=0 gives symmetric distribution."""
        m = MeixnerProcess(alpha=0.4, beta=0.0, delta=0.5)
        u_vals = np.array([0.5, 1.0, 2.0])
        cf_vals = m.char_func_exact(u=u_vals, t=1.0)
        assert np.all(np.abs(cf_vals.imag) < 1e-10)

    @pytest.mark.exactness
    def test_meixner_cumulant_1(self) -> None:
        """κ₁ = δα tan(β/2) + m. Schoutens & Teugels (1998), eq. (5)."""
        a, b, d, m = 0.4, -1.0, 0.5, 0.1
        mx = MeixnerProcess(alpha=a, beta=b, delta=d, m=m)
        expected = d * a * np.tan(b / 2) + m
        assert mx.cumulant_exact(1) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_meixner_cumulant_2(self) -> None:
        """κ₂ = δα²/(2cos²(β/2)). Schoutens & Teugels (1998), eq. (6)."""
        a, b, d = 0.4, -1.0, 0.5
        mx = MeixnerProcess(alpha=a, beta=b, delta=d)
        expected = d * a**2 / (2 * np.cos(b / 2)**2)
        assert mx.cumulant_exact(2) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_meixner_cumulant_3(self) -> None:
        """κ₃ = δα³ sin(β/2)/(2cos³(β/2)). Schoutens & Teugels (1998), eq. (7)."""
        a, b, d = 0.4, -1.0, 0.5
        mx = MeixnerProcess(alpha=a, beta=b, delta=d)
        expected = d * a**3 * np.sin(b / 2) / (2 * np.cos(b / 2)**3)
        assert mx.cumulant_exact(3) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_meixner_cumulant_4(self) -> None:
        """κ₄ = δα⁴(2+cosβ)/(8cos⁴(β/2)). Schoutens & Teugels (1998), eq. (8)."""
        a, b, d = 0.4, -1.0, 0.5
        mx = MeixnerProcess(alpha=a, beta=b, delta=d)
        expected = d * a**4 * (2 + np.cos(b)) / (8 * np.cos(b / 2)**4)
        assert mx.cumulant_exact(4) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_meixner_simulate_mean(self) -> None:
        """Empirical mean converges to κ₁."""
        rng = np.random.default_rng(90)
        m = MeixnerProcess(alpha=0.4, beta=-1.0, delta=0.5)
        paths = m.simulate(n_steps=20, n_paths=3000, T=1.0, rng=rng)
        assert paths[:, -1].mean() == pytest.approx(m.cumulant_exact(1), abs=0.05)

    @pytest.mark.exactness
    def test_meixner_simulate_variance(self) -> None:
        """Empirical variance converges to κ₂."""
        rng = np.random.default_rng(91)
        m = MeixnerProcess(alpha=0.4, beta=-1.0, delta=0.5)
        paths = m.simulate(n_steps=20, n_paths=3000, T=1.0, rng=rng)
        assert paths[:, -1].var() == pytest.approx(m.cumulant_exact(2), rel=0.15)
