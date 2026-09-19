"""
Exactness tests for TemperedStable and NegativeBinomialProcess.

Each test verifies a closed-form result against a published formula.
These tests must never be skipped or marked xfail.
"""

import numpy as np
import pytest

from spxa.zoo.levy.tempered_stable import TemperedStable
from spxa.zoo.levy.negative_binomial import NegativeBinomialProcess
from spxa.ops.subordination import bernstein_function


# ── TemperedStable ────────────────────────────────────────────────────────────

class TestTemperedStable:
    @pytest.mark.exactness
    def test_ts_invalid_alpha_raises(self) -> None:
        """α must be in (0,1). Rosiński (2007): TS subordinator requires α∈(0,1)."""
        with pytest.raises(ValueError):
            TemperedStable(alpha=0.0, C=1.0, lam=1.0)
        with pytest.raises(ValueError):
            TemperedStable(alpha=1.0, C=1.0, lam=1.0)
        with pytest.raises(ValueError):
            TemperedStable(alpha=1.5, C=1.0, lam=1.0)

    @pytest.mark.exactness
    def test_ts_is_subordinator(self) -> None:
        """TemperedStable is non-decreasing. Rosiński (2007): support ⊂ (0,∞)."""
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        assert ts.properties.is_subordinator is True

    @pytest.mark.exactness
    def test_ts_no_gaussian_component(self) -> None:
        """TemperedStable is pure-jump: σ²=0. Cont & Tankov (2004), Table 4.1."""
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        assert ts.triplet.sigma_sq == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_ts_infinite_activity(self) -> None:
        """TemperedStable has infinite activity: ν(ℝ\\{0})=∞.
        Rosiński (2007): ∫_0^1 x^{-1-α} dx diverges at 0 for α>0."""
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        assert np.isinf(ts.triplet.nu.total_mass())

    @pytest.mark.exactness
    def test_ts_cumulant_1(self) -> None:
        """κ₁ = C·Γ(1-α)/λ^{1-α}. Rosiński (2007), Proposition 2.1."""
        from scipy.special import gamma as gamma_fn
        alpha, C, lam = 0.5, 1.0, 2.0
        ts = TemperedStable(alpha=alpha, C=C, lam=lam)
        expected = C * float(gamma_fn(1 - alpha)) / lam ** (1 - alpha)
        assert ts.cumulant_exact(1) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_ts_cumulant_2(self) -> None:
        """κ₂ = C·Γ(2-α)/λ^{2-α}. Rosiński (2007), Proposition 2.1."""
        from scipy.special import gamma as gamma_fn
        alpha, C, lam = 0.5, 1.0, 2.0
        ts = TemperedStable(alpha=alpha, C=C, lam=lam)
        expected = C * float(gamma_fn(2 - alpha)) / lam ** (2 - alpha)
        assert ts.cumulant_exact(2) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_ts_cumulant_3(self) -> None:
        """κ₃ = C·Γ(3-α)/λ^{3-α}. Rosiński (2007), Proposition 2.1."""
        from scipy.special import gamma as gamma_fn
        alpha, C, lam = 0.5, 1.0, 2.0
        ts = TemperedStable(alpha=alpha, C=C, lam=lam)
        expected = C * float(gamma_fn(3 - alpha)) / lam ** (3 - alpha)
        assert ts.cumulant_exact(3) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_ts_bernstein_at_zero(self) -> None:
        """φ(0) = 0. Every Bernstein function satisfies φ(0)=0.
        Schilling, Song & Vondraček (2012), Definition 3.1."""
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        assert float(ts.bernstein_function(0.0)) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_ts_bernstein_positive(self) -> None:
        """φ(s) ≥ 0 for s ≥ 0. Schilling et al. (2012), Definition 3.1."""
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        s_vals = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        phi = ts.bernstein_function(s_vals)
        assert np.all(phi >= 0)

    @pytest.mark.exactness
    def test_ts_bernstein_exact_formula(self) -> None:
        """φ(s) = -C·Γ(-α)·((λ+s)^α - λ^α). Schilling et al. (2012), Example 3.8."""
        from scipy.special import gamma as gamma_fn
        alpha, C, lam = 0.5, 1.0, 2.0
        ts = TemperedStable(alpha=alpha, C=C, lam=lam)
        s = 1.0
        expected = -C * float(gamma_fn(-alpha)) * ((lam + s)**alpha - lam**alpha)
        assert float(ts.bernstein_function(s)) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_ts_bernstein_dispatched(self) -> None:
        """bernstein_function() from ops dispatches to TS closed-form."""
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        phi_ops = float(bernstein_function(ts, lam=1.0))
        phi_exact = float(ts.bernstein_function(1.0))
        assert phi_ops == pytest.approx(phi_exact, rel=1e-10)

    @pytest.mark.exactness
    def test_ts_char_func_at_zero(self) -> None:
        """φ(0;t) = 1. Rosiński (2007), equation (1.3)."""
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        assert abs(complex(ts.char_func_exact(0.0)) - 1.0) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_ts_char_func_modulus_leq_1(self) -> None:
        """|φ(u)| ≤ 1. Standard property of characteristic functions."""
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        u_vals = np.linspace(-5.0, 5.0, 50)
        cf = ts.char_func_exact(u=u_vals, t=1.0)
        assert np.all(np.abs(cf) <= 1.0 + 1e-10)

    @pytest.mark.exactness
    def test_ts_char_func_time_additivity(self) -> None:
        """φ(u;s+t) = φ(u;s)·φ(u;t). Lévy process property, Sato (1999) Def. 1.6."""
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        u, s, t = 1.0, 0.3, 0.7
        cf_s = complex(ts.char_func_exact(u=u, t=s))
        cf_t = complex(ts.char_func_exact(u=u, t=t))
        cf_st = complex(ts.char_func_exact(u=u, t=s + t))
        assert abs(cf_s * cf_t - cf_st) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_ts_simulate_non_decreasing(self) -> None:
        """TemperedStable paths are non-decreasing (subordinator)."""
        rng = np.random.default_rng(100)
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        paths = ts.simulate(n_steps=50, n_paths=30, T=1.0, rng=rng)
        assert np.all(np.diff(paths, axis=1) >= -1e-12)

    @pytest.mark.exactness
    def test_ts_simulate_non_negative(self) -> None:
        """TemperedStable paths are non-negative."""
        rng = np.random.default_rng(101)
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        paths = ts.simulate(n_steps=50, n_paths=30, T=1.0, rng=rng)
        assert np.all(paths >= -1e-12)

    @pytest.mark.exactness
    def test_ts_simulate_mean(self) -> None:
        """Empirical mean converges to κ₁ = C·Γ(1-α)/λ^{1-α}.
        Rosiński (2007), Proposition 2.1."""
        rng = np.random.default_rng(102)
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        paths = ts.simulate(n_steps=1, n_paths=10000, T=1.0, rng=rng)
        assert paths[:, -1].mean() == pytest.approx(ts.cumulant_exact(1), rel=0.05)

    @pytest.mark.exactness
    def test_ts_simulate_variance(self) -> None:
        """Empirical variance converges to κ₂ = C·Γ(2-α)/λ^{2-α}.
        Rosiński (2007), Proposition 2.1."""
        rng = np.random.default_rng(103)
        ts = TemperedStable(alpha=0.5, C=1.0, lam=2.0)
        paths = ts.simulate(n_steps=1, n_paths=10000, T=1.0, rng=rng)
        assert paths[:, -1].var() == pytest.approx(ts.cumulant_exact(2), rel=0.07)


# ── NegativeBinomialProcess ───────────────────────────────────────────────────

class TestNegativeBinomialProcess:
    @pytest.mark.exactness
    def test_nb_invalid_params_raise(self) -> None:
        """r>0 and p∈(0,1) required. Johnson et al. (2005), Chapter 5."""
        with pytest.raises(ValueError):
            NegativeBinomialProcess(r=0.0, p=0.3)
        with pytest.raises(ValueError):
            NegativeBinomialProcess(r=1.0, p=0.0)
        with pytest.raises(ValueError):
            NegativeBinomialProcess(r=1.0, p=1.0)

    @pytest.mark.exactness
    def test_nb_is_subordinator(self) -> None:
        """NegBin increments are non-negative integers → subordinator.
        Kozubowski & Podgórski (2009), Section 2."""
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        assert nb.properties.is_subordinator is True

    @pytest.mark.exactness
    def test_nb_finite_activity(self) -> None:
        """NegBin Lévy measure has finite total mass = r·log(1/(1-p)).
        Quenouille (1949): the Lévy measure is a discrete measure on ℕ."""
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        expected_mass = 2.0 * np.log(1 / 0.7)
        assert nb.triplet.nu.total_mass() == pytest.approx(expected_mass, rel=1e-10)

    @pytest.mark.exactness
    def test_nb_no_gaussian_component(self) -> None:
        """NegBin process is pure-jump: σ²=0."""
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        assert nb.triplet.sigma_sq == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.exactness
    def test_nb_char_func_at_zero(self) -> None:
        """φ(0;t) = 1. Quenouille (1949): ((1-p)/(1-p·e^0))^{rt} = 1."""
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        assert abs(complex(nb.char_func_exact(0.0)) - 1.0) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.exactness
    def test_nb_char_func_modulus_leq_1(self) -> None:
        """|φ(u)| ≤ 1. Standard property of characteristic functions."""
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        u_vals = np.linspace(-5.0, 5.0, 50)
        cf = nb.char_func_exact(u=u_vals, t=1.0)
        assert np.all(np.abs(cf) <= 1.0 + 1e-10)

    @pytest.mark.exactness
    def test_nb_char_func_time_additivity(self) -> None:
        """φ(u;s+t) = φ(u;s)·φ(u;t). Lévy process property, Sato (1999) Def. 1.6."""
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        u, s, t = 0.5, 0.4, 0.6
        cf_s = complex(nb.char_func_exact(u=u, t=s))
        cf_t = complex(nb.char_func_exact(u=u, t=t))
        cf_st = complex(nb.char_func_exact(u=u, t=s + t))
        assert abs(cf_s * cf_t - cf_st) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.exactness
    def test_nb_cumulant_1(self) -> None:
        """κ₁ = r·p/(1-p). Johnson et al. (2005), Table 5.2."""
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        expected = 2.0 * 0.3 / 0.7
        assert nb.cumulant_exact(1) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_nb_cumulant_2(self) -> None:
        """κ₂ = r·p/(1-p)². Johnson et al. (2005), Table 5.2."""
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        expected = 2.0 * 0.3 / 0.7**2
        assert nb.cumulant_exact(2) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.exactness
    def test_nb_cumulant_3(self) -> None:
        """κ₃ = r·p·(1+p)/(1-p)³. Johnson et al. (2005), Table 5.2."""
        r, p = 2.0, 0.3
        nb = NegativeBinomialProcess(r=r, p=p)
        expected = r * p * (1 + p) / (1 - p)**3
        assert nb.cumulant_exact(3) == pytest.approx(expected, rel=1e-5)

    @pytest.mark.exactness
    def test_nb_cumulant_4(self) -> None:
        """κ₄ = r·p·(1+4p+p²)/(1-p)⁴. Johnson et al. (2005), Table 5.2."""
        r, p = 2.0, 0.3
        nb = NegativeBinomialProcess(r=r, p=p)
        expected = r * p * (1 + 4*p + p**2) / (1 - p)**4
        assert nb.cumulant_exact(4) == pytest.approx(expected, rel=1e-4)

    @pytest.mark.exactness
    def test_nb_simulate_non_decreasing(self) -> None:
        """NegBin paths are non-decreasing (non-negative integer increments)."""
        rng = np.random.default_rng(110)
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        paths = nb.simulate(n_steps=100, n_paths=50, T=1.0, rng=rng)
        assert np.all(np.diff(paths, axis=1) >= 0)

    @pytest.mark.exactness
    def test_nb_simulate_integer_values(self) -> None:
        """NegBin paths take non-negative integer values."""
        rng = np.random.default_rng(111)
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        paths = nb.simulate(n_steps=50, n_paths=20, T=1.0, rng=rng)
        assert np.all(paths >= 0)
        assert np.all(paths == np.round(paths))

    @pytest.mark.exactness
    def test_nb_simulate_mean(self) -> None:
        """Empirical mean converges to r·p/(1-p). Johnson et al. (2005)."""
        rng = np.random.default_rng(112)
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        paths = nb.simulate(n_steps=1, n_paths=50000, T=1.0, rng=rng)
        assert paths[:, -1].mean() == pytest.approx(nb.cumulant_exact(1), rel=0.02)

    @pytest.mark.exactness
    def test_nb_simulate_variance(self) -> None:
        """Empirical variance converges to r·p/(1-p)². Johnson et al. (2005)."""
        rng = np.random.default_rng(113)
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        paths = nb.simulate(n_steps=1, n_paths=50000, T=1.0, rng=rng)
        assert paths[:, -1].var() == pytest.approx(nb.cumulant_exact(2), rel=0.03)

    @pytest.mark.exactness
    def test_nb_overdispersed(self) -> None:
        """NegBin is overdispersed: Var > Mean (κ₂ > κ₁) for p > 0.
        This is the defining property distinguishing NegBin from Poisson."""
        nb = NegativeBinomialProcess(r=2.0, p=0.3)
        assert nb.cumulant_exact(2) > nb.cumulant_exact(1)
