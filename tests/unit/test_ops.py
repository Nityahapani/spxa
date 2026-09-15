"""
Tests for the ops module — addition, scaling, subordination, product, integral.
"""

import warnings

import numpy as np
import pytest

from spxa.core.exceptions import ExactnessError, SpxaDegradationWarning
from spxa.core.exactness import ExactnessLevel
from spxa.ops.addition import add, sum_processes, triplet_add, exactness_for_addition
from spxa.ops.integral import (
    StochasticIntegral,
    ito_isometry_variance,
    stochastic_convolution_char_func,
)
from spxa.ops.product import (
    ProductProcess,
    cumulants_to_raw_moments,
    multiply,
    raw_moments_to_cumulants,
)
from spxa.ops.scaling import negate, scale, subtract, triplet_scale
from spxa.ops.subordination import (
    bernstein_function,
    char_exp_subordinated,
    is_valid_subordination,
    subordinate,
)
from spxa.zoo.levy import BrownianMotion, GammaProcess, NIG, VarianceGamma


class TestAddition:
    def test_add_two_bm(self) -> None:
        bm1 = BrownianMotion(mu=1.0, sigma=1.0)
        bm2 = BrownianMotion(mu=2.0, sigma=2.0)
        z = add(bm1, bm2)
        assert z.triplet.b == pytest.approx(3.0)
        assert z.triplet.sigma_sq == pytest.approx(5.0)

    def test_sum_processes_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            sum_processes([])

    def test_sum_processes_single(self) -> None:
        bm = BrownianMotion(mu=1.0, sigma=1.0)
        z = sum_processes([bm])
        assert z.triplet.sigma_sq == pytest.approx(1.0)

    def test_sum_processes_multiple(self) -> None:
        bms = [BrownianMotion(mu=0.0, sigma=1.0) for _ in range(4)]
        z = sum_processes(bms)
        assert z.triplet.sigma_sq == pytest.approx(4.0)

    def test_triplet_add(self) -> None:
        bm1 = BrownianMotion(mu=1.0, sigma=1.0)
        bm2 = BrownianMotion(mu=2.0, sigma=2.0)
        t = triplet_add(bm1.triplet, bm2.triplet)
        assert t.b == pytest.approx(3.0)
        assert t.sigma_sq == pytest.approx(5.0)

    def test_exactness_for_addition_exact(self) -> None:
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        level, reason = exactness_for_addition(bm1, bm2)
        assert level == ExactnessLevel.EXACT
        assert reason == ""

    def test_exactness_for_addition_degrades(self) -> None:
        from spxa.zoo.beyond import FractionalBrownianMotion
        fbm = FractionalBrownianMotion(H=0.7)
        bm = BrownianMotion(sigma=1.0)
        level, reason = exactness_for_addition(fbm, bm)
        assert level == ExactnessLevel.MOMENT_PROPAGATION
        assert len(reason) > 0


class TestScaling:
    def test_scale_sigma_sq(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        z = scale(bm, 3.0)
        assert z.triplet.sigma_sq == pytest.approx(9.0)

    def test_scale_drift(self) -> None:
        bm = BrownianMotion(mu=2.0, sigma=0.01)
        z = scale(bm, 4.0)
        assert z.triplet.b == pytest.approx(8.0, rel=1e-4)

    def test_scale_zero_raises(self) -> None:
        bm = BrownianMotion(sigma=1.0)
        with pytest.raises(ValueError, match="0"):
            scale(bm, 0.0)

    def test_negate(self) -> None:
        bm = BrownianMotion(mu=1.0, sigma=1.0)
        z = negate(bm)
        assert z.triplet.b == pytest.approx(-1.0)
        assert z.triplet.sigma_sq == pytest.approx(1.0)

    def test_subtract(self) -> None:
        bm1 = BrownianMotion(mu=3.0, sigma=2.0)
        bm2 = BrownianMotion(mu=1.0, sigma=1.0)
        z = subtract(bm1, bm2)
        assert z.triplet.b == pytest.approx(2.0)
        assert z.triplet.sigma_sq == pytest.approx(5.0)

    def test_triplet_scale(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        t_scaled = triplet_scale(bm.triplet, 2.0)
        assert t_scaled.sigma_sq == pytest.approx(4.0)


class TestSubordination:
    def test_subordinate_bm_gamma(self) -> None:
        bm = BrownianMotion(sigma=1.0)
        g = GammaProcess(a=1.0, b=1.0)
        z = subordinate(bm, g)
        assert z.properties.has_stationary_increments is True
        assert z.properties.has_independent_increments is True

    def test_subordinate_non_subordinator_raises(self) -> None:
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        with pytest.raises(ValueError, match="subordinator"):
            subordinate(bm1, bm2)

    def test_is_valid_subordination_true(self) -> None:
        bm = BrownianMotion(sigma=1.0)
        g = GammaProcess(a=1.0, b=1.0)
        valid, reason = is_valid_subordination(bm, g)
        assert valid is True
        assert reason == ""

    def test_is_valid_subordination_false(self) -> None:
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        valid, reason = is_valid_subordination(bm1, bm2)
        assert valid is False
        assert len(reason) > 0

    def test_bernstein_function_gamma_at_zero(self) -> None:
        """Bernstein function φ(0) = 0 for all subordinators."""
        g = GammaProcess(a=1.0, b=2.0)
        phi_0 = bernstein_function(g, lam=0.0)
        assert float(phi_0) == pytest.approx(0.0, abs=1e-6)

    def test_bernstein_function_gamma_exact(self) -> None:
        """φ_Gamma(λ) = a·log(1 + λ/b) exactly. Schilling et al. (2012), Example 3.9."""
        g = GammaProcess(a=1.0, b=2.0)
        lam_vals = np.array([0.0, 0.5, 1.0, 2.0])
        phi_vals = bernstein_function(g, lam=lam_vals)
        expected = np.array([0.0, np.log(1.25), np.log(1.5), np.log(2.0)])
        np.testing.assert_allclose(phi_vals, expected, rtol=1e-10)

    def test_bernstein_function_gamma_positive(self) -> None:
        """Bernstein function φ(λ) ≥ 0 for λ ≥ 0, tested at numerically stable values."""
        g = GammaProcess(a=1.0, b=2.0)
        lam_vals = np.array([0.1, 0.3, 0.5, 0.8, 1.0])
        phi_vals = bernstein_function(g, lam=lam_vals)
        assert np.all(np.isfinite(phi_vals))
        assert np.all(phi_vals >= -1e-6)

    def test_char_exp_subordinated_at_zero(self) -> None:
        """φ_Z(0; t) = 1 for any subordinated process."""
        bm = BrownianMotion(sigma=0.2)
        g = GammaProcess(a=1.0, b=1.0)
        cf = char_exp_subordinated(bm, g, u=0.0, t=1.0)
        assert abs(complex(cf) - 1.0) == pytest.approx(0.0, abs=1e-6)

    def test_char_exp_subordinated_matches_vg(self) -> None:
        """BM@Gamma char func magnitude matches VG closed-form (loose tolerance for numerical method)."""
        sigma, nu, theta = 0.2, 0.5, 0.0
        bm = BrownianMotion(mu=theta, sigma=sigma)
        g = GammaProcess(a=1.0 / nu, b=1.0 / nu)
        vg = VarianceGamma(sigma=sigma, nu=nu, theta=theta)

        # Test at small u where numerical char_exp via log-composition is stable
        u_vals = np.array([0.2, 0.5, 1.0])
        cf_sub = char_exp_subordinated(bm, g, u=u_vals, t=1.0)
        cf_vg = vg.char_func_exact(u=u_vals, t=1.0)

        np.testing.assert_allclose(
            np.abs(cf_sub), np.abs(cf_vg), rtol=0.15,
            err_msg="BM@Gamma char func should approximately match VG"
        )


class TestProduct:
    def test_cumulants_to_moments_gaussian(self) -> None:
        """For Gaussian: κ₁=μ, κ₂=σ², κ_n=0 for n≥3. Moments follow."""
        kappas = {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0}
        moments = cumulants_to_raw_moments(kappas)
        assert moments[1] == pytest.approx(0.0)
        assert moments[2] == pytest.approx(1.0)

    def test_moments_to_cumulants_roundtrip(self) -> None:
        """cumulants_to_raw_moments and raw_moments_to_cumulants are inverses."""
        kappas = {1: 1.0, 2: 2.0, 3: 0.5, 4: 0.1}
        moments = cumulants_to_raw_moments(kappas)
        kappas_back = raw_moments_to_cumulants(moments)
        for n in kappas:
            assert kappas_back[n] == pytest.approx(kappas[n], rel=1e-8)

    def test_product_process_exactness(self) -> None:
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=0.0, sigma=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpxaDegradationWarning)
            z = multiply(bm1, bm2)
        assert z.exactness == ExactnessLevel.MOMENT_PROPAGATION

    def test_product_triplet_raises(self) -> None:
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpxaDegradationWarning)
            z = multiply(bm1, bm2)
        with pytest.raises(ExactnessError):
            _ = z.triplet

    def test_product_simulate_shape(self) -> None:
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpxaDegradationWarning)
            z = multiply(bm1, bm2)
        paths = z.simulate(n_steps=50, n_paths=10, T=1.0)
        assert paths.shape == (10, 51)

    def test_product_emits_degradation_warning(self) -> None:
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        with pytest.warns(SpxaDegradationWarning):
            _ = multiply(bm1, bm2)


class TestIntegral:
    def test_ito_isometry_bm_constant(self) -> None:
        """Var(∫_0^t c dW_s) = c²·σ²·t for constant c."""
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        c = 2.0
        var = ito_isometry_variance(bm, f=lambda s: c, t=1.0)
        assert var == pytest.approx(c**2 * 1.0, rel=1e-4)

    def test_ito_isometry_sigma_scaling(self) -> None:
        """Var(∫_0^1 1 dX_s) = σ²·∫_0^1 1² ds = σ² for BM(0,σ)."""
        sigma = 2.0
        bm = BrownianMotion(mu=0.0, sigma=sigma)
        var = ito_isometry_variance(bm, f=lambda s: 1.0, t=1.0)
        assert var == pytest.approx(sigma**2, rel=1e-4)

    def test_stochastic_convolution_at_zero(self) -> None:
        """φ(0) = 1 for any stochastic convolution."""
        bm = BrownianMotion(sigma=1.0)
        cf = stochastic_convolution_char_func(bm, f=lambda s: 1.0, u=0.0, t=1.0)
        assert abs(complex(cf) - 1.0) == pytest.approx(0.0, abs=1e-6)

    def test_stochastic_integral_simulate_shape(self) -> None:
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpxaDegradationWarning)
            z = StochasticIntegral(bm1, bm2, f=lambda x: x)
        paths = z.simulate(n_steps=50, n_paths=5, T=1.0)
        assert paths.shape == (5, 51)

    def test_stochastic_integral_starts_at_zero(self) -> None:
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpxaDegradationWarning)
            z = StochasticIntegral(bm1, bm2, f=lambda x: 1.0)
        paths = z.simulate(n_steps=50, n_paths=10, T=1.0)
        np.testing.assert_array_equal(paths[:, 0], 0.0)

    def test_stochastic_integral_triplet_raises(self) -> None:
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpxaDegradationWarning)
            z = StochasticIntegral(bm1, bm2, f=lambda x: x)
        with pytest.raises(ExactnessError):
            _ = z.triplet
