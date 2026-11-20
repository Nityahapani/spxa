"""Unit tests for core algebraic engine."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.levy import BrownianMotion, GammaProcess


class TestExactnessLevel:
    def test_ordering(self) -> None:
        assert ExactnessLevel.EXACT < ExactnessLevel.MOMENT_PROPAGATION
        assert ExactnessLevel.MOMENT_PROPAGATION < ExactnessLevel.SIMULATION_ONLY
        assert not (ExactnessLevel.EXACT > ExactnessLevel.MOMENT_PROPAGATION)

    def test_combine_returns_minimum(self) -> None:
        assert (
            ExactnessLevel.combine(ExactnessLevel.EXACT, ExactnessLevel.MOMENT_PROPAGATION)
            == ExactnessLevel.MOMENT_PROPAGATION
        )

        assert (
            ExactnessLevel.combine(ExactnessLevel.EXACT, ExactnessLevel.EXACT)
            == ExactnessLevel.EXACT
        )

        assert (
            ExactnessLevel.combine(
                ExactnessLevel.SIMULATION_ONLY, ExactnessLevel.MOMENT_PROPAGATION
            )
            == ExactnessLevel.SIMULATION_ONLY
        )


class TestLevyTripletArithmetic:
    def test_addition_drifts_add(self) -> None:
        bm1 = BrownianMotion(mu=1.0, sigma=1.0)
        bm2 = BrownianMotion(mu=2.0, sigma=1.0)
        z = bm1 + bm2
        assert z.triplet.b == pytest.approx(3.0)

    def test_addition_sigma_sq_adds(self) -> None:
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=0.0, sigma=2.0)
        z = bm1 + bm2
        assert z.triplet.sigma_sq == pytest.approx(5.0)

    def test_scaling_sigma_sq(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        z = 3.0 * bm
        assert z.triplet.sigma_sq == pytest.approx(9.0)

    def test_scaling_drift(self) -> None:
        bm = BrownianMotion(mu=2.0, sigma=0.01)
        z = 4.0 * bm
        assert z.triplet.b == pytest.approx(8.0)

    def test_negation_reverses_drift(self) -> None:
        bm = BrownianMotion(mu=1.5, sigma=1.0)
        z = -bm
        assert z.triplet.b == pytest.approx(-1.5)

    def test_subtraction(self) -> None:
        bm1 = BrownianMotion(mu=3.0, sigma=2.0)
        bm2 = BrownianMotion(mu=1.0, sigma=1.0)
        z = bm1 - bm2
        assert z.triplet.b == pytest.approx(2.0)
        assert z.triplet.sigma_sq == pytest.approx(5.0)

    def test_radd_zero(self) -> None:
        bm = BrownianMotion(mu=1.0, sigma=1.0)
        result = sum([bm, bm])
        assert result.triplet.b == pytest.approx(2.0)

    def test_rmul(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        z = 2.0 * bm
        assert z.triplet.sigma_sq == pytest.approx(4.0)


class TestProcessComposition:
    def test_composed_exactness_remains_exact(self) -> None:
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=0.0, sigma=1.0)
        z = bm1 + bm2
        assert z.exactness == ExactnessLevel.EXACT

    def test_subordination_requires_subordinator(self) -> None:
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=0.0, sigma=1.0)
        with pytest.raises(ValueError, match="subordinator"):
            _ = bm1 @ bm2

    def test_subordination_with_gamma(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        g = GammaProcess(a=1.0, b=1.0)
        z = bm @ g
        assert z.exactness == ExactnessLevel.EXACT
        assert z.properties.has_stationary_increments is True
        assert z.properties.has_independent_increments is True

    def test_triplet_unavailable_after_subordination(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        g = GammaProcess(a=1.0, b=1.0)
        z = bm @ g
        with pytest.raises(ExactnessError, match="triplet"):
            _ = z.triplet


class TestPropertyPropagation:
    def test_bm_martingale_plus_bm_martingale_is_martingale(self) -> None:
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=0.0, sigma=2.0)
        z = bm1 + bm2
        assert z.properties.is_martingale is True

    def test_bm_martingale_plus_drifted_bm_not_martingale(self) -> None:
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=1.0, sigma=1.0)
        z = bm1 + bm2
        assert z.properties.is_martingale is False

    def test_tail_index_takes_minimum(self) -> None:
        from spxa.core.properties import ProcessProperties

        p1 = ProcessProperties(tail_index=1.5)
        p2 = ProcessProperties(tail_index=2.0)
        combined = ProcessProperties.combine_addition(p1, p2)
        assert combined.tail_index == pytest.approx(1.5)

    def test_subordinator_plus_subordinator_is_subordinator(self) -> None:
        g1 = GammaProcess(a=1.0, b=1.0)
        g2 = GammaProcess(a=2.0, b=1.0)
        z = g1 + g2
        assert z.properties.is_subordinator is True

    def test_negative_scale_loses_subordinator(self) -> None:
        g = GammaProcess(a=1.0, b=1.0)
        z = -1.0 * g
        assert z.properties.is_subordinator is False


class TestStory:
    def test_story_contains_primitive_name(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        story = bm.__story__()
        assert "BrownianMotion" in story

    def test_story_contains_exactness(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        story = bm.__story__()
        assert "EXACT" in story

    def test_story_records_addition(self) -> None:
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=0.0, sigma=1.0)
        z = bm1 + bm2
        story = z.__story__()
        assert "Addition" in story

    def test_story_records_scaling(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        z = 2.5 * bm
        story = z.__story__()
        assert "Scaling" in story

    def test_story_records_subordination(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        g = GammaProcess(a=1.0, b=1.0)
        z = bm @ g
        story = z.__story__()
        assert "Subordination" in story


class TestSimulation:
    def test_simulate_shape(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        paths = bm.simulate(n_steps=50, n_paths=10, T=1.0)
        assert paths.shape == (10, 51)

    def test_simulate_starts_at_zero(self) -> None:
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        paths = bm.simulate(n_steps=50, n_paths=10, T=1.0)
        assert np.all(paths[:, 0] == 0.0)

    def test_composed_simulate_shape(self) -> None:
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=1.0, sigma=0.5)
        z = bm1 + bm2
        paths = z.simulate(n_steps=30, n_paths=5, T=2.0)
        assert paths.shape == (5, 31)

    def test_scaled_simulate_is_scaled(self) -> None:
        rng = np.random.default_rng(99)
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        rng2 = np.random.default_rng(99)
        z = 3.0 * bm
        paths_bm = bm.simulate(n_steps=20, n_paths=5, T=1.0, rng=rng)
        paths_z = z.simulate(n_steps=20, n_paths=5, T=1.0, rng=rng2)
        np.testing.assert_allclose(paths_z, 3.0 * paths_bm)


class TestJumpExponentComplexFrequencies:
    """The default quadrature works off the real axis (Laplace exponents)."""

    def test_gamma_laplace_exponent(self):
        # K(z) = -a log(1 - z/b) for Re z < b, via ψ(-iz)
        a, b = 1.5, 3.0
        nu = GammaProcess(a=a, b=b).triplet.nu
        for z in (-5.0, -0.5, 1.0, -2.0 + 1.0j):
            trip = GammaProcess(a=a, b=b).triplet
            got = 1j * trip.b * (-1j * z) + nu.jump_exponent(-1j * z)
            assert np.isfinite(got)
            assert got == pytest.approx(-a * np.log(1 - z / b), rel=1e-7)

    def test_no_nan_from_the_empty_half_line(self):
        # e^{iux} overflows for x → -∞ when Im u < 0, where the density is 0
        nu = GammaProcess(a=2.0, b=1.0).triplet.nu
        assert np.isfinite(nu.jump_exponent(5j))
        assert np.isfinite(nu.jump_exponent(-0.5j))


class TestSubordinatedProcesses:
    """X @ T is a Lévy process: exact cf and cumulants, no triplet."""

    def _vg_pair(self):  # type: ignore[no-untyped-def]
        from spxa.zoo.levy import VarianceGamma

        sigma, nu, theta = 0.2, 0.3, -0.1
        Z = BrownianMotion(mu=theta, sigma=sigma) @ GammaProcess(a=1 / nu, b=1 / nu)
        return Z, VarianceGamma(sigma, nu, theta)

    def test_brownian_motion_in_gamma_time_is_variance_gamma(self):
        Z, V = self._vg_pair()
        u = np.array([-3.0, 0.5, 2.0, 10.0])
        np.testing.assert_allclose(Z.char_func(u, 1.3), V.char_func(u, 1.3), rtol=1e-8)
        kz, kv = Z.cumulants(4), V.cumulants(4)
        for n in range(1, 5):
            assert kz[n] == pytest.approx(kv[n], rel=1e-12)

    def test_cumulant_composition_formulas(self):
        from spxa.zoo.levy import InverseGaussianProcess

        X, T = BrownianMotion(mu=0.3, sigma=0.5), InverseGaussianProcess(mu=1.5, lam=2.0)
        kx, kt, kz = X.cumulants(3), T.cumulants(3), (X @ T).cumulants(3)
        assert kz[1] == pytest.approx(kt[1] * kx[1], rel=1e-12)
        assert kz[2] == pytest.approx(kt[1] * kx[2] + kt[2] * kx[1] ** 2, rel=1e-12)
        k3 = kt[1] * kx[3] + 3 * kt[2] * kx[1] * kx[2] + kt[3] * kx[1] ** 3
        assert kz[3] == pytest.approx(k3, rel=1e-12)

    def test_large_drift_has_no_branch_problem(self):
        # Im ψ_X(u) = μu = 20 >> π: a log of φ_X would pick the wrong branch
        Z = BrownianMotion(mu=2.0, sigma=0.1) @ GammaProcess(a=2.0, b=2.0)
        u = 10.0
        psi_x = 1j * 2.0 * u - 0.5 * 0.01 * u**2
        expected = (1 - psi_x / 2.0) ** (-2.0)
        assert Z.char_func(u) == pytest.approx(expected, rel=1e-6)

    def test_composites_of_subordinated_processes(self):
        Z, V = self._vg_pair()
        B = BrownianMotion(mu=0.05, sigma=0.1)
        u = np.array([0.7, -2.0])
        np.testing.assert_allclose((2.5 * Z).char_func(u), V.char_func(2.5 * u), rtol=1e-8)
        np.testing.assert_allclose((Z + B).char_func(u), V.char_func(u) * B.char_func(u), rtol=1e-8)
        assert (-Z).cumulants(3)[3] == pytest.approx(-V.cumulants(3)[3], rel=1e-12)
        assert (Z + B).cumulants(2)[2] == pytest.approx(V.cumulants(2)[2] + 0.01, rel=1e-12)
        # nested time changes: Z in Gamma time
        W = Z @ GammaProcess(a=4.0, b=4.0)
        assert W.cumulants(1)[1] == pytest.approx(Z.cumulants(1)[1], rel=1e-12)

    def test_triplet_error_is_not_self_contradictory(self):
        from spxa.core.exceptions import TripletUnavailableError

        Z, _ = self._vg_pair()
        with pytest.raises(TripletUnavailableError, match="although it is exact") as e:
            _ = Z.triplet
        assert (
            "requires ExactnessLevel.EXACT, but this process has ExactnessLevel.EXACT"
            not in str(e.value)
        )


class TestSubordinatorsCompose:
    def test_subordinator_in_subordinator_time_is_a_subordinator(self):
        from spxa.zoo.levy import InverseGaussianProcess

        T, S = GammaProcess(a=1.0, b=1.0), InverseGaussianProcess(mu=1.0, lam=2.0)
        assert (T @ S).properties.is_subordinator is True
        assert (BrownianMotion(mu=0.0, sigma=1.0) @ T).properties.is_subordinator is False

    def test_subordination_associates(self):
        X = BrownianMotion(mu=0.1, sigma=0.4)
        T, S = GammaProcess(a=2.0, b=2.0), GammaProcess(a=3.0, b=1.5)
        u = np.array([0.5, -1.7, 3.0])
        np.testing.assert_allclose(
            ((X @ T) @ S).char_func(u), (X @ (T @ S)).char_func(u), rtol=1e-8
        )
