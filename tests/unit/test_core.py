"""Unit tests for core algebraic engine."""

import warnings

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError, SpxaDegradationWarning
from spxa.zoo.levy import BrownianMotion, GammaProcess, VarianceGamma


class TestExactnessLevel:
    def test_ordering(self) -> None:
        assert ExactnessLevel.EXACT < ExactnessLevel.MOMENT_PROPAGATION
        assert ExactnessLevel.MOMENT_PROPAGATION < ExactnessLevel.SIMULATION_ONLY
        assert not (ExactnessLevel.EXACT > ExactnessLevel.MOMENT_PROPAGATION)

    def test_combine_returns_minimum(self) -> None:
        assert ExactnessLevel.combine(
            ExactnessLevel.EXACT, ExactnessLevel.MOMENT_PROPAGATION
        ) == ExactnessLevel.MOMENT_PROPAGATION

        assert ExactnessLevel.combine(
            ExactnessLevel.EXACT, ExactnessLevel.EXACT
        ) == ExactnessLevel.EXACT

        assert ExactnessLevel.combine(
            ExactnessLevel.SIMULATION_ONLY, ExactnessLevel.MOMENT_PROPAGATION
        ) == ExactnessLevel.SIMULATION_ONLY


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
