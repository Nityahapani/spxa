"""Additional regression tests for the N-state MMPP."""

import numpy as np
import pytest

from spxa.zoo.beyond._mmpp_theory import stationary_distribution
from spxa.zoo.beyond.mmpp import MarkovModulatedPoissonProcess


def test_zero_horizon_has_zero_moments():
    p = MarkovModulatedPoissonProcess([[-2, 2], [1, -1]], [3, 8])
    assert p.count_moments(0) == pytest.approx((0, 0))


def test_one_state_poisson_moments():
    p = MarkovModulatedPoissonProcess([[0]], [4])
    assert p.count_moments(2.5) == pytest.approx((10, 10), abs=1e-8)


def test_stationary_distribution_two_state():
    q = np.array([[-2.0, 2.0], [3.0, -3.0]])
    pi = stationary_distribution(q)

    assert pi == pytest.approx([0.6, 0.4], abs=1e-12)
    assert pi @ q == pytest.approx([0, 0], abs=1e-12)


def test_nonunique_stationary_distribution_rejected():
    with pytest.raises(ValueError, match="not unique"):
        stationary_distribution([[0, 0], [0, 0]])


def test_negative_arrival_rate_rejected():
    with pytest.raises(ValueError):
        MarkovModulatedPoissonProcess([[-1, 1], [1, -1]], [2, -3])


def test_invalid_initial_probabilities_rejected():
    with pytest.raises(ValueError):
        MarkovModulatedPoissonProcess(
            [[-1, 1], [1, -1]],
            [2, 3],
            initial_probabilities=[0.3, 0.3],
        )


def test_seeded_event_simulation_reproducible_and_sorted():
    p = MarkovModulatedPoissonProcess([[-2, 2], [1, -1]], [4, 10])

    a = p.simulate_events(3, rng=np.random.default_rng(17))
    b = p.simulate_events(3, rng=np.random.default_rng(17))

    assert np.array_equal(a.arrival_times, b.arrival_times)
    assert np.array_equal(a.transition_times, b.transition_times)
    assert np.array_equal(a.states, b.states)
    assert np.all(np.diff(a.arrival_times) >= 0)
    assert np.all(a.arrival_times <= 3)


def test_simulated_paths_zero_horizon():
    p = MarkovModulatedPoissonProcess([[-1, 1], [1, -1]], [5, 9])

    paths = p.simulate(
        12,
        n_paths=4,
        T=0,
        rng=np.random.default_rng(1),
    )

    assert paths.shape == (4, 13)
    assert np.all(paths == 0)
