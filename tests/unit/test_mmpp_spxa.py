"""SPXA-facing contract tests for finite-state MMPP."""

import numpy as np
import pytest

from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.mmpp import MarkovModulatedPoissonProcess


def test_grid_shape_and_monotonicity():
    p = MarkovModulatedPoissonProcess([[-2, 2], [1, -1]], [3, 9])
    a = p.simulate(100, n_paths=20, T=3, rng=np.random.default_rng(19))
    assert a.shape == (20, 101)
    assert np.all(a[:, 0] == 0)
    assert np.all(np.diff(a, axis=1) >= 0)


def test_absorbing_zero_rate():
    p = MarkovModulatedPoissonProcess([[0]], [0])
    assert not p.simulate(10, n_paths=3, T=10).any()


def test_constant_rate_poisson_moments():
    p = MarkovModulatedPoissonProcess([[-3, 3], [2, -2]], [5, 5])
    mean, var = p.count_moments(2.0)
    assert mean == pytest.approx(10, abs=1e-8)
    assert var == pytest.approx(10, abs=1e-8)


def test_nonstationary_initial_distribution():
    p = MarkovModulatedPoissonProcess([[-1, 1], [0, 0]], [0, 7], initial_state=0)
    mean, var = p.count_moments(2.0)
    assert mean > 0 and var >= mean


def test_triplet_rejected():
    p = MarkovModulatedPoissonProcess([[-1, 1], [1, -1]], [2, 10])
    with pytest.raises(ExactnessError):
        p._triplet()


def test_seeded_reproducibility():
    p = MarkovModulatedPoissonProcess([[-1, 1], [2, -2]], [4, 8])
    a = p.simulate(40, 5, 3, rng=np.random.default_rng(5))
    b = p.simulate(40, 5, 3, rng=np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_invalid_generator_rejected():
    with pytest.raises(ValueError):
        MarkovModulatedPoissonProcess([[1, -1], [0, 0]], [1, 2])


def test_higher_order_cumulants_match_high_precision_reference():
    # Reference: 60-digit mpmath derivatives of log E[exp(theta N_1)] at theta = 0.
    p = MarkovModulatedPoissonProcess(
        [[-1, 1], [2, -2]], [4, 10], initial_probabilities=[2 / 3, 1 / 3]
    )
    ref = {
        1: 6.0,
        2: 9.64406589932065,
        3: 21.3728566911684,
        4: 47.4672936132737,
        5: -20.005738768914,
        6: -1449.73336947646,
    }
    got = p.cumulants(order=6)
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, rel=1e-9)


def test_constant_rate_all_cumulants_equal_poisson():
    p = MarkovModulatedPoissonProcess([[-3, 3], [2, -2]], [5, 5])
    got = p.cumulants(order=8)
    assert all(got[k] == pytest.approx(5.0, rel=1e-8) for k in range(1, 9))


def test_ill_conditioned_cumulants_raise():
    p = MarkovModulatedPoissonProcess([[-1, 1], [2, -2]], [400, 1000])
    with pytest.raises(ValueError, match="ill-conditioned"):
        p.cumulants(order=6)
    assert set(p.cumulants(order=2)) == {1, 2}


def test_cumulants_invalid_order():
    p = MarkovModulatedPoissonProcess([[-1, 1], [2, -2]], [4, 10])
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            p.cumulants(order=bad)


def test_stationary_distribution_and_mean_intensity():
    p = MarkovModulatedPoissonProcess([[-1, 1], [2, -2]], [4, 10])
    assert p.stationary_distribution() == pytest.approx([2 / 3, 1 / 3])
    assert p.mean_intensity == pytest.approx(6.0)
    stationary = MarkovModulatedPoissonProcess(
        [[-1, 1], [2, -2]], [4, 10], initial_probabilities=[2 / 3, 1 / 3]
    )
    mean, _ = stationary.count_moments(5.0)
    assert mean == pytest.approx(5.0 * p.mean_intensity)


def test_reducible_chain_has_no_unique_stationary_law():
    p = MarkovModulatedPoissonProcess([[0, 0], [0, 0]], [1, 2])
    with pytest.raises(ValueError):
        p.stationary_distribution()
