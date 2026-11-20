"""Exact tempered stable draws at every horizon.

The FFT sampler could not resolve the narrow bulk of a tempered stable law at
short horizons (at t = 1e-4 the variance was off by a factor of 7000). The
tilted-stable rejection sampler is exact. These tests pin it from 1e-6 to 100,
across the switch to cf inversion at long horizons, and for its users: the NTS
clock and CGMY with 0 < Y < 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxa.laws.stats import ecf_test, mean_test, variance_test
from spxa.zoo.levy import TemperedStable
from spxa.zoo.levy.tempered_stable import sample_positive_stable

TIMES = [1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0]
LEVEL = 1e-5


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_positive_stable_laplace_transform(alpha: float) -> None:
    rng = np.random.default_rng(0)
    s = sample_positive_stable(alpha, np.full(100_000, 0.7), rng)
    for lam in (0.5, 1.0, 3.0):
        e = np.exp(-lam * s)
        target = np.exp(-0.7 * lam**alpha)
        assert abs(e.mean() - target) < 5 * e.std() / np.sqrt(e.size)


def _check(X, t: float, n: int = 40_000, seed: int = 0) -> None:
    x = X.sample_at(np.full(n, t), np.random.default_rng(seed))
    k = X.cumulants(4)
    s = np.sqrt(k[2] * t)
    assert mean_test(x, k[1] * t, k[2] * t) > LEVEL
    assert variance_test(x, k[2] * t, k[4] * t) > LEVEL
    assert ecf_test(x, lambda u: X.char_func(u, t), [0.5 / s, 2.0 / s]) > LEVEL


@pytest.mark.parametrize("t", TIMES)
@pytest.mark.parametrize(
    "X", [TemperedStable(0.6, 1.0, 1.0), TemperedStable(0.3, 2.0, 5.0)], ids=repr
)
def test_tempered_stable_is_exact_at_every_horizon(X, t: float) -> None:
    _check(X, t)


@pytest.mark.parametrize("t", [1e-4, 0.5])
def test_simulate_uses_the_exact_sampler(t: float) -> None:
    X = TemperedStable(0.6, 1.0, 1.0)
    x = X.simulate(4, 20_000, 4 * t, np.random.default_rng(1))
    d = np.diff(x, axis=1).ravel()
    k = X.cumulants(4)
    assert mean_test(d, k[1] * t, k[2] * t) > LEVEL
    assert variance_test(d, k[2] * t, k[4] * t) > LEVEL


def test_draws_are_non_negative_and_zero_at_zero() -> None:
    x = TemperedStable(0.5, 1.0, 2.0).sample_at(np.array([0.0, 1e-8, 1.0, 200.0]))
    assert x[0] == 0.0 and np.all(x >= 0.0)
