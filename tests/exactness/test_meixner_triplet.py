"""Meixner triplet: overflow-free Lévy density and a singularity-free drift."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.zoo.levy import MeixnerProcess

PARAMS = [(0.4, -0.7, 1.3, 0.05), (2.0, 1.2, 0.5, 0.0), (0.05, 0.3, 4.0, -0.1)]


@pytest.mark.exactness
@pytest.mark.parametrize(("a", "b", "d", "m"), PARAMS)
def test_triplet_reproduces_the_meixner_cumulants(a: float, b: float, d: float, m: float) -> None:
    mx = MeixnerProcess(alpha=a, beta=b, delta=d, m=m)
    trip = mx.triplet
    for n in range(1, 7):
        assert trip.cumulant(n) == pytest.approx(mx.cumulant_exact(n), rel=1e-7, abs=1e-12)
    for u in (0.4, -3.0):
        assert abs(trip.char_exp(u, 1.0) - complex(mx.char_func(u))) < 1e-7


@pytest.mark.exactness
def test_density_has_no_overflow_and_matches_the_naive_formula() -> None:
    a, b, d = 0.05, 0.3, 4.0
    nu = MeixnerProcess(alpha=a, beta=b, delta=d).triplet.nu
    x = np.array([-30.0, -1.0, -1e-3, 1e-3, 0.02, 1.0, 30.0])
    k = nu.density(x)
    assert np.all(np.isfinite(k)) and np.all(k >= 0)
    moderate = np.abs(x) < 1.0
    naive = d * np.exp(b * x[moderate] / a) / (x[moderate] * np.sinh(np.pi * x[moderate] / a))
    np.testing.assert_allclose(k[moderate], naive, rtol=1e-12)
    assert nu.density(np.array([0.0]))[0] == 0.0
