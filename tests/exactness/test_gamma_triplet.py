"""Gamma subordinator triplet: drift ∫_0^1 x ν(dx) = a(1 − e^{−b})/b."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spxa.zoo.levy import BrownianMotion, GammaProcess


@pytest.mark.exactness
@pytest.mark.parametrize(("a", "b"), [(2.0, 4.0), (0.5, 0.3), (10.0, 25.0)])
def test_triplet_reproduces_the_gamma_cumulants(a: float, b: float) -> None:
    g = GammaProcess(a=a, b=b)
    trip = g.triplet
    assert trip.b == pytest.approx(a * (1 - np.exp(-b)) / b, rel=1e-14)
    for n in range(1, 5):
        assert trip.cumulant(n) == pytest.approx(g.cumulant_exact(n), rel=1e-8)
    for u in (0.4, -3.0):
        assert abs(trip.char_exp(u, 1.0) - complex(g.char_func(u))) < 1e-8


@pytest.mark.exactness
def test_compositions_with_a_gamma_component_have_the_right_mean() -> None:
    g = GammaProcess(a=2.0, b=4.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert (g + BrownianMotion(0.1, 0.2)).cumulants(1)[1] == pytest.approx(0.6, rel=1e-8)
        assert (-g).cumulants(2)[1] == pytest.approx(-0.5, rel=1e-8)
        assert (0.5 * g).cumulants(2)[1] == pytest.approx(0.25, rel=1e-8)
