"""Poisson triplet with truncation x·1{|x| ≤ 1}: (λc·1{|c| ≤ 1}, 0, λδ_c)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spxa.zoo.levy import BrownianMotion, PoissonProcess


@pytest.mark.exactness
@pytest.mark.parametrize("c", [0.3, 1.0, 2.0, -1.5])
def test_poisson_triplet(c: float) -> None:
    p = PoissonProcess(rate=1.5, jump_size=c)
    trip = p.triplet
    assert trip.b == pytest.approx(1.5 * c if abs(c) <= 1 else 0.0)
    for n in range(1, 5):
        assert trip.cumulant(n) == pytest.approx(1.5 * c**n, rel=1e-12)
    for u in (0.4, -3.0):
        assert abs(trip.char_exp(u, 1.0) - np.exp(1.5 * np.expm1(1j * u * c))) < 1e-12


@pytest.mark.exactness
def test_poisson_in_compositions() -> None:
    p = PoissonProcess(rate=1.5, jump_size=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = p + BrownianMotion(0.1, 0.5)
        assert s.cumulants(2) == pytest.approx({1: 3.1, 2: 6.25}, rel=1e-12)
        assert (0.25 * p).cumulants(2)[1] == pytest.approx(0.75, rel=1e-12)
        assert (-p).cumulants(2)[1] == pytest.approx(-3.0, rel=1e-12)
