"""The NTS triplet describes X, not its tempered-stable clock.

It used to carry the clock's density delta*exp(-lambda|x|)/|x|^(1+alpha),
mirrored onto both half-lines, with b = kappa_1. Its kappa_2 was then 0.80
instead of 0.14, which made NTS + Y wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate  # type: ignore[import-untyped]

from spxa.core.levy_measure import LevyMeasure
from spxa.zoo.levy import BrownianMotion, NormalTemperedStable

CASES = [
    NormalTemperedStable(alpha=0.6, lambd=1.5, delta=0.8, theta=0.1, sigma=0.3),
    NormalTemperedStable(alpha=0.3, lambd=0.5, delta=1.2, theta=-0.4, sigma=0.8),
]


def _moment(nu: LevyMeasure, n: int) -> float:
    def f(x: float) -> float:
        return x**n * float(nu.density(np.array([x]))[0])

    edges = [-np.inf, -1.0, 0.0, 1.0, np.inf]
    return sum(integrate.quad(f, a, b, limit=400)[0] for a, b in zip(edges[:-1], edges[1:]))


@pytest.mark.parametrize("X", CASES, ids=repr)
def test_density_moments_are_the_cumulants(X: NormalTemperedStable) -> None:
    nu = X.triplet.nu
    for n in (2, 3, 4):
        assert _moment(nu, n) == pytest.approx(X.cumulant(n), rel=1e-8)


@pytest.mark.parametrize("X", CASES, ids=repr)
def test_triplet_cumulants_match(X: NormalTemperedStable) -> None:
    tr = X.triplet
    assert tr.cumulant(1) == pytest.approx(X.cumulant(1), rel=1e-9)
    assert tr.cumulant(2) == pytest.approx(X.cumulant(2), rel=1e-12)


@pytest.mark.parametrize("X", CASES, ids=repr)
def test_levy_khintchine_quadrature_reproduces_the_cf(X: NormalTemperedStable) -> None:
    tr = X.triplet
    u = np.array([0.5, 1.0, 3.0])
    quad = np.array([1j * tr.b * v + LevyMeasure.jump_exponent(tr.nu, complex(v)) for v in u])
    np.testing.assert_allclose(quad, X._char_exponent(u), rtol=1e-5, atol=1e-6)


def test_sum_with_nts_has_the_right_variance() -> None:
    X = CASES[0]
    Z = X + BrownianMotion(0.0, 0.2)
    assert Z.cumulants(2)[2] == pytest.approx(X.cumulant(2) + 0.04, rel=1e-10)
