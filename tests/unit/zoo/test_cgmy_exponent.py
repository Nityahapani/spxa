"""CGMY: the Lévy measure's jump exponent agrees with the closed form."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.core.levy_measure import LevyMeasure
from spxa.zoo.levy import CGMY, BrownianMotion

U = np.array([-5.0, -0.73, 0.0, 0.4, 2.3, 8.0])


@pytest.mark.parametrize("Y", [0.3, 0.9, 1.2, 1.5, 1.8])
def test_triplet_reproduces_the_closed_form(Y: float) -> None:
    X = CGMY(C=0.8, G=1.5, M=3.0, Y=Y)
    np.testing.assert_allclose(X.triplet.char_exp(U, 0.7), X.char_func(U, 0.7), rtol=1e-13)


@pytest.mark.parametrize("Y", [1.5, 1.8])
def test_closed_form_is_what_quadrature_approximates(Y: float) -> None:
    # quadrature from |x| = 1e-8 misses ~ (1e-8)^(2-Y) of a singular integral,
    # so it only agrees loosely for Y near 2
    nu = CGMY(C=1.0, G=1.0, M=2.0, Y=Y).triplet.nu
    exact = nu.jump_exponent(2.0)
    approx = LevyMeasure.jump_exponent(nu, 2.0)
    assert abs(approx - exact) < 1e-3 * abs(exact)


def test_composites_of_cgmy_are_exact() -> None:
    X = CGMY(C=1.0, G=1.0, M=2.0, Y=1.2144526178975437)
    u = np.array([-0.7277422846205255, 2.3])
    np.testing.assert_allclose((2.0 * X).char_func(u), X.char_func(2.0 * u), rtol=1e-12)
    np.testing.assert_allclose((-X).char_func(u), np.conj(X.char_func(u)), rtol=1e-12)
    B = BrownianMotion(mu=0.1, sigma=0.3)
    np.testing.assert_allclose((X + B).char_func(u), X.char_func(u) * B.char_func(u), rtol=1e-12)


def test_laplace_exponent_inside_the_strip() -> None:
    # K(z) = Φ(-iz) is real on -G < z < M
    X = CGMY(C=0.5, G=2.0, M=4.0, Y=1.4)
    tr = X.triplet
    for z in (-1.5, 0.5, 3.5):
        k = 1j * tr.b * (-1j * z) + tr.nu.jump_exponent(-1j * z)
        assert abs(k.imag) < 1e-12
        assert np.isfinite(k.real)
