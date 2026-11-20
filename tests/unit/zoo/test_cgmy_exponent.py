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


@pytest.mark.parametrize("k", [0.0, 1.0])
def test_poles_of_gamma_use_the_limits(k: float) -> None:
    X = CGMY(C=1.0, G=5.0, M=8.0, Y=k)
    phi = X.char_func(U)
    assert np.all(np.isfinite(phi))
    cum = X.cumulants(4)
    assert sorted(cum) == [1, 2, 3, 4] and all(np.isfinite(v) for v in cum.values())
    np.testing.assert_allclose(X.triplet.char_exp(U), phi, rtol=1e-12, atol=1e-15)
    assert X.triplet.cumulant(1) == pytest.approx(cum[1], rel=1e-10)
    # continuity: the closed form a small distance away differs by O(distance)
    for d in (1e-3, 1e-5):
        near = CGMY(C=1.0, G=5.0, M=8.0, Y=k + d)
        assert np.max(np.abs(near.char_func(U) / phi - 1)) < 50 * d
        assert abs(near.cumulants(1)[1] - cum[1]) < 5 * d


def test_y_one_first_cumulant_is_c_log_g_over_m() -> None:
    assert CGMY(C=1.5, G=5.0, M=8.0, Y=1.0).cumulants(1)[1] == pytest.approx(1.5 * np.log(5 / 8))


def test_y_zero_is_variance_gamma() -> None:
    from spxa.zoo.levy import VarianceGamma

    sigma, nu, theta = 0.2, 0.3, -0.1
    root = np.sqrt(theta**2 * nu**2 / 4 + sigma**2 * nu / 2)
    G, M = 1 / (root - theta * nu / 2), 1 / (root + theta * nu / 2)
    np.testing.assert_allclose(
        CGMY(C=1 / nu, G=G, M=M, Y=0.0).char_func(U),
        VarianceGamma(sigma, nu, theta).char_func(U),
        rtol=1e-13,
    )
