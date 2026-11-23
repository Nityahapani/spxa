"""The GH Lévy measure is the Lévy measure (not the GH probability density).

Up to 1.2.0 ``_gh_levy_measure`` returned the GH marginal density at t = 1,
so the triplet's exponent (used by ``GH + Drift`` and ``c * GH``) was off by
O(1). The references here are the NIG Lévy density at lambda = -1/2 and the
closed-form exponent, recovered from the measure by quadrature.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special as sp

from spxa.zoo.levy import Drift, GeneralizedHyperbolic
from spxa.zoo.levy.gh import _gh_levy_bracket, _gh_levy_measure

A, B, D = 3.0, -0.5, 0.8


def test_general_formula_at_lambda_minus_half_is_the_nig_levy_density() -> None:
    xs = np.array([-4.0, -0.7, -1e-3, 1e-5, 0.2, 1.5, 9.0])
    general = np.array(
        [math.exp(B * x - A * abs(x)) / abs(x) * _gh_levy_bracket(abs(x), -0.5, A, D) for x in xs]
    )
    nig = A * D / np.pi * np.exp(B * xs) * sp.kv(1.0, A * np.abs(xs)) / np.abs(xs)
    np.testing.assert_allclose(general, nig, rtol=1e-10)
    np.testing.assert_allclose(_gh_levy_measure(-0.5, A, B, D).density(xs), nig, rtol=1e-13)


def _exponent_from_measure(lam: float, u: np.ndarray) -> np.ndarray:
    xg, wg = np.polynomial.legendre.leggauss(80)
    edges = [-25, -8, -3, -1, -0.3, -0.05, -0.005, 0, 0.005, 0.05, 0.3, 1, 3, 8, 25]
    x = np.concatenate([(h - lo) / 2 * xg + (h + lo) / 2 for lo, h in zip(edges[:-1], edges[1:])])
    w = np.concatenate([(h - lo) / 2 * wg for lo, h in zip(edges[:-1], edges[1:])])
    nu = _gh_levy_measure(lam, A, B, D).density(x)
    g = math.sqrt(A * A - B * B)
    mean = B * D / g * sp.kv(lam + 1, D * g) / sp.kv(lam, D * g)
    kern = np.exp(1j * np.outer(u, x)) - 1 - 1j * np.outer(u, x)
    return 1j * mean * u + kern @ (w * nu)


@pytest.mark.parametrize("lam", [0.0, 1.2])
def test_measure_reproduces_the_closed_form_exponent(lam: float) -> None:
    u = np.array([0.5, 2.0, 5.0])
    closed = np.asarray(GeneralizedHyperbolic(lam, A, B, D, 0.0).char_exponent(u))
    np.testing.assert_allclose(_exponent_from_measure(lam, u), closed, rtol=1e-8)


def test_compositions_use_the_closed_form() -> None:
    gh = GeneralizedHyperbolic(1.2, A, B, D, 0.03)
    u = np.linspace(-6.0, 6.0, 13)
    psi = np.asarray(gh.char_exponent(u))
    np.testing.assert_allclose(gh._char_exponent(u.astype(complex)), psi, rtol=1e-15)
    np.testing.assert_allclose(
        np.log((gh + Drift(0.1)).char_func(u, 1.0)), psi + 0.1j * u, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        (0.5 * gh)._char_exponent(u.astype(complex)), gh.char_exponent(0.5 * u), rtol=1e-15
    )
