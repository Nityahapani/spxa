"""Inverse Gaussian process: Lévy measure c x^{-3/2} e^{-ηx} with η = λ/(2μ²)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from spxa.zoo.levy import InverseGaussianProcess

PARAMS = [(0.8, 1.7), (1.5, 2.0), (0.2, 5.0)]


@pytest.mark.exactness
@pytest.mark.parametrize(("mu", "lam"), PARAMS)
def test_triplet_reproduces_the_ig_cumulants(mu: float, lam: float) -> None:
    ig = InverseGaussianProcess(mu=mu, lam=lam)
    trip = ig.triplet
    for n in range(1, 7):
        assert trip.cumulant(n) == pytest.approx(ig.cumulant_exact(n), rel=1e-8)
    for u in (0.4, -3.0, 10.0):
        assert abs(trip.char_exp(u, 1.0) - complex(ig.char_func(u))) < 1e-7


@pytest.mark.exactness
@pytest.mark.parametrize(("mu", "lam"), PARAMS)
def test_measure_laplace_exponent_tail_and_moments(mu: float, lam: float) -> None:
    nu = InverseGaussianProcess(mu=mu, lam=lam).triplet.nu
    # ∫ (1 − e^{−sx}) k(x) dx is the Bernstein function (λ/μ)(√(1 + 2μ²s/λ) − 1)
    for s_val in (0.5, 3.0):
        val, _ = integrate.quad(
            lambda x: -np.expm1(-s_val * x) * float(nu.density(np.array([x]))[0]),
            0,
            np.inf,
            limit=400,
        )
        assert val == pytest.approx(
            (lam / mu) * (np.sqrt(1 + 2 * mu**2 * s_val / lam) - 1), rel=1e-7
        )
    for x0 in (0.05, 1.0, 4.0):
        val, _ = integrate.quad(
            lambda x: float(nu.density(np.array([x]))[0]), x0, np.inf, limit=400
        )
        assert nu.tail(x0) == pytest.approx(val, rel=1e-8)
    for n in (2, 3, 7):
        val, _ = integrate.quad(
            lambda x: x**n * float(nu.density(np.array([x]))[0]), 0, np.inf, limit=400
        )
        assert nu.moment(n) == pytest.approx(val, rel=1e-7)


@pytest.mark.exactness
def test_drift_is_mu_erf() -> None:
    from scipy.special import erf

    ig = InverseGaussianProcess(mu=0.8, lam=1.7)
    assert ig.triplet.b == pytest.approx(0.8 * erf(np.sqrt(1.7 / (2 * 0.64))), rel=1e-14)
    assert ig.triplet.nu.first_moment_between(1.0, np.inf) == pytest.approx(
        0.8 - ig.triplet.b, rel=1e-7
    )
