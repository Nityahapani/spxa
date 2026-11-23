"""Closed-form characteristic exponents agree with char_func and with the triplet.

Before these overrides, ``_char_exponent`` integrated the Lévy measure point by
point. Subordinated characteristic functions were slow (13 s for 64 VG@TS
frequencies) and accurate only to about 1e-9. Every override is checked on the
real axis against ``char_func``, and inside the exponential-moment strip against
the Lévy–Khintchine quadrature.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from spxa.core.process import Process
from spxa.laws.quantities import exponential_moment_strip
from spxa.zoo.levy import (
    NIG,
    GammaProcess,
    InverseGaussianProcess,
    MeixnerProcess,
    TemperedStable,
    VarianceGamma,
)

CASES = [
    GammaProcess(2.0, 3.0),
    InverseGaussianProcess(1.2, 2.0),
    VarianceGamma(0.3, 0.4, -0.2),
    NIG(5.0, -1.5, 0.7, 0.1),
    MeixnerProcess(1.3, -1.1, 0.4, 0.05),
    MeixnerProcess(0.3, 2.5, 1.0, 0.0),
    TemperedStable(0.6, 0.8, 1.5),
]
U_REAL = np.array([-7.0, -1.3, 0.0, 0.4, 2.2, 30.0])


@pytest.mark.parametrize("X", CASES, ids=repr)
def test_exponent_reproduces_char_func(X: Process) -> None:
    for t in (0.3, 1.0, 2.5):
        np.testing.assert_allclose(
            np.exp(t * X._char_exponent(U_REAL)), X.char_func(U_REAL, t), rtol=1e-13, atol=1e-15
        )


@pytest.mark.parametrize("X", CASES, ids=repr)
def test_exponent_continues_into_the_strip(X: Process) -> None:
    lo, hi = exponential_moment_strip(X)
    z = np.array([0.9 * max(lo, -5.0), 0.9 * min(hi, 5.0), 0.5 * max(lo, -5.0)])
    u = np.concatenate([-1j * z, 1.7 - 1j * z, U_REAL[[1, 3]]])
    reference = Process._char_exponent(X, u)  # Lévy–Khintchine quadrature
    np.testing.assert_allclose(X._char_exponent(u), reference, rtol=1e-7, atol=1e-8)


def test_meixner_exponent_does_not_overflow() -> None:
    X = MeixnerProcess(1.3, -1.1, 0.4, 0.05)
    psi = X._char_exponent(np.array([1e4, -1e4]))
    assert np.all(np.isfinite(psi))
    # |psi| grows like delta * alpha * |u| for large |u|
    assert psi[0].real == pytest.approx(-0.4 * 1.3 * 1e4, rel=1e-3)


def test_subordinated_cf_is_fast() -> None:
    Z = VarianceGamma(0.32, 0.79, 0.22) @ TemperedStable(0.49, 0.22, 0.55)
    start = time.perf_counter()
    Z.char_func(np.linspace(-500, 500, 4096), 0.25)
    assert time.perf_counter() - start < 1.0


def test_zero_measure_needs_no_quadrature() -> None:
    """Brownian motion and Drift carry nu = 0; their jump exponent is exactly 0."""
    from spxa.zoo.levy import BrownianMotion, Drift

    for X in (BrownianMotion(0.1, 0.3), Drift(0.4)):
        nu = X.triplet.nu
        assert nu.jump_exponent(1.0 + 2.0j) == 0j
        assert nu.first_moment_between(1.0, np.inf) == 0.0
    Z = (BrownianMotion(0.1, 0.3) @ GammaProcess(2.0, 2.0)) @ InverseGaussianProcess(1.0, 2.0)
    start = time.perf_counter()
    Z.char_func(np.linspace(-50, 50, 2**14))
    assert time.perf_counter() - start < 1.0  # was about 30 s with a quadrature per point
