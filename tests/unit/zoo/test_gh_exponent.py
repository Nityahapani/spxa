"""The GH characteristic exponent is the continuous logarithm.

The principal log of K_lambda(delta zeta(u)) wrapped by 2 pi once
delta |beta| was of order one or more. That flipped the sign of phi(u, t) for
non-integer t. At lambda = -1/2 the GH law is NIG, whose exponent is
elementary, so it provides an independent reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxa.zoo.levy import NIG, GeneralizedHyperbolic

PARAMS = [(3.0, 1.5, 3.0), (2.0, -1.2, 4.0), (5.0, 4.0, 2.0), (1.0, 0.5, 0.5)]


@pytest.mark.parametrize("alpha, beta, delta", PARAMS)
def test_lambda_minus_half_is_nig_at_fractional_times(alpha, beta, delta) -> None:
    gh = GeneralizedHyperbolic(-0.5, alpha, beta, delta, 0.1)
    nig = NIG(alpha, beta, delta, 0.1)
    u = np.linspace(-15.0, 15.0, 301)
    for t in (0.3, 0.5, 1.7):
        np.testing.assert_allclose(gh.char_func(u, t), nig.char_func(u, t), rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("lam", [-1.3, 1.0, 2.5])
@pytest.mark.parametrize("alpha, beta, delta", PARAMS)
def test_exponent_is_continuous_and_finite(lam, alpha, beta, delta) -> None:
    psi = np.asarray(
        GeneralizedHyperbolic(lam, alpha, beta, delta, 0.0).char_exponent(
            np.linspace(0.0, 400.0, 400_001)
        )
    )
    assert np.all(np.isfinite(psi))
    assert psi[0] == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(np.diff(psi.imag))) < 0.05  # no 2 pi jumps
