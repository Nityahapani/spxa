"""Meixner cf beyond the overflow of cosh.

np.cosh overflows once alpha |u| / 2 > 710, and char_func then returned 0.
At short horizons the true cf is still close to 1 there:
phi(1e5, 1e-5) = 0.74 for (alpha, beta, delta) = (0.3, -0.5, 1). The cf
sampler took the 0 as "decayed" and inverted an under-resolved grid.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spxa.zoo.levy import MeixnerProcess


@pytest.mark.parametrize("t", [1e-5, 0.01, 1.0])
def test_cf_is_exp_t_psi_at_large_frequencies(t: float) -> None:
    X = MeixnerProcess(0.3, -0.5, 1.0, 0.2)
    u = np.array([1.0, 5e3, 2e4, 1e5, 1e7])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        phi = X.char_func(u, t)
    np.testing.assert_allclose(phi, np.exp(t * X._char_exponent(u)), rtol=1e-12, atol=1e-300)


def test_short_horizon_cf_does_not_collapse() -> None:
    X = MeixnerProcess(0.3, -0.5, 1.0, 0.0)
    # |phi(u, t)| ~ (cos(beta/2) * 2 e^{-alpha u / 2})^{2 delta t}
    expected = (2 * np.cos(0.25)) ** 2e-5 * np.exp(-0.3 * 1e5 * 1e-5)
    assert abs(X.char_func(1e5, 1e-5)) == pytest.approx(expected, rel=1e-6)
