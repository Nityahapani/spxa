"""Fourier call pricers under a non-zero risk-free rate.

Regression tests: the share-measure probability P1 must normalise
phi(u - i) by phi(-i) = E[S_T] = S0 * exp(rT), not by S0.
"""

import math

import numpy as np
import pytest

from spxa.calibration.surface.bates import bates_call_price
from spxa.calibration.surface.double_heston import double_heston_call_price
from spxa.calibration.surface.heston import bs_call_price, heston_call_price

pytestmark = pytest.mark.exactness

S0, T = 100.0, 1.0


@pytest.mark.parametrize("r", [0.0, 0.03, 0.08])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_heston_degenerates_to_black_scholes(r, strike):
    # sigma -> 0 and v0 = theta: variance is frozen at 0.04, i.e. BS with vol 0.2
    price = heston_call_price(S0, strike, T, r, 0.04, 1.0, 0.04, 1e-4, 0.0)
    assert price == pytest.approx(bs_call_price(S0, strike, T, r, 0.2), abs=1e-4)


@pytest.mark.parametrize("r", [0.0, 0.05])
def test_double_heston_degenerates_to_black_scholes(r):
    params = np.array([0.02, 1.0, 0.02, 1e-4, 0.0, 0.02, 1.0, 0.02, 1e-4, 0.0])
    price = double_heston_call_price(S0, 95.0, T, r, params)
    assert price == pytest.approx(bs_call_price(S0, 95.0, T, r, 0.2), abs=1e-4)


@pytest.mark.parametrize("r", [0.0, 0.05])
def test_bates_degenerates_to_merton_series(r):
    sig, lam, mu_j, sd_j, strike = 0.2, 0.5, -0.1, 0.15, 95.0
    k_bar = math.exp(mu_j + 0.5 * sd_j**2) - 1.0
    lam_p = lam * (1.0 + k_bar)
    merton = sum(
        math.exp(-lam_p * T)
        * (lam_p * T) ** n
        / math.factorial(n)
        * bs_call_price(
            S0,
            strike,
            T,
            r - lam * k_bar + n * math.log(1.0 + k_bar) / T,
            math.sqrt(sig**2 + n * sd_j**2 / T),
        )
        for n in range(60)
    )
    price = bates_call_price(S0, strike, T, r, 0.04, 1.0, 0.04, 1e-4, 0.0, lam, mu_j, sd_j)
    assert price == pytest.approx(merton, abs=1e-4)


def test_heston_put_call_parity_with_rates():
    r, strike = 0.05, 105.0
    args = (0.04, 1.5, 0.05, 0.4, -0.6)
    call = heston_call_price(S0, strike, T, r, *args, n_quad=128)
    # Put via parity must be non-negative and the call must respect the lower bound
    put = call - S0 + strike * math.exp(-r * T)
    assert put > 0.0
    assert call >= S0 - strike * math.exp(-r * T)
