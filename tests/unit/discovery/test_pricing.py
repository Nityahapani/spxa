"""COS prices of European options under composite models."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.discovery import BROWNIAN, DRIFT, GAMMA, VARIANCE_GAMMA, Model, leaf
from spxa.discovery.pricing import cos_prices, martingale_correction
from spxa.market.black import black_scholes_price

STRIKES = np.array([60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 150.0])


@pytest.mark.parametrize("T", [0.02, 0.5, 3.0])
def test_brownian_motion_gives_black_scholes(T: float) -> None:
    m = Model(leaf(BROWNIAN) + leaf(DRIFT), [(0.25,), (0.7,)])  # the drift cancels
    for call in (True, False):
        p = cos_prices(m, 100.0, STRIKES, T, rate=0.03, dividend=0.01, call=call)
        ref = black_scholes_price(100.0, STRIKES, T, 0.03, 0.25, 0.01, "call" if call else "put")
        np.testing.assert_allclose(p, ref, atol=1e-11)


@pytest.mark.parametrize(
    ("T", "ref", "tol"), [(0.1, 10.993703187, 1e-8), (1.0, 19.099354724, 1e-9)]
)
def test_variance_gamma_reference_values(T: float, ref: float, tol: float) -> None:
    # VG call of Fang & Oosterlee (2008): S0 = 100, K = 90, r = 0.1, q = 0,
    # sigma = 0.12, theta = -0.14, nu = 0.2; reference values to 9 decimals
    vg = Model(leaf(VARIANCE_GAMMA), [(0.12, 0.2, -0.14)])
    p = cos_prices(vg, 100.0, [90.0], T, rate=0.1, call=True, n_terms=4096)
    assert p[0] == pytest.approx(ref, abs=tol)


def test_parity_martingale_and_mixed_types() -> None:
    m = Model(leaf(VARIANCE_GAMMA) + leaf(GAMMA), [(0.2, 0.3, -0.1), (2.0, 20.0)])
    T, r, q = 0.75, 0.02, 0.005
    calls = cos_prices(m, 100.0, STRIKES, T, r, q, call=True)
    puts = cos_prices(m, 100.0, STRIKES, T, r, q, call=False)
    np.testing.assert_allclose(calls - puts, 100 * np.exp(-q * T) - STRIKES * np.exp(-r * T))
    mixed = cos_prices(m, 100.0, STRIKES, T, r, q, call=STRIKES >= 100)
    np.testing.assert_allclose(mixed, np.where(STRIKES >= 100, calls, puts))
    # a call struck near zero is worth the discounted forward: E[S_T] = S0 e^{(r-q)T}
    tiny = cos_prices(m, 100.0, [1e-6], T, r, q, call=True, n_terms=4096)
    assert tiny[0] == pytest.approx(100 * np.exp(-q * T), rel=1e-7)
    # convexity and monotonicity in the strike
    slopes = np.diff(calls) / np.diff(STRIKES)
    assert np.all(slopes < 0) and np.all(np.diff(slopes) > 0)
    assert slopes[0] > -np.exp(-r * T)


def test_martingale_correction() -> None:
    m = Model(leaf(BROWNIAN), [(0.3,)])
    assert martingale_correction(m) == pytest.approx(-0.045)
    g = Model(leaf(GAMMA), [(2.0, 4.0)])
    assert martingale_correction(g) == pytest.approx(2.0 * np.log(1 - 1 / 4.0))
    heavy = Model(leaf(GAMMA), [(2.0, 0.8)])  # E[e^{G_1}] infinite for rate b <= 1
    with pytest.raises(ValueError, match="no risk-neutral price"):
        cos_prices(heavy, 100.0, [100.0], 1.0)
    with pytest.raises(ValueError):
        cos_prices(m, 100.0, [100.0], 0.0)
    with pytest.raises(ValueError):
        cos_prices(m, 100.0, [-1.0], 1.0)
