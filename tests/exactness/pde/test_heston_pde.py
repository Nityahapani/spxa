"""Exactness tests: Heston PDE (ADI) against the Lewis formula and published American values."""

import numpy as np
import pytest

from spxa.pde.heston import HestonPDE
from spxa.pde.reference import heston_cf, lewis_call_price

pytestmark = pytest.mark.exactness

# in 't Hout & Foulon (2010), case 1
KAPPA, THETA, SIGMA, RHO, R, T, K, V0 = 1.5, 0.04, 0.3, -0.9, 0.025, 1.0, 100.0, 0.04
CASE1 = HestonPDE(R, KAPPA, THETA, SIGMA, RHO)


def lewis(S, v0=V0, option="call", model=(KAPPA, THETA, SIGMA, RHO), r=R, q=0.0, T=T, K=K):
    return lewis_call_price(heston_cf(T, v0, *model), S, K, T, r, q, option=option)


@pytest.mark.parametrize("S", [90.0, 100.0, 110.0])
def test_european_call_second_order(S):
    ref = lewis(S)
    errs = [
        abs(CASE1.european(K, T, S, V0, n_s=ns, n_v=nv, n_time=nt).price - ref)
        for ns, nv, nt in ((51, 26, 25), (101, 51, 50), (201, 101, 100))
    ]
    assert 0.5 * np.log2(errs[0] / errs[2]) > 1.7
    assert errs[2] < 1e-3


def test_european_put_and_dividends():
    model = HestonPDE(0.03, 2.0, 0.05, 0.4, -0.6, q=0.02)
    for S in (85.0, 100.0, 120.0):
        got = model.european(K, 0.5, S, 0.06, "put", n_s=201, n_v=101, n_time=100).price
        ref = lewis(S, 0.06, "put", (2.0, 0.05, 0.4, -0.6), r=0.03, q=0.02, T=0.5)
        assert got == pytest.approx(ref, abs=1e-3)


def test_feller_violated():
    model = HestonPDE(0.05, 0.6, 0.04, 0.8, -0.5)
    assert not model.feller
    for S in (90.0, 100.0, 110.0):
        got = model.european(K, 1.0, S, 0.03, n_s=201, n_v=101, n_time=100).price
        ref = lewis(S, 0.03, model=(0.6, 0.04, 0.8, -0.5), r=0.05)
        assert got == pytest.approx(ref, abs=2e-3)


@pytest.mark.parametrize("scheme", ["douglas", "craig_sneyd", "modified_craig_sneyd"])
def test_other_schemes(scheme):
    got = CASE1.european(K, T, 100.0, V0, n_s=201, n_v=101, n_time=100, scheme=scheme).price
    assert got == pytest.approx(lewis(100.0), abs=1.5e-3)


def test_greeks():
    res = CASE1.european(K, T, 100.0, V0, n_s=201, n_v=101, n_time=100)
    h, hv = 0.5, 1e-3
    delta = (lewis(100 + h) - lewis(100 - h)) / (2 * h)
    gamma = (lewis(100 + h) - 2 * lewis(100.0) + lewis(100 - h)) / h**2
    dv = (lewis(100.0, V0 + hv) - lewis(100.0, V0 - hv)) / (2 * hv)
    assert res.delta == pytest.approx(delta, abs=2e-4)
    assert res.gamma == pytest.approx(gamma, abs=2e-4)
    assert res.dv == pytest.approx(dv, rel=2e-3)
    assert res.vega == pytest.approx(2 * np.sqrt(V0) * res.dv)
    # calendar theta from the backward equation
    ht = 1e-3
    theta = -(lewis(100.0, T=T + ht) - lewis(100.0, T=T - ht)) / (2 * ht)
    assert res.theta == pytest.approx(theta, rel=3e-3)  # first order in dt
