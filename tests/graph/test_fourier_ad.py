"""Exact parameter derivatives of Fourier option prices (COS, Carr–Madan, Lewis)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from spxa.graph.autodiff.exactness import GradientExactness
from spxa.graph.autodiff.fourier import (
    carr_madan_price,
    cos_price,
    fourier_nodes,
    fourier_price_node,
    lewis_price,
    martingale_correction,
    price_and_gradient,
)
from spxa.graph.autodiff.process import DiffProcess, UnsupportedProcessError
from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ParameterNode
from spxa.zoo.levy.brownian import BrownianMotion
from spxa.zoo.levy.cgmy import CGMY
from spxa.zoo.levy.drift import Drift
from spxa.zoo.levy.gamma import GammaProcess
from spxa.zoo.levy.merton import MertonJumpDiffusion
from spxa.zoo.levy.nig import NIG
from spxa.zoo.levy.stable import AlphaStable
from spxa.zoo.levy.vg import VarianceGamma

S, T, R, Q = 100.0, 0.75, 0.03, 0.01
K = np.array([55.0, 80.0, 95.0, 100.0, 105.0, 120.0, 160.0])
METHODS = ["cos", "carr_madan", "lewis"]
PRICERS = {"cos": cos_price, "carr_madan": carr_madan_price, "lewis": lewis_price}


def _bs(sig: float) -> tuple[np.ndarray, ...]:
    sq = sig * math.sqrt(T)
    d1 = (np.log(S / K) + (R - Q + sig**2 / 2) * T) / sq
    d2 = d1 - sq
    call = S * math.exp(-Q * T) * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)
    vega = S * math.exp(-Q * T) * norm.pdf(d1) * math.sqrt(T)
    volga = vega * d1 * d2 / sig
    return call, vega, volga


@pytest.mark.parametrize("method", METHODS)
def test_black_scholes_prices_vega_volga(method: str) -> None:
    sig = 0.25
    dp = DiffProcess(BrownianMotion(0.07, sig))
    call, vega, volga = _bs(sig)
    res = price_and_gradient(dp, S, K, T, method=method, hessian=True, rate=R, dividend=Q)
    assert res.exactness == GradientExactness.EXACT and res.names == dp.names
    np.testing.assert_allclose(res.prices, call, rtol=0, atol=2e-12 * S)
    np.testing.assert_allclose(res.jacobian[:, 1], vega, rtol=0, atol=2e-12 * S)
    np.testing.assert_allclose(res.jacobian[:, 0], 0.0, atol=1e-12)  # drift cancels
    assert res.hessian is not None
    np.testing.assert_allclose(res.hessian[:, 1, 1], volga, rtol=0, atol=5e-11 * S)
    np.testing.assert_allclose(res.hessian[:, 0, :], 0.0, atol=1e-11)
    puts = price_and_gradient(dp, S, K, T, method=method, rate=R, dividend=Q, call=False)
    parity = S * math.exp(-Q * T) - K * math.exp(-R * T)
    np.testing.assert_allclose(res.prices - puts.prices, parity, rtol=0, atol=1e-12 * S)
    np.testing.assert_allclose(res.jacobian, puts.jacobian, rtol=0, atol=1e-12 * S)


def _merton_series(theta: np.ndarray, call: bool = True) -> np.ndarray:
    """Merton (1976) as a Poisson mixture of Black–Scholes prices (60 terms)."""
    _, sig, lam, m, s = theta
    kbar = math.exp(m + s * s / 2) - 1
    out = np.zeros_like(K)
    for n in range(60):
        w = math.exp(-lam * T) * (lam * T) ** n / math.factorial(n)
        var = sig**2 + n * s**2 / T
        sq = math.sqrt(var * T)
        # log(S_T/S0) | N=n ~ Normal((r-q)T - lam kbar T - sig^2 T/2 + n m, sig^2 T + n s^2)
        mean = (R - Q) * T - lam * kbar * T - sig**2 * T / 2 + n * m
        f = S * math.exp(mean + var * T / 2)
        d1 = (np.log(f / K) + var * T / 2) / sq
        d2 = d1 - sq
        out += w * math.exp(-R * T) * (f * norm.cdf(d1) - K * norm.cdf(d2))
    return out


@pytest.mark.parametrize("method", METHODS)
def test_merton_against_poisson_series(method: str) -> None:
    proc = MertonJumpDiffusion(0.02, 0.15, 0.8, -0.1, 0.2)
    dp = DiffProcess(proc)
    th = dp.values
    res = price_and_gradient(dp, S, K, T, method=method, rate=R, dividend=Q)
    np.testing.assert_allclose(res.prices, _merton_series(th), rtol=0, atol=1e-11 * S)
    for k in range(len(th)):
        h = 1e-3 * max(1.0, abs(th[k]))
        e = np.eye(len(th))[k] * h
        d1 = (_merton_series(th + e) - _merton_series(th - e)) / (2 * h)
        d2 = (_merton_series(th + e / 2) - _merton_series(th - e / 2)) / h
        ref = (4 * d2 - d1) / 3
        np.testing.assert_allclose(res.jacobian[:, k], ref, rtol=1e-8, atol=1e-9 * S)


MODELS = [
    pytest.param(NIG(8.0, -2.0, 0.5, 0.0), 1e-9, id="nig"),
    pytest.param(CGMY(0.5, 6.0, 9.0, 1.2), 1e-9, id="cgmy"),
    pytest.param(0.8 * NIG(8.0, -2.0, 0.5, 0.0), 1e-9, id="scaled-nig"),
    pytest.param(
        BrownianMotion(0.0, 0.2) @ GammaProcess(1 / 0.2, 1 / 0.2) + Drift(0.3), 5e-9, id="sub+drift"
    ),
    # VG's density is not smooth at 0 (|x|^{2T/nu - 1}), so COS converges algebraically
    pytest.param(VarianceGamma(0.2, 0.3, -0.1) + Drift(0.01), 3e-7, id="vg+drift"),
]


@pytest.mark.parametrize("proc, gtol", MODELS)
def test_methods_agree_on_prices_and_gradients(proc: Any, gtol: float) -> None:
    dp = DiffProcess(proc)
    calls = K >= S
    res = {
        m: price_and_gradient(dp, S, K, T, method=m, rate=R, dividend=Q, call=calls)
        for m in METHODS
    }
    for m in METHODS[1:]:
        np.testing.assert_allclose(res[m].prices, res["cos"].prices, rtol=0, atol=2e-10 * S)
        np.testing.assert_allclose(res[m].jacobian, res["cos"].jacobian, rtol=0, atol=gtol * S)
    np.testing.assert_allclose(
        res["lewis"].jacobian, res["carr_madan"].jacobian, rtol=0, atol=1e-9 * S
    )
    drift = [i for i, n in enumerate(dp.names) if n.startswith("Drift")]
    for m in METHODS:
        for i in drift:
            np.testing.assert_allclose(res[m].jacobian[:, i], 0.0, atol=1e-11 * S)


@pytest.mark.parametrize("method", ["carr_madan", "lewis"])
def test_gradient_is_the_derivative_of_the_discretised_price(method: str) -> None:
    # nodes do not depend on theta here, so a Richardson difference of the pricer
    # must reproduce the Jacobian to difference accuracy
    dp = DiffProcess(NIG(8.0, -2.0, 0.5, 0.01) + Drift(0.0))
    th = dp.values
    J = price_and_gradient(dp, S, K, T, method=method, rate=R, dividend=Q).jacobian
    f = PRICERS[method]
    for k in range(len(th)):
        h = 1e-3 * max(1.0, abs(th[k]))
        e = np.eye(len(th))[k]

        def d(hh: float) -> np.ndarray:
            return (
                f(dp, S, K, T, R, Q, theta=th + hh * e) - f(dp, S, K, T, R, Q, theta=th - hh * e)
            ) / (2 * hh)

        ref = (4 * d(h / 2) - d(h)) / 3
        np.testing.assert_allclose(J[:, k], ref, rtol=1e-7, atol=1e-9 * S)


def test_cos_hessian_against_difference_of_exact_gradient() -> None:
    dp = DiffProcess(NIG(8.0, -2.0, 0.5, 0.0))
    th = dp.values
    res = price_and_gradient(dp, S, K, T, method="lewis", hessian=True, rate=R, dividend=Q)
    assert res.hessian is not None
    np.testing.assert_allclose(res.hessian, np.swapaxes(res.hessian, 1, 2), atol=1e-12 * S)
    for k in range(len(th)):
        h = 1e-4 * max(1.0, abs(th[k]))
        e = np.eye(len(th))[k] * h
        up = price_and_gradient(dp, S, K, T, method="lewis", theta=th + e, rate=R, dividend=Q)
        dn = price_and_gradient(dp, S, K, T, method="lewis", theta=th - e, rate=R, dividend=Q)
        fd = (up.jacobian - dn.jacobian) / (2 * h)
        np.testing.assert_allclose(res.hessian[:, :, k], fd, rtol=1e-6, atol=1e-7 * S)


def test_graph_node_gradient_is_exact() -> None:
    dp = DiffProcess(NIG(8.0, -2.0, 0.5, 0.0))
    params = [ParameterNode(v, name=n) for n, v in zip(dp.names, dp.values)]
    node = fourier_price_node(dp, params, S, K, T, method="lewis", rate=R, dividend=Q)
    g = ComputationalGraph()
    g.add_output(node, "out")
    res = GradientTape(g).gradient("out", sources=params, strict=True)
    J = price_and_gradient(dp, S, K, T, method="lewis", rate=R, dividend=Q).jacobian
    assert res.exactness == GradientExactness.EXACT
    for k, p in enumerate(params):
        assert float(np.real(res[p])) == pytest.approx(J[:, k].sum(), rel=1e-12)
    out = g.evaluate()["out"]
    np.testing.assert_allclose(out, lewis_price(dp, S, K, T, R, Q), rtol=1e-15)


def test_refusals() -> None:
    with pytest.raises(ValueError, match="infinite"):
        martingale_correction(DiffProcess(GammaProcess(1.0, 0.8)))
    with pytest.raises(UnsupportedProcessError):
        cos_price(DiffProcess(AlphaStable(1.5, 0.0, 0.2, 0.0)), S, K, T)
    dp = DiffProcess(NIG(8.0, -2.0, 0.5, 0.0))
    with pytest.raises(ValueError):
        cos_price(dp, S, [-1.0], T)
    with pytest.raises(ValueError):
        lewis_price(dp, S, K, 0.0)
    with pytest.raises(ValueError, match="method"):
        price_and_gradient(dp, S, K, T, method="fft")
    x, w = fourier_nodes(10.0)
    assert x.min() > 0 and x.max() < 10 and w.sum() == pytest.approx(10.0, rel=1e-14)


def test_cos_gradient_tracks_the_interval() -> None:
    # VG's COS expansion is not converged to machine precision, so the interval's
    # dependence on theta matters; the tracked derivative is that of cos_price itself
    dp = DiffProcess(VarianceGamma(0.2, 0.3, -0.1) + Drift(0.01))
    th = dp.values
    Kq = np.array([80.0, 100.0, 120.0])
    J = price_and_gradient(dp, S, Kq, 0.25, method="cos", rate=R, dividend=Q).jacobian
    for k in range(len(th)):
        h = 1e-3 * max(1.0, abs(th[k]))
        e = np.eye(len(th))[k]

        def d(hh: float) -> np.ndarray:
            up = cos_price(dp, S, Kq, 0.25, R, Q, theta=th + hh * e)
            dn = cos_price(dp, S, Kq, 0.25, R, Q, theta=th - hh * e)
            return (up - dn) / (2 * hh)

        np.testing.assert_allclose(J[:, k], (4 * d(h / 2) - d(h)) / 3, rtol=1e-7, atol=1e-10 * S)
    frozen = price_and_gradient(
        dp, S, Kq, 0.25, method="cos", rate=R, dividend=Q, track_interval=False
    ).jacobian
    np.testing.assert_allclose(frozen, J, rtol=0, atol=1e-5 * S)
