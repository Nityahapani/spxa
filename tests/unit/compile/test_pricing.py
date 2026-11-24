"""COS prices from compiled kernels and their exact Jacobian."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from spxa.compile import CompiledProcess, CompileError
from spxa.compile.pricing import cos_price
from spxa.graph.autodiff.fourier import cos_price_surface, price_and_gradient
from spxa.graph.autodiff.process import DiffProcess
from spxa.zoo.levy import (
    CGMY,
    NIG,
    AlphaStable,
    BrownianMotion,
    GammaProcess,
    KouJumpDiffusion,
    MeixnerProcess,
    MertonJumpDiffusion,
    VarianceGamma,
)

STRIKES = np.linspace(70.0, 140.0, 15)
COMPOSITE = (
    BrownianMotion(0.05, 0.2) + 0.5 * KouJumpDiffusion(0.05, 0.2, 2.0, 0.4, 8.0, 5.0)
) @ GammaProcess(4.0, 4.0) + NIG(10.0, -2.0, 0.5)

CASES = {
    "bm": BrownianMotion(0.0, 0.2),
    "vg": VarianceGamma(0.2, 0.3, -0.1),
    "merton": MertonJumpDiffusion(0.05, 0.2, 1.5, -0.1, 0.15),
    "cgmy": CGMY(1.0, 5.0, 6.0, 0.5),
    "meixner": MeixnerProcess(0.3, -0.5, 1.0, 0.0),
    "composite": COMPOSITE,
}


@pytest.mark.parametrize("key", sorted(CASES))
def test_matches_the_graph_cos_pricer(key: str) -> None:
    proc = CASES[key]
    cpr, dp = CompiledProcess(proc), DiffProcess(proc)
    assert cpr.names == dp.names
    kw = dict(rate=0.03, dividend=0.01, call=STRIKES > 100)
    p, j = cpr.cos_price(100.0, STRIKES, 0.7, grad=True, **kw)
    ref = price_and_gradient(dp, 100.0, STRIKES, 0.7, method="cos", **kw)
    np.testing.assert_allclose(p, ref.prices, rtol=0, atol=1e-12)
    scale = max(1.0, float(np.abs(ref.jacobian).max()))
    np.testing.assert_allclose(j, ref.jacobian, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(cpr.cos_price(100.0, STRIKES, 0.7, **kw), p, rtol=0, atol=1e-12)


def test_black_scholes() -> None:
    s0, r, q, sig, T = 100.0, 0.03, 0.01, 0.25, 0.8
    cpr = CompiledProcess(BrownianMotion(0.4, sig))  # the drift cancels
    calls = cpr.cos_price(s0, STRIKES, T, rate=r, dividend=q)
    d1 = (np.log(s0 / STRIKES) + (r - q + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    bs = s0 * np.exp(-q * T) * norm.cdf(d1) - STRIKES * np.exp(-r * T) * norm.cdf(d2)
    np.testing.assert_allclose(calls, bs, rtol=0, atol=1e-10)
    _, j = cpr.cos_price(s0, STRIKES, T, rate=r, dividend=q, grad=True)
    vega = s0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    np.testing.assert_allclose(j[:, 0], 0.0, atol=1e-10)  # d price / d mu
    np.testing.assert_allclose(j[:, 1], vega, rtol=0, atol=1e-9)


def test_put_call_parity() -> None:
    cpr = CompiledProcess(CASES["merton"])
    c = cpr.cos_price(100.0, STRIKES, 1.3, rate=0.02, dividend=0.01, call=True)
    p = cpr.cos_price(100.0, STRIKES, 1.3, rate=0.02, dividend=0.01, call=False)
    fwd = 100.0 * np.exp(-0.01 * 1.3) - STRIKES * np.exp(-0.02 * 1.3)
    np.testing.assert_allclose(c - p, fwd, rtol=0, atol=1e-12)


def test_batched_surface_and_finite_differences() -> None:
    cpr, dp = CompiledProcess(COMPOSITE), DiffProcess(COMPOSITE)
    th = np.tile(cpr.values, (3, 1))
    th[:, 1] = [0.1, 0.2, 0.3]
    th[:, -1] = [0.3, 0.5, 0.9]
    mats = np.array([0.1, 0.5, 1.0, 2.0] * 3)
    ks = np.repeat([80.0, 100.0, 125.0], 4)
    p, j = cpr.cos_price(100.0, ks, mats, rate=0.02, theta=th, grad=True)
    assert p.shape == (3, 12) and j.shape == (3, 12, cpr.n_params)
    for b in range(3):
        ref = cos_price_surface(dp, 100.0, ks, mats, rate=0.02, theta=th[b])
        np.testing.assert_allclose(p[b], ref, rtol=0, atol=1e-12)
    h = 1e-6
    for i in range(cpr.n_params):
        up, dn = th.copy(), th.copy()
        up[:, i] += h
        dn[:, i] -= h
        fd = (
            cpr.cos_price(100.0, ks, mats, rate=0.02, theta=up)
            - cpr.cos_price(100.0, ks, mats, rate=0.02, theta=dn)
        ) / (2 * h)
        np.testing.assert_allclose(j[..., i], fd, rtol=0, atol=2e-7)


def test_free_subset() -> None:
    full = CompiledProcess(CASES["merton"])
    sub = CompiledProcess(CASES["merton"], free=["MertonJumpDiffusion.sigma"])
    p1, j1 = full.cos_price(100.0, STRIKES, 0.5, grad=True)
    p2, j2 = sub.cos_price(100.0, STRIKES, 0.5, grad=True)
    np.testing.assert_array_equal(p1, p2)
    assert j2.shape == (STRIKES.size, 1)
    np.testing.assert_allclose(j2[:, 0], j1[:, 1], rtol=0, atol=1e-13)


@pytest.mark.parametrize("backend", ["numba", "jax"])
def test_backends(backend: str) -> None:
    pytest.importorskip(backend)
    ref = CompiledProcess(CASES["cgmy"])
    cpr = CompiledProcess(CASES["cgmy"], backend)
    th = np.tile(cpr.values, (2, 1))
    th[:, 0] = [0.5, 1.0]
    for a, b in zip(
        cpr.cos_price(100.0, STRIKES, 0.5, theta=th, grad=True),
        ref.cos_price(100.0, STRIKES, 0.5, theta=th, grad=True),
    ):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-11)


def test_errors() -> None:
    cpr = CompiledProcess(CASES["vg"])
    with pytest.raises(ValueError, match="spot"):
        cos_price(cpr, -1.0, 100.0, 1.0)
    with pytest.raises(ValueError, match="strikes must be positive"):
        cos_price(cpr, 100.0, [0.0], 1.0)
    with pytest.raises(ValueError, match="maturities must be positive"):
        cos_price(cpr, 100.0, 100.0, [0.0])
    with pytest.raises(ValueError, match="1-D"):
        cos_price(cpr, 100.0, np.ones((2, 2)), 1.0)
    with pytest.raises(ValueError, match="n_terms"):
        cos_price(cpr, 100.0, 100.0, 1.0, n_terms=1)
    with pytest.raises(ValueError, match="infinite"):
        CompiledProcess(GammaProcess(2.0, 0.5)).cos_price(100.0, 100.0, 1.0)
    with pytest.raises(CompileError, match="exponential moments"):
        CompiledProcess(AlphaStable(1.5, 0.0, 0.2, 0.0)).cos_price(100.0, 100.0, 1.0)
    with pytest.raises(ValueError, match="degenerate"):
        CompiledProcess(BrownianMotion(0.0, 0.2), fixed=None).cos_price(
            100.0, 100.0, 1.0, theta=[0.0, 0.0]
        )
