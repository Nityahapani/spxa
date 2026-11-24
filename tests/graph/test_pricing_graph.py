"""Tests for graph pricing, automated Greeks, and Monte Carlo pricing."""

import numpy as np
import pytest
from scipy.stats import norm

from spxa.graph.pricing.greeks import compute_graph_greeks
from spxa.graph.pricing.pricer_graph import (
    build_black_scholes_pricing_graph,
    monte_carlo_graph_price,
)


def test_black_scholes_pricing_graph_and_greeks():
    s0 = 100.0
    k = 100.0
    t = 1.0
    r = 0.05
    sigma = 0.20

    g = build_black_scholes_pricing_graph(s0=s0, k=k, t=t, r=r, sigma=sigma, is_call=True)
    greeks = compute_graph_greeks(g)

    # Analytical Black-Scholes benchmark:
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    bs_price = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    bs_delta = norm.cdf(d1)
    bs_gamma = norm.pdf(d1) / (s0 * sigma * np.sqrt(t))
    bs_vega = s0 * norm.pdf(d1) * np.sqrt(t)

    assert np.isclose(greeks["price"], bs_price, atol=1e-4)
    assert np.isclose(greeks["delta"], bs_delta, atol=1e-4)
    assert np.isclose(greeks["gamma"], bs_gamma, atol=1e-3)
    assert np.isclose(greeks["vega"], bs_vega, atol=1e-3)


def test_monte_carlo_graph_price_convergence():
    np.random.seed(42)
    n_paths = 50000
    s0 = 100.0
    k = 100.0
    r = 0.05
    sigma = 0.20
    t = 1.0

    z = np.random.randn(n_paths)
    s_t = s0 * np.exp((r - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)

    res = monte_carlo_graph_price(
        paths=s_t,
        payoff_fn=lambda s: np.maximum(s - k, 0.0),
        discount_rate=r,
        maturity=t,
    )

    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    bs_price = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    assert res["ci_95_lower"] <= bs_price <= res["ci_95_upper"]
    assert np.isclose(res["price"], bs_price, atol=0.15)


# ── exact Greeks ────────────────────────────────────────────────────────────


def _bs_closed_form(s0, k, t, r, sig, call):  # type: ignore[no-untyped-def]
    d1 = (np.log(s0 / k) + (r + 0.5 * sig**2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    pdf, disc = norm.pdf(d1), k * np.exp(-r * t)
    common_theta = -s0 * pdf * sig / (2 * np.sqrt(t))
    if call:
        price = s0 * norm.cdf(d1) - disc * norm.cdf(d2)
        delta, rho = norm.cdf(d1), k * t * np.exp(-r * t) * norm.cdf(d2)
        theta = common_theta - r * disc * norm.cdf(d2)
    else:
        price = disc * norm.cdf(-d2) - s0 * norm.cdf(-d1)
        delta, rho = norm.cdf(d1) - 1.0, -k * t * np.exp(-r * t) * norm.cdf(-d2)
        theta = common_theta + r * disc * norm.cdf(-d2)
    return {
        "price": price,
        "delta": delta,
        "gamma": pdf / (s0 * sig * np.sqrt(t)),
        "vega": s0 * pdf * np.sqrt(t),
        "theta": theta,
        "rho": rho,
        "vanna": -pdf * d2 / sig,
        "volga": s0 * pdf * np.sqrt(t) * d1 * d2 / sig,
    }


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("k", [70.0, 100.0, 135.0])
def test_black_scholes_graph_greeks_are_exact(call: bool, k: float) -> None:
    from spxa.graph import graph_greeks
    from spxa.graph.autodiff.exactness import GradientExactness

    s0, t, r, sig = 100.0, 0.7, 0.03, 0.25
    g = build_black_scholes_pricing_graph(s0=s0, k=k, t=t, r=r, sigma=sig, is_call=call)
    res = graph_greeks(g, strict=True)
    assert res.exactness == GradientExactness.EXACT and not res.notes
    ref = _bs_closed_form(s0, k, t, r, sig, call)
    for key, want in ref.items():
        assert res[key] == pytest.approx(want, rel=1e-12, abs=1e-13), key
    assert compute_graph_greeks(g) == res.values


def test_inexact_greeks_are_labelled_and_strict_raises() -> None:
    from spxa.graph import graph_greeks
    from spxa.graph.autodiff.exactness import GradientExactness, InexactGradientError
    from spxa.graph.graph import ComputationalGraph
    from spxa.graph.node import ParameterNode
    from spxa.graph.pde.nodes import bs_pde_price_node

    s, v, r = (
        ParameterNode(100.0, "s0"),
        ParameterNode(0.2, "volatility"),
        ParameterNode(0.05, "rate"),
    )
    node = bs_pde_price_node(s, v, r, strike=105.0, maturity=1.0, n_space=201, n_time=100)
    g = ComputationalGraph("pde")
    g.add_output(node, name="price")
    res = graph_greeks(g)
    assert res.exactness != GradientExactness.EXACT and res.notes
    with pytest.raises(InexactGradientError):
        graph_greeks(g, strict=True)
