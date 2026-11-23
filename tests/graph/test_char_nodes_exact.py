"""Exactness of the characteristic-function nodes: explicit parameter inputs, moments, CGF."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.graph.autodiff.exactness import GradientExactness
from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.char_nodes import (
    CharacteristicExponentNode,
    CharacteristicFunctionNode,
    CumulantGeneratingFunctionNode,
    GeneratorApplicationNode,
    ProcessMomentNode,
)
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode
from spxa.graph.stochastic_nodes import BrownianMotionNode, HestonNode, MertonJumpDiffusionNode

EXACT = GradientExactness.EXACT


def _g(node, name="out"):
    g = ComputationalGraph()
    g.add_output(node, name)
    return g


def _fd_param(g, node_p, h=1e-6, name="out"):
    v = node_p.value
    up = g.evaluate(feed_dict={node_p: v + h})[name]
    dn = g.evaluate(feed_dict={node_p: v - h})[name]
    return (up - dn) / (2 * h)


def _merton():
    return MertonJumpDiffusionNode(s0=100.0, mu=0.03, sigma=0.2, lam=0.7, mu_j=-0.1, sigma_j=0.15)


def _heston():
    return HestonNode(s0=100.0, v0=0.04, mu=0.05, kappa=2.0, theta=0.05, xi=0.4, rho=-0.6)


@pytest.mark.parametrize("make", [lambda: BrownianMotionNode(0.05, 0.2), _merton, _heston])
def test_char_function_gradients_reach_process_parameters(make):
    proc = make()
    u = ConstantNode(np.array([-1.3, 0.4, 2.0]), name="u")
    t = ConstantNode(1.7, name="t")
    cf = CharacteristicFunctionNode(proc, u, t)
    g = _g(cf)
    params = cf.param_nodes
    assert all(p in cf.inputs for p in params)
    res = GradientTape(g).gradient("out", sources=params)
    assert res.exactness == EXACT and not res.notes
    for p in params:
        ref = _fd_param(g, p)
        ref = np.sum(ref)  # gradient of sum(target) for a complex array target
        assert abs(res[p] - ref) <= 1e-7 * (1 + abs(ref)), p.name


def test_char_exponent_merton_gradient():
    proc = _merton()
    cx = CharacteristicExponentNode(proc, ConstantNode(0.8))
    g = _g(cx)
    res = GradientTape(g).gradient("out", sources=cx.param_nodes)
    for p in cx.param_nodes:
        assert abs(res[p] - _fd_param(g, p)) < 1e-8


def test_moments_all_orders_exact_brownian():
    mu, sig, T = 0.05, 0.2, 2.0
    bm = BrownianMotionNode(mu, sig)
    t = ConstantNode(T, name="t")
    cf = CharacteristicFunctionNode(bm, ConstantNode(0.0), t)
    m = mu * T
    s2 = sig**2 * T
    ref = [m, m**2 + s2, m**3 + 3 * m * s2, m**4 + 6 * m**2 * s2 + 3 * s2**2]
    for k in (1, 2, 3, 4):
        node = ProcessMomentNode(cf, k, t)
        g = _g(node)
        assert np.isclose(g.evaluate()["out"], ref[k - 1], rtol=1e-15, atol=1e-18)
        res = GradientTape(g).gradient("out", sources=cf.param_nodes)
        assert res.exactness == EXACT
        for p in cf.param_nodes:
            assert abs(res[p] - _fd_param(g, p)) < 1e-8


def test_merton_moments_match_contour_integral():
    proc = _merton()
    t = ConstantNode(1.3, name="t")
    cf = CharacteristicFunctionNode(proc, ConstantNode(0.0), t)
    from spxa.graph.char_nodes import _contour_moment

    for k in (1, 2, 3, 4):
        exact = _g(ProcessMomentNode(cf, k, t)).evaluate()["out"]
        contour = _contour_moment(lambda u: np.exp(cf.log_char(u, 1.3)), k)
        assert np.isclose(exact, contour, rtol=1e-11, atol=1e-14)


def test_heston_first_moment_and_labels():
    proc = _heston()
    T = 1.5
    t = ConstantNode(T, name="t")
    cf = CharacteristicFunctionNode(proc, ConstantNode(0.0), t)
    node = ProcessMomentNode(cf, 1, t)
    g = _g(node)
    kappa, theta, v0 = 2.0, 0.05, 0.04
    ev = theta * T + (v0 - theta) * (1 - np.exp(-kappa * T)) / kappa
    ref = np.log(100.0) + 0.05 * T - 0.5 * ev
    assert np.isclose(g.evaluate()["out"], ref, rtol=1e-12)
    res = GradientTape(g).gradient("out", sources=cf.param_nodes)
    assert res.exactness == GradientExactness.NUMERICAL


def test_cgf_is_continuous_log_and_differentiates():
    proc = _merton()
    th = ConstantNode(np.array([-2.0, 0.5, 3.0]), name="theta")
    t = ConstantNode(4.0, name="t")
    cf = CharacteristicFunctionNode(proc, ConstantNode(0.0), t)
    node = CumulantGeneratingFunctionNode(cf, th, t)
    g = _g(node)
    mu, sig, lam, mj, sj = 0.03, 0.2, 0.7, -0.1, 0.15
    x = th.value
    ref = 4.0 * (mu * x + 0.5 * sig**2 * x**2 + lam * (np.exp(mj * x + 0.5 * sj**2 * x**2) - 1))
    assert np.allclose(g.evaluate()["out"], ref, rtol=1e-14)
    res = GradientTape(g).gradient("out", sources=[th, *cf.param_nodes])
    assert res.exactness == EXACT
    d_theta = 4.0 * (mu + sig**2 * x + lam * (mj + sj**2 * x) * np.exp(mj * x + 0.5 * sj**2 * x**2))
    assert np.allclose(res[th], d_theta, rtol=1e-13)


def test_generator_exact_and_parameter_gradient():
    bm = BrownianMotionNode(mu=0.03, sigma=0.25)
    x = ConstantNode(2.0, name="x")
    node = GeneratorApplicationNode(bm, lambda v: v**3, x)
    g = _g(node)
    assert np.isclose(g.evaluate()["out"], 0.03 * 12 + 0.5 * 0.0625 * 12, rtol=1e-15)
    res = GradientTape(g).gradient("out", sources=list(bm.params.values()))
    mu_n, sig_n = bm.params["mu"], bm.params["sigma"]
    assert np.isclose(res[mu_n], 12.0, rtol=1e-12)
    assert np.isclose(res[sig_n], 0.25 * 12, rtol=1e-9)


def test_generator_falls_back_for_opaque_functions():
    import math

    bm = BrownianMotionNode(mu=0.03, sigma=0.25)
    node = GeneratorApplicationNode(bm, lambda v: math.sin(float(v)), ConstantNode(0.4))
    val = _g(node).evaluate()["out"]
    ref = 0.03 * math.cos(0.4) - 0.5 * 0.0625 * math.sin(0.4)
    assert abs(val - ref) < 1e-6
    assert node.gradient_exactness == GradientExactness.NUMERICAL
