"""Derivatives at kinks: max/min ties, reductions and payoff strikes.

Every rule returns the midpoint (centroid) of the Clarke subdifferential. For a
kink of two branches this is the average of the one-sided derivatives, so the
gradient times any direction equals the symmetric difference quotient.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxa.graph import math_ops as mo
from spxa.graph.autodiff.jet import Jet, variable_vector
from spxa.graph.node import ConstantNode
from spxa.graph.pricing.payoff_nodes import EuropeanCallPayoffNode, EuropeanPutPayoffNode

_RNG = np.random.default_rng(7)


def _central(f, x: np.ndarray, d: np.ndarray, h: float = 1e-2) -> float:  # type: ignore[no-untyped-def]
    # the branches are linear, so the symmetric quotient is exact up to rounding
    return float((f(x + h * d) - f(x - h * d)) / (2 * h))


# -- binary max/min -------------------------------------------------------------


@pytest.mark.parametrize("largest", [True, False])
def test_tape_vjp_splits_ties_and_is_symmetric(largest: bool) -> None:
    cls = mo.MaximumNode if largest else mo.MinimumNode
    node = cls(ConstantNode(0.0), ConstantNode(0.0))
    a = np.array([1.0, 2.0, 3.0, -1.0])
    b = np.array([1.0, 1.0, 4.0, -1.0])
    ct = np.array([2.0, 3.0, 5.0, 7.0])
    ga, gb = node.vjp_fn(ct, a, b)
    win_a = (a > b) if largest else (a < b)
    want_a = np.where(a == b, 0.5, win_a.astype(float)) * ct
    np.testing.assert_array_equal(ga, want_a)
    np.testing.assert_array_equal(ga + gb, ct)  # d max(x, x) / dx = 1
    gb2, ga2 = node.vjp_fn(ct, b, a)
    np.testing.assert_array_equal(ga, ga2)
    np.testing.assert_array_equal(gb, gb2)


@pytest.mark.parametrize("fn", [np.maximum, np.minimum, np.fmax, np.fmin])
def test_jet_ties_equal_the_symmetric_difference_quotient(fn) -> None:  # type: ignore[no-untyped-def]
    # a = x0 + x1 and b = 2 x0 tie at x = (1, 1); the kink is crossed in every direction
    x0 = np.array([1.0, 1.0])
    th = variable_vector(x0, order=2)
    j = fn(th[0] + th[1], 2.0 * th[0])
    assert float(j.value) == 2.0

    def f(x: np.ndarray) -> float:
        return float(fn(x[0] + x[1], 2.0 * x[0]))

    for _ in range(20):
        d = _RNG.normal(size=2)
        assert float(j.grad @ d) == pytest.approx(_central(f, x0, d), rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(j.grad, [1.5, 0.5], rtol=0, atol=0)
    np.testing.assert_array_equal(j.hess, np.zeros((2, 2)))
    # symmetric in the arguments
    k = fn(2.0 * th[0], th[0] + th[1])
    np.testing.assert_array_equal(k.grad, j.grad)


def test_jet_and_tape_agree_at_ties() -> None:
    a0, b0 = np.array([1.0, 2.0, 5.0]), np.array([1.0, 3.0, 5.0])
    th = variable_vector(np.r_[a0, b0])
    j = np.maximum(th[:3], th[3:])
    ga, gb = mo.MaximumNode(ConstantNode(0.0), ConstantNode(0.0)).vjp_fn(np.ones(3), a0, b0)
    np.testing.assert_array_equal(np.sum(j.grad, axis=0), np.r_[ga, gb])


def test_nan_still_propagates() -> None:
    th = variable_vector(np.array([1.0, np.nan]))
    assert np.isnan(np.maximum(th[0], th[1]).value)
    assert float(np.fmax(th[0], th[1]).value) == 1.0
    np.testing.assert_array_equal(np.fmax(th[0], th[1]).grad, [1.0, 0.0])


# -- reductions -------------------------------------------------------------------


@pytest.mark.parametrize("red", [np.max, np.min])
def test_reductions_share_the_derivative_over_ties(red) -> None:  # type: ignore[no-untyped-def]
    x0 = np.array([[3.0, 1.0, 3.0, 3.0], [0.0, 5.0, -2.0, 4.0]])
    if red is np.min:
        x0 = -x0
    th = variable_vector(x0.ravel(), order=2)
    m = red(th.reshape(2, 4), axis=1)
    np.testing.assert_array_equal(m.value, red(x0, axis=1))
    g = m.grad.reshape(2, 2, 4)
    np.testing.assert_allclose(g[0, 0], [1 / 3, 0, 1 / 3, 1 / 3], rtol=1e-15)
    np.testing.assert_array_equal(g[1, 1], [0, 1, 0, 0] if red is np.max else [0, 1, 0, 0])
    np.testing.assert_array_equal(g[0, 1], np.zeros(4))
    # two-way tie: centroid = average of the one-sided derivatives
    y0 = np.array([2.0, 2.0, -1.0])
    ty = variable_vector(y0)
    for _ in range(10):
        d = _RNG.normal(size=3)
        got = float(red(ty).grad @ d)
        assert got == pytest.approx(_central(lambda v: red(v), y0, d), rel=1e-9, abs=1e-12)


def test_running_max_shares_ties_so_far() -> None:
    x0 = np.array([1.0, 3.0, 3.0, 2.0, 3.0, 4.0])
    th = variable_vector(x0, order=2)
    acc = np.maximum.accumulate(th)
    np.testing.assert_array_equal(acc.value, np.maximum.accumulate(x0))
    want = np.zeros((6, 6))
    want[0, 0] = 1
    want[1, 1] = 1
    want[2, [1, 2]] = 0.5
    want[3, [1, 2]] = 0.5
    want[4, [1, 2, 4]] = 1 / 3
    want[5, 5] = 1
    np.testing.assert_allclose(acc.grad, want, rtol=1e-15, atol=0)
    assert acc.hess is not None and not np.any(acc.hess)
    # agrees with the reduction on every prefix
    for i in range(6):
        np.testing.assert_allclose(acc.grad[i], np.max(th[: i + 1]).grad, rtol=1e-15)


def test_running_min_and_axis() -> None:
    x0 = np.array([[2.0, 1.0, 1.0], [0.0, 0.0, -1.0]])
    th = variable_vector(x0.ravel())
    acc = np.minimum.accumulate(th.reshape(2, 3), axis=1)
    assert isinstance(acc, Jet)
    g = acc.grad.reshape(2, 3, 6)
    np.testing.assert_allclose(g[0, 2], [0, 0.5, 0.5, 0, 0, 0], rtol=1e-15)
    np.testing.assert_allclose(g[1, 1], [0, 0, 0, 0.5, 0.5, 0], rtol=1e-15)
    np.testing.assert_allclose(g[1, 2], [0, 0, 0, 0, 0, 1], rtol=1e-15)


# -- payoffs at the strike ---------------------------------------------------------


def test_call_and_put_payoffs_take_one_half_at_the_strike() -> None:
    s = np.array([90.0, 100.0, 110.0])
    ct = np.ones(3)
    call = EuropeanCallPayoffNode(ConstantNode(0.0), ConstantNode(100.0))
    put = EuropeanPutPayoffNode(ConstantNode(0.0), ConstantNode(100.0))
    gs, gk = call.vjp_fn(ct, s, 100.0)
    np.testing.assert_array_equal(gs, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(gk, [-0.0, -0.5, -1.0])
    gs, gk = put.vjp_fn(ct, s, 100.0)
    np.testing.assert_array_equal(gs, [-1.0, -0.5, -0.0])
    np.testing.assert_array_equal(gk, [1.0, 0.5, 0.0])
    # call - put = S - K: the strike conventions are consistent with parity
    np.testing.assert_array_equal(call.vjp_fn(ct, s, 100.0)[0] - put.vjp_fn(ct, s, 100.0)[0], ct)


def test_abs_at_zero_is_the_midpoint() -> None:
    (g,) = mo.AbsNode(ConstantNode(0.0)).vjp_fn(np.ones(3), np.array([-1.0, 0.0, 2.0]))
    np.testing.assert_array_equal(g, [-1.0, 0.0, 1.0])
    j = np.abs(variable_vector(np.array([0.0])))
    np.testing.assert_array_equal(j.grad, [[0.0]])
