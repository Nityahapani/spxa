"""Tests for graph-based calibration, loss nodes, constraints, and optimization."""

import numpy as np
import pytest

from spxa.graph import math_ops as mo
from spxa.graph.autodiff.exactness import GradientExactness, InexactGradientError
from spxa.graph.autodiff.hessian import graph_hessian
from spxa.graph.autodiff.jet import variable_vector
from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.calibration.constraints import FellerConstraintNode
from spxa.graph.calibration.loss_nodes import ArbitragePenaltyNode, MSELossNode
from spxa.graph.calibration.optimizer import GraphCalibrator
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode, OperatorNode, ParameterNode
from spxa.graph.types import ParameterBounds


def test_linear_graph_calibration_recovering_parameters():
    # Model: y = a * x + b
    # Target: a=3.0, b=5.0
    x_val = ConstantNode(2.0, name="x")
    a = ParameterNode(value=1.0, name="a")
    b = ParameterNode(value=1.0, name="b")

    y_pred = a * x_val + b
    y_target = ConstantNode(3.0 * 2.0 + 5.0, name="y_true")  # 11.0

    loss = MSELossNode(model_output=y_pred, market_quotes=y_target)

    g = ComputationalGraph()
    g.add_output(loss, "loss")

    calibrator = GraphCalibrator(g, loss_output_name="loss")
    res = calibrator.fit(parameters=[a, b], method="L-BFGS-B")

    assert res["success"]
    assert res["final_loss"] < 1e-6
    # 2a + b = 11
    assert np.isclose(a.value * 2.0 + b.value, 11.0, atol=1e-3)


def test_feller_constraint_penalty():
    # CIR parameters: kappa=1.0, theta=0.04, sigma=0.5
    # 2 * kappa * theta = 0.08 < 0.25 = sigma^2 (violates Feller!)
    kappa = ParameterNode(1.0, "kappa")
    theta = ParameterNode(0.04, "theta")
    sigma = ParameterNode(0.5, "sigma")

    feller_node = FellerConstraintNode(kappa, theta, sigma, penalty_weight=100.0)
    g = ComputationalGraph()
    g.add_output(feller_node, "penalty")

    res = g.evaluate()
    # Violation = 0.25 - 0.08 = 0.17 -> Penalty = 100 * 0.17^2 = 2.89
    assert np.isclose(res["penalty"], 100.0 * (0.17**2), atol=1e-4)


# ── exact derivatives through the loss and penalty nodes; the calibrator ─────

X = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # equally spaced, step h = 0.5
H_STEP = 0.5
Q = np.array([0.1, -0.2, 0.05, 0.3, 0.9])
W = np.array([1.0, 2.0, 0.5, 1.0, 3.0])


def _graph(node):
    g = ComputationalGraph()
    g.add_output(node, "out")
    return g


def _quadratic(a, b):
    # y_i = b x_i^2 - a x_i, linear in (a, b), so every Hessian below is in closed form
    x = ConstantNode(X, name="x")
    return mo.AddNode(
        mo.MulNode(b, mo.MulNode(x, x)), mo.MulNode(ConstantNode(-1.0), mo.MulNode(a, x))
    )


def _second_difference_reference(p):
    return np.array([p[i] - 2 * p[i + 1] + p[i + 2] for i in range(len(p) - 2)])


@pytest.mark.parametrize("weighted", [False, True])
def test_mse_gradient_and_hessian_are_exact_closed_forms(weighted):
    a, b = ParameterNode(1.3, "a"), ParameterNode(-0.4, "b")
    loss = MSELossNode(_quadratic(a, b), ConstantNode(Q), ConstantNode(W) if weighted else None)
    g = _graph(loss)
    w = W / W.sum() if weighted else np.full(X.size, 1.0 / X.size)
    r = -0.4 * X**2 - 1.3 * X - Q
    jac = np.stack([-X, X**2])  # dr/d(a, b)
    grad = GradientTape(g).gradient("out", sources=[a, b])
    assert grad.exactness == GradientExactness.EXACT and not grad.notes
    np.testing.assert_allclose([grad[a], grad[b]], 2 * jac @ (w * r), rtol=1e-13)
    hess = graph_hessian(g, "out", [a, b], strict=True)
    assert hess.exactness == GradientExactness.EXACT
    np.testing.assert_allclose(hess.hessian, 2 * (jac * w) @ jac.T, rtol=1e-13)
    np.testing.assert_allclose(hess.gradient, [grad[a], grad[b]], rtol=1e-13)


@pytest.mark.parametrize(
    "prices", [[3.0, 1.0, 0.4, 0.5, 0.2], [1.0, 2.0, 2.5, 2.6, 4.0], [3, 2, 1]]
)
def test_arbitrage_penalty_vjp_matches_its_closed_form_and_the_jet_route(prices):
    p = ConstantNode(np.array(prices, dtype=float), name="p")
    lam = 7.0
    g = _graph(ArbitragePenaltyNode(p, lam))
    pv = np.array(prices, dtype=float)
    v = np.maximum(-_second_difference_reference(pv), 0.0)
    assert g.evaluate()["out"] == pytest.approx(lam * np.sum(v**2), rel=1e-15)
    # -2 lambda D^T v, assembled entry by entry
    ref = np.zeros_like(pv)
    for i, vi in enumerate(v):
        ref[i : i + 3] += -2 * lam * vi * np.array([1.0, -2.0, 1.0])
    vjp = GradientTape(g).gradient("out", sources=[p])
    assert vjp.exactness == GradientExactness.EXACT and not vjp.notes
    np.testing.assert_allclose(vjp[p], ref, rtol=1e-14, atol=1e-14)
    # the forward runs on jets, an independent route to the same gradient
    jet_out = ArbitragePenaltyNode(ConstantNode(pv), lam).op_fn(variable_vector(pv))
    np.testing.assert_allclose(jet_out.grad, ref, rtol=1e-14, atol=1e-14)
    # and central differences agree. [3, 2, 1] sits on the convexity boundary,
    # where max(0, x)^2 is C^1 with derivative 0 but one-sided, so differences
    # carry an O(h) bias of about lam * 4h
    h = 1e-6
    fd = np.array(
        [
            (
                g.evaluate(feed_dict={p: pv + h * e})["out"]
                - g.evaluate(feed_dict={p: pv - h * e})["out"]
            )
            / (2 * h)
            for e in np.eye(pv.size)
        ]
    )
    np.testing.assert_allclose(vjp[p], fd, rtol=1e-6, atol=8 * lam * h)


def test_arbitrage_penalty_hessian_is_exact_and_short_inputs_carry_no_penalty():
    a, b = ParameterNode(1.3, "a"), ParameterNode(-0.4, "b")
    lam = 3.0
    g = _graph(ArbitragePenaltyNode(_quadratic(a, b), lam))
    # D(x) = 0 and D(x^2) = 2h^2 on an even grid: P = lam (n-2) (2 b h^2)^2 for b < 0
    n = X.size
    assert g.evaluate()["out"] == pytest.approx(lam * (n - 2) * (2 * -0.4 * H_STEP**2) ** 2)
    hess = graph_hessian(g, "out", [a, b], strict=True)
    assert hess.exactness == GradientExactness.EXACT
    np.testing.assert_allclose(
        hess.hessian, [[0.0, 0.0], [0.0, 8 * lam * (n - 2) * H_STEP**4]], rtol=1e-13, atol=1e-13
    )
    short = _graph(ArbitragePenaltyNode(ConstantNode(np.array([1.0, 0.0])), lam))
    assert short.evaluate()["out"] == 0.0
    with pytest.raises(ValueError, match="one-dimensional"):
        _graph(ArbitragePenaltyNode(ConstantNode(np.ones((3, 3))), lam)).evaluate()


@pytest.mark.parametrize("sigma", [0.2, 1.2])  # Feller satisfied / violated
def test_feller_penalty_gradient_and_hessian_are_exact(sigma):
    kappa, theta = ParameterNode(1.3, "kappa"), ParameterNode(0.4, "theta")
    lam = 5.0
    g = _graph(FellerConstraintNode(kappa, theta, ConstantNode(sigma), lam))
    e = sigma**2 - 2 * 1.3 * 0.4
    u = max(e, 0.0)
    de = np.array([-2 * 0.4, -2 * 1.3])
    grad = GradientTape(g).gradient("out", sources=[kappa, theta])
    np.testing.assert_allclose([grad[kappa], grad[theta]], 2 * lam * u * de, rtol=1e-14)
    hess = graph_hessian(g, "out", [kappa, theta], strict=True)
    expected = (
        2 * lam * (np.outer(de, de) + e * np.array([[0.0, -2.0], [-2.0, 0.0]]))
        if e > 0
        else np.zeros((2, 2))
    )
    np.testing.assert_allclose(hess.hessian, expected, rtol=1e-13, atol=1e-14)


def test_penalised_calibration_reaches_the_closed_form_optimum():
    # q = -x^2 is concave; with y = b x^2 the penalty acts for b < 0 and
    # L(b) = mean((b+1)^2 x^4) + lam (n-2) (2 b h^2)^2, minimised at
    # b* = -m4 / (m4 + 4 lam (n-2) h^4), m4 = mean(x^4)
    b = ParameterNode(0.5, "b")
    y = mo.MulNode(b, mo.MulNode(ConstantNode(X), ConstantNode(X)))
    lam = 0.8
    total = mo.AddNode(MSELossNode(y, ConstantNode(-(X**2))), ArbitragePenaltyNode(y, lam))
    g = ComputationalGraph()
    g.add_output(total, "loss")
    m4, n = np.mean(X**4), X.size
    b_star = -m4 / (m4 + 4 * lam * (n - 2) * H_STEP**4)
    res = GraphCalibrator(g).fit([b], tol=1e-14, strict=True)
    assert res["success"]
    assert res["gradient_exactness"] == GradientExactness.EXACT and res["gradient_notes"] == []
    assert b.value == pytest.approx(b_star, rel=1e-8)
    b.value = 0.5
    res = GraphCalibrator(g).fit_adam([b], lr=0.05, max_iter=600, strict=True)
    assert res["gradient_exactness"] == GradientExactness.EXACT
    assert b.value == pytest.approx(b_star, abs=1e-4)


class _FloatSquare(OperatorNode):
    """A node whose forward calls float(): no VJP and no jets, so the tape differences it."""

    def __init__(self, x):
        super().__init__(op_fn=lambda v: float(v) ** 2, inputs=[x], name="float_square")


def test_calibrator_reports_inexact_gradients_and_strict_raises():
    a = ParameterNode(2.0, "a")
    g = ComputationalGraph()
    g.add_output(MSELossNode(_FloatSquare(a), ConstantNode(9.0)), "loss")
    res = GraphCalibrator(g).fit([a])
    assert res["gradient_exactness"] == GradientExactness.NUMERICAL
    assert [n.node for n in res["gradient_notes"]] == ["float_square"]
    assert abs(a.value) == pytest.approx(3.0, rel=1e-4)
    a.value = 2.0
    with pytest.raises(InexactGradientError, match="float_square"):
        GraphCalibrator(g).fit([a], strict=True)
    with pytest.raises(InexactGradientError):
        GraphCalibrator(g).fit_adam([a], max_iter=2, strict=True)


def test_calibrator_rejects_a_non_scalar_loss_and_projects_adam_onto_bounds():
    a = ParameterNode(1.0, "a")
    g = ComputationalGraph()
    g.add_output(mo.MulNode(a, ConstantNode(X)), "loss")
    with pytest.raises(ValueError, match="must be a scalar"):
        GraphCalibrator(g).fit([a])
    with pytest.raises(ValueError, match="must be a scalar"):
        GraphCalibrator(g).fit_adam([a])
    with pytest.raises(ValueError, match="no parameters"):
        GraphCalibrator(ComputationalGraph()).fit([])
    # the loss decreases without bound as c falls; Adam stops at the lower bound
    c = ParameterNode(1.0, "c", bounds=ParameterBounds(lower=0.25, upper=4.0))
    g = ComputationalGraph()
    g.add_output(mo.MulNode(c, ConstantNode(3.0)), "loss")
    res = GraphCalibrator(g).fit_adam([c], lr=0.1, max_iter=100)
    assert c.value == 0.25 and res["final_loss"] == pytest.approx(0.75)
