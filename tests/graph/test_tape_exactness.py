"""Stress tests for the graph tape: routes, shapes, complex values, exactness labels."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.graph import math_ops as mo
from spxa.graph.autodiff.exactness import (
    GradientExactness,
    GradientResult,
    InexactGradientError,
)
from spxa.graph.autodiff.hessian import compute_graph_hessian, graph_hessian
from spxa.graph.autodiff.jet import jet_hessian, variable_vector
from spxa.graph.autodiff.tape import (
    GradientTape,
    _local_jet_vjp,
    _unbroadcast,
    jvp,
    value_and_grad,
)
from spxa.graph.calibration.loss_nodes import MSELossNode
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode, OperatorNode, ParameterNode
from spxa.graph.subgraph import SubgraphNode

EXACT = GradientExactness.EXACT
NUMERICAL = GradientExactness.NUMERICAL


def _graph(*nodes, out):
    g = ComputationalGraph()
    for n in nodes:
        g.add_node(n)
    g.add_output(out, "out")
    return g


def _add_all(g, node, seen=None):
    seen = set() if seen is None else seen
    if node.node_id in seen:
        return
    seen.add(node.node_id)
    for i in node.inputs:
        _add_all(g, i, seen)
    g.add_node(node)


def _build(out):
    g = ComputationalGraph()
    _add_all(g, out)
    g.add_output(out, "out")
    return g


# ── the 1.2 finite-difference fallback bug and its replacement ───────────────


def test_array_input_without_vjp_is_exact_per_element():
    # sum(w * [1, 2, 3]) through a node with no vjp: gradient is [1, 2, 3].
    w = ConstantNode(np.array([0.5, -1.0, 2.0]), name="w")
    node = OperatorNode(lambda x: np.sum(x * np.array([1.0, 2.0, 3.0])), [w], name="dot")
    g = _build(node)
    res = GradientTape(g).gradient("out", sources=[w])
    assert isinstance(res, GradientResult)
    assert np.array_equal(res[w], [1.0, 2.0, 3.0])
    assert res.exactness == EXACT and res.notes == []


def test_jet_incapable_node_falls_back_per_element_and_is_labelled():
    w = ConstantNode(np.array([0.3, 0.7]), name="w")

    def opaque(x):  # math.* forces floats: no jets
        import math

        return math.exp(float(x[0])) * math.sin(float(x[1]))

    node = OperatorNode(opaque, [w], name="opaque")
    g = _build(node)
    res = GradientTape(g).gradient("out", sources=[w])
    ref = [np.exp(0.3) * np.sin(0.7), np.exp(0.3) * np.cos(0.7)]
    assert np.allclose(res[w], ref, rtol=1e-9, atol=0)
    assert res.exactness == NUMERICAL
    assert res.source_exactness[w] == NUMERICAL
    assert any(n.node == "opaque" for n in res.notes)
    with pytest.raises(InexactGradientError, match="opaque"):
        GradientTape(g).gradient("out", sources=[w], strict=True)
    with pytest.raises(InexactGradientError):
        GradientTape(g, strict=True).gradient("out", sources=[w])


def test_node_that_ignores_jets_is_detected():
    # a forward that swallows the jet and recomputes from a cached float
    a = ParameterNode(1.2, "a")
    cache = {"v": 1.2}

    def sneaky(x):
        try:
            cache["v"] = float(x)
        except TypeError:
            pass
        return np.exp(cache["v"])

    node = OperatorNode(sneaky, [a], name="sneaky")
    g = _build(node)
    res = GradientTape(g).gradient("out", sources=[a])
    assert np.isclose(res[a], np.exp(1.2), rtol=1e-9)
    assert res.exactness == NUMERICAL  # the jet result was rejected, not trusted


# ── shapes ──────────────────────────────────────────────────────────────────


def test_scalar_parameter_times_array_gets_scalar_gradient():
    a = ParameterNode(0.7, "a")
    c = ConstantNode(np.array([1.0, 2.0, 3.0]))
    out = mo.SumNode(mo.MulNode(a, c))
    g = _build(out)
    res = GradientTape(g).gradient("out", sources=[a])
    assert isinstance(res[a], float) and res[a] == 6.0


def test_broadcast_row_column():
    a = ConstantNode(np.array([[1.0], [2.0]]), name="col")
    b = ConstantNode(np.array([[3.0, 4.0, 5.0]]), name="row")
    out = mo.SumNode(mo.MulNode(a, b))
    g = _build(out)
    res = GradientTape(g).gradient("out", sources=[a, b])
    assert res[a].shape == (2, 1) and np.allclose(res[a], [[12.0], [12.0]])
    assert res[b].shape == (1, 3) and np.allclose(res[b], [[3.0, 3.0, 3.0]])


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1)])
def test_sum_and_mean_with_axis(axis):
    x = ConstantNode(np.arange(6.0).reshape(2, 3) + 1.0, name="x")
    for cls in (mo.SumNode, mo.MeanNode):
        red = cls(mo.MulNode(x, x), axis=axis)
        out = mo.SumNode(mo.MulNode(red, red))
        g = _build(out)
        res = GradientTape(g).gradient("out", sources=[x])
        # reference by a full jet over the 6 elements
        vj = variable_vector(x.value.ravel()).reshape(2, 3)
        f = np.sum if cls is mo.SumNode else np.mean
        r = f(vj * vj, axis=axis)
        ref = np.sum(r * r).grad.reshape(2, 3)
        assert np.allclose(res[x], ref, rtol=1e-13, atol=1e-13)


# ── authored VJPs against local jets ─────────────────────────────────────────

_RNG = np.random.default_rng(7)
_P = _RNG.uniform(0.3, 2.0, (3, 4))
_Q = _RNG.uniform(0.3, 2.0, (3, 4))
_R = _RNG.uniform(-1.5, 1.5, (3, 4))

UNARY_NODES = [
    mo.NegNode,
    mo.ExpNode,
    mo.LogNode,
    mo.SqrtNode,
    mo.AbsNode,
    mo.SinNode,
    mo.CosNode,
    mo.SinhNode,
    mo.CoshNode,
    mo.TanhNode,
    mo.ErfNode,
    mo.GammaNode,
    mo.TransposeNode,
]
BINARY_NODES = [mo.AddNode, mo.SubNode, mo.MulNode, mo.DivNode, mo.PowNode]


def _audit(node, vals, ct):
    authored = node.vjp_fn(ct, *vals)
    ref = _local_jet_vjp(node, vals, node.forward(*vals), ct, list(range(len(vals))))
    for k in range(len(vals)):
        a = _unbroadcast(authored[k], np.shape(vals[k]), False)  # as the tape does
        assert np.allclose(a, ref[k], rtol=1e-12, atol=1e-12), (node.name, k)


@pytest.mark.parametrize("cls", UNARY_NODES, ids=lambda c: c.__name__)
def test_unary_vjps_match_jets(cls):
    x = ConstantNode(0.0)
    node = cls(x)
    vals = [_P if cls not in (mo.AbsNode, mo.SinNode) else _R]
    ct = _RNG.normal(size=np.shape(node.forward(*vals)))
    _audit(node, vals, ct)


@pytest.mark.parametrize("cls", BINARY_NODES, ids=lambda c: c.__name__)
def test_binary_vjps_match_jets(cls):
    node = cls(ConstantNode(0.0), ConstantNode(0.0))
    ct = _RNG.normal(size=_P.shape)
    _audit(node, [_P, _Q], ct)


def test_max_min_vjps_off_ties():
    for cls in (mo.MaximumNode, mo.MinimumNode):
        node = cls(ConstantNode(0.0), ConstantNode(0.0))
        _audit(node, [_P, _Q], _RNG.normal(size=_P.shape))


def test_normal_nodes():
    for cls in (mo.NormalCDFNode, mo.NormalPDFNode):
        node = cls(ConstantNode(0.0))
        h = 1e-6
        ct = _RNG.normal(size=_R.shape)
        (g,) = node.vjp_fn(ct, _R)
        fd = ct * (node.forward(_R + h) - node.forward(_R - h)) / (2 * h)
        assert np.allclose(g, fd, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "sa,sb", [((3, 4), (4, 2)), ((4,), (4, 2)), ((3, 4), (4,)), ((4,), (4,)), ((2, 3, 4), (4, 5))]
)
def test_matmul_vjp_all_ranks(sa, sb):
    a, b = _RNG.normal(size=sa), _RNG.normal(size=sb)
    node = mo.MatMulNode(ConstantNode(0.0), ConstantNode(0.0))
    ct = _RNG.normal(size=np.shape(np.matmul(a, b)))
    _audit(node, [a, b], ct)


def test_cholesky_vjp_against_differences():
    B = _RNG.normal(size=(4, 4))
    A = B @ B.T + 4 * np.eye(4)
    node = mo.CholeskyNode(ConstantNode(0.0))
    Lb = np.tril(_RNG.normal(size=(4, 4)))
    (got,) = node.vjp_fn(Lb, A)
    ref = np.zeros((4, 4))
    h = 1e-6
    for i in range(4):
        for j in range(4):
            E = np.zeros((4, 4))
            E[i, j] = h
            ref[i, j] = np.sum(Lb * (np.linalg.cholesky(A + E) - np.linalg.cholesky(A - E))) / (
                2 * h
            )
    assert np.allclose(got, ref, rtol=1e-7, atol=1e-8)
    assert np.allclose(np.triu(got, 1), 0.0)
    # batched
    (gb,) = node.vjp_fn(np.stack([Lb, 2 * Lb]), np.stack([A, A]))
    assert np.allclose(gb[1], 2 * got, rtol=1e-12)


def test_pow_vjp_at_zero_and_negative_base():
    node = mo.PowNode(ConstantNode(0.0), ConstantNode(0.0))
    da, db = node.vjp_fn(np.ones(3), np.array([0.0, -2.0, 4.0]), np.array(2.0))
    assert np.allclose(da, [0.0, -4.0, 8.0])
    assert db[0] == 0.0 and np.isnan(db[1]) and np.isclose(db[2], 16 * np.log(4.0))
    # a**2 with a negative base and a constant exponent differentiates cleanly
    a = ParameterNode(-3.0, "a")
    g = _build(mo.PowNode(a, ConstantNode(2.0)))
    assert GradientTape(g).gradient("out", sources=[a])[a] == -6.0


def test_floored_log_and_sqrt_derivatives_match_what_is_computed():
    log = mo.LogNode(ConstantNode(0.0))
    assert np.array_equal(log.vjp_fn(np.ones(2), np.array([0.0, 2.0]))[0], [0.0, 0.5])
    sq = mo.SqrtNode(ConstantNode(0.0))
    d = sq.vjp_fn(np.ones(3), np.array([-1.0, 0.0, 4.0]))[0]
    assert d[0] == 0.0 and np.isinf(d[1]) and d[2] == 0.25


# ── complex values (JAX cotangent convention) ────────────────────────────────


def test_complex_intermediate_real_target():
    u = np.linspace(-2.0, 2.0, 5)
    a = ParameterNode(0.4, "a")
    z = mo.ExpNode(mo.MulNode(a, ConstantNode(1j * u)))  # authored holomorphic vjp
    re = OperatorNode(np.real, [z], name="re")  # local jet
    out = mo.SumNode(re)
    g = _build(out)
    res = GradientTape(g).gradient("out", sources=[a])
    assert np.isclose(res[a], np.sum(-u * np.sin(0.4 * u)), rtol=1e-14)
    assert isinstance(res[a], float) and res.exactness == EXACT


def test_complex_target_gives_complex_derivative_for_real_source():
    u = np.linspace(-2.0, 2.0, 5)
    a = ParameterNode(0.4, "a")
    z = OperatorNode(lambda x: np.sum(np.exp(1j * u * x) / (1.0 + x * x)), [a], name="phi")
    g = _build(z)
    res = GradientTape(g).gradient("out", sources=[a])

    def f(x):
        return np.sum(np.exp(1j * u * x) / (1.0 + x * x))

    ref = (f(0.4 + 1e-7) - f(0.4 - 1e-7)) / 2e-7
    assert abs(res[a] - ref) < 1e-7
    J = GradientTape(g).jacobian(["out"], [a])
    assert np.allclose(J[0, 0], res[a], rtol=1e-13)


def test_non_holomorphic_abs_and_conj():
    a = ParameterNode(0.3, "a")
    b = ParameterNode(-0.8, "b")
    z = OperatorNode(lambda x, y: x + 1j * y * y, [a, b], name="z")
    w = OperatorNode(lambda q: np.abs(q) ** 2 + np.real(np.conj(q) * 1j), [z], name="w")
    g = _build(w)
    res = GradientTape(g).gradient("out", sources=[a, b])
    # w = a^2 + b^4 + b^2
    assert np.isclose(res[a], 2 * 0.3, rtol=1e-14)
    assert np.isclose(res[b], 4 * (-0.8) ** 3 + 2 * (-0.8), rtol=1e-14)


def test_complex_source_cotangent_convention():
    # L = Re(z^2) with z a fed complex source: cotangent = dL/dx - i dL/dy = 2 z.
    z = ConstantNode(0.0, name="z")
    sq = OperatorNode(lambda q: np.real(q * q), [z], name="sq")
    g = _build(sq)
    z0 = 0.6 + 0.9j
    res = GradientTape(g).gradient("out", sources=[z], feed_dict={z: z0})
    assert np.isclose(res[z], 2 * z0, rtol=1e-14)


# ── feed_dict, subgraphs, losses ──────────────────────────────────────────────


def test_fed_nodes_are_leaves():
    a = ParameterNode(2.0, "a")
    mid = mo.MulNode(a, a)
    out = mo.MulNode(mid, ConstantNode(3.0))
    g = _build(out)
    tape = GradientTape(g)
    assert tape.gradient("out", sources=[a])[a] == 12.0
    # feeding mid cuts the dependence on a
    assert tape.gradient("out", sources=[a], feed_dict={mid: 5.0})[a] == 0.0
    # mid fed and differentiated: d out / d mid = 3
    assert tape.gradient("out", sources=[mid], feed_dict={mid: 5.0})[mid] == 3.0
    # feeding a source moves the evaluation point
    assert tape.gradient("out", sources=[a], feed_dict={a: 1.5})[a] == 9.0


def test_subgraph_node_gradient():
    x_in = ParameterNode(1.0, "x_in")
    inner_out = mo.ExpNode(mo.MulNode(x_in, x_in))
    inner = _build(inner_out)
    p = ParameterNode(0.7, "p")
    sub = SubgraphNode(inner, [p], [x_in], "out")
    g = _build(mo.MulNode(sub, ConstantNode(2.0)))
    res = GradientTape(g).gradient("out", sources=[p])
    assert np.isclose(res[p], 2 * 2 * 0.7 * np.exp(0.49), rtol=1e-14)
    assert res.exactness == EXACT
    assert x_in.value == 1.0  # the inner graph is not mutated


def test_weighted_mse_gradient_includes_weights():
    m = ConstantNode(np.array([1.0, 2.0, 4.0]), name="m")
    q = ConstantNode(np.array([1.5, 1.0, 3.0]), name="q")
    w = ConstantNode(np.array([1.0, 2.0, 0.5]), name="w")
    node = MSELossNode(m, q, w)
    g = _build(node)
    res = GradientTape(g).gradient("out", sources=[m, q, w])
    for src in (m, q, w):
        h = 1e-6
        ref = np.zeros(3)
        for i in range(3):
            fd = {}
            for sgn in (1, -1):
                v = src.value.copy()
                v[i] += sgn * h
                fd[sgn] = g.evaluate(feed_dict={src: v})["out"]
            ref[i] = (fd[1] - fd[-1]) / (2 * h)
        assert np.allclose(res[src], ref, rtol=1e-7, atol=1e-9)


# ── Jacobians, JVPs, Hessians ─────────────────────────────────────────────────


def _model_graph():
    a = ParameterNode(0.6, "a")
    b = ParameterNode(1.3, "b")
    x = ConstantNode(np.linspace(0.0, 1.0, 4))
    y = mo.ExpNode(mo.MulNode(mo.MulNode(a, x), mo.SinNode(b)))
    loss = mo.SumNode(mo.MulNode(y, y))
    g = ComputationalGraph()
    _add_all(g, loss)
    _add_all(g, y)
    g.add_output(loss, "loss")
    g.add_output(y, "y")
    return g, a, b, x


def _y(a, b, x):
    return np.exp(a * x * np.sin(b))


def test_jacobian_forward_and_reverse_agree_with_closed_form():
    g, a, b, x = _model_graph()
    tape = GradientTape(g)
    J = tape.jacobian(["y", "loss"], [a, b])
    xv = x.value
    y = _y(0.6, 1.3, xv)
    dy_da = y * xv * np.sin(1.3)
    dy_db = y * 0.6 * xv * np.cos(1.3)
    ref = np.vstack([np.c_[dy_da, dy_db], [np.sum(2 * y * dy_da), np.sum(2 * y * dy_db)]])
    assert J.shape == (5, 2)
    assert np.allclose(J, ref, rtol=1e-14, atol=1e-15)
    Jr, level, _ = tape._reverse_jacobian(["y", "loss"], [a, b], None)
    assert np.allclose(Jr, ref, rtol=1e-14, atol=1e-15) and level == EXACT


def test_jvp_matches_jacobian():
    g, a, b, _ = _model_graph()
    J = GradientTape(g).jacobian(["y"], [a, b])
    y, dy, level = jvp(g, "y", {a: 0.25, b: -2.0})
    assert np.allclose(dy, J @ np.array([0.25, -2.0]), rtol=1e-14)
    assert level == EXACT


def test_exact_hessian_and_legacy_signature():
    g, a, b, x = _model_graph()
    H, names = compute_graph_hessian(g, "loss", [a, b])
    assert names == ["a", "b"]
    xv = x.value

    def loss(t):
        return np.sum(_y(t[0], t[1], xv) ** 2)

    _, _, H_ref = jet_hessian(lambda t: np.sum(np.exp(t[0] * xv * np.sin(t[1])) ** 2), [0.6, 1.3])
    assert np.allclose(H, H_ref, rtol=1e-13, atol=1e-14)
    res = graph_hessian(g, "loss", [a, b])
    assert res.exactness == EXACT
    assert np.isclose(res.value, loss([0.6, 1.3]))


def test_hessian_falls_back_honestly():
    a = ParameterNode(0.6, "a")
    b = ParameterNode(1.3, "b")

    def opaque(x, y):
        import math

        return math.exp(float(x) * float(y)) + float(x) ** 3

    node = OperatorNode(opaque, [a, b], name="opaque")
    g = _build(node)
    res = graph_hessian(g, "out", [a, b])
    e = np.exp(0.6 * 1.3)
    ref = np.array([[1.3**2 * e + 6 * 0.6, e * (1 + 0.78)], [e * (1 + 0.78), 0.6**2 * e]])
    assert np.allclose(res.hessian, ref, rtol=1e-6)
    assert res.exactness == NUMERICAL and res.notes
    with pytest.raises(InexactGradientError):
        graph_hessian(g, "out", [a, b], strict=True)
    # parameter bounds cannot clip the perturbation (the graph is never mutated)
    assert a.value == 0.6 and b.value == 1.3


def test_value_and_grad_types():
    g, a, b, _ = _model_graph()
    val, grads = value_and_grad(g, "loss", ["a", "b", "missing"])
    assert isinstance(val, float) and set(grads) == {"a", "b"}
    assert all(isinstance(v, float) for v in grads.values())


def test_gradient_result_is_a_dict():
    g, a, b, _ = _model_graph()
    res = GradientTape(g).gradient("loss")
    assert set(res) == {a, b}
    assert res.get(a, 0.0) == res[a]
    assert set(res.by_name()) == {"a", "b"}
    assert "EXACT" in repr(res)


def test_random_graphs_reverse_equals_forward():
    rng = np.random.default_rng(11)
    unary = [mo.SinNode, mo.CosNode, mo.TanhNode, mo.ExpNode, mo.ErfNode]
    binary = [mo.AddNode, mo.SubNode, mo.MulNode]
    for _ in range(40):
        params = [ParameterNode(float(rng.uniform(-1, 1)), f"p{i}") for i in range(3)]
        pool = list(params) + [ConstantNode(rng.uniform(-1, 1, 3))]
        for _ in range(8):
            if rng.random() < 0.5:
                pool.append(
                    unary[int(rng.integers(len(unary)))](pool[int(rng.integers(len(pool)))])
                )
            else:
                i, j = rng.integers(len(pool), size=2)
                pool.append(binary[int(rng.integers(len(binary)))](pool[int(i)], pool[int(j)]))
        out = mo.SumNode(pool[-1])
        g = _build(out)
        tape = GradientTape(g)
        rev = tape.gradient("out", sources=params)
        J = tape.jacobian(["out"], params)
        assert np.allclose([rev[p] for p in params], J[0], rtol=1e-13, atol=1e-14)
        H, _ = compute_graph_hessian(g, "out", params)
        assert np.allclose(H, H.T, atol=1e-13)
        assert isinstance(J, np.ndarray)
        assert all(isinstance(rev[p], float) for p in params)
