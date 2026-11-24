"""spxa.grad / value_and_grad / jacobian / hessian and the derivative stories."""

from __future__ import annotations

import numpy as np
import pytest

import spxa
from spxa.graph.autodiff.api import EXACTNESS, grad, hessian, jacobian, value_and_grad
from spxa.graph.autodiff.exactness import GradientExactness
from spxa.graph.autodiff.fourier import price_and_gradient
from spxa.graph.autodiff.jet import JetError
from spxa.zoo.levy import VarianceGamma


def f(x):  # type: ignore[no-untyped-def]
    return np.sum(np.sin(x) * x**2) + np.prod(x[:2])


X = np.array([0.3, 1.2, -0.7])


def test_grad_and_hessian_closed_form() -> None:
    g = np.cos(X) * X**2 + 2 * X * np.sin(X) + np.array([X[1], X[0], 0.0])
    np.testing.assert_allclose(grad(f)(X), g, rtol=1e-15, atol=1e-16)
    v, g2 = value_and_grad(f)(X)
    assert v == pytest.approx(f(X), rel=1e-15)
    np.testing.assert_array_equal(g, g2)
    d2 = -np.sin(X) * X**2 + 4 * X * np.cos(X) + 2 * np.sin(X)
    H = np.diag(d2)
    H[0, 1] = H[1, 0] = 1.0
    np.testing.assert_allclose(hessian(f)(X), H, rtol=1e-14, atol=1e-15)


def test_shapes_scalars_and_constants() -> None:
    assert grad(lambda t: t**3)(2.0) == pytest.approx(12.0)
    assert hessian(lambda t: t**3)(2.0) == pytest.approx(12.0)
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert grad(lambda m: np.sum(m**2))(M).shape == (2, 2)
    assert hessian(lambda m: np.sum(m**2))(M).shape == (2, 2, 2, 2)
    J = jacobian(lambda v: v[:, None] * v[None, :])(np.array([1.0, 2.0]))
    assert J.shape == (2, 2, 2)
    np.testing.assert_array_equal(J[0, 1], [2.0, 1.0])
    np.testing.assert_array_equal(grad(lambda t: 3.0)(np.ones(2)), [0.0, 0.0])


def test_refuses_to_lose_derivatives_and_bad_outputs() -> None:
    with pytest.raises(JetError):
        grad(lambda t: float(t) * 2)(1.0)
    with pytest.raises(ValueError, match="scalar"):
        grad(lambda v: v * 2)(np.ones(2))
    assert EXACTNESS is GradientExactness.EXACT


def test_top_level_names_and_process_gradient() -> None:
    assert spxa.grad is grad and spxa.jacobian is jacobian and spxa.hessian is hessian
    vg = spxa.DiffProcess(VarianceGamma(0.2, 0.3, -0.1))

    def exkurt(th):  # type: ignore[no-untyped-def]
        k = vg.cumulants(4, th)
        return k[3] / k[1] ** 2

    g = spxa.grad(exkurt)(vg.values)
    h = 1e-5
    for i in range(3):
        e = np.eye(3)[i] * h
        fd = (exkurt(vg.values + e) - exkurt(vg.values - e)) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-8)


def test_pricing_story_names_the_interval_route() -> None:
    vg = spxa.DiffProcess(VarianceGamma(0.2, 0.3, -0.1))
    a = price_and_gradient(vg, 100.0, np.array([100.0]), 0.5)
    assert "differentiated with the cumulants" in a.__story__()
    b = price_and_gradient(vg, 100.0, np.array([100.0]), 0.5, track_interval=False)
    assert "frozen" in b.__story__()
    c = price_and_gradient(vg, 100.0, np.array([100.0]), 0.5, method="lewis")
    assert "nodes fixed" in c.__story__()
    assert "EXACT" in vg.__story__()


def test_prod_rule_with_zero_factors_and_axes() -> None:
    x = np.array([[1.0, 2.0, 0.0], [3.0, -1.0, 2.0]])
    g = grad(lambda m: np.prod(m))(x)
    expect = np.zeros_like(x)
    expect[0, 2] = 1.0 * 2.0 * 3.0 * -1.0 * 2.0
    np.testing.assert_array_equal(g, expect)
    J = jacobian(lambda m: np.prod(m, axis=0))(x)
    np.testing.assert_array_equal(J[1], [[0.0, -1.0, 0.0], [0.0, 2.0, 0.0]])
    assert jacobian(lambda m: np.prod(m, axis=1, keepdims=True))(x).shape == (2, 1, 2, 3)
    H = hessian(lambda v: np.prod(v))(np.array([2.0, 3.0, 5.0]))
    np.testing.assert_array_equal(H, [[0, 5, 3], [5, 0, 2], [3, 2, 0]])
