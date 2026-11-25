"""Tests for symbolic differentiation of the compile IR."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from spxa.compile import ir as I
from spxa.compile.diff import derivative, gradient
from spxa.compile.evaluate import evaluate
from spxa.compile.ir import CompileError

u = I.U
a = I.param(0, "a")
b = I.param(1, "b")
THETA = [0.7, 1.9]
UU = np.array([-3.0, -0.4, 0.1, 0.3, 2.5]) + 0.05j  # off the branch cuts

#: Expressions covering every differentiable op, in terms of a, b and u.
CASES: dict = {
    "add_mul": lambda: a * u + b * a,
    "div": lambda: (a + u) / (b - 1j * u),
    "pow_const": lambda: (b + u * u) ** 2.0 + (a + u) ** 0.5,
    "pow_complex_const": lambda: (b - 1j * u) ** I.const(1.5 + 0.25j),
    "pow_param": lambda: (b + 1j * u) ** a,
    "pow_param_base": lambda: b ** (a * u),
    "exp_expm1": lambda: I.exp(a * u) + I.expm1(1j * b * u),
    "log_log1p": lambda: I.log(b - 1j * u) + I.log1p(a * u * u),
    "sqrt": lambda: I.sqrt(a * a - (b + 1j * u) ** 2.0),
    "trig": lambda: I.sin(a * u) * I.cos(b) + I.tan(0.3 * a * u),
    "sinc": lambda: I.sinc(a * u + b),
    "sinc_d1": lambda: I.sinc_d1(a * u - b) + I.sinc_d1(a * I.real(u)),
    "gamma_family": lambda: I.gamma(-a) * u + I.digamma(b) + I.polygamma(2, a + b),
    "kve": lambda: I.kve(a - 0.2, b * (2.0 - 1j * u)),
    "where": lambda: I.where(I.ge(a, 0.5), a * u, b * u * u),
    "real_imag_abs": lambda: I.real(a * u) + I.imag(b * u) + I.abs_(a - 1j * b),
    "abs_real": lambda: I.abs_(a - b) * u,
    "neg": lambda: -(a * I.exp(-b * u)),
}


def _num_dparam(e: I.Expr, k: int, h: float = 1e-6) -> np.ndarray:
    tp = list(THETA)
    tm = list(THETA)
    tp[k] += h
    tm[k] -= h
    return (evaluate(e, UU, tp) - evaluate(e, UU, tm)) / (2 * h)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("k", [0, 1])
def test_parameter_derivative_matches_central_differences(name: str, k: int) -> None:
    e = CASES[name]()
    d = derivative(e, I.param(k, "ab"[k]))
    got = np.broadcast_to(evaluate(d, UU, THETA), UU.shape)
    ref = _num_dparam(e, k)
    np.testing.assert_allclose(got, ref, rtol=2e-7, atol=2e-7)


ANALYTIC = [n for n in sorted(CASES) if n not in ("real_imag_abs", "sinc_d1")]


@pytest.mark.parametrize("name", ANALYTIC)
def test_frequency_derivative_matches_complex_differences(name: str) -> None:
    e = CASES[name]()
    d = derivative(e, u)
    h = 1e-6
    got = np.broadcast_to(evaluate(d, UU, THETA), UU.shape)
    # an analytic function has the same derivative along both axes
    for step in (h, 1j * h):
        ref = (evaluate(e, UU + step, THETA) - evaluate(e, UU - step, THETA)) / (2 * step)
        np.testing.assert_allclose(got, ref, rtol=2e-7, atol=2e-7)


def test_second_derivatives_compose() -> None:
    e = a * I.log(1.0 - 1j * u / b)
    d2 = derivative(derivative(e, b), b)
    h = 1e-4
    vals = [evaluate(e, UU, [THETA[0], THETA[1] + s * h]) for s in (-1, 0, 1)]
    ref = (vals[2] - 2 * vals[1] + vals[0]) / h**2
    np.testing.assert_allclose(evaluate(d2, UU, THETA), ref, rtol=1e-6, atol=1e-6)


def test_sinc_second_derivative_at_zero() -> None:
    # d/du sinc'(a u) = a sinc''(a u), and sinc''(0) = -1/3
    d2 = derivative(I.sinc_d1(a * u), u)
    np.testing.assert_allclose(evaluate(d2, np.array([0.0, 1e-9]), [2.0]), [-2.0 / 3.0] * 2)


def test_derivative_shares_the_value_nodes() -> None:
    s = I.sqrt(1.0 - 2j * u * a / b)
    e = (b / a) * (1.0 - s)
    da = derivative(e, a)
    both = I.postorder([e, da])
    assert sum(1 for n in both if n.op == "sqrt") == 1


def test_independent_parameters_vanish() -> None:
    e = I.exp(a * u)
    assert derivative(e, b).is_const_equal(0)
    assert derivative(I.ge(a, 0.0), a).is_const_equal(0)


def test_sign_has_zero_parameter_derivative() -> None:
    assert derivative(I.sign(a - b), a).is_const_equal(0)


def test_sequences_and_gradient() -> None:
    e1, e2 = a * u, b * u
    d = derivative([e1, e2], a)
    assert isinstance(d, list) and d[0] is u and d[1].is_const_equal(0)
    g = gradient(a * b * u, [a, b])
    assert g[0] is b * u and g[1] is a * u


def test_piecewise_constant_guards_need_no_derivative() -> None:
    # Re(u) has no complex derivative, but a guard on it is locally constant
    e = I.where(I.ge(I.real(u), 0.0), u * u, -(u * u))
    d = derivative(e, u)
    z = np.array([-1.5 + 0.2j, 0.7 - 0.1j])
    got = np.asarray(evaluate(d, z, []), dtype=complex)
    np.testing.assert_allclose(got, np.where(z.real >= 0, 2 * z, -2 * z), rtol=1e-15)
    # used outside a guard it still raises
    with pytest.raises(CompileError, match="not analytic"):
        derivative(I.where(I.ge(I.real(u), 0.0), I.real(u), u), u)


def test_errors() -> None:
    with pytest.raises(CompileError):
        derivative(a * u, a * u)
    with pytest.raises(CompileError):
        derivative(I.real(u) * a, u)
    with pytest.raises(CompileError):
        derivative(I.abs_(u), u)

    class Opaque:
        pass

    c = I.call(Opaque(), a * u)
    with pytest.raises(CompileError, match="opaque"):
        derivative(c, a)
    # a call whose argument does not depend on the parameter has derivative zero
    assert derivative(I.call(Opaque(), u), a).is_const_equal(0)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: I.exp(x),
        lambda x: I.log(x),
        lambda x: I.sqrt(x),
        lambda x: I.expm1(x),
    ],
)
def test_derivative_expression_stays_small(fn: Callable[[I.Expr], I.Expr]) -> None:
    e = fn(a * u + b)
    d = derivative(e, a)
    assert I.count_nodes([e, d]) <= I.count_nodes(e) + 6
