"""Stress tests for forward-mode jets (spxa.graph.autodiff.jet).

References used, from strongest to weakest:

* complex step, f'(x) = Im f(x + ih)/h with h = 1e-30, exact to rounding for
  real-analytic f; applied to a first-order jet it gives the exact Hessian;
* mpmath at 30 digits (Bessel-K order derivatives);
* fourth-order central differences (non-analytic operations such as abs,
  maximum, clip, interp), with tolerances set by their truncation error.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.special as sp

from spxa.graph.autodiff.jet import (
    Jet,
    JetError,
    constant_like,
    is_jet,
    jet_hessian,
    jet_jacobian,
    kv_order_derivative,
    stack,
    value_of,
    variable_vector,
    variables,
)
from spxa.graph.autodiff.jet import _kv_order_derivatives

H_CS = 1e-30


def _fd(f, x, h=1e-5):
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / (1.0 + np.abs(b))))


# ── elementwise rules: value, first and second derivative ────────────────────

# (function, domain, analytic in x so complex step applies)
UNARY = {
    "exp": (np.exp, (-2, 2), True),
    "expm1": (np.expm1, (-2, 2), True),
    "exp2": (np.exp2, (-2, 2), True),
    "log": (np.log, (0.2, 3), True),
    "log1p": (np.log1p, (-0.5, 3), True),
    "log2": (np.log2, (0.2, 3), True),
    "log10": (np.log10, (0.2, 3), True),
    "sqrt": (np.sqrt, (0.2, 3), True),
    "cbrt": (np.cbrt, (0.2, 3), False),
    "square": (np.square, (-2, 2), True),
    "reciprocal": (lambda v: np.reciprocal(v) if is_jet(v) else 1.0 / v, (0.3, 3), True),
    "sin": (np.sin, (-3, 3), True),
    "cos": (np.cos, (-3, 3), True),
    "tan": (np.tan, (-1, 1), True),
    "arcsin": (np.arcsin, (-0.9, 0.9), True),
    "arccos": (np.arccos, (-0.9, 0.9), True),
    "arctan": (np.arctan, (-3, 3), True),
    "sinh": (np.sinh, (-2, 2), True),
    "cosh": (np.cosh, (-2, 2), True),
    "tanh": (np.tanh, (-2, 2), True),
    "arcsinh": (np.arcsinh, (-2, 2), True),
    "arccosh": (np.arccosh, (1.2, 3), True),
    "arctanh": (np.arctanh, (-0.9, 0.9), True),
    "abs": (np.abs, (-3, 3), False),
    "pow2.5": (lambda v: v**2.5, (0.2, 3), True),
    "pow_int3": (lambda v: v**3, (-2, 2), True),
    "rpow": (lambda v: 2.3**v, (-2, 2), True),
    "gamma": (sp.gamma, (0.3, 4), True),
    "rgamma": (sp.rgamma, (0.3, 4), True),
    "gammaln": (sp.gammaln, (0.3, 4), False),
    "loggamma": (sp.loggamma, (0.3, 4), True),
    "digamma": (sp.digamma, (0.3, 4), False),
    "erf": (sp.erf, (-2, 2), True),
    "erfc": (sp.erfc, (-2, 2), True),
    "erfcx": (sp.erfcx, (-2, 3), False),
    "ndtr": (sp.ndtr, (-3, 3), False),
    "log_ndtr": (sp.log_ndtr, (-8, 3), False),
    "ndtri": (sp.ndtri, (0.05, 0.95), False),
    "expit": (sp.expit, (-3, 3), False),
    "logit": (sp.logit, (0.1, 0.9), False),
    "exprel": (sp.exprel, (-2, 2), False),
}


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary_rules_value_grad_hess(name):
    f, (lo, hi), analytic = UNARY[name]
    x = np.random.default_rng(abs(hash(name)) % 2**32).uniform(lo, hi, 9)
    if name == "exprel":
        x = np.r_[x, 0.0, 1e-9, -3e-6, 2e-3, -0.02]  # series branch near 0
    (j,) = variables([x], order=2)
    out = f(j)
    assert np.array_equal(out.value, f(x))
    g, hh = out.grad[..., 0], out.hess[..., 0, 0]
    if analytic:
        g_ref = np.imag(f(x + 1j * H_CS)) / H_CS
        assert _rel(g, g_ref) < 1e-13
        # Hessian = complex step applied to the first-order jet derivative
        (jc,) = variables([x + 1j * H_CS], order=1)
        h_ref = np.imag(f(jc).grad[..., 0]) / H_CS
        assert _rel(hh, h_ref) < 1e-12
    else:
        assert _rel(g, _fd(f, x)) < 1e-8
        assert _rel(hh, _fd(lambda y: _fd(f, y, 1e-4), x, 1e-3)) < 1e-5


def test_exprel_near_zero_is_exact():
    # exprel(x) = sum x^k/(k+1)!, so exprel'(0) = 1/2, exprel''(0) = 1/3.
    (j,) = variables([np.array([0.0, 1e-12, -1e-12])], order=2)
    out = sp.exprel(j)
    assert np.allclose(out.grad[..., 0], 0.5, rtol=0, atol=1e-12)
    assert np.allclose(out.hess[..., 0, 0], 1.0 / 3.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "name,f",
    [
        ("exp", np.exp),
        ("log", np.log),
        ("sqrt", np.sqrt),
        ("pow", lambda z: z**0.7),
        ("loggamma", sp.loggamma),
        ("gamma", sp.gamma),
        ("erf", sp.erf),
        ("tanh", np.tanh),
        ("log1p", np.log1p),
    ],
)
def test_complex_intermediates(name, f):
    # z(t) = z0 + c t is complex; d/dt f(z(t)) = f'(z) c.
    z0, c = 1.3 + 0.7j, 0.4 - 0.2j
    (t,) = variables([0.0], order=2)
    out = f(z0 + c * t)
    ref = _fd(lambda s: f(z0 + c * s), 0.0, 1e-4)
    ref2 = _fd(lambda s: _fd(lambda r: f(z0 + c * r), s, 1e-4), 0.0, 1e-3)
    assert abs(out.grad[0] - ref) / (1 + abs(ref)) < 1e-9
    assert abs(out.hess[0, 0] - ref2) / (1 + abs(ref2)) < 1e-5


def test_complex_polygamma_against_mpmath():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    from spxa.graph.autodiff.jet import _polygamma_complex

    for z in [0.3 + 0.2j, 1.7 - 2j, 25 + 3j, -3.3 + 0.4j, -0.5 + 5j, 4j, -20.7 - 0.1j, 100 + 100j]:
        for k in (1, 2):
            ref = complex(mp.psi(k, mp.mpc(z)))
            assert abs(_polygamma_complex(k, np.array(z)) - ref) <= 1e-13 * abs(ref)


# ── multivariate operations against references ───────────────────────────────

U = np.linspace(-3.0, 3.0, 7)
XS = np.linspace(0.0, 1.0, 9)
M = np.random.default_rng(1).normal(size=(3, 7))


def _check_multi(f, theta, tolg=1e-8, tolh=1e-5):
    theta = np.asarray(theta, float)
    n = theta.size
    out = f(*variables(list(theta), order=2))
    assert np.allclose(out.value, f(*theta), rtol=0, atol=0)
    H = np.asarray(out.hess)
    assert np.max(np.abs(H - np.swapaxes(H, -1, -2))) < 1e-12
    for k in range(n):
        ek = np.eye(n)[k]
        gk = _fd(lambda s: np.asarray(f(*(theta + s * ek))), 0.0)
        assert _rel(out.grad[..., k], gk) < tolg
        for m in range(n):
            em = np.eye(n)[m]
            hkm = _fd(
                lambda s: _fd(lambda r: np.asarray(f(*(theta + s * em + r * ek))), 0.0, 1e-4),
                0.0,
                1e-3,
            )
            assert _rel(H[..., k, m], hkm) < tolh


MULTI = {
    "rational": lambda a, b: a * b / (a + b**2),
    "pow_ab": lambda a, b: a**b,
    "char_fn_real": lambda a, b: np.real(np.exp(1j * U * a - 0.5 * b**2 * U**2)),
    "abs_complex": lambda a, b: np.abs(np.log(1 - 1j * U * a + 0.5 * b**2 * U**2)),
    "angle": lambda a, b: np.angle(1 - 1j * U * a + b),
    "conj": lambda a, b: np.imag(np.conj(np.exp(1j * U * a)) * b),
    "arctan2": lambda a, b: np.arctan2(a * U, b + U),
    "hypot": lambda a, b: np.hypot(a * U, b),
    "max_min": lambda a, b: np.maximum(a * U, b) + np.minimum(a, b * U),
    "where": lambda a, b: np.where(U > 0, a * U**2, np.sin(b * U)),
    "sum": lambda a, b: np.sum(np.exp(a * U) * b),
    "sum_axis": lambda a, b: np.sum(np.outer(np.exp(a * U), U * b), axis=1),
    "cumsum": lambda a, b: np.cumsum(np.exp(a * U) * b),
    "mean": lambda a, b: np.mean(np.sin(a * U) * b),
    "max": lambda a, b: np.max(np.sin(a * U) * b),
    "matmul": lambda a, b: M @ np.exp(a * U) * b,
    "matmul_jj": lambda a, b: np.outer(np.exp(a * U[:3]), U[:3] * b) @ np.cos(b * U[:3]),
    "dot": lambda a, b: np.dot(np.exp(a * U), np.cos(b * U)),
    "fft": lambda a, b: np.real(np.fft.fft(np.exp(1j * a * U) * b)),
    "ifft": lambda a, b: np.imag(np.fft.ifft(np.exp(1j * a * U) * b)),
    "interp_fp": lambda a, b: np.interp(np.array([0.13, 0.5, 0.97]), XS, np.exp(a * XS) * b),
    "interp_x": lambda a, b: np.interp(0.31 + 0.1 * a, XS, np.exp(XS) * b),
    "accmax": lambda a, b: np.maximum.accumulate(np.sin(a * U + b)),
    "clip": lambda a, b: np.clip(a * U, -b, b),
    "diff": lambda a, b: np.diff(np.exp(a * U) * b),
    "trapezoid": lambda a, b: np.trapezoid(np.exp(a * U) * b, U),
    "getitem": lambda a, b: (np.exp(a * U) * b)[2:5],
    "fancy_index": lambda a, b: (np.exp(a * U) * b)[[0, 3, 3, 6]],
    "logaddexp": lambda a, b: np.logaddexp(a * U, b),
    "reshape_T": lambda a, b: np.transpose(np.reshape(np.exp(a * np.arange(6.0)) * b, (2, 3))),
    "concat": lambda a, b: np.concatenate([np.sin(a * U), np.cos(b * U)]),
    "flip": lambda a, b: np.flip(np.exp(a * U) * b),
    "kv_z": lambda a, b: sp.kv(1.3, a * U**2 + b),
    "kve_z": lambda a, b: sp.kve(0.4, a * U**2 + b),
    "kv_nu": lambda a, b: sp.kv(a, b * (1 + U**2)),
    "kve_nu_z": lambda a, b: sp.kve(3 * a - 0.5, b * (1 + U**2)),
    "kv_complex": lambda a, b: np.real(sp.kv(a, b * (1 + U**2) + 0.5j * U)),
    "gh_like": lambda a, b: np.imag(np.log(sp.kve(a, b * np.sqrt(4 - (0.3 + 1j * U) ** 2)))),
}


@pytest.mark.parametrize("name", sorted(MULTI))
def test_multivariate_ops(name):
    theta = [0.6, 0.9] if name in ("kv_complex", "gh_like") else [0.3, 0.4]
    _check_multi(MULTI[name], theta)


def test_stack_and_object_array_coercion():
    a, b = variables([0.3, 0.4], order=2)
    s = stack([a * b, np.sin(a), 2.0])
    assert s.shape == (3,)
    assert np.allclose(s.grad, [[0.4, 0.3], [np.cos(0.3), 0.0], [0.0, 0.0]])
    assert np.allclose(s.hess[1], [[-np.sin(0.3), 0.0], [0.0, 0.0]])
    # np.array([jet, jet]) is an object array; jets coerce it back on use
    obj = np.array([a, b * 2.0], dtype=object)
    assert obj.dtype == object and obj.shape == (2,)
    r = a * obj  # any jet rule restacks the object array into one Jet
    assert is_jet(r) and r.shape == (2,)
    assert np.allclose(r.grad, [[2 * 0.3, 0.0], [0.8, 2 * 0.3]])
    assert np.allclose(r.hess[1], [[0.0, 2.0], [2.0, 0.0]])
    # a bare object array never reaches the jet protocol; NumPy refuses it loudly
    with pytest.raises(TypeError):
        np.exp(obj)
    # without dtype=object, building a numeric array from jets is refused
    with pytest.raises(JetError):
        np.array([a, b])


def test_setitem_copies_on_write_and_is_exact():
    (a,) = variables([0.5], order=2)
    base = np.exp(a * np.arange(4.0))
    view = base[1:3]
    other = base.reshape(2, 2)
    snap_v, snap_g = view.value.copy(), view.grad.copy()
    base[1] = a**3
    assert np.array_equal(view.value, snap_v) and np.array_equal(view.grad, snap_g)
    assert np.array_equal(other.value.ravel()[1], np.exp(0.5))
    assert np.isclose(base.grad[1, 0], 3 * 0.25)
    assert np.isclose(base.hess[1, 0, 0], 6 * 0.5)
    base[2] = 7.0  # constant
    assert base.grad[2, 0] == 0.0 and base.hess[2, 0, 0] == 0.0
    base[3] = 1j * a  # promotes to complex
    assert np.iscomplexobj(base.value) and base.grad[3, 0] == 1j


def test_jacobian_and_hessian_helpers_match_closed_form():
    def f(x):
        return np.sum(x**2 * np.arange(1.0, 4.0)) + x[0] * x[1] * x[2]

    x0 = np.array([0.3, -0.7, 1.1])
    v, g, h = jet_hessian(f, x0)
    w = np.arange(1.0, 4.0)
    assert np.isclose(v, f(x0))
    g_ref = 2 * w * x0 + np.array([x0[1] * x0[2], x0[0] * x0[2], x0[0] * x0[1]])
    h_ref = np.diag(2 * w) + np.array([[0, x0[2], x0[1]], [x0[2], 0, x0[0]], [x0[1], x0[0], 0]])
    assert np.allclose(g, g_ref, rtol=0, atol=1e-14)
    assert np.allclose(h, h_ref, rtol=0, atol=1e-14)
    val, J = jet_jacobian(lambda x: np.exp(x) * x[::-1], x0)
    J_ref = np.diag(np.exp(x0) * x0[::-1])
    for i in range(3):
        J_ref[i, 2 - i] += np.exp(x0[i])
    assert np.allclose(J, J_ref, rtol=0, atol=1e-14)
    assert np.allclose(val, np.exp(x0) * x0[::-1])


# ── randomized compositions vs complex step ──────────────────────────────────

_U_OPS = [
    np.sin,
    np.cos,
    np.tanh,
    np.arctan,
    lambda x: np.exp(0.3 * x),
    lambda x: np.log(1.0 + x * x),
    lambda x: np.sqrt(1.0 + x * x),
    lambda x: x / (1.0 + x * x),
    lambda x: sp.erf(0.5 * x),
]
_B_OPS = [np.add, np.subtract, np.multiply, lambda a, b: a / (2.0 + b * b)]


def _random_expr(rng, depth, nvar):
    if depth == 0 or rng.random() < 0.2:
        k = int(rng.integers(nvar))
        c = float(rng.uniform(-1.5, 1.5))
        return lambda xs, k=k, c=c: c * xs[k]
    if rng.random() < 0.5:
        op = _U_OPS[int(rng.integers(len(_U_OPS)))]
        e = _random_expr(rng, depth - 1, nvar)
        return lambda xs, op=op, e=e: op(e(xs))
    op = _B_OPS[int(rng.integers(len(_B_OPS)))]
    e1, e2 = _random_expr(rng, depth - 1, nvar), _random_expr(rng, depth - 1, nvar)
    return lambda xs, op=op, e1=e1, e2=e2: op(e1(xs), e2(xs))


def test_random_compositions_exact_to_rounding():
    rng = np.random.default_rng(20261023)
    nvar = 3
    for _ in range(150):
        expr = _random_expr(rng, 5, nvar)
        x0 = rng.uniform(-1.0, 1.0, nvar)
        out = expr(variables(list(x0), order=2))
        if not is_jet(out):
            continue
        for k in range(nvar):
            xc = x0.astype(complex)
            xc[k] += 1j * H_CS
            g_ref = np.imag(expr(list(xc))) / H_CS
            assert abs(out.grad[k] - g_ref) <= 1e-13 * (1 + abs(g_ref))
            # exact Hessian column: complex step through a first-order jet
            hj = expr(variables(list(xc), order=1))
            h_ref = np.imag(hj.grad) / H_CS
            assert np.allclose(out.hess[k], h_ref, rtol=1e-12, atol=1e-12)


# ── Bessel K order derivatives vs mpmath ─────────────────────────────────────


@pytest.mark.parametrize("nu", [0.0, 0.3, 1.0, 2.5, 7.0, -3.2, 20.0])
def test_kv_order_derivatives_against_mpmath(nu):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    zs = [1e-3, 0.05, 0.7, 3.0, 30.0, 300.0, 2 + 1j, 10 - 5j, 0.5 * np.exp(0.78j)]
    if abs(nu) < 10:
        zs += [0.5 + 0.8j, 0.1 + 0.15j]
    for z in zs:
        d = _kv_order_derivatives(np.array(nu), np.array(z), False, True)

        def k(v, w):
            return mp.besselk(v, w)

        zz = mp.mpc(z)
        r1 = complex(mp.diff(lambda v: k(v, zz), nu))
        r2 = complex(mp.diff(lambda v: k(v, zz), nu, 2))
        r3 = complex(mp.diff(k, (nu, zz), (1, 1)))
        for got, ref in ((d["d_nu"], r1), (d["d_nu2"], r2), (d["d_nu_dz"], r3)):
            assert abs(got - ref) <= 1e-11 * abs(ref), (nu, z)


def test_kv_order_derivative_refuses_cancellation():
    with pytest.raises(JetError, match="cancellation"):
        _kv_order_derivatives(np.array(45.0), np.array(0.5 + 0.8j), False, True)


def test_kv_order_derivative_public_and_symmetric_in_nu():
    # K_{-nu} = K_nu, so d/dnu is odd in nu and vanishes at 0.
    z = np.array([0.4, 2.0, 9.0])
    d_pos = kv_order_derivative(1.7, z)
    d_neg = kv_order_derivative(-1.7, z)
    assert np.allclose(d_pos, -d_neg, rtol=1e-14, atol=0)
    assert np.allclose(kv_order_derivative(0.0, z), 0.0, atol=1e-15)
    # K_{1/2}(z) = sqrt(pi/2z) e^{-z} has d/dnu at 1/2 = sqrt(pi/2z) e^{-z} E1(2z) e^{2z}
    ref = np.sqrt(np.pi / (2 * z)) * np.exp(-z) * sp.exp1(2 * z) * np.exp(2 * z)
    assert np.allclose(kv_order_derivative(0.5, z), ref, rtol=1e-13, atol=0)


# ── guards against silently dropping derivatives ─────────────────────────────


def test_conversions_that_drop_derivatives_raise():
    (a,) = variables([np.array([0.1, 0.2])])
    for bad in (
        lambda: float(a[0]),
        lambda: complex(a[0]),
        lambda: int(a[0]),
        lambda: bool(a[0]),
        lambda: np.asarray(a, dtype=float),
        lambda: np.asarray(a),
        lambda: np.exp(a, out=np.empty(2)),
        lambda: np.exp(a, dtype=float),
        lambda: np.real(1j * a).astype(int),
        lambda: (1j * a).astype(float),
        lambda: np.sort(a),
    ):
        with pytest.raises(JetError):
            bad()
    with pytest.raises((JetError, ValueError)):  # never a silent float() of the jet
        np.zeros(2)[0] = a[0]


def test_mismatched_jets_cannot_combine():
    (a,) = variables([1.0], order=1)
    (b,) = variables([1.0], order=2)
    c, d = variables([1.0, 2.0])
    with pytest.raises(JetError):
        _ = a + b
    with pytest.raises(JetError):
        _ = a + c


def test_comparisons_act_on_values_and_branches_are_exact():
    (a,) = variables([np.array([-1.0, 2.0])], order=2)
    assert np.array_equal(a > 0, [False, True])
    y = np.where(a > 0, a**2, -(a**3))
    assert np.allclose(y.grad[:, 0], [-3.0, 4.0])
    assert np.allclose(y.hess[:, 0, 0], [6.0, 2.0])


def test_constant_like_and_value_of():
    (a,) = variables([np.ones(3)], order=2)
    c = constant_like(np.arange(3.0), a)
    assert c.n == 1 and c.order == 2 and not np.any(c.grad) and not np.any(c.hess)
    assert np.array_equal(value_of(c), np.arange(3.0))
    assert value_of(4.0) == 4.0


def test_variable_vector_and_array_variables():
    x = variable_vector([1.0, 2.0, 3.0], order=2)
    assert x.grad.shape == (3, 3) and np.array_equal(x.grad, np.eye(3))
    (a,) = variables([np.array([1.0, 2.0])])
    # an array variable is c + theta with one scalar theta
    assert np.isclose(np.sum(a**2).grad[0], 2 * (1.0 + 2.0))
    assert isinstance(a, Jet)
