r"""
Interpreters for the IR: a NumPy reference evaluator and a Taylor-mode evaluator.

:func:`evaluate` walks the DAG once with the NumPy routines of
:mod:`spxa.compile._numeric`; the generated kernels must agree with it to
the last bit (the NumPy backend) or to rounding (other backends).

:func:`taylor` evaluates the DAG on truncated Taylor series in :math:`u`
(:class:`~spxa.graph.autodiff.taylor.TSeries`), which gives exact cumulants

.. math::
    \kappa_n = n!\,[z^n]\,\psi(-iz)

of any lowered composition, to any order, batched over parameter sets.
Nodes that do not depend on :math:`u` stay ordinary arrays. Branches
(``where``) whose condition depends on :math:`u` are resolved at the
expansion point, which is exact because the selected branch is analytic in
a neighbourhood of it. ``real``, ``imag``, ``abs`` and ``sign`` of a
:math:`u`-dependent value may only feed comparisons; used as values they are
not analytic, and :class:`~spxa.graph.autodiff.taylor.TaylorError` is raised.
Opaque ``call`` nodes expand through the cumulants of their process,
:math:`\psi^{(n)}(0) = i^n\kappa_n`.
"""

from __future__ import annotations

from math import factorial
from typing import Any, Dict, List, Sequence, Union

import numpy as np

from spxa.compile import ir as I
from spxa.compile._numeric import apply_op
from spxa.compile.ir import CompileError, Expr

__all__ = ["evaluate", "taylor", "cumulants_from_ir", "call_exponent"]


def call_exponent(process: Any, arg: Any) -> np.ndarray:
    """``process._char_exponent`` on an array of any shape (complex)."""
    a = np.asarray(arg, dtype=complex)
    flat = np.atleast_1d(a).ravel()
    out = np.asarray(process._char_exponent(flat), dtype=complex)
    return out.reshape(a.shape)


def evaluate(roots: Union[Expr, Sequence[Expr]], u: Any, params: Sequence[Any]) -> Any:
    """
    Evaluate expressions at frequency ``u`` and parameter values ``params``.

    ``params[k]`` is the value of parameter index ``k`` (a float or an array
    that broadcasts against ``u``). Returns one array per root.
    """
    single = isinstance(roots, Expr)
    rs: List[Expr] = [roots] if single else list(roots)  # type: ignore[list-item]
    vals: Dict[int, Any] = {}
    uu = np.asarray(u, dtype=complex)
    with np.errstate(all="ignore"):
        for node in I.postorder(rs):
            op = node.op
            if op == "const":
                vals[id(node)] = node.payload
            elif op == "param":
                vals[id(node)] = params[node.payload[0]]
            elif op == "var":
                vals[id(node)] = uu
            elif op == "call":
                vals[id(node)] = call_exponent(node.payload, vals[id(node.args[0])])
            else:
                vals[id(node)] = apply_op(op, [vals[id(a)] for a in node.args], node.payload)
    out = [vals[id(r)] for r in rs]
    return out[0] if single else out


# ── Taylor mode ─────────────────────────────────────────────────────────────


class _NonAnalytic:
    """``real``/``imag``/``abs``/``sign`` of a series: usable only in comparisons."""

    __slots__ = ("value", "op")

    def __init__(self, value: Any, op: str) -> None:
        self.value = value
        self.op = op


def _is_series(x: Any) -> bool:
    from spxa.graph.autodiff.taylor import TSeries

    return isinstance(x, TSeries)


def _c0(x: Any) -> Any:
    if _is_series(x):
        return x.coeffs[0]
    if isinstance(x, _NonAnalytic):
        return x.value
    return x


def _lift(x: Any, order: int) -> Any:
    from spxa.graph.autodiff.taylor import TSeries

    if _is_series(x):
        return x
    x = np.asarray(x)[()]
    return TSeries([x] + [0.0 * x] * order)


def _series_tan(a: Any) -> Any:
    return np.sin(a) / np.cos(a)


def _series_sinc(a: Any, order: int) -> Any:
    from spxa.graph.autodiff.taylor import compose

    z0 = a.coeffs[0]
    if np.all(np.asarray(z0) == 0):
        # derivatives of sin z / z at 0: (-1)^k / (2k + 1) at even orders, 0 at odd
        derivs: List[Any] = []
        for n in range(order + 1):
            derivs.append(0.0 if n % 2 else (-1.0) ** (n // 2) / (n + 1))
        return compose(a, derivs)
    if np.any(np.asarray(z0) == 0):
        raise CompileError("sinc series at a mix of zero and non-zero expansion points")
    return np.sin(a) / a


def _series_call(process: Any, a: Any, order: int) -> Any:
    from spxa.graph.autodiff.taylor import TaylorError, compose

    a0 = np.asarray(a.coeffs[0])
    if np.any(np.abs(a0) > 1e-14):
        raise TaylorError(
            f"the opaque exponent of {type(process).__name__} expands only at 0 (through its "
            "cumulants)"
        )
    kappa = process.cumulants(order)
    derivs = [0.0] + [(1j**n) * kappa[n] for n in range(1, order + 1)]
    if not all(np.isfinite(d) for d in derivs):
        raise TaylorError(f"{type(process).__name__} has no finite cumulants up to order {order}")
    return compose(a, derivs)


def _small_int(x: Any) -> bool:
    v = np.asarray(x)
    if v.ndim != 0 or np.iscomplexobj(v) and np.imag(v) != 0:
        return False
    r = float(np.real(v))
    return r.is_integer() and 0 <= r <= 8


def taylor(
    root: Expr, order: int, params: Sequence[Any], u0: Any = 0.0, du: Any = 1.0
) -> List[Any]:
    r"""
    Taylor coefficients of :math:`h \mapsto e(u_0 + du\cdot h)` at :math:`h = 0`.

    Returns ``order + 1`` coefficients, each with the broadcast shape of the
    parameter values. With ``u0 = 0, du = -1j`` the coefficients are those of
    :math:`K(z) = \psi(-iz)`.
    """
    from spxa.graph.autodiff.taylor import TaylorError, TSeries

    if order < 1:
        raise ValueError("order must be >= 1")
    useries = TSeries([np.asarray(u0, dtype=complex)[()], complex(du)] + [0.0] * (order - 1))
    vals: Dict[int, Any] = {}
    with np.errstate(all="ignore"):
        for node in I.postorder([root]):
            op = node.op
            if op == "const":
                vals[id(node)] = node.payload
                continue
            if op == "param":
                vals[id(node)] = params[node.payload[0]]
                continue
            if op == "var":
                vals[id(node)] = useries
                continue
            args = [vals[id(a)] for a in node.args]
            if not node.depends_u:
                vals[id(node)] = apply_op(op, args, node.payload)
                continue
            if any(isinstance(x, _NonAnalytic) for x in args) and op not in ("ge", "gt"):
                raise TaylorError(
                    "real/imag/abs/sign of a u-dependent value is not analytic; "
                    "it can only feed a comparison"
                )
            if op in ("ge", "gt"):
                vals[id(node)] = apply_op(op, [np.real(_c0(x)) for x in args])
            elif op in ("real", "imag", "abs", "sign"):
                x = args[0]
                vals[id(node)] = _NonAnalytic(apply_op(op, [_c0(x)]), op)
            elif op == "where":
                c, a, b = args
                if not (_is_series(a) or _is_series(b)):
                    vals[id(node)] = np.where(c, a, b)
                else:
                    sa, sb = _lift(a, order), _lift(b, order)
                    vals[id(node)] = TSeries(
                        [np.where(c, x, y)[()] for x, y in zip(sa.coeffs, sb.coeffs)]
                    )
            elif op == "tan":
                vals[id(node)] = _series_tan(args[0])
            elif op == "sinc":
                vals[id(node)] = _series_sinc(args[0], order)
            elif op == "call":
                vals[id(node)] = _series_call(node.payload, _lift(args[0], order), order)
            elif op == "pow" and not _is_series(args[1]) and _small_int(args[1]):
                # integer powers by repeated products: exact at a zero constant term,
                # where the general power recurrence divides by it
                vals[id(node)] = args[0] ** int(np.real(args[1]))
            elif op in ("gamma", "digamma", "polygamma", "sinc_d1", "kve_dnu"):
                raise TaylorError(f"no Taylor rule for {op} of a u-dependent argument")
            else:
                vals[id(node)] = apply_op(op, args, node.payload)
    out = vals[id(root)]
    if not _is_series(out):
        out = _lift(out, order)
    return [np.asarray(c)[()] for c in out.coeffs]


def cumulants_from_ir(root: Expr, order: int, params: Sequence[Any]) -> np.ndarray:
    r"""
    :math:`\kappa_1..\kappa_{order}` of :math:`X_1` from its exponent ``root``.

    Shape ``(order,)`` for scalar parameters, ``batch + (order,)`` for arrays.
    """
    coeffs = taylor(root, order, params, 0.0, -1j)
    ks = [np.real(factorial(n) * np.asarray(coeffs[n])) for n in range(1, order + 1)]
    return np.stack(np.broadcast_arrays(*ks), axis=-1)
