r"""
Truncated Taylor series in one complex variable, with jet-valued coefficients.

A :class:`TSeries` holds :math:`f(z_0 + h) = \sum_{k=0}^{N} c_k h^k + O(h^{N+1})`.
Arithmetic and the elementary functions act on the coefficients by the
standard recurrences (Griewank & Walther, *Evaluating Derivatives*, ch. 13),
so the coefficients are exact up to rounding. Coefficients may be
:class:`~spxa.graph.autodiff.jet.Jet` objects, which makes the derivatives of
every Taylor coefficient with respect to parameters exact as well. This is
how :mod:`spxa.graph.autodiff.process` obtains cumulants
:math:`\kappa_n = n!\,[z^n]K(z)` and their parameter gradients in closed form
for any composition of Lévy primitives.

Supported: ``+ - * /``, ``**`` (constant or jet exponent; series exponents
via ``exp(b log a)``), ``np.exp``, ``np.log``, ``np.sqrt``, ``np.cosh``,
``np.sinh``, ``np.cos``, ``np.sin``, and ``scipy.special.kve`` / ``kv`` in
the argument. Logarithms and powers use the principal branch at
:math:`z_0` and its analytic continuation, which is what the closed forms in
:mod:`spxa.zoo` evaluate.
"""

from __future__ import annotations

from math import factorial
from typing import Any, Callable, List, Sequence

import numpy as np
import scipy.special as sps

__all__ = ["TSeries", "TaylorError", "taylor_variable"]


class TaylorError(TypeError):
    """An operation is not defined on truncated Taylor series."""


class TSeries:
    """Truncated Taylor series; ``coeffs[k]`` multiplies ``h**k``."""

    __slots__ = ("coeffs",)
    __array_priority__ = 2000.0
    __taylor__ = True  # jets defer to series (see Jet.__array_ufunc__)
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, coeffs: Sequence[Any]) -> None:
        if len(coeffs) == 0:
            raise TaylorError("a series needs at least one coefficient")
        self.coeffs: List[Any] = list(coeffs)

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    def _lift(self, x: Any) -> TSeries:
        if isinstance(x, TSeries):
            if x.order != self.order:
                raise TaylorError("series of different orders")
            return x
        return TSeries([x] + [0.0 * x] * self.order)

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, o: Any) -> TSeries:
        o = self._lift(o)
        return TSeries([a + b for a, b in zip(self.coeffs, o.coeffs)])

    __radd__ = __add__

    def __neg__(self) -> TSeries:
        return TSeries([-a for a in self.coeffs])

    def __sub__(self, o: Any) -> TSeries:
        return self + (-self._lift(o))

    def __rsub__(self, o: Any) -> TSeries:
        return self._lift(o) - self

    def __mul__(self, o: Any) -> TSeries:
        if not isinstance(o, TSeries):
            return TSeries([a * o for a in self.coeffs])
        o = self._lift(o)
        a, b = self.coeffs, o.coeffs
        return TSeries([sum(a[j] * b[k - j] for j in range(k + 1)) for k in range(len(a))])

    __rmul__ = __mul__

    def reciprocal(self) -> TSeries:
        a = self.coeffs
        r: List[Any] = [1.0 / a[0]]
        for k in range(1, len(a)):
            r.append(-sum(a[j] * r[k - j] for j in range(1, k + 1)) / a[0])
        return TSeries(r)

    def __truediv__(self, o: Any) -> TSeries:
        if not isinstance(o, TSeries):
            return TSeries([a / o for a in self.coeffs])
        return self * self._lift(o).reciprocal()

    def __rtruediv__(self, o: Any) -> TSeries:
        return self._lift(o) * self.reciprocal()

    def __pow__(self, p: Any) -> TSeries:
        if isinstance(p, TSeries):
            return _exp(p * _log(self))
        if isinstance(p, (int, np.integer)) and 0 <= int(p) <= 8:
            out = self._lift(1.0 + 0.0 * self.coeffs[0])
            for _ in range(int(p)):
                out = out * self
            return out
        return _power_const(self, p)

    def __rpow__(self, b: Any) -> TSeries:
        return _exp(self * np.log(b))

    # -- conversions -----------------------------------------------------------
    def __float__(self) -> float:
        raise TaylorError("float() of a Taylor series")

    def __array__(self, *a: Any, **k: Any) -> np.ndarray:
        raise TaylorError("converting a Taylor series to an array drops its coefficients")

    def __bool__(self) -> bool:
        raise TaylorError("the truth value of a Taylor series is ambiguous")

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__" or kwargs.get("out") not in (None, (None,)):
            raise TaylorError(f"{ufunc.__name__}.{method} is not supported on Taylor series")
        rule = _UFUNCS.get(ufunc)
        if rule is None:
            raise TaylorError(f"no Taylor rule for {ufunc.__name__!r}")
        return rule(*inputs)

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        raise TaylorError(f"no Taylor rule for numpy.{func.__name__}")

    def derivative(self, k: int) -> Any:
        """The ``k``-th derivative at the expansion point, ``k! c_k``."""
        return factorial(k) * self.coeffs[k]


def taylor_variable(z0: Any, order: int) -> TSeries:
    """The identity series ``z0 + h``."""
    if order < 1:
        raise TaylorError("order must be >= 1")
    return TSeries([z0, 1.0] + [0.0] * (order - 1))


# ── elementary functions ──────────────────────────────────────────────────────


def _exp(a: TSeries) -> TSeries:
    c = a.coeffs
    e: List[Any] = [np.exp(c[0])]
    for k in range(1, len(c)):
        e.append(sum(j * c[j] * e[k - j] for j in range(1, k + 1)) / k)
    return TSeries(e)


def _log(a: TSeries) -> TSeries:
    c = a.coeffs
    out: List[Any] = [np.log(c[0])]
    for k in range(1, len(c)):
        s = c[k] - sum(j * out[j] * c[k - j] for j in range(1, k)) / k
        out.append(s / c[0])
    return TSeries(out)


def _power_const(a: TSeries, p: Any) -> TSeries:
    # (a^p)' = p a^p a'/a  =>  a0 y_k = (1/k) sum_{j=1}^k (p j - (k - j)) a_j y_{k-j}
    c = a.coeffs
    y: List[Any] = [c[0] ** p]
    for k in range(1, len(c)):
        s = sum((p * j - (k - j)) * c[j] * y[k - j] for j in range(1, k + 1))
        y.append(s / (k * c[0]))
    return TSeries(y)


def _sqrt(a: TSeries) -> TSeries:
    return _power_const(a, 0.5)


def _cosh(a: TSeries) -> TSeries:
    return 0.5 * (_exp(a) + _exp(-a))


def _sinh(a: TSeries) -> TSeries:
    return 0.5 * (_exp(a) - _exp(-a))


def _cos(a: TSeries) -> TSeries:
    return 0.5 * (_exp(1j * a) + _exp(-1j * a))


def _sin(a: TSeries) -> TSeries:
    return (_exp(1j * a) - _exp(-1j * a)) * (-0.5j)


def compose(a: TSeries, derivs: Sequence[Any]) -> TSeries:
    r"""
    :math:`f(a(h))` from :math:`f^{(k)}(a_0)`, ``k = 0..N``.

    :math:`f(a_0 + d) = \sum_k f^{(k)}(a_0) d^k/k!` with :math:`d = a - a_0`
    (no constant term), so powers of ``d`` beyond ``N`` vanish.
    """
    n = a.order
    d = TSeries([0.0 * a.coeffs[0]] + a.coeffs[1:])
    out = d._lift(derivs[0])
    power = d._lift(1.0 + 0.0 * a.coeffs[0])
    for k in range(1, n + 1):
        power = power * d
        out = out + power * (derivs[k] / factorial(k))
    return out


def _bessel_derivs(nu: Any, z0: Any, n: int, scaled: bool) -> List[Any]:
    r"""
    :math:`\partial_z^k` of :math:`K_\nu(z)` (or :math:`e^zK_\nu(z)`) at ``z0``, ``k = 0..n``.

    :math:`K_\nu' = -(K_{\nu-1} + K_{\nu+1})/2`. For the scaled function,
    :math:`(e^zK_\nu)' = e^zK_\nu - (e^zK_{\nu-1} + e^zK_{\nu+1})/2`. Each derivative is a
    combination :math:`\sum_j c_j\,F_{\nu+j}(z_0)`, built by applying the rule to the
    previous combination.
    """
    fn = sps.kve if scaled else sps.kv
    combo = {0: 1.0}
    cache: dict[int, Any] = {}

    def F(j: int) -> Any:
        if j not in cache:
            cache[j] = fn(nu + j, z0)
        return cache[j]

    out = []
    for _ in range(n + 1):
        out.append(sum(c * F(j) for j, c in combo.items()))
        nxt: dict[int, float] = {}
        for j, c in combo.items():
            nxt[j - 1] = nxt.get(j - 1, 0.0) - 0.5 * c
            nxt[j + 1] = nxt.get(j + 1, 0.0) - 0.5 * c
            if scaled:
                nxt[j] = nxt.get(j, 0.0) + c
        combo = nxt
    return out


def _bessel(scaled: bool) -> Callable[[Any, Any], TSeries]:
    def rule(nu: Any, z: Any) -> TSeries:
        if isinstance(nu, TSeries):
            raise TaylorError("Bessel K with a series in the order is not supported")
        return compose(z, _bessel_derivs(nu, z.coeffs[0], z.order, scaled))

    return rule


def _with_constant(a: TSeries, c0: Any) -> TSeries:
    """``a`` with its constant term replaced (keeps expm1/log1p accurate near 0)."""
    return TSeries([c0] + list(a.coeffs[1:]))


def _binary_power(a: Any, b: Any) -> TSeries:
    if isinstance(a, TSeries):
        return a.__pow__(b)
    return b.__rpow__(a)


_UFUNCS: dict[Any, Callable[..., Any]] = {
    np.add: lambda a, b: a + b if isinstance(a, TSeries) else b + a,
    np.subtract: lambda a, b: a - b if isinstance(a, TSeries) else b.__rsub__(a),
    np.multiply: lambda a, b: a * b if isinstance(a, TSeries) else b * a,
    np.true_divide: lambda a, b: a / b if isinstance(a, TSeries) else b.__rtruediv__(a),
    np.negative: lambda a: -a,
    np.positive: lambda a: a,
    np.power: _binary_power,
    np.square: lambda a: a * a,
    np.reciprocal: lambda a: a.reciprocal(),
    np.exp: _exp,
    np.log: _log,
    np.sqrt: _sqrt,
    np.cosh: _cosh,
    np.sinh: _sinh,
    np.cos: _cos,
    np.sin: _sin,
    np.expm1: lambda a: _with_constant(_exp(a) - 1.0, np.expm1(a.coeffs[0])),
    np.log1p: lambda a: _with_constant(_log(1.0 + a), np.log1p(a.coeffs[0])),
    sps.kve: _bessel(True),
    sps.kv: _bessel(False),
}
