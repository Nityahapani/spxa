r"""
Forward-mode automatic differentiation with jets.

A :class:`Jet` carries a value together with its exact first derivatives
(and, optionally, its exact second derivatives) with respect to ``n`` real
input directions:

.. math::
    x \;\mapsto\; \big(x,\; \partial_k x,\; \partial_k\partial_l x\big),
    \qquad k, l = 1, \dots, n.

The value may be real or complex and of any shape ``S``. The gradient has
shape ``S + (n,)`` and the Hessian ``S + (n, n)``. Derivatives are always
taken with respect to *real* inputs. A complex intermediate :math:`z(\theta)`
therefore carries :math:`\partial z/\partial\theta_k \in \mathbb C`, which is
well defined whether or not later operations are holomorphic, so ``np.real``,
``np.imag``, ``np.conj`` and ``np.abs`` all differentiate correctly.

Jets implement the NumPy ufunc and array-function protocols, so code written
with NumPy (and SciPy special-function ufuncs) evaluates derivatives exactly
without modification: ``np.exp``, ``np.log``, ``np.sqrt``, ``np.where``,
``np.sum``, ``scipy.special.gammaln``, ``scipy.special.kv`` (including the
derivative in the order), and so on. Every rule is the analytic derivative of
the operation; nothing is approximated by differences. Special functions and
their derivatives are evaluated to working precision. The derivative of the
Bessel function :math:`K_\nu(z)` in its order :math:`\nu`, which has no closed
form, comes from the integral representation
:math:`K_\nu(z) = \int_0^\infty e^{-z\cosh t}\cosh(\nu t)\,dt` with the
trapezoid rule, whose error for this entire, doubly-exponentially decaying
integrand is far below rounding (see :func:`_kv_order_derivatives`).

Anything that would silently drop derivative information raises
:class:`JetError` instead: ``float(jet)``, ``np.asarray(jet, dtype=float)``,
``bool(jet)``, in-place ufuncs (``out=``) and NumPy functions without a rule.
Comparisons act on values and return plain boolean arrays, so branches are
taken on the value (the derivative is then that of the branch taken, which is
correct away from the switching set).

Second-order rules: for a scalar function :math:`f` applied elementwise,

.. math::
    \partial_k f(x) = f'(x)\,\partial_k x, \qquad
    \partial_{kl} f(x) = f'(x)\,\partial_{kl}x + f''(x)\,\partial_k x\,\partial_l x,

and for a bilinear operation :math:`B(a, b)`,
:math:`\partial_{kl}B = B(\partial_{kl}a, b) + B(a, \partial_{kl}b) +
B(\partial_k a, \partial_l b) + B(\partial_l a, \partial_k b)`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.special as sps

__all__ = [
    "Jet",
    "JetError",
    "is_jet",
    "value_of",
    "grad_of",
    "hess_of",
    "variables",
    "variable_vector",
    "constant_like",
    "stack",
    "jet_jacobian",
    "jet_hessian",
    "kv_order_derivative",
]


class JetError(TypeError):
    """An operation would drop derivative information, or has no differentiation rule."""


_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_LOG_SQRT_2PI = float(0.5 * np.log(2.0 * np.pi))
_TWO_OVER_SQRT_PI = float(2.0 / np.sqrt(np.pi))


# ── helpers ───────────────────────────────────────────────────────────────────


def is_jet(x: Any) -> bool:
    """Whether ``x`` is a :class:`Jet`."""
    return isinstance(x, Jet)


def _coerce(x: Any) -> Any:
    """Turn an object array holding jets (``np.array([jet, jet])``) into one Jet."""
    if isinstance(x, np.ndarray) and x.dtype == object:
        flat = list(x.ravel())
        if any(isinstance(e, Jet) for e in flat):
            j = stack(flat)
            return j.reshape(x.shape + j.shape[1:]) if x.ndim != 1 else j
    return x


def value_of(x: Any) -> Any:
    """The value of a jet, or ``x`` itself."""
    x = _coerce(x)
    return x.value if isinstance(x, Jet) else x


def grad_of(x: Any, n: int) -> np.ndarray:
    """The gradient of a jet (shape ``S + (n,)``), or zeros for a constant."""
    x = _coerce(x)
    if isinstance(x, Jet):
        return x.grad
    v = np.asarray(x)
    return np.zeros(v.shape + (n,), dtype=np.result_type(v.dtype, float))


def hess_of(x: Any, n: int) -> np.ndarray:
    """The Hessian of a jet (shape ``S + (n, n)``), or zeros for a constant or first-order jet."""
    x = _coerce(x)
    if isinstance(x, Jet) and x.hess is not None:
        return x.hess
    v = np.asarray(value_of(x))
    return np.zeros(v.shape + (n, n), dtype=np.result_type(v.dtype, float))


def _meta(*args: Any) -> Tuple[int, int]:
    """(n, order) shared by the jets among ``args``; raises on a mismatch."""
    n = order = None
    for a in args:
        if isinstance(a, Jet):
            if n is None:
                n, order = a.n, a.order
            elif a.n != n or a.order != order:
                raise JetError(
                    f"cannot combine jets with (n, order) = ({n}, {order}) and ({a.n}, {a.order})"
                )
    if n is None:
        raise JetError("no jet among the arguments")
    return n, order  # type: ignore[return-value]


def _outer(ga: np.ndarray, gb: np.ndarray) -> np.ndarray:
    return ga[..., :, None] * gb[..., None, :]


def _a(x: Any) -> np.ndarray:
    return np.asarray(x)


# ── the Jet type ──────────────────────────────────────────────────────────────


class Jet:
    r"""
    A value with exact first (and optionally second) derivatives.

    Parameters
    ----------
    value :
        Real or complex array (or scalar) of shape ``S``.
    grad :
        Array broadcastable to ``S + (n,)``: :math:`\partial x/\partial\theta_k`.
    hess :
        Optional array broadcastable to ``S + (n, n)``. Present on second-order
        jets only. Jets of different orders, or with different ``n``, cannot
        be combined.

    Use :func:`variables` or :func:`variable_vector` to create independent
    inputs, then compute with ordinary NumPy code.
    """

    __slots__ = ("value", "grad", "hess", "_owned")
    __array_priority__ = 1000.0
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, value: Any, grad: Any, hess: Any = None) -> None:
        v = np.asarray(value)
        if v.dtype == object:
            raise JetError("a Jet value cannot be an object array")
        if v.dtype.kind in "biu":
            v = v.astype(float)
        g = np.asarray(grad)
        if g.ndim == 0:
            raise JetError("grad needs a trailing direction axis")
        n = g.shape[-1]
        if g.shape != v.shape + (n,):
            g = np.broadcast_to(g, v.shape + (n,))
        h = None
        if hess is not None:
            h = np.asarray(hess)
            if h.shape != v.shape + (n, n):
                h = np.broadcast_to(h, v.shape + (n, n))
        self.value = v
        self.grad = g
        self.hess = h
        self._owned = False

    # -- metadata ---------------------------------------------------------
    @property
    def n(self) -> int:
        """Number of input directions."""
        return int(self.grad.shape[-1])

    @property
    def order(self) -> int:
        """1 for gradient-only jets, 2 when the Hessian is carried."""
        return 1 if self.hess is None else 2

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.value.shape)

    @property
    def ndim(self) -> int:
        return int(self.value.ndim)

    @property
    def size(self) -> int:
        return int(self.value.size)

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    @property
    def real(self) -> Jet:
        return _real(self)

    @property
    def imag(self) -> Jet:
        return _imag(self)

    @property
    def T(self) -> Jet:
        return self.transpose()

    def conj(self) -> Jet:
        return np.conjugate(self)  # type: ignore[no-any-return]

    conjugate = conj

    def copy(self) -> Jet:
        return Jet(
            self.value.copy(),
            np.array(self.grad),
            None if self.hess is None else np.array(self.hess),
        )

    def reshape(self, *shape: Any) -> Jet:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        v = self.value.reshape(shape)
        g = self.grad.reshape(v.shape + (self.n,))
        h = None if self.hess is None else self.hess.reshape(v.shape + (self.n, self.n))
        return Jet(v, g, h)

    def ravel(self) -> Jet:
        return self.reshape(-1)

    def flatten(self) -> Jet:
        return self.reshape(-1).copy()

    def transpose(self, *axes: Any) -> Jet:
        nd = self.ndim
        if not axes or axes == (None,):
            perm = tuple(range(nd))[::-1]
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(int(a) % nd for a in axes[0])
        else:
            perm = tuple(int(a) % nd for a in axes)
        g = np.transpose(self.grad, perm + (nd,))
        h = None if self.hess is None else np.transpose(self.hess, perm + (nd, nd + 1))
        return Jet(np.transpose(self.value, perm), g, h)

    def sum(self, axis: Any = None, keepdims: bool = False) -> Jet:
        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Any = None, keepdims: bool = False) -> Jet:
        return _mean(self, axis=axis, keepdims=keepdims)

    def astype(self, dtype: Any) -> Jet:
        dt = np.dtype(dtype)
        if dt.kind == "c":
            return Jet(
                self.value.astype(dt),
                self.grad.astype(dt),
                None if self.hess is None else self.hess.astype(dt),
            )
        if dt.kind == "f":
            if self.value.dtype.kind == "c":
                raise JetError(
                    "casting a complex Jet to real drops its imaginary part; use np.real"
                )
            return Jet(
                self.value.astype(dt),
                self.grad.astype(dt),
                None if self.hess is None else self.hess.astype(dt),
            )
        raise JetError(f"cannot cast a Jet to {dt}")

    # -- conversions that would drop derivatives ----------------------------
    def __float__(self) -> float:
        raise JetError("float() of a Jet drops its derivatives; use jet.value explicitly")

    def __complex__(self) -> complex:
        raise JetError("complex() of a Jet drops its derivatives; use jet.value explicitly")

    def __int__(self) -> int:
        raise JetError("int() of a Jet drops its derivatives")

    def __index__(self) -> int:
        raise JetError("a Jet cannot be used as an index")

    def __bool__(self) -> bool:
        raise JetError("the truth value of a Jet is ambiguous; compare jet.value instead")

    def __array__(self, dtype: Any = None, *args: Any, **kwargs: Any) -> np.ndarray:
        if dtype is not None and np.dtype(dtype) == object:
            # np.array([jet, jet], dtype=object): an object array of element jets,
            # which every jet rule turns back into a single Jet (see _coerce).
            box = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(*self.shape):
                box[idx] = self[idx]
            return box
        raise JetError(
            "converting a Jet to a plain array drops its derivatives; use jet.value explicitly"
        )

    # -- indexing ------------------------------------------------------------
    def _gidx(self, idx: Any, extra: int) -> Any:
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = tuple(value_of(i) if isinstance(i, Jet) else i for i in idx)
        if any(i is Ellipsis for i in idx):
            return idx + (slice(None),) * extra
        return idx

    def __getitem__(self, idx: Any) -> Jet:
        v = self.value[self._gidx(idx, 0)]
        g = self.grad[self._gidx(idx, 1)]
        h = None if self.hess is None else self.hess[self._gidx(idx, 2)]
        return Jet(v, g, h)

    def __setitem__(self, idx: Any, val: Any) -> None:
        val = _coerce(val)
        if not self._owned:
            # Rules may share arrays between jets (views, broadcasts, pass-through),
            # so the first write copies: assigning into one jet never alters another.
            self.value = np.array(self.value)
            self.grad = np.array(self.grad)
            if self.hess is not None:
                self.hess = np.array(self.hess)
            self._owned = True
        n = self.n
        vv = value_of(val)
        if np.iscomplexobj(vv) and not np.iscomplexobj(self.value):
            self.value = self.value.astype(complex)
            self.grad = self.grad.astype(complex)
            if self.hess is not None:
                self.hess = self.hess.astype(complex)
        if isinstance(val, Jet):
            _meta(self, val)
        self.value[self._gidx(idx, 0)] = vv
        tgt = self.value[self._gidx(idx, 0)]
        self.grad[self._gidx(idx, 1)] = np.broadcast_to(grad_of(val, n), np.shape(tgt) + (n,))
        if self.hess is not None:
            self.hess[self._gidx(idx, 2)] = np.broadcast_to(hess_of(val, n), np.shape(tgt) + (n, n))

    # -- arithmetic via ufuncs ------------------------------------------------
    def __add__(self, o: Any) -> Any:
        return np.add(self, o)

    def __radd__(self, o: Any) -> Any:
        return np.add(o, self)

    def __sub__(self, o: Any) -> Any:
        return np.subtract(self, o)

    def __rsub__(self, o: Any) -> Any:
        return np.subtract(o, self)

    def __mul__(self, o: Any) -> Any:
        return np.multiply(self, o)

    def __rmul__(self, o: Any) -> Any:
        return np.multiply(o, self)

    def __truediv__(self, o: Any) -> Any:
        return np.true_divide(self, o)

    def __rtruediv__(self, o: Any) -> Any:
        return np.true_divide(o, self)

    def __pow__(self, o: Any) -> Any:
        return np.power(self, o)

    def __rpow__(self, o: Any) -> Any:
        return np.power(o, self)

    def __matmul__(self, o: Any) -> Any:
        return np.matmul(self, o)

    def __rmatmul__(self, o: Any) -> Any:
        return np.matmul(o, self)

    def __neg__(self) -> Any:
        return np.negative(self)

    def __pos__(self) -> Any:
        return self

    def __abs__(self) -> Any:
        return np.absolute(self)

    def __lt__(self, o: Any) -> Any:
        return np.less(self.value, value_of(o))

    def __le__(self, o: Any) -> Any:
        return np.less_equal(self.value, value_of(o))

    def __gt__(self, o: Any) -> Any:
        return np.greater(self.value, value_of(o))

    def __ge__(self, o: Any) -> Any:
        return np.greater_equal(self.value, value_of(o))

    def __eq__(self, o: Any) -> Any:  # type: ignore[override]
        return np.equal(self.value, value_of(o))

    def __ne__(self, o: Any) -> Any:  # type: ignore[override]
        return np.not_equal(self.value, value_of(o))

    def __repr__(self) -> str:
        kind = "Jet2" if self.hess is not None else "Jet"
        return f"{kind}(value={self.value!r}, n={self.n})"

    # -- protocols ---------------------------------------------------------
    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        out = kwargs.get("out")
        if out is not None and not (isinstance(out, tuple) and all(o is None for o in out)):
            raise JetError(f"in-place {ufunc.__name__}(out=...) is not supported on Jets")
        inputs = tuple(_coerce(i) for i in inputs)
        if method == "__call__":
            for k, val in kwargs.items():
                if k == "out" or (k == "where" and val is True) or (k == "casting"):
                    continue
                raise JetError(f"{ufunc.__name__}(..., {k}=...) is not supported on Jets")
            if ufunc in _VALUE_ONLY:
                return ufunc(*[value_of(i) for i in inputs])
            rule = _UFUNC_RULES.get(ufunc)
            if rule is None:
                raise JetError(f"no differentiation rule for ufunc {ufunc.__name__!r}")
            if not any(isinstance(i, Jet) for i in inputs):
                return ufunc(*inputs)
            return rule(*inputs)
        if method == "reduce" and ufunc in (np.add, np.maximum, np.minimum):
            axis = kwargs.get("axis", 0)
            keepdims = kwargs.get("keepdims", False) is True
            if ufunc is np.add:
                return _sum(inputs[0], axis=axis, keepdims=keepdims)
            return _extreme(inputs[0], axis=axis, keepdims=keepdims, largest=ufunc is np.maximum)
        if method == "accumulate" and ufunc in (np.add, np.maximum, np.minimum):
            axis = kwargs.get("axis", 0)
            if ufunc is np.add:
                return _cumsum(inputs[0], axis=axis)
            return _accumulate_extreme(inputs[0], axis=axis, largest=ufunc is np.maximum)
        raise JetError(f"{ufunc.__name__}.{method} is not supported on Jets")

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        impl = _FUNCTIONS.get(func)
        if impl is None:
            raise JetError(f"np.{getattr(func, '__name__', func)} has no Jet implementation")
        return impl(*args, **kwargs)


# ── construction ──────────────────────────────────────────────────────────────


def variables(values: Sequence[Any], order: int = 1) -> Tuple[Jet, ...]:
    r"""
    Independent jets, one per entry of ``values``.

    Entry ``k`` (a scalar or an array) is seeded with direction ``k``, so with
    ``n = len(values)`` every later result carries :math:`\partial/\partial
    x_k`. Array entries share their direction across elements: they are
    treated as ``values[k] = c_k + \theta_k`` with a scalar :math:`\theta_k`.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    n = len(values)
    out = []
    for k, v in enumerate(values):
        arr = np.asarray(v)
        if arr.dtype.kind in "biu":
            arr = arr.astype(float)
        g = np.zeros(arr.shape + (n,), dtype=arr.dtype if arr.dtype.kind in "fc" else float)
        g[..., k] = 1.0
        h = np.zeros(arr.shape + (n, n), dtype=g.dtype) if order == 2 else None
        out.append(Jet(arr, g, h))
    return tuple(out)


def variable_vector(x: Any, order: int = 1) -> Jet:
    """A vector jet: element ``k`` of ``x`` is seeded with direction ``k``."""
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.size
    g = np.eye(n)
    h = np.zeros((n, n, n)) if order == 2 else None
    return Jet(arr, g, h)


def constant_like(x: Any, ref: Jet) -> Jet:
    """``x`` as a jet with zero derivatives, matching ``ref``'s ``n`` and order."""
    v = np.asarray(value_of(x))
    n = ref.n
    g = np.zeros(v.shape + (n,), dtype=np.result_type(v.dtype, float))
    h = None if ref.hess is None else np.zeros(v.shape + (n, n), dtype=g.dtype)
    return Jet(v, g, h)


def stack(items: Sequence[Any], axis: int = 0) -> Any:
    """``np.stack`` for a mixture of jets and constants."""
    items = [_coerce(i) for i in items]
    if not any(isinstance(i, Jet) for i in items):
        return np.stack([np.asarray(i) for i in items], axis=axis)
    n, order = _meta(*items)
    vals = [np.asarray(value_of(i)) for i in items]
    shape = np.broadcast_shapes(*[v.shape for v in vals])
    vals = [np.broadcast_to(v, shape) for v in vals]
    nd = len(shape) + 1
    ax = axis % nd
    v = np.stack(vals, axis=ax)
    g = np.stack([np.broadcast_to(grad_of(i, n), shape + (n,)) for i in items], axis=ax)
    h = None
    if order == 2:
        h = np.stack([np.broadcast_to(hess_of(i, n), shape + (n, n)) for i in items], axis=ax)
    return Jet(v, g, h)


# ── elementwise rules ─────────────────────────────────────────────────────────


def _unary(
    x: Jet,
    f: Callable[[np.ndarray], Any],
    df: Callable[[np.ndarray, np.ndarray], Any],
    d2f: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], Any]] = None,
) -> Jet:
    v = x.value
    fv = np.asarray(f(v))
    d1 = np.asarray(df(v, fv))
    g = d1[..., None] * x.grad
    h = None
    if x.hess is not None:
        if d2f is None:
            raise JetError("no second-order rule for this operation")
        d2 = np.asarray(d2f(v, fv, d1))
        h = d1[..., None, None] * x.hess + d2[..., None, None] * _outer(x.grad, x.grad)
    return Jet(fv, g, h)


def _linear_unary(x: Jet, f: Callable[[np.ndarray], Any]) -> Jet:
    h = None if x.hess is None else f(x.hess)
    return Jet(f(x.value), f(x.grad), h)


def _real(x: Any) -> Any:
    x = _coerce(x)
    if not isinstance(x, Jet):
        return np.real(x)
    return _linear_unary(x, np.real)


def _imag(x: Any) -> Any:
    x = _coerce(x)
    if not isinstance(x, Jet):
        return np.imag(x)
    return _linear_unary(x, np.imag)


def _add(a: Any, b: Any, sign: float = 1.0) -> Jet:
    n, order = _meta(a, b)
    av, bv = value_of(a), value_of(b)
    v = np.add(av, bv) if sign > 0 else np.subtract(av, bv)
    shp = np.shape(v)
    if isinstance(a, Jet) and isinstance(b, Jet):
        g = a.grad + b.grad if sign > 0 else a.grad - b.grad
        h = None
        if order == 2:
            h = a.hess + b.hess if sign > 0 else a.hess - b.hess
    elif isinstance(a, Jet):
        g, h = a.grad, a.hess
    else:
        g = b.grad if sign > 0 else -b.grad
        h = None if b.hess is None else (b.hess if sign > 0 else -b.hess)
    g = np.broadcast_to(g, shp + (n,))
    if h is not None:
        h = np.broadcast_to(h, shp + (n, n))
    return Jet(v, g, h)


def _mul(a: Any, b: Any) -> Jet:
    n, order = _meta(a, b)
    av, bv = _a(value_of(a)), _a(value_of(b))
    v = av * bv
    if isinstance(a, Jet) and isinstance(b, Jet):
        g = av[..., None] * b.grad + bv[..., None] * a.grad
        h = None
        if order == 2:
            h = (
                av[..., None, None] * b.hess
                + bv[..., None, None] * a.hess
                + _outer(a.grad, b.grad)
                + _outer(b.grad, a.grad)
            )
        return Jet(v, g, h)
    j, c = (a, bv) if isinstance(a, Jet) else (b, av)
    g = c[..., None] * j.grad
    h = None if j.hess is None else c[..., None, None] * j.hess
    return Jet(v, g, h)


def _reciprocal(x: Jet) -> Jet:
    return _unary(x, lambda v: 1.0 / v, lambda v, f: -(f * f), lambda v, f, d: 2.0 * f * f * f)


def _div(a: Any, b: Any) -> Jet:
    if isinstance(b, Jet):
        # value from a direct division (not a * (1/b)) for bitwise agreement with NumPy
        r = _mul(a, _reciprocal(b))
        return Jet(np.true_divide(value_of(a), b.value), r.grad, r.hess)
    bv = _a(b)
    return Jet(
        np.true_divide(a.value, bv),
        a.grad / bv[..., None],
        None if a.hess is None else a.hess / bv[..., None, None],
    )


def _power(a: Any, b: Any) -> Jet:
    if not isinstance(b, Jet):
        p = _a(b)
        if p.ndim == 0 and float(np.real(p)) == np.real(p) and np.imag(p) == 0:
            pf = float(np.real(p))
            if pf == 0.0:
                return constant_like(np.ones_like(a.value), a)
            if pf == 1.0:
                return a  # type: ignore[no-any-return]
            if pf == 2.0:
                return _mul(a, a)
        return _unary(
            a,
            lambda v: np.power(v, p),
            lambda v, f: p * np.power(v, p - 1.0),
            lambda v, f, d: p * (p - 1.0) * np.power(v, p - 2.0),
        )
    if not isinstance(a, Jet):
        base = _a(a)
        logb = np.log(base.astype(complex)) if np.any(np.real(base) < 0) else np.log(base)
        return _unary(
            b,
            lambda v: np.power(base, v),
            lambda v, f: f * logb,
            lambda v, f, d: f * logb * logb,
        )
    # a^b = exp(b log a), value from np.power for agreement with NumPy
    r = np.exp(b * np.log(a))
    return Jet(np.power(a.value, b.value), r.grad, r.hess)


def _select(mask: np.ndarray, a: Any, b: Any) -> Any:
    """Elementwise ``a`` where ``mask`` else ``b`` (values and derivatives)."""
    a, b = _coerce(a), _coerce(b)
    if not isinstance(a, Jet) and not isinstance(b, Jet):
        return np.where(mask, a, b)
    n, order = _meta(a, b)
    av, bv = value_of(a), value_of(b)
    v = np.where(mask, av, bv)
    m = np.broadcast_to(np.asarray(mask), np.shape(v))
    g = np.where(m[..., None], grad_of(a, n), grad_of(b, n))
    h = None
    if order == 2:
        h = np.where(m[..., None, None], hess_of(a, n), hess_of(b, n))
    return Jet(v, np.broadcast_to(g, np.shape(v) + (n,)), h)


def _maximum(a: Any, b: Any) -> Any:
    av, bv = _a(value_of(a)), _a(value_of(b))
    mask = (av >= bv) | np.isnan(av)
    return _select(mask, a, b)


def _minimum(a: Any, b: Any) -> Any:
    av, bv = _a(value_of(a)), _a(value_of(b))
    mask = (av <= bv) | np.isnan(av)
    return _select(mask, a, b)


def _fmax(a: Any, b: Any) -> Any:
    av, bv = _a(value_of(a)), _a(value_of(b))
    mask = (av >= bv) | np.isnan(bv)
    return _select(mask, a, b)


def _fmin(a: Any, b: Any) -> Any:
    av, bv = _a(value_of(a)), _a(value_of(b))
    mask = (av <= bv) | np.isnan(bv)
    return _select(mask, a, b)


def _absolute(x: Jet) -> Jet:
    if np.iscomplexobj(x.value):
        # r = |z|: dr = Re(conj(z) dz)/r; d2r = (Re(conj(dz_l) dz_k) + Re(conj(z) d2z) - dr_k dr_l)/r.
        # At z = 0 the derivative does not exist; 0 is returned (the real-|x| convention).
        z = x.value
        r = np.abs(z)
        safe = np.where(r > 0, r, 1.0)
        live = (r > 0)[..., None]
        g = np.where(live, np.real(np.conj(z)[..., None] * x.grad) / safe[..., None], 0.0)
        h = None
        if x.hess is not None:
            num = (
                np.real(np.conj(x.grad)[..., :, None] * x.grad[..., None, :])
                + np.real(np.conj(z)[..., None, None] * x.hess)
                - g[..., :, None] * g[..., None, :]
            )
            h = np.where(live[..., None], num / safe[..., None, None], 0.0)
        return Jet(r, g, h)
    s = np.sign(x.value)
    return Jet(
        np.abs(x.value),
        s[..., None] * x.grad,
        None if x.hess is None else s[..., None, None] * x.hess,
    )


def _arctan2(y: Any, x: Any) -> Jet:
    yv, xv = _a(value_of(y)), _a(value_of(x))
    v = np.arctan2(yv, xv)
    use_x = np.abs(xv) >= np.abs(yv)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_x = _select(use_x, x, 1.0) if isinstance(x, Jet) else np.where(use_x, x, 1.0)
        safe_y = _select(~use_x, y, 1.0) if isinstance(y, Jet) else np.where(~use_x, y, 1.0)
        j1 = np.arctan(y / safe_x)
        j2 = -np.arctan(x / safe_y)
    r = _select(use_x, j1, j2)
    return Jet(v, r.grad, r.hess)


def _hypot(a: Any, b: Any) -> Jet:
    r = np.sqrt(a * a + b * b)
    return Jet(np.hypot(value_of(a), value_of(b)), r.grad, r.hess)


def _logaddexp(a: Any, b: Any) -> Jet:
    m = np.maximum(value_of(a), value_of(b))
    m = np.where(np.isfinite(m), m, 0.0)
    r = m + np.log(np.exp(a - m) + np.exp(b - m))
    return Jet(np.logaddexp(value_of(a), value_of(b)), r.grad, r.hess)


_BERNOULLI_2J = (
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
)


def _polygamma_complex(k: int, z: np.ndarray) -> np.ndarray:
    r"""
    :math:`\psi^{(k)}(z)` for complex ``z`` and ``k`` in {1, 2} (SciPy has real ``k``-th only).

    Reflection maps :math:`\operatorname{Re} z < 1/2` to the right half-plane
    (:math:`\psi_1(z) = \pi^2/\sin^2\pi z - \psi_1(1-z)`,
    :math:`\psi_2(z) = \psi_2(1-z) - 2\pi^3\cos\pi z/\sin^3\pi z`); the recurrence
    :math:`\psi^{(k)}(z) = \psi^{(k)}(z+N) + (-1)^{k+1}k!\sum_{j<N}(z+j)^{-k-1}` moves
    to :math:`|w| \ge 20`, where the Stirling series with eight Bernoulli terms is
    accurate to below :math:`10^{-20}` relative.
    """
    if k not in (1, 2):
        raise JetError(f"polygamma({k}, z) for complex z is not available")
    z = np.asarray(z, dtype=complex)
    left = z.real < 0.5
    zz = np.where(left, 1.0 - z, z)
    sign = 1.0 if k == 1 else -1.0  # (-1)^{k+1}
    fact = 1.0 if k == 1 else 2.0  # k!
    acc = np.zeros_like(zz)
    nshift = 20
    for j in range(nshift):
        acc += 1.0 / (zz + j) ** (k + 1)
    w = zz + nshift
    # (-1)^{k+1} [ (k-1)!/w^k + k!/(2 w^{k+1}) + sum_j B_{2j} (2j+k-1)!/((2j)! w^{2j+k}) ]
    series = 1.0 / w**k + fact / (2.0 * w ** (k + 1))  # (k-1)! = 1 for k = 1, 2
    for jj, b in enumerate(_BERNOULLI_2J, start=1):
        coef = b
        for m in range(2 * jj + 1, 2 * jj + k):  # (2j+k-1)!/(2j)!
            coef *= m
        series = series + coef / w ** (2 * jj + k)
    right = sign * (fact * acc + series)
    if np.any(left):
        sp_ = np.sin(np.pi * z)
        if k == 1:
            refl = np.pi**2 / sp_**2 - right
        else:
            refl = right - 2.0 * np.pi**3 * np.cos(np.pi * z) / sp_**3
        right = np.where(left, refl, right)
    return right


def _polygamma_real(k: int, v: np.ndarray) -> np.ndarray:
    """Polygamma of real or (for k = 1, 2) complex argument."""
    if np.iscomplexobj(v):
        return _polygamma_complex(k, v)
    return sps.polygamma(k, v)


def _digamma_any(v: np.ndarray) -> np.ndarray:
    return sps.digamma(v)


def _phi(v: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * v * v) / _SQRT_2PI


def _exprel_d1(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    small = np.abs(v) < 1e-2
    with np.errstate(all="ignore"):
        exact = (np.exp(v) * (v - 1.0) + 1.0) / (v * v)
    series = 0.5 + v / 3.0 + v * v / 8.0 + v**3 / 30.0 + v**4 / 144.0 + v**5 / 840.0
    return np.where(small, series, exact)


def _exprel_d2(v: np.ndarray, f: np.ndarray, d: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    small = np.abs(v) < 1e-2
    with np.errstate(all="ignore"):
        exact = (np.exp(v) * (v * v - 2.0 * v + 2.0) - 2.0) / v**3
    series = 1.0 / 3.0 + v / 4.0 + v * v / 10.0 + v**3 / 36.0 + v**4 / 168.0 + v**5 / 960.0
    return np.where(small, series, exact)


def _log_ndtr_d1(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * v * v - _LOG_SQRT_2PI - f)


# ── Bessel K and its order derivative ─────────────────────────────────────────


def _kv_order_derivatives(
    nu: np.ndarray, z: np.ndarray, scaled: bool, second: bool
) -> Dict[str, np.ndarray]:
    r"""
    :math:`\partial_\nu`, :math:`\partial_\nu^2` and :math:`\partial_\nu\partial_z` of
    :math:`K_\nu(z)` (or of :math:`e^zK_\nu(z)` when ``scaled``).

    Uses :math:`e^zK_\nu(z) = \int_0^\infty e^{-z(\cosh t - 1)}\cosh(\nu t)\,dt`
    (:math:`\operatorname{Re}z > 0`) differentiated under the integral:

    .. math::
        \partial_\nu = \int_0^\infty t\sinh(\nu t)\,w(t)\,dt, \quad
        \partial_\nu^2 = \int_0^\infty t^2\cosh(\nu t)\,w(t)\,dt, \quad
        \partial_\nu\partial_z = \int_0^\infty (1 - \cosh t)\,t\sinh(\nu t)\,w(t)\,dt,

    with :math:`w(t) = e^{-z(\cosh t - 1)}`. The integrands are entire in
    :math:`t` and decay doubly exponentially, so the trapezoid rule with step
    :math:`h` has error of order :math:`e^{-2\pi d/h}`, where :math:`d \approx
    \pi/2 - |\arg z|` is the half-width of the strip in which :math:`w` decays.
    With :math:`h = 1/32` this is below :math:`10^{-30}` for
    :math:`|\arg z| < 1.2`. The range is cut where the integrand has fallen by
    :math:`e^{-60}` from its maximum.
    """
    nu = np.asarray(nu, dtype=float)
    z = np.asarray(z)
    if np.any(np.real(z) <= 0):
        raise JetError("the order derivative of K_nu(z) needs Re z > 0")
    if np.any(np.abs(np.angle(z)) > 1.2):
        raise JetError("the order derivative of K_nu(z) is implemented for |arg z| <= 1.2")
    shape = np.broadcast_shapes(nu.shape, z.shape)
    nu_b = np.broadcast_to(nu, shape).ravel()
    z_b = np.broadcast_to(z, shape).ravel()
    anu = np.abs(nu_b)
    sgn = np.sign(nu_b)
    rz = np.real(z_b)
    # truncation: rz (cosh T - 1) - |nu| T >= 60 + log(T^2 + 1), solved by fixed point
    T = np.full(nu_b.shape, 1.0)
    for _ in range(60):
        T = np.arccosh(1.0 + (60.0 + anu * T + 2.0 * np.log1p(T)) / rz)
    T = np.minimum(T, 50.0)
    h = 1.0 / 32.0
    m = int(np.ceil(float(np.max(T)) / h)) + 1 if T.size else 1
    t = np.arange(m + 1) * h
    tt = t[None, :]
    expo = -z_b[:, None] * (np.cosh(tt) - 1.0) + anu[:, None] * tt
    mask = tt <= T[:, None] + h
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        e = np.where(mask, np.exp(expo), 0.0)
        damp = np.exp(-2.0 * anu[:, None] * tt)
        ch = 0.5 * (1.0 + damp)  # cosh(nu t) e^{-|nu| t}
        sh = 0.5 * (1.0 - damp) * sgn[:, None]  # sinh(nu t) e^{-|nu| t}
    w = np.full(t.shape, h)
    w[0] = 0.5 * h
    out: Dict[str, np.ndarray] = {}
    a = (tt * sh * e) @ w
    # cancellation guard: for complex z the integrand oscillates, and for large |nu| its
    # magnitude can exceed the integral by many orders. Refuse rather than lose digits.
    mag = (tt * np.abs(sh * e)) @ w
    cond = mag / np.maximum(np.abs(a), 1e-300)
    if np.any(cond > 1e5):
        raise JetError(
            "d/dnu K_nu(z) would lose more than 5 digits to cancellation here "
            f"(|nu| up to {float(np.max(anu)):.3g}, arg z up to {float(np.max(np.abs(np.angle(z_b)))):.3g})"
        )
    out["d_nu"] = a
    if second:
        out["d_nu2"] = (tt * tt * ch * e) @ w
        c = ((1.0 - np.cosh(tt)) * tt * sh * e) @ w
        out["d_nu_dz"] = c
    if not scaled:
        ez = np.exp(-z_b)
        out["d_nu"] = ez * out["d_nu"]
        if second:
            out["d_nu2"] = ez * out["d_nu2"]
            out["d_nu_dz"] = ez * (out["d_nu_dz"] - a)
    return {k: v.reshape(shape) for k, v in out.items()}


def kv_order_derivative(nu: Any, z: Any, scaled: bool = False) -> np.ndarray:
    r""":math:`\partial K_\nu(z)/\partial\nu` (or of :math:`e^zK_\nu(z)` when ``scaled``)."""
    return _kv_order_derivatives(np.asarray(nu), np.asarray(z), scaled, False)["d_nu"]


def _bessel_k(nu: Any, z: Any, scaled: bool) -> Jet:
    r"""``kv`` / ``kve`` with jets in the order and/or the argument."""
    n, order = _meta(nu, z)
    fn = sps.kve if scaled else sps.kv
    nv, zv = _a(value_of(nu)), _a(value_of(z))
    if np.iscomplexobj(nv):
        raise JetError("Bessel K with a complex order is not supported")
    val = fn(nv, zv)
    shp = np.shape(val)
    # argument derivatives: K' = -(K_{v-1} + K_{v+1})/2; kve' = kve - (kve_{v-1} + kve_{v+1})/2
    km, kp = fn(nv - 1.0, zv), fn(nv + 1.0, zv)
    dz = -0.5 * (km + kp) + (val if scaled else 0.0)
    g = np.zeros(shp + (n,), dtype=np.result_type(val, float))
    gz = grad_of(z, n)
    gn = grad_of(nu, n)
    zj, nj = isinstance(z, Jet), isinstance(nu, Jet)
    if zj:
        g = g + dz[..., None] * gz
    od: Dict[str, np.ndarray] = {}
    if nj:
        od = _kv_order_derivatives(nv, zv, scaled, order == 2)
        g = g + od["d_nu"][..., None] * gn
    h = None
    if order == 2:
        k2m, k2p = fn(nv - 2.0, zv), fn(nv + 2.0, zv)
        # K'' = (K_{v-2} + 2K_v + K_{v+2})/4
        dzz_k = 0.25 * (k2m + 2.0 * val + k2p) if not scaled else None
        if scaled:
            # kve = e^z K: kve'' = kve + 2 e^z K' + e^z K''
            K = val
            Kp = -0.5 * (km + kp)
            Kpp = 0.25 * (k2m + 2.0 * K + k2p)
            dzz = K + 2.0 * Kp + Kpp
        else:
            dzz = dzz_k
        h = np.zeros(shp + (n, n), dtype=g.dtype)
        if zj:
            h = h + dz[..., None, None] * hess_of(z, n) + dzz[..., None, None] * _outer(gz, gz)
        if nj:
            h = (
                h
                + od["d_nu"][..., None, None] * hess_of(nu, n)
                + od["d_nu2"][..., None, None] * _outer(gn, gn)
            )
        if zj and nj:
            cross = od["d_nu_dz"]
            h = h + cross[..., None, None] * (_outer(gn, gz) + _outer(gz, gn))
    return Jet(val, np.broadcast_to(g, shp + (n,)), h)


def _polygamma(k: Any, x: Any) -> Jet:
    if isinstance(k, Jet):
        raise JetError("polygamma: the order cannot be a Jet")
    kk = _a(k)
    return _unary(
        x,
        lambda v: sps.polygamma(kk, v),
        lambda v, f: _polygamma_real(int(kk) + 1, v),
        lambda v, f, d: _polygamma_real(int(kk) + 2, v),
    )


_VALUE_ONLY = {
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
    np.equal,
    np.not_equal,
    np.logical_and,
    np.logical_or,
    np.logical_not,
    np.logical_xor,
    np.isnan,
    np.isinf,
    np.isfinite,
    np.signbit,
    # piecewise constant: derivative zero almost everywhere
    np.sign,
    np.floor,
    np.ceil,
    np.trunc,
    np.rint,
}

_UFUNC_RULES: Dict[Any, Callable[..., Any]] = {
    np.add: lambda a, b: _add(a, b, 1.0),
    np.subtract: lambda a, b: _add(a, b, -1.0),
    np.multiply: _mul,
    np.true_divide: _div,
    np.divide: _div,
    np.power: _power,
    np.float_power: _power,
    np.matmul: lambda a, b: _matmul(a, b),
    np.maximum: _maximum,
    np.minimum: _minimum,
    np.fmax: _fmax,
    np.fmin: _fmin,
    np.arctan2: _arctan2,
    np.hypot: _hypot,
    np.logaddexp: _logaddexp,
    np.negative: lambda x: _linear_unary(x, np.negative),
    np.positive: lambda x: x,
    np.conjugate: lambda x: _linear_unary(x, np.conjugate),
    np.absolute: _absolute,
    np.reciprocal: _reciprocal,
    np.square: lambda x: _mul(x, x),
    np.exp: lambda x: _unary(x, np.exp, lambda v, f: f, lambda v, f, d: f),
    np.expm1: lambda x: _unary(x, np.expm1, lambda v, f: np.exp(v), lambda v, f, d: d),
    np.exp2: lambda x: _unary(
        x, np.exp2, lambda v, f: f * np.log(2.0), lambda v, f, d: d * np.log(2.0)
    ),
    np.log: lambda x: _unary(x, np.log, lambda v, f: 1.0 / v, lambda v, f, d: -d * d),
    np.log1p: lambda x: _unary(x, np.log1p, lambda v, f: 1.0 / (1.0 + v), lambda v, f, d: -d * d),
    np.log2: lambda x: _unary(
        x,
        np.log2,
        lambda v, f: 1.0 / (v * np.log(2.0)),
        lambda v, f, d: -1.0 / (v * v * np.log(2.0)),
    ),
    np.log10: lambda x: _unary(
        x,
        np.log10,
        lambda v, f: 1.0 / (v * np.log(10.0)),
        lambda v, f, d: -1.0 / (v * v * np.log(10.0)),
    ),
    np.sqrt: lambda x: _unary(x, np.sqrt, lambda v, f: 0.5 / f, lambda v, f, d: -0.25 / (v * f)),
    np.cbrt: lambda x: _unary(
        x, np.cbrt, lambda v, f: f / (3.0 * v), lambda v, f, d: -2.0 * f / (9.0 * v * v)
    ),
    np.sin: lambda x: _unary(x, np.sin, lambda v, f: np.cos(v), lambda v, f, d: -f),
    np.cos: lambda x: _unary(x, np.cos, lambda v, f: -np.sin(v), lambda v, f, d: -f),
    np.tan: lambda x: _unary(x, np.tan, lambda v, f: 1.0 + f * f, lambda v, f, d: 2.0 * f * d),
    np.arcsin: lambda x: _unary(
        x,
        np.arcsin,
        lambda v, f: 1.0 / np.sqrt(1.0 - v * v),
        lambda v, f, d: v * d**3,
    ),
    np.arccos: lambda x: _unary(
        x,
        np.arccos,
        lambda v, f: -1.0 / np.sqrt(1.0 - v * v),
        lambda v, f, d: v * d**3,
    ),
    np.arctan: lambda x: _unary(
        x, np.arctan, lambda v, f: 1.0 / (1.0 + v * v), lambda v, f, d: -2.0 * v * d * d
    ),
    np.sinh: lambda x: _unary(x, np.sinh, lambda v, f: np.cosh(v), lambda v, f, d: f),
    np.cosh: lambda x: _unary(x, np.cosh, lambda v, f: np.sinh(v), lambda v, f, d: f),
    np.tanh: lambda x: _unary(x, np.tanh, lambda v, f: 1.0 - f * f, lambda v, f, d: -2.0 * f * d),
    np.arcsinh: lambda x: _unary(
        x,
        np.arcsinh,
        lambda v, f: 1.0 / np.sqrt(1.0 + v * v),
        lambda v, f, d: -v * d**3,
    ),
    np.arccosh: lambda x: _unary(
        x,
        np.arccosh,
        lambda v, f: 1.0 / np.sqrt(v * v - 1.0),
        lambda v, f, d: -v * d**3,
    ),
    np.arctanh: lambda x: _unary(
        x, np.arctanh, lambda v, f: 1.0 / (1.0 - v * v), lambda v, f, d: 2.0 * v * d * d
    ),
    np.deg2rad: lambda x: _mul(x, np.pi / 180.0),
    np.rad2deg: lambda x: _mul(x, 180.0 / np.pi),
    # scipy.special
    sps.gamma: lambda x: _unary(
        x,
        sps.gamma,
        lambda v, f: f * _digamma_any(v),
        lambda v, f, d: f * (_digamma_any(v) ** 2 + _polygamma_real(1, v)),
    ),
    sps.rgamma: lambda x: _unary(
        x,
        sps.rgamma,
        lambda v, f: -f * _digamma_any(v),
        lambda v, f, d: f * (_digamma_any(v) ** 2 - _polygamma_real(1, v)),
    ),
    sps.gammaln: lambda x: _unary(
        x, sps.gammaln, lambda v, f: sps.digamma(v), lambda v, f, d: _polygamma_real(1, v)
    ),
    sps.loggamma: lambda x: _unary(
        x, sps.loggamma, lambda v, f: sps.digamma(v), lambda v, f, d: _polygamma_real(1, v)
    ),
    sps.digamma: lambda x: _unary(
        x,
        sps.digamma,
        lambda v, f: _polygamma_real(1, v),
        lambda v, f, d: _polygamma_real(2, v),
    ),
    sps.polygamma: _polygamma,
    sps.erf: lambda x: _unary(
        x,
        sps.erf,
        lambda v, f: _TWO_OVER_SQRT_PI * np.exp(-v * v),
        lambda v, f, d: -2.0 * v * d,
    ),
    sps.erfc: lambda x: _unary(
        x,
        sps.erfc,
        lambda v, f: -_TWO_OVER_SQRT_PI * np.exp(-v * v),
        lambda v, f, d: -2.0 * v * d,
    ),
    sps.erfcx: lambda x: _unary(
        x,
        sps.erfcx,
        lambda v, f: 2.0 * v * f - _TWO_OVER_SQRT_PI,
        lambda v, f, d: 2.0 * f + 2.0 * v * d,
    ),
    sps.ndtr: lambda x: _unary(x, sps.ndtr, lambda v, f: _phi(v), lambda v, f, d: -v * d),
    sps.log_ndtr: lambda x: _unary(x, sps.log_ndtr, _log_ndtr_d1, lambda v, f, d: -d * (v + d)),
    sps.ndtri: lambda x: _unary(
        x, sps.ndtri, lambda v, f: 1.0 / _phi(f), lambda v, f, d: f * d * d
    ),
    sps.expit: lambda x: _unary(
        x, sps.expit, lambda v, f: f * (1.0 - f), lambda v, f, d: d * (1.0 - 2.0 * f)
    ),
    sps.logit: lambda x: _unary(
        x,
        sps.logit,
        lambda v, f: 1.0 / (v * (1.0 - v)),
        lambda v, f, d: (2.0 * v - 1.0) * d * d,
    ),
    sps.exprel: lambda x: _unary(x, sps.exprel, _exprel_d1, _exprel_d2),
    sps.kv: lambda nu, z: _bessel_k(nu, z, scaled=False),
    sps.kve: lambda nu, z: _bessel_k(nu, z, scaled=True),
}


# ── array functions ───────────────────────────────────────────────────────────


def _norm_axes(axis: Any, nd: int) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(nd))
    if isinstance(axis, (tuple, list)):
        return tuple(int(a) % nd for a in axis)
    return (int(axis) % nd,)


def _sum(x: Any, axis: Any = None, keepdims: Any = False, **kw: Any) -> Any:
    keepdims = keepdims is True
    x = _coerce(x)
    if not isinstance(x, Jet):
        return np.sum(x, axis=axis, keepdims=keepdims)
    axes = _norm_axes(axis, x.ndim)
    v = np.sum(x.value, axis=axes, keepdims=keepdims)
    g = np.sum(x.grad, axis=axes, keepdims=keepdims)
    h = None if x.hess is None else np.sum(x.hess, axis=axes, keepdims=keepdims)
    return Jet(v, g, h)


def _mean(x: Any, axis: Any = None, keepdims: Any = False, **kw: Any) -> Any:
    keepdims = keepdims is True
    x = _coerce(x)
    if not isinstance(x, Jet):
        return np.mean(x, axis=axis, keepdims=keepdims)
    axes = _norm_axes(axis, x.ndim)
    cnt = int(np.prod([x.shape[a] for a in axes])) if axes else 1
    return _sum(x, axis=axis, keepdims=keepdims) / float(cnt)


def _cumsum(x: Any, axis: Any = None, **kw: Any) -> Any:
    x = _coerce(x)
    if not isinstance(x, Jet):
        return np.cumsum(x, axis=axis)
    if axis is None:
        x = x.ravel()
        axis = 0
    ax = int(axis) % x.ndim
    h = None if x.hess is None else np.cumsum(x.hess, axis=ax)
    return Jet(np.cumsum(x.value, axis=ax), np.cumsum(x.grad, axis=ax), h)


def _extreme(x: Any, axis: Any = None, keepdims: Any = False, largest: bool = True) -> Any:
    keepdims = keepdims is True
    x = _coerce(x)
    if not isinstance(x, Jet):
        return (np.max if largest else np.min)(x, axis=axis, keepdims=keepdims)
    if axis is None:
        x = x.ravel()
        axis = 0
    if isinstance(axis, (tuple, list)):
        raise JetError("max/min of a Jet over several axes is not supported")
    ax = int(axis) % x.ndim
    idx = np.argmax(x.value, axis=ax) if largest else np.argmin(x.value, axis=ax)
    idx = np.expand_dims(idx, ax)
    v = np.take_along_axis(x.value, idx, axis=ax)
    g = np.take_along_axis(x.grad, idx[..., None], axis=ax)
    h = None if x.hess is None else np.take_along_axis(x.hess, idx[..., None, None], axis=ax)
    j = Jet(v, g, h)
    return j if keepdims else _squeeze(j, axis=ax)


def _accumulate_extreme(x: Any, axis: int = 0, largest: bool = True) -> Any:
    x = _coerce(x)
    ax = int(axis) % x.ndim
    vm = np.moveaxis(x.value, ax, 0)
    m = vm.shape[0]
    idx = np.zeros(vm.shape, dtype=np.intp)
    best = vm[0].copy()
    cur = np.zeros(vm.shape[1:], dtype=np.intp)
    for i in range(1, m):
        upd = vm[i] > best if largest else vm[i] < best
        cur = np.where(upd, i, cur)
        best = np.where(upd, vm[i], best)
        idx[i] = cur
    gm = np.moveaxis(x.grad, ax, 0)
    v = np.take_along_axis(vm, idx, axis=0)
    g = np.take_along_axis(gm, idx[..., None], axis=0)
    h = None
    if x.hess is not None:
        hm = np.moveaxis(x.hess, ax, 0)
        h = np.moveaxis(np.take_along_axis(hm, idx[..., None, None], axis=0), 0, ax)
    return Jet(np.moveaxis(v, 0, ax), np.moveaxis(g, 0, ax), h)


def _squeeze(x: Any, axis: Any = None) -> Any:
    x = _coerce(x)
    if not isinstance(x, Jet):
        return np.squeeze(x, axis=axis)
    v = np.squeeze(x.value, axis=axis)
    return x.reshape(v.shape)


def _where(cond: Any, a: Any = None, b: Any = None) -> Any:
    if a is None and b is None:
        return np.where(value_of(cond))
    return _select(np.asarray(value_of(cond), dtype=bool), a, b)


def _reshape(x: Any, newshape: Any = None, order: str = "C", shape: Any = None, **kw: Any) -> Any:
    shp = shape if shape is not None else newshape
    x = _coerce(x)
    return x.reshape(shp) if isinstance(x, Jet) else np.reshape(x, shp)


def _ravel(x: Any, order: str = "C") -> Any:
    x = _coerce(x)
    return x.ravel() if isinstance(x, Jet) else np.ravel(x)


def _transpose(x: Any, axes: Any = None) -> Any:
    x = _coerce(x)
    if not isinstance(x, Jet):
        return np.transpose(x, axes)
    return x.transpose() if axes is None else x.transpose(axes)


def _moveaxis(x: Any, source: Any, destination: Any) -> Any:
    x = _coerce(x)
    nd = x.ndim
    order = [i for i in range(nd)]
    src = [int(s) % nd for s in np.atleast_1d(source)]
    dst = [int(d) % nd for d in np.atleast_1d(destination)]
    rest = [i for i in order if i not in src]
    perm: list = [None] * nd
    for s, d in zip(src, dst):
        perm[d] = s
    it = iter(rest)
    perm = [p if p is not None else next(it) for p in perm]
    return x.transpose(tuple(perm))


def _swapaxes(x: Any, a1: int, a2: int) -> Any:
    x = _coerce(x)
    perm = list(range(x.ndim))
    a1, a2 = a1 % x.ndim, a2 % x.ndim
    perm[a1], perm[a2] = perm[a2], perm[a1]
    return x.transpose(tuple(perm))


def _expand_dims(x: Any, axis: Any) -> Any:
    x = _coerce(x)
    v = np.expand_dims(x.value, axis)
    return x.reshape(v.shape)


def _atleast_1d(*xs: Any) -> Any:
    out = []
    for x in xs:
        x = _coerce(x)
        if isinstance(x, Jet):
            out.append(x.reshape(-1) if x.ndim == 0 else x)
        else:
            out.append(np.atleast_1d(x))
    return out[0] if len(out) == 1 else out


def _broadcast_to(x: Any, shape: Any, subok: bool = False) -> Any:
    x = _coerce(x)
    if not isinstance(x, Jet):
        return np.broadcast_to(x, shape)
    shape = tuple(np.atleast_1d(shape)) if not isinstance(shape, tuple) else shape
    n = x.n
    v = np.broadcast_to(x.value, shape)
    if x.ndim > len(shape):
        raise JetError("cannot broadcast to fewer dimensions")
    # align: the value broadcasts on the left, so the grad needs the same leading padding
    gv = x.grad.reshape((1,) * (len(shape) - x.ndim) + x.grad.shape)
    g = np.broadcast_to(gv, tuple(shape) + (n,))
    h = None
    if x.hess is not None:
        hv = x.hess.reshape((1,) * (len(shape) - x.ndim) + x.hess.shape)
        h = np.broadcast_to(hv, tuple(shape) + (n, n))
    return Jet(v, g, h)


def _broadcast_arrays(*args: Any, subok: bool = False) -> Any:
    shape = np.broadcast_shapes(*[np.shape(value_of(_coerce(a))) for a in args])
    return [_broadcast_to(_coerce(a), shape) for a in args]


def _concatenate(seq: Sequence[Any], axis: int = 0, **kw: Any) -> Any:
    seq = [_coerce(s) for s in seq]
    if not any(isinstance(s, Jet) for s in seq):
        return np.concatenate(seq, axis=axis)
    n, order = _meta(*seq)
    vals = [np.asarray(value_of(s)) for s in seq]
    if axis is None:
        return _concatenate([s.ravel() if isinstance(s, Jet) else np.ravel(s) for s in seq], 0)
    ax = int(axis) % vals[0].ndim
    v = np.concatenate(vals, axis=ax)
    g = np.concatenate([grad_of(s, n) for s in seq], axis=ax)
    h = None
    if order == 2:
        h = np.concatenate([hess_of(s, n) for s in seq], axis=ax)
    return Jet(v, g, h)


def _stack(seq: Sequence[Any], axis: int = 0, **kw: Any) -> Any:
    return stack(list(seq), axis=axis)


def _hstack(seq: Sequence[Any], **kw: Any) -> Any:
    seq = [_atleast_1d(_coerce(s)) for s in seq]
    nd = np.ndim(value_of(seq[0]))
    return _concatenate(seq, axis=0 if nd == 1 else 1)


def _diff(x: Any, n: int = 1, axis: int = -1, prepend: Any = None, append: Any = None) -> Any:
    x = _coerce(x)
    if prepend is not None or append is not None:
        parts = []
        if prepend is not None:
            pv = np.asarray(value_of(prepend))
            shp = list(x.shape)
            shp[axis] = 1 if pv.ndim == 0 else pv.shape[axis]
            parts.append(
                _broadcast_to(prepend, tuple(shp))
                if isinstance(prepend, Jet)
                else np.broadcast_to(pv, shp)
            )
        parts.append(x)
        if append is not None:
            av = np.asarray(value_of(append))
            shp = list(x.shape)
            shp[axis] = 1 if av.ndim == 0 else av.shape[axis]
            parts.append(
                _broadcast_to(append, tuple(shp))
                if isinstance(append, Jet)
                else np.broadcast_to(av, shp)
            )
        x = _concatenate(parts, axis=axis)
    if not isinstance(x, Jet):
        return np.diff(x, n=n, axis=axis)
    ax = int(axis) % x.ndim
    out = x
    for _ in range(n):
        sl1 = [slice(None)] * out.ndim
        sl0 = [slice(None)] * out.ndim
        sl1[ax] = slice(1, None)
        sl0[ax] = slice(None, -1)
        out = out[tuple(sl1)] - out[tuple(sl0)]
    return out


def _dot(a: Any, b: Any, out: Any = None) -> Any:
    a, b = _coerce(a), _coerce(b)
    av, bv = np.asarray(value_of(a)), np.asarray(value_of(b))
    if av.ndim == 0 or bv.ndim == 0:
        return a * b
    return _matmul(a, b)


def _bilinear(op: Callable[[np.ndarray, np.ndarray], np.ndarray], a: Any, b: Any) -> Jet:
    """``op`` bilinear and broadcasting over leading axes (matmul, outer products)."""
    n, order = _meta(a, b)
    av, bv = np.asarray(value_of(a)), np.asarray(value_of(b))
    v = op(av, bv)
    nd = v.ndim

    rank = max(av.ndim, bv.ndim)

    def lead(x: np.ndarray, k: int) -> np.ndarray:
        # move the k trailing direction axes first, then pad the value part with leading
        # singletons so that batch axes of both operands stay right-aligned
        y = np.moveaxis(x, list(range(x.ndim - k, x.ndim)), list(range(k)))
        pad = rank - (x.ndim - k)
        return y.reshape(y.shape[:k] + (1,) * pad + y.shape[k:]) if pad > 0 else y

    def trail(x: np.ndarray, k: int) -> np.ndarray:
        return np.moveaxis(x, list(range(k)), list(range(x.ndim - k, x.ndim)))

    g = np.zeros(v.shape + (n,), dtype=np.result_type(v, float))
    if isinstance(a, Jet):
        g = g + trail(op(lead(a.grad, 1), bv), 1)
    if isinstance(b, Jet):
        g = g + trail(op(av, lead(b.grad, 1)), 1)
    h = None
    if order == 2:
        h = np.zeros(v.shape + (n, n), dtype=g.dtype)
        if isinstance(a, Jet):
            h = h + trail(op(lead(a.hess, 2), bv), 2)
        if isinstance(b, Jet):
            h = h + trail(op(av, lead(b.hess, 2)), 2)
        if isinstance(a, Jet) and isinstance(b, Jet):
            ga = lead(a.grad, 1)  # (n,) + Sa
            gb = lead(b.grad, 1)
            cross = op(ga[:, None], gb[None, :])  # (n, n) + S
            h = h + trail(cross, 2) + np.swapaxes(trail(cross, 2), -1, -2)
    assert g.shape[:nd] == v.shape
    return Jet(v, g, h)


def _matmul(a: Any, b: Any) -> Any:
    a, b = _coerce(a), _coerce(b)
    if not isinstance(a, Jet) and not isinstance(b, Jet):
        return np.matmul(a, b)
    av, bv = np.asarray(value_of(a)), np.asarray(value_of(b))
    # vector cases: promote to matrices so the leading direction axes broadcast correctly
    if av.ndim == 1 and bv.ndim == 1:
        r = _sum(a * b, axis=-1)
        return Jet(np.matmul(av, bv), r.grad, r.hess)  # value bit-identical to NumPy
    if av.ndim == 1:
        r = _bilinear(np.matmul, _expand_dims(a, 0) if isinstance(a, Jet) else av[None, :], b)
        return _squeeze(r, axis=-2)
    if bv.ndim == 1:
        r = _bilinear(np.matmul, a, _expand_dims(b, -1) if isinstance(b, Jet) else bv[:, None])
        return _squeeze(r, axis=-1)
    return _bilinear(np.matmul, a, b)


def _outer_fn(a: Any, b: Any, out: Any = None) -> Any:
    a, b = _coerce(a), _coerce(b)
    a1 = a.ravel() if isinstance(a, Jet) else np.ravel(a)
    b1 = b.ravel() if isinstance(b, Jet) else np.ravel(b)
    return np.multiply(a1[:, None], b1[None, :])


def _clip(a: Any, a_min: Any = None, a_max: Any = None, out: Any = None, **kw: Any) -> Any:
    a_min = kw.pop("min", a_min)
    a_max = kw.pop("max", a_max)
    r = a
    if a_min is not None:
        r = _maximum(r, a_min)
    if a_max is not None:
        r = _minimum(r, a_max)
    return r


def _fft_like(fn: Callable[..., np.ndarray]) -> Callable[..., Any]:
    def impl(a: Any, n: Any = None, axis: int = -1, norm: Any = None, out: Any = None) -> Any:
        a = _coerce(a)
        if not isinstance(a, Jet):
            return fn(a, n=n, axis=axis, norm=norm)
        ax = int(axis) % a.ndim
        v = fn(a.value, n=n, axis=ax, norm=norm)
        g = fn(a.grad, n=n, axis=ax, norm=norm)
        h = None if a.hess is None else fn(a.hess, n=n, axis=ax, norm=norm)
        return Jet(v, g, h)

    return impl


def _interp(
    x: Any, xp: Any, fp: Any, left: Any = None, right: Any = None, period: Any = None
) -> Any:
    if isinstance(_coerce(xp), Jet):
        raise JetError("np.interp: the abscissae xp cannot be Jets")
    if period is not None:
        raise JetError("np.interp(period=...) is not supported on Jets")
    x, fp = _coerce(x), _coerce(fp)
    xpv = np.asarray(xp, dtype=float)
    xv = np.asarray(value_of(x), dtype=float)
    m = xpv.size
    j = np.clip(np.searchsorted(xpv, xv, side="right") - 1, 0, m - 2)
    x0, x1 = xpv[j], xpv[j + 1]
    w = (xv - x0) / (x1 - x0)
    below, above = xv < xpv[0], xv > xpv[-1]
    if isinstance(fp, Jet):
        f0, f1 = fp[j], fp[j + 1]
    else:
        fpv = np.asarray(fp)
        f0, f1 = fpv[j], fpv[j + 1]
    wj: Any = w
    if isinstance(x, Jet):
        wj = (x - x0) / (x1 - x0)
    res = f0 + (f1 - f0) * wj
    lv = fp[0] if left is None else left
    rv = fp[-1] if right is None else right
    res = _select(below, lv, res)
    res = _select(above, rv, res)
    return res


def _trapezoid(y: Any, x: Any = None, dx: float = 1.0, axis: int = -1) -> Any:
    y = _coerce(y)
    if isinstance(_coerce(x), Jet):
        raise JetError("trapezoid: the abscissae cannot be Jets")
    ax = int(axis) % np.ndim(value_of(y))
    m = np.shape(value_of(y))[ax]
    if x is None:
        w = np.full(m, float(dx))
        w[0] = w[-1] = 0.5 * dx
    else:
        xv = np.asarray(x, dtype=float)
        d = np.diff(xv)
        w = np.zeros(m)
        w[:-1] += 0.5 * d
        w[1:] += 0.5 * d
    shape = [1] * np.ndim(value_of(y))
    shape[ax] = m
    return _sum(y * w.reshape(shape), axis=ax)


def _like(fn: Callable[..., np.ndarray]) -> Callable[..., Any]:
    def impl(a: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(value_of(_coerce(a)), *args, **kwargs)

    return impl


def _copy(a: Any, *args: Any, **kwargs: Any) -> Any:
    a = _coerce(a)
    return a.copy() if isinstance(a, Jet) else np.copy(a)


def _angle(z: Any, deg: bool = False) -> Any:
    z = _coerce(z)
    r = _arctan2(_imag(z), _real(z))
    return r * (180.0 / np.pi) if deg else r


def _polyval(p: Any, x: Any) -> Any:
    coeffs = list(p) if not isinstance(p, Jet) else [p[i] for i in range(p.shape[0])]
    y: Any = coeffs[0] * (x * 0 + 1) if coeffs else 0.0
    for c in coeffs[1:]:
        y = y * x + c
    return y


def _flip(m: Any, axis: Any = None) -> Any:
    m = _coerce(m)
    if not isinstance(m, Jet):
        return np.flip(m, axis)
    axes = _norm_axes(axis, m.ndim)
    sl = [slice(None)] * m.ndim
    for a in axes:
        sl[a] = slice(None, None, -1)
    return m[tuple(sl)]


def _isclose(a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:
    return np.isclose(value_of(a), value_of(b), *args, **kwargs)


def _allclose(a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:
    return np.allclose(value_of(a), value_of(b), *args, **kwargs)


def _iscomplexobj(x: Any) -> bool:
    return bool(np.iscomplexobj(value_of(_coerce(x))))


def _real_fn(x: Any) -> Any:
    return _real(x)


def _imag_fn(x: Any) -> Any:
    return _imag(x)


_FUNCTIONS: Dict[Any, Callable[..., Any]] = {
    np.sum: _sum,
    np.mean: _mean,
    np.cumsum: _cumsum,
    np.max: lambda a, axis=None, keepdims=False, **k: _extreme(a, axis, keepdims, True),
    np.min: lambda a, axis=None, keepdims=False, **k: _extreme(a, axis, keepdims, False),
    np.amax: lambda a, axis=None, keepdims=False, **k: _extreme(a, axis, keepdims, True),
    np.amin: lambda a, axis=None, keepdims=False, **k: _extreme(a, axis, keepdims, False),
    np.where: _where,
    np.real: _real_fn,
    np.imag: _imag_fn,
    np.angle: _angle,
    np.reshape: _reshape,
    np.ravel: _ravel,
    np.transpose: _transpose,
    np.moveaxis: _moveaxis,
    np.swapaxes: _swapaxes,
    np.squeeze: lambda a, axis=None: _squeeze(a, axis),
    np.expand_dims: _expand_dims,
    np.atleast_1d: _atleast_1d,
    np.broadcast_to: _broadcast_to,
    np.broadcast_arrays: _broadcast_arrays,
    np.concatenate: _concatenate,
    np.stack: _stack,
    np.hstack: _hstack,
    np.diff: _diff,
    np.dot: _dot,
    np.outer: _outer_fn,
    np.clip: _clip,
    np.fft.fft: _fft_like(np.fft.fft),
    np.fft.ifft: _fft_like(np.fft.ifft),
    np.interp: _interp,
    np.zeros_like: _like(np.zeros_like),
    np.ones_like: _like(np.ones_like),
    np.empty_like: _like(np.empty_like),
    np.full_like: lambda a, fill_value, *args, **kw: np.full_like(
        value_of(_coerce(a)), value_of(fill_value), *args, **kw
    ),
    np.shape: lambda a: np.shape(value_of(_coerce(a))),
    np.ndim: lambda a: np.ndim(value_of(_coerce(a))),
    np.size: lambda a, axis=None: np.size(value_of(_coerce(a)), axis),
    np.copy: _copy,
    np.polyval: _polyval,
    np.flip: _flip,
    np.isclose: _isclose,
    np.allclose: _allclose,
    np.iscomplexobj: _iscomplexobj,
    np.result_type: lambda *a: np.result_type(*[value_of(_coerce(x)) for x in a]),
}
if hasattr(np, "trapezoid"):
    _FUNCTIONS[np.trapezoid] = _trapezoid
if hasattr(np, "trapz"):
    try:
        _FUNCTIONS[np.trapz] = _trapezoid  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass


# ── convenience ───────────────────────────────────────────────────────────────


def jet_jacobian(fun: Callable[[Any], Any], x: Any) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Value and exact Jacobian of ``fun`` at ``x`` by forward mode.

    ``fun`` maps a 1-d array to an array (or scalar) and must be written with
    NumPy operations that have jet rules. Returns ``(f(x), J)`` with
    ``J[..., k] = ∂f/∂x_k``. Raises :class:`JetError` if ``fun`` drops
    derivatives anywhere.
    """
    xj = variable_vector(x)
    out = _coerce(fun(xj))
    if not isinstance(out, Jet):
        raise JetError("the function returned a value without derivatives")
    return np.array(out.value), np.array(out.grad)


def jet_hessian(fun: Callable[[Any], Any], x: Any) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Value, exact gradient and exact Hessian of a scalar ``fun`` at ``x`` (second-order jets)."""
    xj = variable_vector(x, order=2)
    out = _coerce(fun(xj))
    if not isinstance(out, Jet) or out.hess is None:
        raise JetError("the function returned a value without second derivatives")
    return out.value, np.array(out.grad), np.array(out.hess)
