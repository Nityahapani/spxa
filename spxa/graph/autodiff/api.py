r"""
Function transforms for exact derivatives: :func:`grad`, :func:`value_and_grad`,
:func:`jacobian`, :func:`hessian`.

Each takes a function written with NumPy / SciPy operations and returns a
function computing exact derivatives by forward-mode jets
(:mod:`spxa.graph.autodiff.jet`), in the style of JAX::

    import numpy as np
    import spxa
    from spxa.graph.autodiff.process import DiffProcess
    from spxa.zoo.levy import VarianceGamma

    vg = DiffProcess(VarianceGamma(0.2, 0.3, -0.1))
    kurt = lambda th: vg.cumulants(4, th)[3] / vg.cumulants(4, th)[1] ** 2
    spxa.grad(kurt)(vg.values)          # d(excess kurtosis)/d(sigma, nu, theta)

The argument is a scalar or an array of any shape; derivatives have the
argument's shape appended (``jacobian``) or are shaped like it (``grad``).
Every result is exact up to rounding (labelled
:attr:`~spxa.graph.autodiff.exactness.GradientExactness.EXACT`): a function
that would lose derivative information (``float()`` of a traced value, an
operation without a jet rule) raises
:class:`~spxa.graph.autodiff.jet.JetError` rather than returning a
difference quotient. Complex outputs are differentiated as holomorphic in
the real argument (``d Re f + i d Im f``); :func:`grad` and
:func:`hessian` need a real scalar output.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

from spxa.graph.autodiff.exactness import GradientExactness
from spxa.graph.autodiff.jet import Jet, JetError, _coerce, variable_vector

__all__ = ["grad", "value_and_grad", "jacobian", "hessian", "EXACTNESS"]

#: How the transforms' derivatives are obtained.
EXACTNESS = GradientExactness.EXACT


def _flat_call(f: Callable[..., Any], x: Any, order: int) -> Tuple[Any, Tuple[int, ...]]:
    x0 = np.asarray(x, dtype=float)
    shape = x0.shape
    xj = variable_vector(x0.ravel(), order=order)
    arg = xj[0] if shape == () else xj.reshape(shape)
    out = _coerce(f(arg))
    return out, shape


def _require_jet(out: Any, n: int, what: str) -> Jet:
    if isinstance(out, Jet):
        return out
    # a constant function: derivatives are exactly zero
    v = np.asarray(out)
    if v.dtype == object:
        raise JetError(f"{what}: the function returned an object array; use jet.stack")
    return Jet(v, np.zeros(v.shape + (n,), dtype=v.dtype if np.iscomplexobj(v) else float))


def _real_scalar(out: Jet, what: str) -> None:
    if np.ndim(out.value) != 0:
        raise ValueError(f"{what} needs a scalar output; use jacobian for arrays")
    if np.iscomplexobj(out.value) and np.imag(out.value) != 0:
        raise ValueError(f"{what} needs a real output")


def value_and_grad(f: Callable[[Any], Any]) -> Callable[[Any], Tuple[float, np.ndarray]]:
    """``x -> (f(x), df/dx)`` for a real scalar ``f``; the gradient has ``x``'s shape."""

    def run(x: Any) -> Tuple[float, np.ndarray]:
        out, shape = _flat_call(f, x, 1)
        n = int(np.prod(shape)) if shape else 1
        out = _require_jet(out, n, "value_and_grad")
        _real_scalar(out, "value_and_grad")
        g = np.real(np.asarray(out.grad)).reshape(shape)
        return float(np.real(out.value)), g

    return run


def grad(f: Callable[[Any], Any]) -> Callable[[Any], np.ndarray]:
    """``x -> df/dx`` for a real scalar ``f`` (exact, forward mode)."""
    vg = value_and_grad(f)
    return lambda x: vg(x)[1]


def jacobian(f: Callable[[Any], Any]) -> Callable[[Any], np.ndarray]:
    """``x -> J`` with ``J[i..., k...] = d f[i...] / d x[k...]`` (shape ``f.shape + x.shape``)."""

    def run(x: Any) -> np.ndarray:
        out, shape = _flat_call(f, x, 1)
        n = int(np.prod(shape)) if shape else 1
        out = _require_jet(out, n, "jacobian")
        g = np.asarray(out.grad)
        return g.reshape(np.shape(out.value) + shape)

    return run


def hessian(f: Callable[[Any], Any]) -> Callable[[Any], np.ndarray]:
    """``x -> d^2 f / dx^2`` for a real scalar ``f`` (shape ``x.shape + x.shape``)."""

    def run(x: Any) -> np.ndarray:
        out, shape = _flat_call(f, x, 2)
        n = int(np.prod(shape)) if shape else 1
        out = _require_jet(out, n, "hessian")
        _real_scalar(out, "hessian")
        if out.hess is None:
            return np.zeros(shape + shape)
        return np.real(np.asarray(out.hess)).reshape(shape + shape)

    return run
