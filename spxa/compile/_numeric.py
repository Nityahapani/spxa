r"""
NumPy implementations of the IR operations.

The reference interpreter, constant folding and the generated NumPy code all
call these, so a folded constant, an interpreted node and a compiled node
agree bit for bit.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Any, Callable, Dict, Sequence

import numpy as np
import scipy.special as sps

__all__ = ["apply_op", "sinc", "sinc_d1", "kve_dnu", "NP_FUNCS"]

#: Below this modulus :func:`sinc_d1` uses its Taylor series (see there).
_SINC_SERIES_RADIUS = 1.0
#: (-1)^k 2k / (2k+1)!, k = 1..10, correctly rounded.
_SINC_D1_COEFFS = tuple(
    float(Fraction((-1) ** k * 2 * k, factorial(2 * k + 1))) for k in range(1, 11)
)


def sinc(z: Any) -> Any:
    r""":math:`\sin z/z` with the value 1 at :math:`z = 0`."""
    z = np.asarray(z)
    small = z == 0
    safe = np.where(small, 1.0, z)
    out = np.where(small, 1.0, np.sin(safe) / safe)
    return out[()]


def sinc_d1(z: Any) -> Any:
    r"""
    :math:`\frac{d}{dz}\frac{\sin z}{z} = \frac{\cos z - \sin z/z}{z}`.

    The difference cancels for small :math:`z` (it is :math:`-z/3 + O(z^3)`).
    For :math:`|z| < 1` the Taylor series
    :math:`\sum_{k\ge1}(-1)^k\,2k\,z^{2k-1}/(2k+1)!` is summed to :math:`z^{19}`
    (coefficients correctly rounded); the first omitted term is below
    :math:`10^{-20}` of the sum there. Outside, the direct form loses at most a
    few units in the last place to the cancellation, :math:`O(\epsilon/|z|^2)`.
    """
    z = np.asarray(z)
    small = np.abs(z) < _SINC_SERIES_RADIUS
    zs = np.where(small, z, 0.0)
    z2 = zs * zs
    acc = _SINC_D1_COEFFS[-1] + 0.0 * z2
    for c in _SINC_D1_COEFFS[-2::-1]:
        acc = c + z2 * acc
    series = zs * acc
    zl = np.where(small, 1.0, z)
    direct = (np.cos(zl) - np.sin(zl) / zl) / zl
    return np.where(small, series, direct)[()]


def kve_dnu(nu: Any, z: Any) -> Any:
    r""":math:`\partial_\nu\,e^zK_\nu(z)` (see :func:`spxa.graph.autodiff.jet.kv_order_derivative`)."""
    from spxa.graph.autodiff.jet import kv_order_derivative

    return kv_order_derivative(nu, z, scaled=True)[()]


def _where(c: Any, a: Any, b: Any) -> Any:
    return np.where(c, a, b)


NP_FUNCS: Dict[str, Callable[..., Any]] = {
    "add": np.add,
    "mul": np.multiply,
    "neg": np.negative,
    "div": np.true_divide,
    "pow": np.power,
    "exp": np.exp,
    "expm1": np.expm1,
    "log": np.log,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinc": sinc,
    "sinc_d1": sinc_d1,
    "abs": np.abs,
    "sign": np.sign,
    "real": np.real,
    "imag": np.imag,
    "gamma": sps.gamma,
    "digamma": sps.digamma,
    "kve": sps.kve,
    "kve_dnu": kve_dnu,
    "ge": np.greater_equal,
    "gt": np.greater,
    "where": _where,
}


def apply_op(op: str, vals: Sequence[Any], payload: Any = None) -> Any:
    """Evaluate one operation on concrete operands."""
    if op == "polygamma":
        return sps.polygamma(payload, vals[0])
    fn = NP_FUNCS.get(op)
    if fn is None:
        raise KeyError(f"no NumPy implementation for op {op!r}")
    return fn(*vals)
