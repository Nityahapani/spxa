r"""
Expectations: how a law states what must hold.

Inside a law, call :func:`expect_close`, :func:`expect` or :func:`implies`.
Any of them raises :class:`LawViolation` when the claim fails. A law may
also return ``False``. :func:`assume` discards an example that does not
meet a precondition, and the runner then draws another.

Numerical tolerance
    :func:`expect_close` compares with
    :math:`|a - b| \le \mathrm{atol} + \mathrm{rtol}\cdot\max(|a|, |b|)`,
    elementwise for arrays and on complex values. It is symmetric in
    ``a`` and ``b``, unlike ``numpy.isclose``. Infinite values must be
    equal. NaN equals NaN only when ``equal_nan=True``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class LawViolation(AssertionError):
    """A law's claim failed for the current example."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})


class Discard(Exception):
    """Raised by :func:`assume`: the example does not satisfy a precondition."""


def assume(condition: Any, reason: str = "") -> None:
    """Discard the current example unless ``condition`` holds."""
    if not condition:
        raise Discard(reason)


def expect(condition: Any, message: str = "expectation failed", **details: Any) -> None:
    """Fail the law unless ``condition`` is true."""
    if not condition:
        raise LawViolation(message, details)


def implies(premise: Any, conclusion: Any, message: str = "") -> None:
    """Fail when ``premise`` holds and ``conclusion`` does not.

    ``conclusion`` may be a zero-argument callable, which is evaluated only
    when ``premise`` holds.
    """
    if not premise:
        return
    ok = conclusion() if callable(conclusion) else conclusion
    if not ok:
        raise LawViolation(message or "premise holds but conclusion fails")


def _fmt(x: Any) -> str:
    a = np.asarray(x)
    if a.ndim == 0:
        return repr(a.item())
    return np.array2string(a, precision=6, threshold=8)


def expect_close(
    a: Any,
    b: Any,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    what: str = "",
    equal_nan: bool = False,
) -> None:
    """Fail unless ``a`` and ``b`` agree within ``atol + rtol * max(|a|, |b|)``.

    On failure the details record both sides, the absolute and relative
    error, and the worst index for arrays.
    """
    x = np.asarray(a, dtype=complex)
    y = np.asarray(b, dtype=complex)
    try:
        x, y = np.broadcast_arrays(x, y)
    except ValueError:
        raise LawViolation(
            f"{what or 'values'}: shapes {x.shape} and {y.shape} differ",
            {"left": a, "right": b},
        ) from None
    nan_x, nan_y = np.isnan(x), np.isnan(y)
    with np.errstate(invalid="ignore", over="ignore"):
        err = np.abs(x - y)
        scale = np.maximum(np.abs(x), np.abs(y))
        finite = np.isfinite(x) & np.isfinite(y)
        # infinities must match exactly: inf <= rtol * inf would accept inf vs -inf
        ok = np.where(finite, err <= atol + rtol * scale, x == y)
    if equal_nan:
        ok |= nan_x & nan_y
    else:
        ok &= ~(nan_x | nan_y)
    if np.all(ok):
        return
    bad = np.flatnonzero(~ok.ravel())
    ranked = np.where(ok.ravel(), -1.0, np.where(np.isfinite(err.ravel()), err.ravel(), np.inf))
    worst = int(np.argmax(ranked))
    xi, yi = x.ravel()[worst], y.ravel()[worst]
    ei = float(err.ravel()[worst])
    rel = ei / float(scale.ravel()[worst]) if scale.ravel()[worst] > 0 else float("inf")

    def real(v: complex) -> Any:
        return v.real if v.imag == 0 else v

    details: dict[str, Any] = {
        "left": real(xi),
        "right": real(yi),
        "abs_error": ei,
        "rel_error": rel,
        "rtol": rtol,
        "atol": atol,
    }
    where = ""
    if x.ndim > 0:
        idx = np.unravel_index(worst, x.shape)
        details["index"] = tuple(int(i) for i in idx)
        details["n_mismatched"] = int(bad.size)
        where = f" at index {details['index']} ({bad.size} of {x.size} mismatched)"
    raise LawViolation(
        f"{what or 'values'} differ{where}: {_fmt(real(xi))} vs {_fmt(real(yi))} "
        f"(abs error {ei:.3g}, rel error {rel:.3g})",
        details,
    )
