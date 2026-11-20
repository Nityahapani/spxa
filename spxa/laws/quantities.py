r"""
The quantities laws talk about, computed from a process.

Each function reads one well-defined quantity off a
:class:`~spxa.core.process.Process`:

- :func:`cumulants`, :func:`mean` and :func:`variance` of :math:`X_t`, from
  the process's own ``cumulants`` (per unit time, multiplied by
  :math:`t`).
- :func:`char_func`: :math:`\varphi_t(u) = E[e^{iuX_t}]`, from the process's
  own ``char_func``.
- :func:`char_exponent`: :math:`\psi(u)` with :math:`\varphi_t(u) = e^{t\psi(u)}`,
  computed from the Lévy–Khintchine triplet :math:`(b, \sigma^2, \nu)`,

  .. math::
      \psi(u) = ibu - \tfrac12\sigma^2u^2
      + \int\big(e^{iux} - 1 - iux\mathbb 1_{|x|\le1}\big)\,\nu(dx).

  It is defined directly, not as :math:`\log\varphi`, so there is no
  branch ambiguity, and it accepts complex :math:`u`.
- :func:`laplace_exponent`: :math:`K(z) = \log E[e^{zX_1}] = \psi(-iz)`, for
  :math:`z` in the strip of finite exponential moments. For a
  subordinator this includes the whole half-plane
  :math:`\operatorname{Re} z \le 0`.

These are deliberately thin. A law that compares, say, :func:`char_func`
of a sum with the product of :func:`char_func` of its parts tests the
process implementations, not this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spxa.core.process import Process


def cumulants(X: Process, order: int = 4, t: float = 1.0) -> np.ndarray:
    r""":math:`\kappa_1, \dots, \kappa_{order}` of :math:`X_t`, as an array."""
    k = X.cumulants(order)
    return np.array([float(k[n]) for n in range(1, order + 1)]) * t


def mean(X: Process, t: float = 1.0) -> float:
    """:math:`E[X_t]`."""
    return float(cumulants(X, 1, t)[0])


def variance(X: Process, t: float = 1.0) -> float:
    r""":math:`\mathrm{Var}(X_t)`."""
    return float(cumulants(X, 2, t)[1])


def char_func(X: Process, u: Any, t: float = 1.0) -> np.ndarray:
    r""":math:`E[e^{iuX_t}]` at each frequency in ``u`` (always an array)."""
    u_arr = np.atleast_1d(np.asarray(u, dtype=float))
    return np.atleast_1d(np.asarray(X.char_func(u_arr, t), dtype=complex))


def char_exponent(X: Process, u: Any) -> np.ndarray:
    r""":math:`\psi(u)` from the Lévy–Khintchine triplet (complex ``u`` allowed)."""
    trip = X.triplet
    u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
    out = np.empty(u_arr.shape, dtype=complex)
    for i, ui in enumerate(u_arr.flat):
        out.flat[i] = (
            1j * trip.b * ui - 0.5 * trip.sigma_sq * ui**2 + trip.nu.jump_exponent(complex(ui))
        )
    return out


def laplace_exponent(X: Process, z: Any) -> np.ndarray:
    r""":math:`K(z) = \log E[e^{zX_1}] = \psi(-iz)`."""
    return char_exponent(X, -1j * np.atleast_1d(np.asarray(z, dtype=complex)))


def mgf(X: Process, z: Any, t: float = 1.0) -> np.ndarray:
    r""":math:`E[e^{zX_t}] = e^{tK(z)}`."""
    return np.exp(t * laplace_exponent(X, z))
