r"""
European option prices under a composite Lévy model, by the COS method.

The log-price is modelled risk-neutrally as

.. math::
    S_T = S_0 \exp\big((r - q)T + \omega T + X_T\big), \qquad \omega = -K_X(1),

where :math:`X` is the composite model and :math:`K_X(z) = \log E[e^{zX_1}]`.
The correction :math:`\omega` makes :math:`e^{-(r-q)t}S_t` a martingale, so it
needs :math:`E[e^{X_1}] < \infty`: the model's strip
(:meth:`~spxa.discovery.expr.Model.strip`) must contain :math:`z = 1`. Any drift in
:math:`X` cancels in :math:`X_T + \omega T`, so drift leaves do not affect prices.

Prices use the COS expansion of Fang & Oosterlee (2008) in the variable
:math:`z = \log(S_T/S_0)` on a fixed interval :math:`[a, b]` per maturity:

.. math::
    P(K) = e^{-rT}\sum_{k=0}^{N-1}{}' \mathrm{Re}\big[\varphi_T(u_k)e^{-iu_k a}\big]
    \frac{2}{b-a}\big[K\psi_k(a, \kappa) - S_0\chi_k(a, \kappa)\big],

with :math:`u_k = k\pi/(b-a)` and :math:`\kappa = \log(K/S_0)` clipped to
:math:`[a, b]`. Here :math:`\psi_k` and :math:`\chi_k` are the cosine
integrals of :math:`1` and :math:`e^z` over :math:`[a, \kappa]`. Puts are
priced this way because their payoff is bounded, and calls follow by
put–call parity. The interval is
:math:`c_1 \pm L\sqrt{c_2 + \sqrt{c_4}}` from the cumulants of :math:`z`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spxa.discovery.expr import Model


def martingale_correction(model: Model) -> float:
    r""":math:`\omega = -K_X(1)`; raises if :math:`E[e^{X_1}]` is infinite."""
    lo, hi = model.strip()
    if not hi > 1.0:
        raise ValueError(
            f"E[exp(X_1)] is infinite for {model.render(3)} (exponential moments only up to "
            f"z = {hi:.4g}); no risk-neutral price exists"
        )
    k1 = complex(np.asarray(model.log_mgf(1.0)).ravel()[0])
    return -float(k1.real)


def _chi_psi(a: float, b: float, c: float, d: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, ...]:
    """Cosine integrals of e^z and 1 over [c, d] on the basis of [a, b]; shape (len(d), len(u))."""
    d = d[:, None]
    w = u[None, :]
    arg_d, arg_c = w * (d - a), w * (c - a)
    chi = (
        np.cos(arg_d) * np.exp(d)
        - np.cos(arg_c) * np.exp(c)
        + w * np.sin(arg_d) * np.exp(d)
        - w * np.sin(arg_c) * np.exp(c)
    ) / (1.0 + w**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(w == 0, d - c, (np.sin(arg_d) - np.sin(arg_c)) / np.where(w == 0, 1.0, w))
    return chi, psi


def cos_prices(
    model: Model,
    spot: float,
    strikes: Any,
    maturity: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    call: Any = True,
    n_terms: int = 1024,
    width: float = 12.0,
) -> np.ndarray:
    """European option prices at one maturity (see the module docstring).

    ``call`` is a boolean or a boolean array matching ``strikes``.
    """
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    is_call = np.broadcast_to(np.asarray(call, dtype=bool), K.shape)
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if np.any(K <= 0) or spot <= 0:
        raise ValueError("spot and strikes must be positive")
    T = float(maturity)
    omega = martingale_correction(model)
    k = model.cumulants(4)
    c1 = (rate - dividend + omega + k[0]) * T
    c2, c4 = max(k[1], 0.0) * T, max(k[3], 0.0) * T
    half = width * np.sqrt(c2 + np.sqrt(c4))
    a, b = c1 - half, c1 + half
    u = np.arange(n_terms) * np.pi / (b - a)
    phi = np.exp(1j * u * (rate - dividend + omega) * T + T * model.char_exponent(u))
    weights = np.real(phi * np.exp(-1j * u * a))
    weights[0] *= 0.5
    kappa = np.clip(np.log(K / spot), a, b)
    chi, psi = _chi_psi(a, b, a, kappa, u)
    payoff = 2.0 / (b - a) * (K[:, None] * psi - spot * chi)
    put = np.exp(-rate * T) * payoff @ weights
    put = np.maximum(put, 0.0)
    forward_value = spot * np.exp(-dividend * T) - K * np.exp(-rate * T)
    return np.where(is_call, put + forward_value, put)
