r"""
COS option prices and their exact parameter Jacobian from compiled kernels.

Conventions follow :mod:`spxa.discovery.pricing` and
:mod:`spxa.graph.autodiff.fourier`: the risk-neutral log-price is

.. math::
    \log(S_T/S_0) = (r - q)T + \omega T + X_T, \qquad \omega = -\psi(-i),

and European prices come from the COS expansion of Fang & Oosterlee (2008)
on :math:`[a, b] = c_1 \pm L\sqrt{c_2 + \sqrt{c_4}}` built from the exact
cumulants. Puts use the bounded payoff coefficients; calls follow by put–call
parity. For a batch of parameter vectors every row gets its own interval
and frequency grid, and all rows are priced by one compiled call.

The Jacobian is the exact derivative of the discretised price, including the
dependence of :math:`[a, b]` (hence of the frequencies
:math:`u_k = k\pi/(b - a)`) on the parameters. With
:math:`g_k = iu_k((r - q + \omega)T - a) + T\psi(u_k)` and
:math:`\mathrm{coef}_k = \mathrm{Re}\,e^{g_k}`,

.. math::
    \partial_j g_k = iu_kT\,\partial_j\omega + T\,\partial_j\psi(u_k)
      + \big(i((r - q + \omega)T - a) + T\psi'(u_k)\big)\partial_j u_k
      - iu_k\,\partial_j a, \qquad
    \partial_j u_k = -u_k\frac{\partial_j b - \partial_j a}{b - a},

and the payoff matrix contributes its closed-form derivatives in :math:`a`
and :math:`b`. :math:`\partial\psi/\partial\theta`, :math:`\psi'` and the
cumulant Jacobian all come from generated code; nothing is differenced.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from spxa.compile.ir import CompileError

__all__ = ["cos_price"]


def _payoff_ab(
    K: np.ndarray, spot: float, a: np.ndarray, b: np.ndarray, n_terms: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Put payoff coefficients :math:`P_{bmk}` and :math:`\partial_a P`, :math:`\partial_b P`.

    :math:`P = \tfrac{2}{L}(K\psi_k - S_0\chi_k)` (halved at :math:`k = 0`),
    :math:`L = b - a`, :math:`w = k\pi/L`, :math:`d = \mathrm{clip}(\log(K/S_0), a, b)`,
    :math:`\vartheta = w(d - a)`, :math:`\chi = (\cos\vartheta\,e^d - e^a +
    w\sin\vartheta\,e^d)/(1 + w^2)`, :math:`\psi = \sin\vartheta/w` (:math:`d - a`
    at :math:`k = 0`). ``a`` and ``b`` have shape ``(B,)``; outputs ``(B, M, N)``.
    """
    k_idx = np.arange(int(n_terms))
    A, Bb = a[:, None, None], b[:, None, None]
    L = Bb - A
    w = k_idx[None, None, :] * np.pi / L
    ell = np.log(K / spot)[None, :, None]
    lo, hi = ell < A, ell > Bb
    d = np.where(lo, A, np.where(hi, Bb, ell))
    half = np.ones(k_idx.size)
    half[0] = 0.5
    first = (k_idx == 0)[None, None, :]
    th = w * (d - A)
    c, s = np.cos(th), np.sin(th)
    ed, ea = np.exp(d), np.exp(A)
    den = 1.0 + w * w
    sed, ced = s * ed, c * ed
    N = ced - ea + w * sed
    ws = np.where(first, 1.0, w)
    psi = np.where(first, d - A, s / ws)
    Kc = K[None, :, None]
    core = Kc * psi - spot * (N / den)
    out = [2.0 / L * core * half]
    for end in (0, 1):
        sign = 1.0 if end == 0 else -1.0  # dw = sign * w / L and dL = -sign
        w_p = sign * w / L
        d_p = (lo if end == 0 else hi).astype(float)
        a_p = 1.0 if end == 0 else 0.0
        th_p = w_p * (d - A) + w * (d_p - a_p)
        N_p = (w * ced - sed) * th_p + (ced + w * sed) * d_p - ea * a_p + w_p * sed
        chi_p = N_p / den - N * (2.0 * w * w_p) / (den * den)
        psi_p = np.where(first, d_p - a_p, (c * th_p - s * w_p / ws) / ws)
        core_p = Kc * psi_p - spot * chi_p
        out.append((2.0 * sign / (L * L) * core + 2.0 / L * core_p) * half)
    return out[0], out[1], out[2]


def _omega(cp: Any, th: np.ndarray, grad: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    r""":math:`\omega = -\psi(-i)` per row (and its gradient); raises without :math:`E[e^{X_1}]`."""
    u = np.array([-1j])
    with np.errstate(all="ignore"):
        if grad:
            v, g = cp.kernel.psi_and_grad(u, th)
            v, g = v[:, 0], g[:, 0, :]
        else:
            v, g = cp.kernel.psi(u, th)[:, 0], None
    bad = ~np.isfinite(v.real) | (np.abs(v.imag) > 1e-10 * np.maximum(1.0, np.abs(v.real)))
    if np.any(bad):
        raise ValueError(
            "E[exp(X_1)] is infinite (psi(-i) is not a finite real number); "
            "no risk-neutral price exists"
        )
    return -v.real, (None if g is None else -g.real)


def _interval(
    kap: np.ndarray,
    jac: Optional[np.ndarray],
    om: np.ndarray,
    dom: Optional[np.ndarray],
    T: float,
    rate: float,
    dividend: float,
    width: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """End points ``a``, ``b`` per row and their gradients (as the graph COS pricer)."""
    c1 = (rate - dividend + om + kap[:, 0]) * T
    pos2, pos4 = kap[:, 1] > 0, kap[:, 3] > 0
    c2 = np.where(pos2, kap[:, 1], 0.0) * T
    c4 = np.where(pos4, kap[:, 3], 0.0) * T
    sq4 = np.sqrt(c4)
    rad = c2 + sq4
    half = width * np.sqrt(rad)
    if not np.all(np.isfinite(half) & (half > 0)):
        raise ValueError("degenerate COS interval (zero variance?)")
    a, b = c1 - half, c1 + half
    if jac is None or dom is None:
        return a, b, None, None
    dc1 = (dom + jac[:, 0, :]) * T
    dc2 = np.where(pos2[:, None], jac[:, 1, :], 0.0) * T
    # sqrt(c4) is treated as a constant zero where c4 vanishes (Brownian motion)
    safe = np.where(pos4, sq4, 1.0)[:, None]
    dsq4 = np.where(pos4[:, None], jac[:, 3, :] * T / (2.0 * safe), 0.0)
    dhalf = width * (dc2 + dsq4) / (2.0 * np.sqrt(rad)[:, None])
    return a, b, dc1 - dhalf, dc1 + dhalf


def _price_group(
    cp: Any,
    th: np.ndarray,
    spot: float,
    K: np.ndarray,
    is_call: np.ndarray,
    T: float,
    rate: float,
    dividend: float,
    n_terms: int,
    width: float,
    kap: np.ndarray,
    jac: Optional[np.ndarray],
    om: np.ndarray,
    dom: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    grad = jac is not None
    a, b, da, db = _interval(kap, jac, om, dom, T, rate, dividend, width)
    L = b - a
    k_idx = np.arange(int(n_terms))
    u = k_idx[None, :] * np.pi / L[:, None]  # (B, N)
    drift = (rate - dividend + om) * T - a  # (B,)
    disc = np.exp(-rate * T)
    shift = np.where(is_call, spot * np.exp(-dividend * T) - K * np.exp(-rate * T), 0.0)
    if not grad:
        psi = cp.kernel.psi_rowwise(u, th)
        E = np.exp(1j * u * drift[:, None] + T * psi)
        P, _, _ = _payoff_ab(K, spot, a, b, n_terms)
        prices = disc * np.einsum("bmk,bk->bm", P, E.real) + shift
        return prices, None
    assert da is not None and db is not None and dom is not None
    psi, dpsi, gpsi = cp.kernel.psi_du_and_grad(u, th, rowwise=True)
    E = np.exp(1j * u * drift[:, None] + T * psi)
    P, Pa, Pb = _payoff_ab(K, spot, a, b, n_terms)
    PE = P * E.real[:, None, :]
    prices = disc * PE.sum(axis=-1) + shift
    # d coef_k / d theta_j = Re(E_k dg_kj). The T dpsi/dtheta term needs the full
    # (B, N, n) kernel output; every other term of dg is f_k(row) * h_j(row), so
    # it is contracted over k first: sum_k P Re(E f) times h.
    PEi = P * E.imag[:, None, :]
    gterm = T * (
        np.einsum("bmk,bkj->bmj", PE, gpsi.real) - np.einsum("bmk,bkj->bmj", PEi, gpsi.imag)
    )

    def contract(f: np.ndarray) -> np.ndarray:
        return np.einsum("bmk,bk->bm", P, (E * f).real)

    s_om = contract(1j * u * T)
    s_l = contract((1j * drift[:, None] + T * dpsi) * (-u / L[:, None]))
    s_a = contract(-1j * u)
    jacp = disc * (
        gterm
        + s_om[:, :, None] * dom[:, None, :]
        + s_l[:, :, None] * (db - da)[:, None, :]
        + s_a[:, :, None] * da[:, None, :]
        + np.einsum("bmk,bk->bm", Pa, E.real)[:, :, None] * da[:, None, :]
        + np.einsum("bmk,bk->bm", Pb, E.real)[:, :, None] * db[:, None, :]
    )
    return prices, jacp


def cos_price(
    cp: Any,
    spot: float,
    strikes: Any,
    maturities: Any,
    rate: float = 0.0,
    dividend: float = 0.0,
    call: Any = True,
    n_terms: int = 1024,
    width: float = 12.0,
    theta: Optional[Any] = None,
    grad: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""
    European option prices by the COS method, with the exact Jacobian on request.

    Parameters
    ----------
    cp :
        A :class:`~spxa.compile.process.CompiledProcess`.
    spot, rate, dividend :
        :math:`S_0`, the continuously compounded rate and dividend yield.
    strikes, maturities, call :
        Quotes; they broadcast to one 1-D shape (``M`` options). Quotes with
        the same maturity share one expansion.
    n_terms, width :
        Number of cosine terms and the interval half-width in standard
        deviations (``L``).
    theta :
        Free-parameter values: one vector, or a ``(B, n_free)`` batch.
    grad :
        Also return :math:`\partial\,\mathrm{price}/\partial\theta`.

    Returns
    -------
    prices, or ``(prices, jacobian)``
        Shapes ``(M,)`` and ``(M, n_free)``, with a leading ``B`` for a batch.
    """
    if not (np.isfinite(spot) and spot > 0):
        raise ValueError("spot must be positive")
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    Tq = np.atleast_1d(np.asarray(maturities, dtype=float))
    if K.ndim != 1 or Tq.ndim != 1:
        raise ValueError("strikes and maturities must be scalars or 1-D arrays")
    K, Tq = np.broadcast_arrays(K, Tq)
    if np.any(~np.isfinite(K)) or np.any(K <= 0):
        raise ValueError("strikes must be positive and finite")
    if np.any(~np.isfinite(Tq)) or np.any(Tq <= 0):
        raise ValueError("maturities must be positive and finite")
    is_call = np.broadcast_to(np.asarray(call, dtype=bool), K.shape)
    if int(n_terms) < 2 or not width > 0:
        raise ValueError("need n_terms >= 2 and width > 0")
    if not cp.analytic:
        raise CompileError("COS pricing needs exponential moments (an analytic exponent)")

    th, batched = cp.kernel._theta(None if theta is None else np.asarray(theta, dtype=float))
    th2 = th if batched else th[None, :]
    if grad:
        kap, jac = cp.kernel.cumulants_and_jacobian(4, th2)
    else:
        kap, jac = cp.kernel.cumulants(4, th2), None
    om, dom = _omega(cp, th2, grad)
    n = cp.kernel.n_free
    prices = np.empty((th2.shape[0], K.size))
    jacp = np.empty((th2.shape[0], K.size, n)) if grad else None
    for T in np.unique(Tq):
        idx = np.flatnonzero(Tq == T)
        p, j = _price_group(
            cp, th2, float(spot), K[idx], is_call[idx], float(T), float(rate), float(dividend),
            int(n_terms), float(width), kap, jac, om, dom,
        )  # fmt: skip
        prices[:, idx] = p
        if jacp is not None and j is not None:
            jacp[:, idx, :] = j
    if not batched:
        prices = prices[0]
        jacp = None if jacp is None else jacp[0]
    return (prices, jacp) if grad and jacp is not None else prices
