r"""
Exact parameter gradients of Hull–White caplet, cap and swaption prices.

Every price depends on the parameters :math:`\theta = (\sigma_0, \dots, \sigma_{n-1}, a)`
only through :math:`B(t, T) = (1 - e^{-a(T-t)})/a` and the variance

.. math::
    y(t) = \sum_j \sigma_j^2\,\frac{e^{-2a(t - u_{j,1})} - e^{-2a(t - u_{j,0})}}{2a}

of the state over the volatility pieces :math:`[u_{j,0}, u_{j,1}) \cap [0, t]`. Both are
evaluated on forward-mode jets, so the derivatives are those of the formulas
:class:`~spxa.rates.hull_white.HullWhite` evaluates.

Caplets
    Black on the lognormal ratio :math:`Q = P(t_f, s)/P(t_f, e)` with log-variance
    :math:`(B(t_f, e) - B(t_f, s))^2 y(t_f)`; the strike :math:`K^*` does not depend on
    :math:`\theta`.
Swaptions, Jamshidian
    :math:`V^+ = \sum_i \pm a_i\,\mathrm{ZBO}(T, t_i, K_i)` with :math:`K_i = P(T, t_i | x^*)`
    and :math:`V(x^*) = 0`. Every bond option is exercised on the same set
    :math:`\{x \gtrless x^*\}`, so :math:`\partial\mathrm{ZBO}_i/\partial K_i` is the same
    :math:`P(0, T)\,Q^T(\text{exercise})` for all :math:`i`, and the strike terms add up to
    that times :math:`\partial_\theta \sum_i a_i K_i = -\partial_\theta\,\mathrm{const} = 0`.
    The gradient is therefore the gradient at frozen strikes (an envelope theorem).
Swaptions, integral
    With :math:`x = \sqrt{y(T)}\,z` the price is
    :math:`P(0, T)\int V^+(\sqrt{y}\,z)\,\phi(z)\,dz` over fixed limits; :math:`V^+` vanishes
    where the region :math:`\{V > 0\}` moves, so the derivative is the integral of
    :math:`\partial_\theta V` over that region. :math:`V\phi` is analytic on each side of the
    root, and composite 32-point Gauss–Legendre on panels of width 0.5 in :math:`z`
    evaluates it to rounding.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy.special import ndtr

from spxa.graph.autodiff.jet import Jet, variable_vector
from spxa.market.curves import YieldCurve
from spxa.rates.capfloor import CapFloor, Caplet
from spxa.rates.hull_white import HullWhite
from spxa.rates.swaptions import Swaption

__all__ = ["hull_white_price_and_gradient"]


def _B(a: Any, t: float, T: Any) -> Any:
    return -np.expm1(-a * (np.asarray(T, float) - t)) / a


def _y(model: HullWhite, th: Jet, t: float) -> Any:
    """:math:`y(t)` on the jet ``th = (sigma_0, ..., sigma_{n-1}, a)`` (same pieces as the model)."""
    st = model.sigma_times
    a = th[-1]
    edges = np.r_[0.0, st[st < t], t]
    idx = np.searchsorted(st, edges[:-1], side="right")
    total: Any = 0.0
    for u0, u1, j in zip(edges[:-1], edges[1:], idx):
        s = th[int(j)]
        total = total + s * s * (np.exp(-2 * a * (t - u1)) - np.exp(-2 * a * (t - u0))) / (2 * a)
    return total


def _zcb_option(model: HullWhite, th: Jet, T: float, S: float, K: float, option: str) -> Any:
    pT, pS = float(model.curve.discount(T)), float(model.curve.discount(S))
    v = _B(th[-1], T, S) * np.sqrt(_y(model, th, T))
    d1 = np.log(pS / (K * pT)) / v + 0.5 * v
    d2 = d1 - v
    if option == "call":
        return pS * ndtr(d1) - K * pT * ndtr(d2)
    return K * pT * ndtr(-d2) - pS * ndtr(-d1)


def _caplet(model: HullWhite, caplet: Any, projection: Optional[YieldCurve], th: Jet) -> Any:
    c = caplet.coupon
    proj = projection if projection is not None else model.curve
    crv = model.curve
    tf = float(crv.time(c.fixing))
    s, e, p = (float(crv.time(d)) for d in (c.start, c.end, c.payment_date))
    tau = c.accrual
    Q0 = float(crv.discount(s) / crv.discount(e))
    b = caplet.forward(proj) - (Q0 - 1.0) / tau
    Kstar = 1.0 + tau * (caplet.strike - b)
    call = caplet.kind == "cap"
    scale = float(c.notional * crv.discount(p))
    if Kstar <= 0 or tf <= 0:
        # intrinsic value: no dependence on the parameters
        return scale * (max(Q0 - Kstar, 0.0) if call else max(Kstar - Q0, 0.0) * (Kstar > 0))
    a = th[-1]
    v = np.abs(_B(a, tf, e) - _B(a, tf, s)) * np.sqrt(_y(model, th, tf))
    d1 = (np.log(Q0 / Kstar) + 0.5 * v * v) / v
    d2 = d1 - v
    und = Q0 * ndtr(d1) - Kstar * ndtr(d2) if call else Kstar * ndtr(-d2) - Q0 * ndtr(-d1)
    return scale * und


def _swaption(
    model: HullWhite, swaption: Any, projection: Optional[YieldCurve], th: Jet, method: str
) -> Any:
    plan = model._swaption_plan(swaption, projection, method)
    if plan["kind"] == "intrinsic":
        return plan["value"]
    T, times, amounts = plan["T"], plan["times"], plan["amounts"]
    if plan["kind"] == "jamshidian":
        total: Any = 0.0
        for t_i, a_i, k_i in zip(times, amounts, plan["K"]):
            if plan["increasing"]:
                total = total - a_i * _zcb_option(model, th, T, float(t_i), float(k_i), "put")
            else:
                total = total + a_i * _zcb_option(model, th, T, float(t_i), float(k_i), "call")
        return total
    # integral: d/dtheta of pT * int V+(sd z) phi(z) dz over the region V > 0. V+ phi is
    # analytic on each side of the root, so composite Gauss-Legendre converges
    # exponentially; all nodes are evaluated on one vector jet.
    const, pT = plan["const"], plan["pT"]
    a = th[-1]
    yT = _y(model, th, T)
    sd = np.sqrt(yT)
    b = _B(a, T, times)  # (m,)
    ratio = np.asarray(model.curve.discount(times), float) / float(model.curve.discount(T))
    sdv = float(np.real(sd.value))
    lo, hi = plan["lo"] / sdv, plan["hi"] / sdv
    pts = [lo, plan["root"] / sdv, hi] if plan["root"] is not None else [lo, hi]
    z, wz = _gl_nodes(pts)
    x = sd * z  # (N,)
    V = const + np.sum(
        (amounts * ratio)[None, :] * np.exp(-b[None, :] * x[:, None] - 0.5 * (b * b * yT)[None, :]),
        axis=1,
    )
    pos = np.asarray(np.real(V.value)) > 0.0  # V+ (the root is a node boundary)
    w = np.where(pos, wz * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi), 0.0)
    return pT * np.sum(V * w)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


def _gl_nodes(edges: list[float], width: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Composite 32-point Gauss-Legendre on each interval between ``edges`` (panels <= ``width``)."""
    zs, ws = [], []
    for z0, z1 in zip(edges[:-1], edges[1:]):
        k = max(1, int(np.ceil((z1 - z0) / width)))
        e = np.linspace(z0, z1, k + 1)
        half, mid = (e[1:] - e[:-1]) / 2.0, (e[1:] + e[:-1]) / 2.0
        zs.append((mid[:, None] + half[:, None] * _GL_X[None, :]).ravel())
        ws.append((half[:, None] * _GL_W[None, :]).ravel())
    return np.concatenate(zs), np.concatenate(ws)


def hull_white_price_and_gradient(
    model: HullWhite,
    inst: Any,
    projection: Optional[YieldCurve] = None,
    method: str = "jamshidian",
) -> Tuple[float, np.ndarray]:
    r"""
    Price of a swaption, cap/floor or caplet and its exact gradient.

    The gradient is with respect to :math:`(\sigma_0, \dots, \sigma_{n-1}, a)`, the
    volatility levels of ``model`` followed by its mean reversion, so it has length
    ``len(model.sigma) + 1``. Jamshidian swaptions, caplets and caps are differentiated in
    closed form; swaptions priced by the integral are differentiated under the integral
    by Gauss–Legendre quadrature, accurate to rounding. ``method`` is passed to
    :meth:`~spxa.rates.hull_white.HullWhite.swaption`.
    """
    if method not in ("jamshidian", "integral"):
        raise ValueError("method must be 'jamshidian' or 'integral'")
    th = variable_vector(np.r_[model.sigma, model.a])
    if isinstance(inst, Swaption):
        out = _swaption(model, inst, projection, th, method)
    elif isinstance(inst, CapFloor):
        out = sum(_caplet(model, c, projection, th) for c in inst.caplets)
    elif isinstance(inst, Caplet):
        out = _caplet(model, inst, projection, th)
    else:
        raise TypeError(f"cannot price {type(inst).__name__}")
    if isinstance(out, Jet):
        return float(np.real(out.value)), np.asarray(np.real(out.grad), float)
    return float(out), np.zeros(th.n)
