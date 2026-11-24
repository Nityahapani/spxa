r"""
Fourier option pricers with exact parameter derivatives.

The risk-neutral log-price follows the convention of :mod:`spxa.discovery.pricing`:

.. math::
    S_T = S_0\exp\big((r - q)T + \omega T + X_T\big), \qquad \omega = -\psi(-i),

so :math:`e^{-(r-q)t}S_t` is a martingale whenever :math:`E[e^{X_1}] < \infty`.
:math:`X` is a :class:`~spxa.graph.autodiff.process.DiffProcess`; drift in
:math:`X` cancels in :math:`X_T + \omega T`, so its derivative is exactly zero.

The returned gradients are the exact derivatives of the discretised price,
computed in forward mode through the closed-form characteristic exponent; no
finite differences are involved. The Carr–Madan and Lewis grids do not depend
on the parameters. The COS interval :math:`[a, b]` does, through the
cumulants; with ``track_interval=True`` (the default) that dependence is
differentiated too, so the gradient is the derivative of the price exactly as
``cos_price`` computes it (and agrees with differences of it to rounding).
``track_interval=False`` freezes the interval at the current values, which
differs from the tracked gradient by the (tiny) truncation sensitivity. As the discretisation is
refined, price and gradient converge together (the integrands are analytic in
the parameters, so the quadrature error of the derivative has the same order
as that of the price).

``cos_price``
    Fang & Oosterlee (2008); puts from bounded payoff coefficients, calls by
    put–call parity. The interval is :math:`c_1 \pm L\sqrt{c_2 + \sqrt{c_4}}`
    from the exact cumulants.
``carr_madan_price``
    Carr & Madan (1999) damped call transform, by composite Gauss–Legendre
    quadrature *at each requested strike* (no FFT grid, no interpolation).
``lewis_price``
    Lewis (2001) formula
    :math:`C = S_0e^{-qT} - \frac{\sqrt{S_0K}e^{-(r+q)T/2}}{\pi}
    \int_0^\infty \mathrm{Re}\big[e^{iuk}\varphi(u - i/2)\big]\frac{du}{u^2 + 1/4}`,
    :math:`k = \log(S_0/K) + (r - q)T`, by the same quadrature.

No price is clipped (a ``max(price, 0)`` floor would put a kink into the
gradient), so tiny negative deep out-of-the-money values at the quadrature
noise level are possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from spxa.graph.autodiff.exactness import GradientExactness
from spxa.graph.autodiff.jet import Jet, stack, value_of, variable_vector
from spxa.graph.autodiff.process import DiffProcess, UnsupportedProcessError

__all__ = [
    "cos_price",
    "carr_madan_price",
    "lewis_price",
    "price_and_gradient",
    "PricingResult",
    "martingale_correction",
    "fourier_price_node",
    "fourier_nodes",
    "cos_interval",
    "cos_price_surface",
]

_METHODS = ("cos", "carr_madan", "lewis")


def _strikes(strikes: Any) -> np.ndarray:
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    if K.ndim != 1:
        raise ValueError("strikes must be a scalar or a 1-D array")
    if np.any(~np.isfinite(K)) or np.any(K <= 0):
        raise ValueError("strikes must be positive and finite")
    return K


def _check(spot: float, maturity: float) -> None:
    if not (np.isfinite(spot) and spot > 0):
        raise ValueError("spot must be positive")
    if not (np.isfinite(maturity) and maturity > 0):
        raise ValueError("maturity must be positive")


def martingale_correction(dp: DiffProcess, theta: Any = None) -> Any:
    r""":math:`\omega = -\psi(-i) = -K(1)`; raises if :math:`E[e^{X_1}]` is infinite."""
    if not dp.analytic:
        raise UnsupportedProcessError("the process has no exponential moments")
    with np.errstate(all="ignore"):
        k1 = dp.char_exponent(np.array(-1j), theta)
    v = complex(np.asarray(value_of(k1)))
    if not np.isfinite(v.real) or abs(v.imag) > 1e-10 * max(1.0, abs(v.real)):
        raise ValueError(
            "E[exp(X_1)] is infinite (psi(-i) is not a finite real number); "
            "no risk-neutral price exists"
        )
    return -np.real(k1)


def _cf_T(
    dp: DiffProcess, u: np.ndarray, T: float, r: float, q: float, theta: Any, omega: Any = None
) -> Any:
    r""":math:`E[e^{iu\log(S_T/S_0)}]` (``u`` may be complex); ``omega`` if already known."""
    if omega is None:
        omega = martingale_correction(dp, theta)
    return np.exp(1j * u * ((r - q) * T + omega * T) + T * dp.char_exponent(u, theta))


def fourier_nodes(
    u_max: float, panel: float = 1.0, order: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Composite Gauss–Legendre nodes and weights on :math:`[0, u_{max}]`.

    Panels double from width 1/16 at the origin (resolving the poles of
    :math:`1/(u^2 + 1/4)` at :math:`\pm i/2` and of the Carr–Madan denominator)
    up to ``panel``, then stay at ``panel``. For integrands analytic in a strip
    of half-width :math:`d` the panel error decays like :math:`\rho^{-2\,order}`
    with :math:`\rho = d/h + \sqrt{1 + d^2/h^2}` and :math:`h` the half-width,
    i.e. below 1e-16 for ``order = 16`` once :math:`d \ge h`; each panel of width
    1 also spans at most :math:`|k|/(2\pi)` oscillations of :math:`e^{iuk}`.
    """
    if not (u_max > 0 and panel > 0 and order >= 2):
        raise ValueError("u_max and panel must be positive and order >= 2")
    edges = [0.0]
    w = 1.0 / 16.0
    while edges[-1] < u_max:
        edges.append(min(edges[-1] + w, u_max))
        w = min(2.0 * w, panel)
    x, wt = np.polynomial.legendre.leggauss(int(order))
    lo, hi = np.array(edges[:-1]), np.array(edges[1:])
    half, mid = (hi - lo) / 2.0, (hi + lo) / 2.0
    nodes = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * wt[None, :]).ravel()
    return nodes, weights


def _theta_values(dp: DiffProcess, theta: Any) -> np.ndarray:
    return dp.values if theta is None else np.asarray(value_of(theta), dtype=float)


# ── COS ─────────────────────────────────────────────────────────────────────


def cos_interval(
    dp: DiffProcess,
    maturity: float,
    rate: float,
    dividend: float,
    width: float,
    theta: Any = None,
    track: bool = False,
) -> Tuple[Any, Any]:
    """
    The COS truncation interval :math:`c_1 \\pm L\\sqrt{c_2 + \\sqrt{c_4}}` for ``log(S_T/S_0)``.

    With ``track=True`` and a Jet ``theta`` the end points are jets (their
    dependence on the parameters is differentiated); otherwise floats.
    """
    th = theta if (track and isinstance(theta, Jet)) else _theta_values(dp, theta)
    k = dp.cumulants(4, th)
    omega = martingale_correction(dp, th)
    return _interval(k, omega, float(maturity), rate, dividend, width)


def _interval(k: Any, omega: Any, T: float, rate: float, dividend: float, width: float) -> Any:
    """:func:`cos_interval` from the unit-time cumulants ``k`` and ``omega`` (Lévy scaling)."""
    c1 = (rate - dividend + omega + k[0]) * T
    c2, c4 = np.maximum(k[1], 0.0) * T, np.maximum(k[3], 0.0) * T
    # sqrt is not differentiable at 0; a vanishing c4 (e.g. Brownian motion) is a
    # constant zero there, so it is treated as such
    sq4 = np.sqrt(c4) if float(np.real(value_of(c4))) > 0.0 else 0.0
    half = width * np.sqrt(c2 + sq4)
    hv = float(np.real(value_of(half)))
    if not (np.isfinite(hv) and hv > 0):
        raise ValueError("degenerate COS interval (zero variance?)")
    if isinstance(half, Jet) or isinstance(c1, Jet):
        return c1 - half, c1 + half
    return float(c1 - half), float(c1 + half)


def cos_price(
    dp: DiffProcess,
    spot: float,
    strikes: Any,
    maturity: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    call: Any = True,
    n_terms: int = 1024,
    width: float = 12.0,
    theta: Any = None,
    track_interval: bool = True,
) -> Any:
    """
    European prices by the COS method; a Jet when ``theta`` is a Jet.

    With ``track_interval`` (default) the derivative includes the dependence
    of :math:`[a, b]` on the parameters through the cumulants, so it is the
    exact derivative of the price *as computed*; with ``False`` the interval is
    frozen at the parameter values (the two agree once the expansion has
    converged).
    """
    _check(spot, maturity)
    K = _strikes(strikes)
    is_call = np.broadcast_to(np.asarray(call, dtype=bool), K.shape)
    moments = _cos_moments(dp, theta, track_interval)
    return _cos_one(
        dp, spot, K, is_call, float(maturity), rate, dividend, n_terms, width, theta, moments
    )


def cos_price_surface(
    dp: DiffProcess,
    spot: float,
    strikes: Any,
    maturities: Any,
    rate: float = 0.0,
    dividend: float = 0.0,
    call: Any = True,
    n_terms: int = 1024,
    width: float = 12.0,
    theta: Any = None,
    track_interval: bool = True,
) -> Any:
    r"""
    :func:`cos_price` for quotes at several maturities, one price per quote.

    ``strikes``, ``maturities`` and ``call`` broadcast to one 1-D shape.
    Element ``i`` equals ``cos_price(..., strikes[i], maturities[i], ...)``
    to rounding. The cumulants and :math:`\omega` of :math:`X_1` do not depend
    on the maturity, so they are computed once and scaled by :math:`T`.
    """
    K = _strikes(strikes)
    Tq = np.atleast_1d(np.asarray(maturities, dtype=float))
    if Tq.ndim != 1:
        raise ValueError("maturities must be a scalar or a 1-D array")
    K, Tq = np.broadcast_arrays(K, Tq)
    is_call = np.broadcast_to(np.asarray(call, dtype=bool), K.shape)
    groups = np.unique(Tq)
    for T in groups:
        _check(spot, float(T))
    moments = _cos_moments(dp, theta, track_interval)
    idxs, parts = [], []
    for T in groups:
        idx = np.flatnonzero(Tq == T)
        p = _cos_one(
            dp, spot, K[idx], is_call[idx], float(T), rate, dividend, n_terms, width, theta, moments
        )
        idxs.append(idx)
        parts.append(p)
    order = np.argsort(np.concatenate(idxs), kind="stable")
    if not any(isinstance(p, Jet) for p in parts):
        return np.concatenate([np.asarray(p, dtype=float) for p in parts])[order]
    return np.concatenate(parts)[order]


def _cos_moments(dp: DiffProcess, theta: Any, track: bool) -> Tuple[Any, Any, Any]:
    """Unit-time cumulants and omega for the interval, and omega for the transform."""
    jet = isinstance(theta, Jet)
    th_iv = theta if (track and jet) else _theta_values(dp, theta)
    k = dp.cumulants(4, th_iv)
    om_iv = martingale_correction(dp, th_iv)
    om_cf = om_iv if (track or not jet) else martingale_correction(dp, theta)
    return k, om_iv, om_cf


def _cos_one(
    dp: DiffProcess,
    spot: float,
    K: np.ndarray,
    is_call: np.ndarray,
    T: float,
    rate: float,
    dividend: float,
    n_terms: int,
    width: float,
    theta: Any,
    moments: Tuple[Any, Any, Any],
) -> Any:
    k, om_iv, om_cf = moments
    a, b = _interval(k, om_iv, T, rate, dividend, width)
    k_idx = np.arange(int(n_terms))
    u = k_idx * np.pi / (b - a)
    phi = _cf_T(dp, u, T, rate, dividend, theta, om_cf)
    coef = np.real(phi * np.exp(-1j * u * a))
    disc = np.exp(-rate * T)
    forward = spot * np.exp(-dividend * T) - K * np.exp(-rate * T)
    shift = np.where(is_call, forward, 0.0)
    if isinstance(a, Jet) and isinstance(coef, Jet) and coef.hess is None:
        # The (M, N) payoff matrix depends on theta only through (a, b): its two
        # partial derivatives in closed form, chained with da and db.
        av, bv = float(np.real(a.value)), float(np.real(b.value))
        P, Pa, Pb = _cos_payoff_ab(K, spot, av, bv, k_idx)
        cv, cg = np.asarray(coef.value), np.asarray(coef.grad)
        val = disc * (P @ cv) + shift
        grad = disc * (
            (Pa @ cv)[:, None] * np.real(a.grad)[None, :]
            + (Pb @ cv)[:, None] * np.real(b.grad)[None, :]
            + P @ cg
        )
        return Jet(val, grad)
    payoff = _cos_payoff(K, spot, a, b, k_idx)
    return disc * (payoff @ coef) + shift


def _cos_payoff_ab(
    K: np.ndarray, spot: float, a: float, b: float, k_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    :func:`_cos_payoff` at float end points, and its partial derivatives in ``a`` and ``b``.

    With :math:`L = b - a`, :math:`w = k\pi/L`, :math:`d = \mathrm{clip}(\log(K/S_0), a, b)`
    and :math:`\vartheta = w(d - a)`:

    - :math:`\partial_a w = w/L` and :math:`\partial_b w = -w/L`;
    - :math:`\partial d` is 1 where the clip is active at that end point. Ties
      keep :math:`\log(K/S_0)`, as ``np.maximum`` and ``np.minimum`` do on jets;
    - the product and quotient rules are applied to
      :math:`\chi = (\cos\vartheta\,e^d - e^a + w\sin\vartheta\,e^d)/(1 + w^2)` and
      :math:`\psi = \sin\vartheta/w` (:math:`d - a` at :math:`k = 0`).
    """
    L = b - a
    w = (k_idx * np.pi / L)[None, :]
    ell = np.log(K / spot)[:, None]
    lo, hi = ell < a, ell > b
    d = np.where(lo, a, np.where(hi, b, ell))
    half = np.ones(k_idx.size)
    half[0] = 0.5
    first = (k_idx == 0)[None, :]
    th = w * (d - a)
    c, s = np.cos(th), np.sin(th)
    ed, ea = np.exp(d), np.exp(a)
    den = 1.0 + w * w
    sed, ced = s * ed, c * ed
    N = ced - ea + w * sed
    ws = np.where(first, 1.0, w)
    psi = np.where(first, d - a, s / ws)
    core = K[:, None] * psi - spot * (N / den)
    out = [2.0 / L * core * half]
    for end in (0, 1):
        sign = 1.0 if end == 0 else -1.0  # dw = sign * w / L and dL = -sign
        w_p = sign * w / L
        d_p = (lo if end == 0 else hi).astype(float)
        a_p = 1.0 if end == 0 else 0.0
        th_p = w_p * (d - a) + w * (d_p - a_p)
        N_p = (w * ced - sed) * th_p + (ced + w * sed) * d_p - ea * a_p + w_p * sed
        chi_p = N_p / den - N * (2.0 * w * w_p) / (den * den)
        psi_p = np.where(first, d_p - a_p, (c * th_p - s * w_p / ws) / ws)
        core_p = K[:, None] * psi_p - spot * chi_p
        out.append((2.0 * sign / (L * L) * core + 2.0 / L * core_p) * half)
    return out[0], out[1], out[2]


def _cos_payoff(K: np.ndarray, spot: float, a: Any, b: Any, k_idx: np.ndarray) -> Any:
    r"""Put payoff coefficients :math:`\frac{2}{b-a}(K\psi_k - S_0\chi_k)` (halved at k = 0)."""
    u = k_idx * np.pi / (b - a)
    half = np.ones(k_idx.size)
    half[0] = 0.5
    kappa = np.minimum(np.maximum(np.log(K / spot), a), b)  # np.clip with jet end points
    d = kappa[:, None]
    w = u[None, :]
    first = (k_idx == 0)[None, :]
    arg_d = w * (d - a)
    chi = (np.cos(arg_d) * np.exp(d) - np.exp(a) + w * np.sin(arg_d) * np.exp(d)) / (1.0 + w**2)
    w_safe = w + first  # u_0 = 0 exactly (value and derivatives)
    psi = np.where(first, d - a, np.sin(arg_d) / w_safe)
    return 2.0 / (b - a) * (K[:, None] * psi - spot * chi) * half[None, :]


# ── Carr–Madan (direct Simpson summation) ────────────────────────────────────


def carr_madan_price(
    dp: DiffProcess,
    spot: float,
    strikes: Any,
    maturity: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    call: Any = True,
    alpha: float = 1.5,
    u_max: float = 400.0,
    panel: float = 1.0,
    order: int = 16,
    theta: Any = None,
) -> Any:
    r"""
    Carr–Madan prices; calls from the damped transform, puts by parity.

    :math:`C(k) = \frac{e^{-\alpha k}}{\pi}\int_0^\infty \mathrm{Re}\big[e^{-ivk}
    \frac{e^{-rT}\varphi(v - (\alpha+1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}\big]dv`
    with :math:`k = \log K` and :math:`\varphi` the characteristic function of
    :math:`\log S_T`. Needs :math:`E[S_T^{1+\alpha}] < \infty`.
    """
    _check(spot, maturity)
    K = _strikes(strikes)
    is_call = np.broadcast_to(np.asarray(call, dtype=bool), K.shape)
    T = float(maturity)
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    v, wts = fourier_nodes(u_max, panel, order)
    z = v - (alpha + 1.0) * 1j
    phi = _cf_T(dp, z, T, rate, dividend, theta) * np.exp(1j * z * np.log(spot))
    den = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
    integrand = np.exp(-rate * T) * phi / den  # (n,)
    k = np.log(K)
    kern = np.exp(-1j * np.outer(k, v)) * wts[None, :]  # (M, n), constant
    callp = np.exp(-alpha * k) / np.pi * np.real(kern @ integrand)
    forward = spot * np.exp(-dividend * T) - K * np.exp(-rate * T)
    return callp - np.where(is_call, 0.0, forward)


# ── Lewis (2001) ────────────────────────────────────────────────────────────


def lewis_price(
    dp: DiffProcess,
    spot: float,
    strikes: Any,
    maturity: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    call: Any = True,
    u_max: float = 400.0,
    panel: float = 1.0,
    order: int = 16,
    theta: Any = None,
) -> Any:
    r"""Lewis (2001) prices by Gauss–Legendre on :math:`[0, u_{max}]`; puts by parity."""
    _check(spot, maturity)
    K = _strikes(strikes)
    is_call = np.broadcast_to(np.asarray(call, dtype=bool), K.shape)
    T = float(maturity)
    u, wts = fourier_nodes(u_max, panel, order)
    # characteristic function of x = log(S_T/S_0) - (r - q)T
    omega = martingale_correction(dp, theta)
    z = u - 0.5j
    phi = np.exp(1j * z * omega * T + T * dp.char_exponent(z, theta))
    k = np.log(spot / K) + (rate - dividend) * T
    kern = np.exp(1j * np.outer(k, u)) * (wts / (u**2 + 0.25))[None, :]
    integral = np.real(kern @ phi)
    callp = (
        spot * np.exp(-dividend * T)
        - np.sqrt(spot * K) * np.exp(-(rate + dividend) * T / 2.0) / np.pi * integral
    )
    forward = spot * np.exp(-dividend * T) - K * np.exp(-rate * T)
    return callp - np.where(is_call, 0.0, forward)


_PRICERS = {"cos": cos_price, "carr_madan": carr_madan_price, "lewis": lewis_price}


@dataclass
class PricingResult:
    """Prices with their exact parameter Jacobian (and optionally Hessians)."""

    prices: np.ndarray
    jacobian: np.ndarray
    names: Tuple[str, ...]
    method: str
    hessian: Optional[np.ndarray] = None
    exactness: GradientExactness = GradientExactness.EXACT
    notes: Dict[str, Any] = field(default_factory=dict)

    def __story__(self) -> str:
        if self.method == "cos" and self.notes.get("track_interval", True):
            grid = "the truncation interval [a, b] differentiated with the cumulants"
        elif self.method == "cos":
            grid = "the truncation interval frozen at the parameter values"
        else:
            grid = "the quadrature nodes fixed"
        return (
            f"{self.method} prices of {self.prices.size} options; d price / d "
            f"({', '.join(self.names)}) by forward-mode jets through the closed-form "
            f"characteristic exponent, {grid} ({self.exactness.name} derivative of the "
            "discretised price)"
        )


def price_and_gradient(
    dp: DiffProcess,
    spot: float,
    strikes: Any,
    maturity: float,
    method: str = "cos",
    hessian: bool = False,
    theta: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> PricingResult:
    """Prices and the exact Jacobian ``d price_i / d theta_k`` (shape ``(M, n)``)."""
    if method not in _PRICERS:
        raise ValueError(f"method must be one of {_METHODS}")
    th = variable_vector(dp.values if theta is None else theta, order=2 if hessian else 1)
    out = _PRICERS[method](dp, spot, strikes, maturity, theta=th, **kwargs)
    n = len(dp.names)
    if isinstance(out, Jet):
        prices = np.asarray(np.real(out.value), dtype=float)
        jac = np.asarray(np.real(out.grad), dtype=float)
        hes = None if out.hess is None else np.asarray(np.real(out.hess), dtype=float)
    else:  # no free parameter reaches the price
        prices = np.asarray(out, dtype=float)
        jac = np.zeros(prices.shape + (n,))
        hes = np.zeros(prices.shape + (n, n)) if hessian else None
    notes = {"track_interval": bool(kwargs.get("track_interval", True))} if method == "cos" else {}
    return PricingResult(prices, jac, dp.names, method, hes, notes=notes)


def fourier_price_node(
    dp: DiffProcess,
    params: Sequence[Any],
    spot: float,
    strikes: Any,
    maturity: float,
    method: str = "cos",
    name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    A graph node pricing options from parameter nodes (one per free parameter).

    Its VJP is :math:`J^\\top g` with the exact Jacobian of :func:`price_and_gradient`,
    so graph gradients through it are EXACT.
    """
    from spxa.graph.node import OperatorNode

    if len(params) != len(dp.names):
        raise ValueError(f"expected {len(dp.names)} parameter nodes for {dp.names}")
    if method not in _PRICERS:
        raise ValueError(f"method must be one of {_METHODS}")

    def op(*vals: Any) -> Any:
        th = stack(list(vals)) if any(isinstance(v, Jet) for v in vals) else np.asarray(vals, float)
        return _PRICERS[method](dp, spot, strikes, maturity, theta=th, **kwargs)

    def vjp(g: Any, *vals: Any) -> Tuple[Any, ...]:
        res = price_and_gradient(
            dp, spot, strikes, maturity, method=method, theta=np.asarray(vals, float), **kwargs
        )
        ct = np.real(np.asarray(g)) * np.ones(res.prices.shape)
        return tuple(float(ct @ res.jacobian[:, k]) for k in range(len(vals)))

    op.__name__ = f"{method}_price"
    return OperatorNode(op, list(params), name=name or f"{method}_price", vjp_fn=vjp)
