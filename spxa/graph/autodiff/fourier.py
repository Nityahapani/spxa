r"""
Fourier option pricers with exact parameter derivatives.

The risk-neutral log-price follows the convention of :mod:`spxa.discovery.pricing`:

.. math::
    S_T = S_0\exp\big((r - q)T + \omega T + X_T\big), \qquad \omega = -\psi(-i),

so :math:`e^{-(r-q)t}S_t` is a martingale whenever :math:`E[e^{X_1}] < \infty`.
:math:`X` is a :class:`~spxa.graph.autodiff.process.DiffProcess`; drift in
:math:`X` cancels in :math:`X_T + \omega T`, so its derivative is exactly zero.

Three pricers share one discretisation contract: **the quadrature (the COS
interval :math:`[a, b]`, the Fourier grid and its weights) depends only on the
current parameter values and is held fixed when differentiating**. The
returned gradients are therefore the exact derivatives of the discretised
price, computed in forward mode through the closed-form characteristic
exponent; no finite differences are involved. As the discretisation is
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


def _cf_T(dp: DiffProcess, u: np.ndarray, T: float, r: float, q: float, theta: Any) -> Any:
    r""":math:`E[e^{iu\log(S_T/S_0)}]` (``u`` may be complex)."""
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
    dp: DiffProcess, maturity: float, rate: float, dividend: float, width: float, theta: Any = None
) -> Tuple[float, float]:
    """The COS truncation interval for ``log(S_T/S_0)`` (parameter values only)."""
    th = _theta_values(dp, theta)
    k = dp.cumulants(4, th)
    omega = float(martingale_correction(dp, th))
    T = float(maturity)
    c1 = (rate - dividend + omega + k[0]) * T
    c2, c4 = max(k[1], 0.0) * T, max(k[3], 0.0) * T
    half = width * np.sqrt(c2 + np.sqrt(c4))
    if not (np.isfinite(half) and half > 0):
        raise ValueError("degenerate COS interval (zero variance?)")
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
) -> Any:
    """European prices by the COS method; a Jet when ``theta`` is a Jet."""
    _check(spot, maturity)
    K = _strikes(strikes)
    is_call = np.broadcast_to(np.asarray(call, dtype=bool), K.shape)
    T = float(maturity)
    a, b = cos_interval(dp, T, rate, dividend, width, theta)
    u = np.arange(int(n_terms)) * np.pi / (b - a)
    phi = _cf_T(dp, u, T, rate, dividend, theta)
    coef = np.real(phi * np.exp(-1j * u * a))
    half = np.ones(u.size)
    half[0] = 0.5
    kappa = np.clip(np.log(K / spot), a, b)
    d = kappa[:, None]
    w = u[None, :]
    arg_d = w * (d - a)
    chi = (np.cos(arg_d) * np.exp(d) - np.exp(a) + w * np.sin(arg_d) * np.exp(d)) / (1.0 + w**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(w == 0, d - a, np.sin(arg_d) / np.where(w == 0, 1.0, w))
    payoff = 2.0 / (b - a) * (K[:, None] * psi - spot * chi) * half[None, :]
    put = np.exp(-rate * T) * (payoff @ coef)
    forward = spot * np.exp(-dividend * T) - K * np.exp(-rate * T)
    return put + np.where(is_call, forward, 0.0)


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
        return (
            f"{self.method} prices of {self.prices.size} options; d price / d "
            f"({', '.join(self.names)}) by forward-mode jets through the closed-form "
            "characteristic exponent with the quadrature held fixed (EXACT derivative of the "
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
    return PricingResult(prices, jac, dp.names, method, hes)


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
