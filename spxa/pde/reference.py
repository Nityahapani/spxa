r"""Closed-form and semi-analytic reference prices used to verify the PDE solvers.

Every function here is independent of the finite-difference machinery, so
it can serve as an external benchmark:

* Black–Scholes prices and Greeks, digitals and asset-or-nothing options.
* Reiner–Rubinstein (1991) single-barrier formulas in Haug's notation,
  with rebates.
* Eigenfunction (sine-series) expansion for double-barrier knock-outs.
* Cox–Ross–Rubinstein and Leisen–Reimer binomial trees for American
  options, and the perpetual American put.
* Lewis (2001) Fourier formula for exponential Lévy and Heston models, and
  Merton's (1976) jump-diffusion series.

Conventions: ``S`` spot, ``K`` strike, ``T`` maturity in years, ``r``
continuously compounded rate, ``q`` continuous dividend yield, ``sigma``
volatility. Prices are per unit notional.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm

from spxa.pde._utils import check_option_type, check_positive

# =====================================================================================
# Black–Scholes
# =====================================================================================


def _d1d2(S: Any, K: Any, T: float, r: float, sigma: float, q: float) -> tuple[Any, Any]:
    sq = sigma * np.sqrt(T)
    d1 = (np.log(np.asarray(S, dtype=float) / K) + (r - q + 0.5 * sigma**2) * T) / sq
    return d1, d1 - sq


def bs_price(
    S: Any, K: Any, T: float, r: float, sigma: float, q: float = 0.0, option: str = "call"
) -> Any:
    """Black–Scholes–Merton price of a European call or put."""
    check_positive("T", T)
    check_positive("sigma", sigma)
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    S = np.asarray(S, dtype=float)
    if check_option_type(option) == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option: str = "call"
) -> dict[str, float]:
    """Price and Greeks (delta, gamma, vega, theta, rho) of a European option.

    ``theta`` is the calendar-time derivative per year and ``vega``/``rho``
    are per unit change of ``sigma``/``r``.
    """
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    pdf = float(norm.pdf(d1))
    gamma = dq * pdf / (S * sigma * math.sqrt(T))
    vega = S * dq * pdf * math.sqrt(T)
    decay = -S * dq * pdf * sigma / (2.0 * math.sqrt(T))
    if check_option_type(option) == "call":
        delta = dq * float(norm.cdf(d1))
        theta = decay - r * K * dr * float(norm.cdf(d2)) + q * S * dq * float(norm.cdf(d1))
        rho = K * T * dr * float(norm.cdf(d2))
    else:
        delta = -dq * float(norm.cdf(-d1))
        theta = decay + r * K * dr * float(norm.cdf(-d2)) - q * S * dq * float(norm.cdf(-d1))
        rho = -K * T * dr * float(norm.cdf(-d2))
    return {
        "price": float(bs_price(S, K, T, r, sigma, q, option)),
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def bs_digital_price(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
    cash: float = 1.0,
) -> Any:
    """Cash-or-nothing digital paying ``cash`` at expiry."""
    _, d2 = _d1d2(S, K, T, r, sigma, q)
    sgn = 1.0 if check_option_type(option) == "call" else -1.0
    return cash * math.exp(-r * T) * norm.cdf(sgn * d2)


def bs_asset_or_nothing_price(
    S: Any, K: float, T: float, r: float, sigma: float, q: float = 0.0, option: str = "call"
) -> Any:
    """Asset-or-nothing digital paying ``S_T`` at expiry."""
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    sgn = 1.0 if check_option_type(option) == "call" else -1.0
    return np.asarray(S, dtype=float) * math.exp(-q * T) * norm.cdf(sgn * d1)


# =====================================================================================
# Reiner–Rubinstein single barriers
# =====================================================================================

_BARRIER_TYPES = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


def barrier_price(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
    barrier_type: str = "down-and-out",
    rebate: float = 0.0,
    rebate_timing: str = "hit",
) -> float:
    r"""Continuously monitored single-barrier option (Reiner & Rubinstein, 1991).

    Parameters
    ----------
    barrier_type : {'down-and-out', 'down-and-in', 'up-and-out', 'up-and-in'}
    rebate : float
        Cash rebate. Knock-in options pay it at expiry if the barrier was
        never hit. Knock-out options pay it when the barrier is hit
        (``rebate_timing='hit'``) or at expiry (``'expiry'``).

    Notes
    -----
    Uses the building blocks :math:`A, \dots, F` of Haug (2007), §4.17.1,
    with cost of carry :math:`b = r - q`. If the barrier has already been
    breached, knock-outs are worth their rebate and knock-ins the vanilla.
    """
    opt = check_option_type(option)
    bt = str(barrier_type).lower()
    if bt not in _BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {_BARRIER_TYPES}")
    if rebate_timing not in ("hit", "expiry"):
        raise ValueError("rebate_timing must be 'hit' or 'expiry'")
    for name, v in (("S", S), ("K", K), ("H", H), ("T", T), ("sigma", sigma)):
        check_positive(name, v)
    down = bt.startswith("down")
    knock_out = bt.endswith("out")
    breached = S <= H if down else S >= H
    vanilla_px = float(bs_price(S, K, T, r, sigma, q, opt))
    if breached:
        if knock_out:
            return rebate if rebate_timing == "hit" else rebate * math.exp(-r * T)
        return vanilla_px

    b = r - q
    sq = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    lam = math.sqrt(mu**2 + 2.0 * r / sigma**2)
    phi = 1.0 if opt == "call" else -1.0
    eta = 1.0 if down else -1.0
    x1 = math.log(S / K) / sq + (1 + mu) * sq
    x2 = math.log(S / H) / sq + (1 + mu) * sq
    y1 = math.log(H**2 / (S * K)) / sq + (1 + mu) * sq
    y2 = math.log(H / S) / sq + (1 + mu) * sq
    z = math.log(H / S) / sq + lam * sq
    N = norm.cdf
    carry, disc = math.exp((b - r) * T), math.exp(-r * T)
    hs = H / S
    A = phi * S * carry * N(phi * x1) - phi * K * disc * N(phi * x1 - phi * sq)
    B = phi * S * carry * N(phi * x2) - phi * K * disc * N(phi * x2 - phi * sq)
    C = phi * S * carry * hs ** (2 * (mu + 1)) * N(eta * y1) - phi * K * disc * hs ** (2 * mu) * N(
        eta * y1 - eta * sq
    )
    D = phi * S * carry * hs ** (2 * (mu + 1)) * N(eta * y2) - phi * K * disc * hs ** (2 * mu) * N(
        eta * y2 - eta * sq
    )
    # E: rebate at expiry if not hit; F: rebate at the hitting time
    E_unit = disc * (N(eta * x2 - eta * sq) - hs ** (2 * mu) * N(eta * y2 - eta * sq))
    F_unit = hs ** (mu + lam) * N(eta * z) + hs ** (mu - lam) * N(eta * z - 2 * eta * lam * sq)
    above = K > H
    if knock_out:
        if opt == "call":
            if down:
                core = A - C if above else B - D
            else:
                core = 0.0 if above else A - B + C - D
        else:
            if down:
                core = A - B + C - D if above else 0.0
            else:
                core = B - D if above else A - C
        if rebate_timing == "hit":
            reb = rebate * F_unit
        else:
            reb = rebate * (disc - E_unit)
        return float(core + reb)
    if opt == "call":
        if down:
            core = C if above else A - B + D
        else:
            core = A if above else B - C + D
    else:
        if down:
            core = B - C + D if above else A
        else:
            core = A - B + D if above else C
    return float(core + rebate * E_unit)


# =====================================================================================
# Double barrier knock-out: eigenfunction expansion
# =====================================================================================


def double_barrier_price(
    S: float,
    K: float,
    L: float,
    U: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
    tol: float = 1e-14,
    max_terms: int = 2000,
) -> float:
    r"""Double knock-out call or put (no rebate) by eigenfunction expansion.

    In :math:`x = \log S` with drift :math:`\nu = r - q - \tfrac12\sigma^2`,
    the transition density killed at :math:`l = \log L`, :math:`u = \log U`
    (:math:`W = u - l`) is

    .. math::
        p(x, y, T) = e^{\nu (y - x)/\sigma^2 - \nu^2 T / (2\sigma^2)}
        \frac{2}{W} \sum_{n \ge 1} e^{-\frac12 \sigma^2 (n\pi/W)^2 T}
        \sin\frac{n\pi (x - l)}{W} \sin\frac{n\pi (y - l)}{W},

    and the price is :math:`e^{-rT} \int_l^u p(x, y, T) g(e^y)\,dy`. The
    payoff integrals use high-order Gauss–Legendre quadrature on the smooth
    part of the payoff, and the series is summed until the terms fall below
    ``tol``.
    """
    opt = check_option_type(option)
    for name, v in (("S", S), ("K", K), ("L", L), ("U", U), ("T", T), ("sigma", sigma)):
        check_positive(name, v)
    if not L < U:
        raise ValueError("need L < U")
    if not (L < S < U):
        return 0.0
    lo, hi = math.log(L), math.log(U)
    W = hi - lo
    x = math.log(S)
    nu = r - q - 0.5 * sigma**2
    # payoff support inside the corridor
    k = math.log(K)
    a, b = (max(lo, k), hi) if opt == "call" else (lo, min(hi, k))
    if a >= b:
        return 0.0
    n_q = 400
    gx, gw = np.polynomial.legendre.leggauss(n_q)
    y = 0.5 * (b - a) * gx + 0.5 * (a + b)
    w = 0.5 * (b - a) * gw
    pay = np.maximum(np.exp(y) - K, 0.0) if opt == "call" else np.maximum(K - np.exp(y), 0.0)
    weight = w * pay * np.exp(nu * (y - x) / sigma**2)
    pref = math.exp(-r * T - nu**2 * T / (2 * sigma**2)) * 2.0 / W
    total = 0.0
    for n in range(1, max_terms + 1):
        kn = n * math.pi / W
        decay = math.exp(-0.5 * sigma**2 * kn**2 * T)
        term = decay * math.sin(kn * (x - lo)) * float(np.dot(weight, np.sin(kn * (y - lo))))
        total += term
        if decay * float(np.sum(np.abs(weight))) < tol and n > 3:
            break
    return float(pref * total)
