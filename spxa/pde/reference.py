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
