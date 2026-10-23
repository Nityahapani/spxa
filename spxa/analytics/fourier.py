"""
Fourier-based option pricing for Lévy process models.

For a log-price process X_t with characteristic function φ(u; t),
European option prices can be computed via the Carr-Madan formula using
FFT. This is the standard approach in quantitative finance for Lévy models.

Reference: Carr, P. & Madan, D.B. (1999). Option valuation using the fast
Fourier transform. *Journal of Computational Finance*, 2(4), 61–73.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spxa.core.process import Process


def european_call_price(
    process: Process,
    S0: float,
    K: float,
    r: float,
    t: float,
    alpha: float = 1.5,
    n_points: int = 4096,
    eta: float = 0.25,
) -> float:
    """
    Price a European call option assuming X_t is the log-return process.

    Uses the Carr-Madan FFT method. The asset price is S0*exp(X_t+r*t-log(E[exp(X_t)])).
    Thus the discounted asset is a martingale. A process whose CF already
    contains risk-neutral log drift is unchanged by this normalization.

    The call price is:
      C(K) = e^{-αk}/(π) · Re[∫_0^∞ e^{-iuk} ψ(u) du]

    where k = log(K/S0), α is a dampening factor, and ψ(u) is the
    modified characteristic function of X_t.

    Parameters
    ----------
    process :
        Log-return Lévy process. Must support char_func.
    S0 :
        Current asset price.
    K :
        Strike price.
    r :
        Risk-free rate (continuously compounded).
    t :
        Time to expiry.
    alpha :
        Dampening factor α > 0. Typically 1.0–2.0. Larger values
        give better accuracy for OTM options but may cause overflow.
    n_points :
        FFT grid size. Must be a power of 2.
    eta :
        Spacing in the integration domain.

    Returns
    -------
    float
        Call price.

    References
    ----------
    Carr, P. & Madan, D.B. (1999). Option valuation using the fast Fourier
    transform. *Journal of Computational Finance*, 2(4), 61–73.
    """
    if (
        S0 <= 0
        or K <= 0
        or not all(np.isfinite(v) for v in (S0, K, r, t, alpha, eta))
        or t < 0
        or alpha <= 0
        or eta <= 0
    ):
        raise ValueError(
            "Finite positive spot/strike/damping/spacing and nonnegative time required"
        )
    if not isinstance(n_points, int) or n_points < 16 or n_points & (n_points - 1):
        raise ValueError("n_points must be a power of two >=16")
    if t == 0:
        return max(S0 - K, 0.0)
    lam = 2 * np.pi / (n_points * eta)
    b = n_points * lam / 2
    k = np.log(K / S0)

    u_grid = np.arange(n_points) * eta
    u_shifted = u_grid - (alpha + 1) * 1j

    # Use char_func_exact if the process provides it (avoids slow LK quadrature)
    cf_fn = getattr(process, "char_func_exact", None) or process.char_func
    moment = complex(cf_fn(u=-1j, t=t))
    if (
        not np.isfinite(moment)
        or abs(moment.imag) > 1e-8 * max(1.0, abs(moment.real))
        or moment.real <= 0
    ):
        raise ValueError("A finite positive exponential moment E[exp(X_t)] is required")
    # S_t = S0*exp(X_t + r*t - log E[exp(X_t)]). An already
    # risk-neutral log-return CF receives no additional correction.
    correction = r * t - np.log(moment.real)
    cf_vals = cf_fn(u=u_shifted, t=t) * np.exp(1j * u_shifted * correction)
    if not np.all(np.isfinite(cf_vals)):
        raise ValueError("Damping requires a finite (alpha+1)-th exponential moment; reduce alpha")

    numerator = np.exp(-r * t) * cf_vals
    denominator = alpha**2 + alpha - u_grid**2 + 1j * (2 * alpha + 1) * u_grid

    psi = numerator / denominator

    simpson_weights = np.ones(n_points)
    simpson_weights[0] = 1 / 3
    simpson_weights[-1] = 1 / 3
    simpson_weights[1:-1:2] = 4 / 3
    simpson_weights[2:-2:2] = 2 / 3

    x = np.exp(1j * b * u_grid) * psi * simpson_weights * eta

    fft_result = np.fft.fft(x)

    k_grid = -b + lam * np.arange(n_points)
    call_prices = (np.exp(-alpha * k_grid) / np.pi * fft_result).real

    call_price = float(np.interp(k, k_grid, call_prices)) * S0
    lower = max(S0 - K * np.exp(-r * t), 0.0)
    tolerance = 1e-4 * max(S0, K)
    if not np.isfinite(call_price) or call_price < lower - tolerance or call_price > S0 + tolerance:
        raise ArithmeticError(
            "FFT grid unresolved: option price violates arbitrage bounds; refine grid"
        )
    return float(np.clip(call_price, lower, S0))


def european_put_price(
    process: Process,
    S0: float,
    K: float,
    r: float,
    t: float,
    **kwargs: Any,
) -> float:
    """
    Price a European put via put-call parity.

    P = C - S0 + K·e^{-rt}

    Parameters
    ----------
    process, S0, K, r, t :
        Same as european_call_price.
    **kwargs :
        Passed to european_call_price.

    Returns
    -------
    float
        Put price.
    """
    call = european_call_price(process=process, S0=S0, K=K, r=r, t=t, **kwargs)
    return float(call - S0 + K * np.exp(-r * t))


def implied_volatility(
    price: float,
    S0: float,
    K: float,
    r: float,
    t: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Compute Black-Scholes implied volatility from an option price via bisection.

    Parameters
    ----------
    price :
        Observed option price.
    S0, K, r, t :
        Asset price, strike, rate, time.
    option_type :
        'call' or 'put'.
    tol :
        Convergence tolerance on price.
    max_iter :
        Maximum bisection iterations.

    Returns
    -------
    float
        Implied volatility σ. Returns NaN if no solution found.
    """
    from scipy.stats import norm

    if option_type not in ("call", "put") or min(S0, K, t, tol) <= 0 or max_iter < 1:
        raise ValueError(
            "Valid option type and positive spot/strike/time/tolerance/iterations required"
        )
    discount = K * np.exp(-r * t)
    lower = max(S0 - discount, 0.0) if option_type == "call" else max(discount - S0, 0.0)
    upper = S0 if option_type == "call" else discount
    if not np.isfinite(price) or price < lower - tol or price >= upper:
        return float("nan")
    if abs(price - lower) <= tol:
        return 0.0

    def bs_price(sigma: float) -> float:
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        if option_type == "call":
            return float(S0 * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2))
        return float(K * np.exp(-r * t) * norm.cdf(-d2) - S0 * norm.cdf(-d1))

    lo, hi = 1e-6, 10.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        diff = bs_price(mid) - price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid

    return float("nan")
