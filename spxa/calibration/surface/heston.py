r"""Heston (1993) Stochastic Volatility Surface Calibrator.

Calibrates the 5 fundamental Heston parameters:
- v0    : initial variance level (> 0)
- kappa : mean reversion speed of variance (> 0)
- theta : long-term variance mean (> 0)
- sigma : volatility of variance (> 0)
- rho   : correlation between spot returns and variance innovations in (-1, 1)

Prices European options via the characteristic function and inverts Black-Scholes
implied volatility using bounded root-finding.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import optimize, special

from spxa.calibration.surface.base import SurfaceCalibrator
from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.loss import ArbitragePenalizedLoss, SurfaceLoss
from spxa.calibration.surface.optimizer import SurfaceOptimizer


def bs_call_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """Standard Black-Scholes European call price formula."""
    if t <= 0.0 or sigma <= 0.0:
        return max(0.0, s - k)
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    n_d1 = 0.5 * (1.0 + special.erf(d1 / math.sqrt(2.0)))
    n_d2 = 0.5 * (1.0 + special.erf(d2 / math.sqrt(2.0)))
    return float(s * n_d1 - k * math.exp(-r * t) * n_d2)


def bs_implied_vol(price: float, s: float, k: float, t: float, r: float) -> float:
    """Invert Black-Scholes call price for implied volatility."""
    intrinsic = max(0.0, s - k * math.exp(-r * t))
    if price <= intrinsic + 1e-8:
        return 0.001
    if price >= s - 1e-8:
        return 5.0

    def obj(sigma: float) -> float:
        return bs_call_price(s, k, t, r, sigma) - price

    try:
        sol = optimize.brentq(obj, 1e-4, 5.0, xtol=1e-6)
        return float(sol)
    except Exception:
        # Fallback to ATM approximation
        return float(max(0.05, math.sqrt(2.0 * math.pi / t) * (price / s)))


def heston_char_func(
    phi: complex,
    s0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    t: float,
    r: float,
) -> complex:
    """Heston continuous characteristic function with Little-Trap branch cut elimination."""
    i = 1j
    d = np.sqrt((rho * sigma * phi * i - kappa) ** 2 + (sigma**2) * (i * phi + phi**2))
    g = (kappa - rho * sigma * phi * i - d) / (kappa - rho * sigma * phi * i + d)

    exp_dt = np.exp(-d * t)
    # Albrecher et al. (2007) Little Trap formulation
    c = r * phi * i * t + (kappa * theta / (sigma**2)) * (
        (kappa - rho * sigma * phi * i - d) * t - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    d_term = (
        (kappa - rho * sigma * phi * i - d) / (sigma**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    )
    return np.exp(c + d_term * v0 + i * phi * np.log(s0))


def heston_call_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    n_quad: int = 40,
) -> float:
    """Price European call option under Heston model using Gauss-Legendre quadrature."""
    # Integrate (e^{-i phi ln K} * cf(phi - i) / (i phi * S0) - K * e^{-r T} e^{-i phi ln K} * cf(phi) / (i phi))
    # Using Gil-Pelaez representation
    u_nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # Map [-1, 1] to [0, 80]
    u_max = 80.0
    u = 0.5 * u_max * (u_nodes + 1.0)
    w = 0.5 * u_max * weights

    # P1 uses the share measure: phi(u - i) / phi(-i), with phi(-i) = E[S_T] = S0 e^{rT}
    forward = s0 * math.exp(r * t)
    cf1 = np.array(
        [heston_char_func(phi - 1j, s0, v0, kappa, theta, sigma, rho, t, r) / forward for phi in u]
    )
    cf2 = np.array([heston_char_func(phi, s0, v0, kappa, theta, sigma, rho, t, r) for phi in u])

    log_k = math.log(k)
    integrand1 = np.real(np.exp(-1j * u * log_k) * cf1 / (1j * u))
    integrand2 = np.real(np.exp(-1j * u * log_k) * cf2 / (1j * u))

    p1 = 0.5 + (1.0 / np.pi) * np.sum(integrand1 * w)
    p2 = 0.5 + (1.0 / np.pi) * np.sum(integrand2 * w)

    call = s0 * p1 - k * math.exp(-r * t) * p2
    return float(max(0.0, call))


class HestonCalibrator(SurfaceCalibrator):
    r"""Calibrates Heston stochastic volatility model parameters to market surfaces.

    Parameters
    ----------
    enforce_feller : bool
        If True, penalizes parameter sets violating Feller condition 2*kappa*theta >= sigma^2.
    """

    def __init__(
        self,
        enforce_feller: bool = True,
        loss_fn: Optional[SurfaceLoss] = None,
        optimizer: Optional[SurfaceOptimizer] = None,
    ) -> None:
        if enforce_feller and loss_fn is None:
            base_loss = ArbitragePenalizedLoss(feller_penalty_weight=50.0)
        else:
            base_loss = loss_fn
        super().__init__(loss_fn=base_loss, optimizer=optimizer)

    @property
    def param_names(self) -> List[str]:
        return ["v0", "kappa", "theta", "sigma", "rho"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (0.001, 1.5),  # v0
            (0.05, 10.0),  # kappa
            (0.001, 1.5),  # theta
            (0.01, 2.5),  # sigma (vol of vol)
            (-0.99, 0.99),  # rho
        ]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([0.04, 1.5, 0.04, 0.3, -0.6])

    def model_quote(
        self,
        strike: float,
        maturity: float,
        params: np.ndarray,
        surface: MarketSurface,
    ) -> float:
        v0, kappa, theta, sigma, rho = params[0], params[1], params[2], params[3], params[4]
        call = heston_call_price(
            s0=surface.spot,
            k=strike,
            t=maturity,
            r=surface.rate,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
        )
        return bs_implied_vol(call, surface.spot, strike, maturity, surface.rate)
