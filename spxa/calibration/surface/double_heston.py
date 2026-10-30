r"""Double Heston (Two-Factor Stochastic Volatility) Surface Calibrator.

Calibrates the 10-parameter Double Heston model (Christoffersen et al. 2009):
- Factor 1 (Fast Mean-Reverting): (v0_1, kappa_1, theta_1, sigma_1, rho_1)
- Factor 2 (Slow Mean-Reverting): (v0_2, kappa_2, theta_2, sigma_2, rho_2)

Total instantaneous variance is the sum v_t = v_{1, t} + v_{2, t}.
The joint characteristic function factors into the product of two individual Heston CFs.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from spxa.calibration.surface.base import SurfaceCalibrator
from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.heston import bs_implied_vol
from spxa.calibration.surface.loss import ArbitragePenalizedLoss, SurfaceLoss
from spxa.calibration.surface.optimizer import SurfaceOptimizer


def heston_component_cf(
    phi: complex,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    t: float,
) -> complex:
    """Characteristic function component for an individual CIR variance factor without drift."""
    i = 1j
    d = np.sqrt((rho * sigma * phi * i - kappa) ** 2 + (sigma**2) * (i * phi + phi**2))
    g = (kappa - rho * sigma * phi * i - d) / (kappa - rho * sigma * phi * i + d)

    exp_dt = np.exp(-d * t)
    c = (kappa * theta / (sigma**2)) * (
        (kappa - rho * sigma * phi * i - d) * t - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    d_term = (
        (kappa - rho * sigma * phi * i - d) / (sigma**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    )
    return np.exp(c + d_term * v0)


def double_heston_char_func(
    phi: complex,
    s0: float,
    t: float,
    r: float,
    v0_1: float,
    kappa_1: float,
    theta_1: float,
    sigma_1: float,
    rho_1: float,
    v0_2: float,
    kappa_2: float,
    theta_2: float,
    sigma_2: float,
    rho_2: float,
) -> complex:
    """Joint characteristic function of log-price under Double Heston model."""
    cf1 = heston_component_cf(phi, v0_1, kappa_1, theta_1, sigma_1, rho_1, t)
    cf2 = heston_component_cf(phi, v0_2, kappa_2, theta_2, sigma_2, rho_2, t)
    drift_term = np.exp(1j * phi * (np.log(s0) + r * t))
    return drift_term * cf1 * cf2


def double_heston_call_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    params: np.ndarray,
    n_quad: int = 40,
) -> float:
    """Price European call option under Double Heston model via Gauss-Legendre quadrature."""
    v0_1, k1, th1, sig1, rho1 = params[0], params[1], params[2], params[3], params[4]
    v0_2, k2, th2, sig2, rho2 = params[5], params[6], params[7], params[8], params[9]

    u_nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    u_max = 80.0
    u = 0.5 * u_max * (u_nodes + 1.0)
    w = 0.5 * u_max * weights

    # P1 uses the share measure: phi(u - i) / phi(-i), with phi(-i) = E[S_T] = S0 e^{rT}
    forward = s0 * math.exp(r * t)
    cf1 = np.array(
        [
            double_heston_char_func(
                phi - 1j, s0, t, r, v0_1, k1, th1, sig1, rho1, v0_2, k2, th2, sig2, rho2
            )
            / forward
            for phi in u
        ]
    )
    cf2 = np.array(
        [
            double_heston_char_func(
                phi, s0, t, r, v0_1, k1, th1, sig1, rho1, v0_2, k2, th2, sig2, rho2
            )
            for phi in u
        ]
    )

    log_k = math.log(k)
    integrand1 = np.real(np.exp(-1j * u * log_k) * cf1 / (1j * u))
    integrand2 = np.real(np.exp(-1j * u * log_k) * cf2 / (1j * u))

    p1 = 0.5 + (1.0 / np.pi) * np.sum(integrand1 * w)
    p2 = 0.5 + (1.0 / np.pi) * np.sum(integrand2 * w)

    call = s0 * p1 - k * math.exp(-r * t) * p2
    return float(max(0.0, call))


class DoubleHestonCalibrator(SurfaceCalibrator):
    r"""Calibrates two-factor Double Heston stochastic volatility model."""

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
        return [
            "v0_1",
            "kappa_1",
            "theta_1",
            "sigma_1",
            "rho_1",
            "v0_2",
            "kappa_2",
            "theta_2",
            "sigma_2",
            "rho_2",
        ]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [
            # Factor 1 (Fast)
            (0.001, 1.0),
            (2.0, 15.0),
            (0.001, 1.0),
            (0.01, 2.5),
            (-0.99, 0.99),
            # Factor 2 (Slow)
            (0.001, 1.0),
            (0.05, 2.0),
            (0.001, 1.0),
            (0.01, 2.5),
            (-0.99, 0.99),
        ]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array(
            [
                0.02,
                5.0,
                0.02,
                0.30,
                -0.70,  # Fast factor
                0.02,
                0.5,
                0.02,
                0.15,
                -0.40,  # Slow factor
            ]
        )

    def model_quote(
        self,
        strike: float,
        maturity: float,
        params: np.ndarray,
        surface: MarketSurface,
    ) -> float:
        call = double_heston_call_price(
            s0=surface.spot,
            k=strike,
            t=maturity,
            r=surface.rate,
            params=params,
        )
        return bs_implied_vol(call, surface.spot, strike, maturity, surface.rate)
