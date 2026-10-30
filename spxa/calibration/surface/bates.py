r"""Bates (1996) Stochastic Volatility Jump-Diffusion Surface Calibrator.

Jointly calibrates Heston CIR stochastic volatility and log-normal Merton jumps:
- v0       : initial variance level (> 0)
- kappa    : mean-reversion speed (> 0)
- theta    : long-term variance (> 0)
- sigma    : volatility of variance (> 0)
- rho      : leverage correlation in (-1, 1)
- jump_rate: Poisson jump arrival intensity lambda (>= 0)
- jump_mean: mean percentage jump size in log-price mu_J
- jump_std : standard deviation of jump size sigma_J (> 0)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from spxa.calibration.surface.base import SurfaceCalibrator
from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.heston import bs_implied_vol, heston_char_func
from spxa.calibration.surface.loss import ArbitragePenalizedLoss, SurfaceLoss
from spxa.calibration.surface.optimizer import SurfaceOptimizer


def bates_char_func(
    phi: complex,
    s0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    jump_rate: float,
    jump_mean: float,
    jump_std: float,
    t: float,
    r: float,
) -> complex:
    """Bates characteristic function combining Heston diffusion and compound Poisson jumps."""
    # Heston continuous diffusion component
    cf_diff = heston_char_func(phi, s0, v0, kappa, theta, sigma, rho, t, r)

    # Jump component: exp( lambda * t * (exp(i * phi * mu_J - 0.5 * sigma_J^2 * phi^2) - 1 - i * phi * k_bar) )
    # where k_bar = exp(mu_J + 0.5 * sigma_J^2) - 1 is the martingale compensator
    i = 1j
    k_bar = np.exp(jump_mean + 0.5 * (jump_std**2)) - 1.0
    jump_exp = np.exp(i * phi * jump_mean - 0.5 * (jump_std**2) * (phi**2)) - 1.0 - i * phi * k_bar
    cf_jump = np.exp(jump_rate * t * jump_exp)

    return cf_diff * cf_jump


def bates_call_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    jump_rate: float,
    jump_mean: float,
    jump_std: float,
    n_quad: int = 40,
) -> float:
    """Price European call option under Bates model via Gauss-Legendre quadrature."""
    u_nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    u_max = 80.0
    u = 0.5 * u_max * (u_nodes + 1.0)
    w = 0.5 * u_max * weights

    # P1 uses the share measure: phi(u - i) / phi(-i), with phi(-i) = E[S_T] = S0 e^{rT}
    forward = s0 * math.exp(r * t)
    cf1 = np.array(
        [
            bates_char_func(
                phi - 1j,
                s0,
                v0,
                kappa,
                theta,
                sigma,
                rho,
                jump_rate,
                jump_mean,
                jump_std,
                t,
                r,
            )
            / forward
            for phi in u
        ]
    )
    cf2 = np.array(
        [
            bates_char_func(
                phi,
                s0,
                v0,
                kappa,
                theta,
                sigma,
                rho,
                jump_rate,
                jump_mean,
                jump_std,
                t,
                r,
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


class BatesCalibrator(SurfaceCalibrator):
    r"""Calibrates Bates (1996) jump-diffusion stochastic volatility model parameters."""

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
        return ["v0", "kappa", "theta", "sigma", "rho", "jump_rate", "jump_mean", "jump_std"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (0.001, 1.5),  # v0
            (0.05, 10.0),  # kappa
            (0.001, 1.5),  # theta
            (0.01, 2.5),  # sigma
            (-0.99, 0.99),  # rho
            (0.0, 5.0),  # jump_rate
            (-0.6, 0.2),  # jump_mean
            (0.01, 0.5),  # jump_std
        ]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([0.04, 1.5, 0.04, 0.3, -0.6, 0.5, -0.05, 0.10])

    def model_quote(
        self,
        strike: float,
        maturity: float,
        params: np.ndarray,
        surface: MarketSurface,
    ) -> float:
        v0, kappa, theta, sigma, rho, j_rate, j_mean, j_std = (
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
        )
        call = bates_call_price(
            s0=surface.spot,
            k=strike,
            t=maturity,
            r=surface.rate,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            jump_rate=j_rate,
            jump_mean=j_mean,
            jump_std=j_std,
        )
        return bs_implied_vol(call, surface.spot, strike, maturity, surface.rate)
