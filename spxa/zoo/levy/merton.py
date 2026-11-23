"""
Merton Jump-Diffusion Process.

Merton (1976) model combines continuous Brownian diffusion with compound
Poisson jump shocks whose sizes follow a normal distribution:

    X_t = μ t + σ W_t + Σ_{i=1}^{N_t} Y_i

where:
  - W_t is standard Brownian motion with diffusion volatility σ > 0.
  - N_t ~ Poisson(λ t) is a homogeneous Poisson process with arrival intensity λ > 0.
  - Y_i ~ N(μ_J, σ_J^2) are i.i.d. Gaussian jump sizes.

The process is an EXACT Lévy process with finite activity jump measure:
    ν(dx) = λ · (1 / (√(2π) σ_J)) · exp(-(x - μ_J)² / (2 σ_J²)) dx

It provides exact closed-form characteristic functions, analytical cumulants of
all orders, exact simulation without discretization error, and Merton's famous
(1976) semi-analytical option pricing series.

References
----------
Merton, R. C. (1976). Option pricing when underlying stock returns are
discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapman & Hall/CRC. Section 3.5.

Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*.
Cambridge University Press. Example 8.6.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy import special
from scipy.optimize import brentq

from spxa.core.exactness import ExactnessLevel
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.sim.exact import sample_merton
from spxa.story.narrator import CompositionNode, NodeKind
from spxa.zoo.levy.compound_poisson import NormalJump, _CompoundPoissonLevyMeasure


class MertonJumpDiffusion(Process):
    r"""
    Merton (1976) jump-diffusion process.

    .. math::

        X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} Y_i

    where :math:`W_t` is standard Brownian motion, :math:`N_t \sim \mathrm{Poisson}(\lambda t)`,
    and :math:`Y_i \sim \mathcal{N}(\mu_J, \sigma_J^2)`.

    Parameters
    ----------
    drift :
        Continuous drift parameter :math:`\mu \in \mathbb{R}`. Default 0.0.
    sigma :
        Continuous diffusion volatility :math:`\sigma > 0`. Default 0.2.
    rate :
        Jump arrival intensity :math:`\lambda > 0`. Default 1.0.
    jump_mean :
        Mean jump size :math:`\mu_J \in \mathbb{R}`. Default -0.05.
    jump_std :
        Jump size standard deviation :math:`\sigma_J > 0`. Default 0.15.
    """

    def __init__(
        self,
        drift: float = 0.0,
        sigma: float = 0.2,
        rate: float = 1.0,
        jump_mean: float = -0.05,
        jump_std: float = 0.15,
        **kwargs,
    ) -> None:
        # Support aliases
        if "mu" in kwargs:
            drift = kwargs.pop("mu")
        if "vol" in kwargs:
            sigma = kwargs.pop("vol")
        if "lam" in kwargs:
            rate = kwargs.pop("lam")
        elif "jump_rate" in kwargs:
            rate = kwargs.pop("jump_rate")
        if "mu_j" in kwargs:
            jump_mean = kwargs.pop("mu_j")
        elif "mean_jump" in kwargs:
            jump_mean = kwargs.pop("mean_jump")
        if "sigma_j" in kwargs:
            jump_std = kwargs.pop("sigma_j")
        elif "std_jump" in kwargs:
            jump_std = kwargs.pop("std_jump")

        if kwargs:
            unexpected = ", ".join(repr(k) for k in kwargs)
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        drift_f = float(drift)
        sigma_f = float(sigma)
        rate_f = float(rate)
        jump_mean_f = float(jump_mean)
        jump_std_f = float(jump_std)

        if sigma_f <= 0.0:
            raise ValueError(f"Diffusion volatility sigma must be strictly positive, got {sigma_f}")
        if rate_f <= 0.0:
            raise ValueError(f"Jump intensity rate must be strictly positive, got {rate_f}")
        if jump_std_f <= 0.0:
            raise ValueError(
                f"Jump standard deviation jump_std must be strictly positive, got {jump_std_f}"
            )

        self._drift = drift_f
        self._sigma = sigma_f
        self._rate = rate_f
        self._jump_mean = jump_mean_f
        self._jump_std = jump_std_f
        self._jump_dist = NormalJump(mean=jump_mean_f, std=jump_std_f)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"Merton(drift={drift_f}, sigma={sigma_f}, rate={rate_f}, jump_mean={jump_mean_f}, jump_std={jump_std_f})",
            math_note=(
                f"Merton jump-diffusion: μ={drift_f}, σ={sigma_f}, λ={rate_f}, "
                f"Y ~ N({jump_mean_f}, {jump_std_f}²)"
            ),
            reference="Merton (1976); Cont & Tankov (2004)",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    # ------------------------------------------------------------------
    # Properties & Parameters
    # ------------------------------------------------------------------

    @property
    def drift(self) -> float:
        r"""Continuous drift parameter :math:`\mu`."""
        return self._drift

    @property
    def mu(self) -> float:
        r"""Alias for continuous drift :math:`\mu`."""
        return self._drift

    @property
    def sigma(self) -> float:
        r"""Continuous diffusion volatility :math:`\sigma`."""
        return self._sigma

    @property
    def rate(self) -> float:
        r"""Jump arrival intensity :math:`\lambda`."""
        return self._rate

    @property
    def jump_mean(self) -> float:
        r"""Mean jump size :math:`\mu_J`."""
        return self._jump_mean

    @property
    def jump_std(self) -> float:
        r"""Standard deviation of jump size :math:`\sigma_J`."""
        return self._jump_std

    @property
    def jump_distribution(self) -> NormalJump:
        r"""The underlying Gaussian jump distribution :class:`NormalJump`."""
        return self._jump_dist

    @property
    def expected_relative_jump(self) -> float:
        r"""
        Expected relative percentage jump size :math:`\bar{k} = \mathbb{E}[e^Y - 1] = e^{\mu_J + \frac{1}{2}\sigma_J^2} - 1`.
        """
        return float(math.exp(self._jump_mean + 0.5 * self._jump_std**2) - 1.0)

    # ------------------------------------------------------------------
    # Lévy Triplet & Properties
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        # Canonical truncation h(x) = x * 1_{|x| <= 1}
        # b = mu + lambda * E[Y * 1_{|Y| <= 1}]
        trunc_mean = self._jump_dist.truncation_mean(cutoff=1.0)
        b = self._drift + self._rate * trunc_mean
        sigma_sq = self._sigma**2
        nu = _CompoundPoissonLevyMeasure(self._rate, self._jump_dist)
        return LevyTriplet(b=b, sigma_sq=sigma_sq, nu=nu)

    def _properties(self) -> ProcessProperties:
        # Martingale iff total expected drift E[X_1] = 0 <=> mu + lambda * mu_J = 0
        total_drift = self._drift + self._rate * self._jump_mean
        is_martingale = bool(np.isclose(total_drift, 0.0, atol=1e-12))

        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=is_martingale,
            has_finite_mean=True,
            has_finite_variance=True,
            is_subordinator=False,
            notes=[
                (
                    f"Merton jump-diffusion: continuous BM(μ={self._drift}, σ²={self._sigma**2}) + "
                    f"CompoundPoisson(λ={self._rate}, Y~N({self._jump_mean}, {self._jump_std}²))."
                ),
                f"Expected relative jump k_bar = {self.expected_relative_jump:.6f}.",
            ],
        )

    # ------------------------------------------------------------------
    # Distributional Analytics (Exponent, Char Func, Cumulants)
    # ------------------------------------------------------------------

    def char_exponent(self, u: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        r"""
        Lévy characteristic exponent :math:`\psi(u)`.

        .. math::

            \psi(u) = i\mu u - \frac{1}{2}\sigma^2 u^2 + \lambda \left( e^{i u \mu_J - \frac{1}{2}\sigma_J^2 u^2} - 1 \right)
        """
        u_arr = np.asarray(u, dtype=complex)
        is_scalar = np.ndim(u) == 0

        drift_term = 1j * self._drift * u_arr
        diff_term = -0.5 * (self._sigma**2) * (u_arr**2)
        jump_term = self._rate * (
            np.exp(1j * self._jump_mean * u_arr - 0.5 * (self._jump_std**2) * (u_arr**2)) - 1.0
        )
        psi = drift_term + diff_term + jump_term

        if is_scalar:
            return complex(psi)
        return psi

    def char_func(self, u: Union[float, np.ndarray], t: float = 1.0) -> Union[complex, np.ndarray]:
        r"""
        Characteristic function :math:`\phi(u, t) = \mathbb{E}[e^{i u X_t}] = e^{t \psi(u)}`.
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        psi = self.char_exponent(u)
        val = np.exp(t * psi)
        if np.ndim(u) == 0:
            return complex(val)
        return val

    def cumulants(self, order: int) -> dict[int, float]:
        r"""
        Analytical cumulants :math:`\kappa_1, \dots, \kappa_{\mathrm{order}}` of :math:`X_1`.

        .. math::

            \kappa_1 = \mu + \lambda \mu_J \\
            \kappa_2 = \sigma^2 + \lambda (\mu_J^2 + \sigma_J^2) \\
            \kappa_n = \lambda \mathbb{E}[Y^n], \quad n \ge 3
        """
        if order < 1:
            raise ValueError(f"Cumulant order must be at least 1, got {order}")

        res: dict[int, float] = {}
        res[1] = float(self._drift + self._rate * self._jump_mean)

        if order >= 2:
            res[2] = float(self._sigma**2 + self._rate * (self._jump_mean**2 + self._jump_std**2))

        for n in range(3, order + 1):
            res[n] = float(self._rate * self._jump_dist.moment(n))

        return res

    # ------------------------------------------------------------------
    # Merton (1976) Analytical Option Pricing Formula
    # ------------------------------------------------------------------

    def call_price(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        S0: float = 100.0,
        r: Optional[float] = None,
        q: float = 0.0,
        max_jumps: int = 60,
        tol: float = 1e-12,
    ) -> Union[float, np.ndarray]:
        r"""
        European call option price via Merton's (1976) Poisson mixture formula.

        .. math::

            C(S_0, K, T) = \sum_{k=0}^\infty \frac{e^{-\lambda' T} (\lambda' T)^k}{k!} \, C_{\mathrm{BS}}(S_0, K, T, \sigma_k, r_k)

        where:
        - :math:`\lambda' = \lambda (1 + \bar{k}) = \lambda e^{\mu_J + \frac{1}{2}\sigma_J^2}`
        - :math:`\sigma_k = \sqrt{\sigma^2 + \frac{k \sigma_J^2}{T}}`
        - :math:`r_k = r - q - \lambda \bar{k} + \frac{k (\mu_J + \frac{1}{2}\sigma_J^2)}{T}`

        Parameters
        ----------
        K :
            Strike price or array of strikes (:math:`K > 0`).
        T :
            Time to maturity in years (:math:`T > 0`). Default 1.0.
        S0 :
            Initial spot asset price (:math:`S_0 > 0`). Default 100.0.
        r :
            Risk-free interest rate. If None, uses risk-neutral drift implied by :math:`\mu`.
        q :
            Continuous dividend yield. Default 0.0.
        max_jumps :
            Maximum Poisson terms in series. Default 60.
        tol :
            Convergence tolerance for Poisson tail summation. Default 1e-12.

        Returns
        -------
        float or np.ndarray
        """
        if T <= 0.0:
            raise ValueError(f"Maturity T must be strictly positive, got {T}")
        if S0 <= 0.0:
            raise ValueError(f"Spot price S0 must be strictly positive, got {S0}")

        k_bar = self.expected_relative_jump
        if r is None:
            r_val = float(self._drift + 0.5 * self._sigma**2 + self._rate * k_bar)
        else:
            r_val = float(r)

        # Risk-neutral continuous drift of log-price under measure Q
        mu_rn = (r_val - q) - 0.5 * (self._sigma**2) - self._rate * k_bar
        ln_s0 = math.log(S0)
        sig2_t = (self._sigma**2) * T
        sig_j2 = self._jump_std**2
        exp_neg_r_t = math.exp(-r_val * T)

        def _price_single(strike: float) -> float:
            if strike <= 0.0:
                raise ValueError(f"Strike K must be strictly positive, got {strike}")

            total_call = 0.0
            ln_k = math.log(strike)
            poisson_weight = math.exp(-self._rate * T)

            for k in range(max_jumps):
                mean_k = ln_s0 + mu_rn * T + k * self._jump_mean
                var_k = sig2_t + k * sig_j2
                std_k = math.sqrt(var_k)

                d1 = (mean_k + var_k - ln_k) / std_k
                d2 = d1 - std_k
                call_k = exp_neg_r_t * (
                    math.exp(mean_k + 0.5 * var_k) * special.ndtr(d1) - strike * special.ndtr(d2)
                )
                total_call += poisson_weight * call_k

                # Iterative update: p_{k+1} = p_k * (rate * T) / (k + 1)
                poisson_weight *= (self._rate * T) / (k + 1)

                if poisson_weight < tol and k > int(self._rate * T):
                    break

            intrinsic = max(0.0, S0 * math.exp(-q * T) - strike * exp_neg_r_t)
            return float(max(total_call, intrinsic))

        K_arr = np.asarray(K)
        if K_arr.ndim == 0:
            return _price_single(float(K_arr))
        return np.array([_price_single(float(k)) for k in K_arr], dtype=float)

    def put_price(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        S0: float = 100.0,
        r: Optional[float] = None,
        q: float = 0.0,
        max_jumps: int = 60,
        tol: float = 1e-12,
    ) -> Union[float, np.ndarray]:
        r"""
        European put option price via Put-Call parity.

        .. math::

            P = C - S_0 e^{-q T} + K e^{-r T}
        """
        k_bar = self.expected_relative_jump
        r_val = (
            float(self._drift + 0.5 * self._sigma**2 + self._rate * k_bar)
            if r is None
            else float(r)
        )
        calls = self.call_price(K, T=T, S0=S0, r=r_val, q=q, max_jumps=max_jumps, tol=tol)
        K_arr = np.asarray(K, dtype=float)
        puts = calls - S0 * math.exp(-q * T) + K_arr * math.exp(-r_val * T)
        intrinsic = np.maximum(0.0, K_arr * math.exp(-r_val * T) - S0 * math.exp(-q * T))
        return np.maximum(puts, intrinsic)

    def implied_volatility(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        S0: float = 100.0,
        r: Optional[float] = None,
        q: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        Black-Scholes implied volatility inverted from Merton call price.
        """
        k_bar = self.expected_relative_jump
        r_val = (
            float(self._drift + 0.5 * self._sigma**2 + self._rate * k_bar)
            if r is None
            else float(r)
        )

        def _bs_call(vol: float, strike: float) -> float:
            if vol <= 1e-8:
                return float(max(0.0, S0 * math.exp(-q * T) - strike * math.exp(-r_val * T)))
            d1 = (math.log(S0 / strike) + (r_val - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
            d2 = d1 - vol * math.sqrt(T)
            return float(
                S0 * math.exp(-q * T) * special.ndtr(d1)
                - strike * math.exp(-r_val * T) * special.ndtr(d2)
            )

        def _iv_single(strike: float) -> float:
            call_mkt = float(self.call_price(strike, T=T, S0=S0, r=r_val, q=q))
            intrinsic = float(max(0.0, S0 * math.exp(-q * T) - strike * math.exp(-r_val * T)))
            upper_bound = float(S0 * math.exp(-q * T))
            if call_mkt <= intrinsic + 1e-10:
                return 0.0
            if call_mkt >= upper_bound - 1e-10:
                return 5.0

            try:
                return float(brentq(lambda v: _bs_call(v, strike) - call_mkt, 1e-4, 5.0, xtol=1e-6))
            except Exception:
                return float(np.nan)

        K_arr = np.asarray(K)
        if K_arr.ndim == 0:
            return _iv_single(float(K_arr))
        return np.array([_iv_single(float(k)) for k in K_arr], dtype=float)

    def implied_vol_smile(
        self,
        strikes: Union[list[float], np.ndarray],
        T: float = 1.0,
        S0: float = 100.0,
        r: Optional[float] = None,
        q: float = 0.0,
    ) -> np.ndarray:
        r"""Compute the Black-Scholes implied volatility smile across an array of strikes."""
        return self.implied_volatility(strikes, T=T, S0=S0, r=r, q=q)

    # ------------------------------------------------------------------
    # Exact Path Simulation
    # ------------------------------------------------------------------

    _vector_sampler = True

    def sample_at(self, times: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Independent draws of X_t for each entry of ``times``: drift*t + sigma*sqrt(t)*Z + N(N*m, N*s^2) with N ~ Poisson(rate*t)."""
        rng = rng or np.random.default_rng()
        t = np.asarray(times, dtype=float)
        n = rng.poisson(self._rate * t)
        diffusion = self._drift * t + self._sigma * np.sqrt(t) * rng.standard_normal(t.shape)
        jumps = self._jump_mean * n + self._jump_std * np.sqrt(n) * rng.standard_normal(t.shape)
        return diffusion + jumps

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate exact sample paths of :math:`X_t`.

        Zero discretization error for any step size :math:`\Delta t`.

        Returns
        -------
        np.ndarray
            Shape ``(n_paths, n_steps + 1)`` with ``paths[:, 0] == 0.0``.
        """
        return sample_merton(
            drift=self._drift,
            sigma=self._sigma,
            rate=self._rate,
            jump_mean=self._jump_mean,
            jump_std=self._jump_std,
            n_steps=n_steps,
            n_paths=n_paths,
            T=T,
            rng=rng,
        )

    def __repr__(self) -> str:
        return (
            f"MertonJumpDiffusion(drift={self._drift}, sigma={self._sigma}, rate={self._rate}, "
            f"jump_mean={self._jump_mean}, jump_std={self._jump_std})"
        )

    def __str__(self) -> str:
        return (
            f"Merton(drift={self._drift}, sigma={self._sigma}, rate={self._rate}, "
            f"jump_mean={self._jump_mean}, jump_std={self._jump_std})"
        )


# Canonical aliases
MertonProcess = MertonJumpDiffusion
Merton76 = MertonJumpDiffusion
