r"""
Constant Elasticity of Variance (CEV) diffusion process.

The CEV process satisfies the stochastic differential equation (SDE):
    dS_t = μ S_t dt + σ S_t^γ dW_t,   S_0 = s_0 > 0

This is the interior SDE with a killed/absorbing extension at zero for γ<1,
including γ=0; it is not unconstrained Brownian motion at that boundary.

where:
  μ ∈ ℝ : Drift rate (under risk-neutral pricing, μ = r - q)
  σ > 0 : Volatility scale parameter
  γ ≥ 0 : Elasticity exponent (power on asset price)
  s_0 > 0 : Initial asset price
  W_t   : Standard 1-dimensional Brownian motion

Local Volatility & Elasticity Structure
---------------------------------------
The return (percentage) volatility is state-dependent:
    σ_loc(S) = σ S^{γ - 1}

The elasticity of return variance with respect to spot price is constant:
    (d(σ_loc(S)²) / σ_loc(S)²) / (dS / S) = 2(γ - 1)

Classification of Regimes
-------------------------
1. Equity leverage regime (0 ≤ γ < 1):
   - Local volatility increases as asset price falls (leverage effect).
   - The origin S = 0 is an absorbing boundary (bankruptcy / default).
   - Once absorbed at zero, the process remains at zero: S_{t+u} = 0 for u ≥ 0.
   - Closed-form European option prices are given by Cox (1975) and Schroder (1989)
     in terms of the non-central chi-squared distribution.

2. Geometric Brownian Motion regime (γ = 1):
   - Local volatility is constant: σ_loc(S) = σ.
   - Reduces to the classical Black–Scholes–Merton model.
   - Zero is strictly inaccessible.

3. Commodity / inverse leverage regime (γ > 1):
   - Local volatility increases as asset price rises (common in agricultural / energy commodities).
   - Zero is strictly inaccessible; closed-form option prices follow Emanuel & MacBeth (1982).

Exactness Level
---------------
ExactnessLevel.MOMENT_PROPAGATION (state-dependent local volatility σ S^γ precludes
a Lévy–Khintchine triplet, but analytical non-central chi-squared option pricing,
absorption probabilities, transition laws, and exact simulation are available).

References
----------
Cox, J.C. (1975). Notes on option pricing I: Constant elasticity of diffusions.
    *Working paper*, Stanford University.
Schroder, M. (1989). Computing the constant elasticity of variance option pricing
    formula. *The Journal of Finance*, 44(1), 211–219.
Emanuel, D.C. & MacBeth, J.D. (1982). Further results of the constant elasticity
    of variance call option pricing model. *Journal of Financial and Quantitative Analysis*,
    17(4), 533–554.
Linetsky, V. & Mendoza, R. (2010). Constant elasticity of variance (CEV) diffusion model.
    In *Encyclopedia of Quantitative Finance*, R. Cont (Ed.), John Wiley & Sons.
Choi, J. & Wu, L. (2021). The equivalent constant-elasticity-of-variance (CEV) volatility
    of the stochastic-alpha-beta-rho (SABR) model. *Journal of Economic Dynamics and Control*,
    128, 104143.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import scipy.stats as stats

from spxa.core._validation import validate_order, validate_time
from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.sim.exact import sample_cev
from spxa.story.narrator import CompositionNode, NodeKind


class CEVProcess(Process):
    r"""
    Constant Elasticity of Variance (CEV) diffusion process.

    dS_t = μ S_t dt + σ S_t^γ dW_t,   S_0 = s_0 > 0.

    Parameters
    ----------
    mu : float, default 0.0
        Drift parameter μ. Under risk-neutral pricing, μ = r - q.
    sigma : float, default 0.2
        Volatility scale parameter σ > 0.
    gamma : float, default 0.5
        Elasticity power exponent γ ≥ 0.
        - γ < 1: Equity leverage regime (absorbing zero, implied volatility skew).
        - γ = 1: Geometric Brownian Motion (constant volatility σ).
        - γ > 1: Commodity / inverse leverage regime.
    s0 : float, default 100.0
        Initial asset spot price s_0 > 0.
    r : float, optional
        Risk-free interest rate r. If specified, sets the default discount rate and
        aligns drift μ = r - q.
    q : float, default 0.0
        Continuous dividend yield q ≥ 0.

    Attributes
    ----------
    elasticity : float
        Elasticity of return variance 2(γ - 1).
    vol_elasticity : float
        Elasticity of return volatility γ - 1.
    is_absorbed_at_zero : bool
        True if the origin is an absorbing boundary (γ < 1).
    regime : str
        Regime name: 'equity_leverage' (γ < 1), 'geometric_brownian' (γ = 1),
        or 'commodity' (γ > 1).

    References
    ----------
    Cox (1975); Schroder (1989), *Journal of Finance* 44(1), 211–219.
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.2,
        gamma: float = 0.5,
        s0: float = 100.0,
        r: float | None = None,
        q: float = 0.0,
    ) -> None:
        if not np.all(np.isfinite([mu, sigma, gamma, s0, q, mu if r is None else r])):
            raise ValueError("CEV parameters must be finite")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if s0 <= 0:
            raise ValueError(f"s0 must be strictly positive, got {s0}")
        if q < 0:
            raise ValueError(f"q must be non-negative, got {q}")

        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.s0 = float(s0)
        self.q = float(q)

        if r is not None:
            self.r = float(r)
            self.mu = float(r - q)
        else:
            self.mu = float(mu)
            self.r = float(mu + q)

        regime_label = (
            "equity leverage (absorbing 0)"
            if self.gamma < 1.0
            else ("GBM (constant vol)" if self.gamma == 1.0 else "commodity (inverse leverage)")
        )

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CEVProcess(mu={self.mu}, sigma={self.sigma}, gamma={self.gamma}, s0={self.s0})",
            math_note=(
                f"dS_t = μ S_t dt + σ S_t^γ dW_t; μ={self.mu}, σ={self.sigma}, "
                f"γ={self.gamma}, s0={self.s0}; regime: {regime_label}; "
                f"variance elasticity 2(γ-1) = {2.0 * (self.gamma - 1.0):.2f}"
            ),
            reference="Cox (1975); Schroder (1989), Journal of Finance 44(1), 211–219",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def __repr__(self) -> str:
        return f"CEVProcess(mu={self.mu}, sigma={self.sigma}, gamma={self.gamma}, s0={self.s0})"

    # ------------------------------------------------------------------
    # Core structural properties
    # ------------------------------------------------------------------

    @property
    def elasticity(self) -> float:
        r"""
        Elasticity of return variance with respect to asset price:
        \frac{d(\sigma(S)^2) / \sigma(S)^2}{dS / S} = 2(\gamma - 1).
        """
        return 2.0 * (self.gamma - 1.0)

    @property
    def vol_elasticity(self) -> float:
        r"""
        Elasticity of return volatility with respect to asset price:
        \frac{d\sigma(S) / \sigma(S)}{dS / S} = \gamma - 1.
        """
        return self.gamma - 1.0

    @property
    def is_absorbed_at_zero(self) -> bool:
        r"""True if boundary at zero is absorbing (γ < 1)."""
        return self.gamma < 1.0

    @property
    def feller_condition(self) -> bool:
        r"""
        True if the origin is strictly inaccessible (γ ≥ 1).
        For γ < 1, the origin is an attainable absorbing boundary.
        """
        return self.gamma >= 1.0

    @property
    def regime(self) -> str:
        r"""
        Regime name:
        - 'equity_leverage' (γ < 1)
        - 'geometric_brownian' (γ = 1)
        - 'commodity' (γ > 1)
        """
        if self.gamma == 1.0:
            return "geometric_brownian"
        elif self.gamma < 1.0:
            return "equity_leverage"
        else:
            return "commodity"

    # ------------------------------------------------------------------
    # Local Volatility & Default / Absorption Probability
    # ------------------------------------------------------------------

    def local_volatility(self, S: float | np.ndarray) -> float | np.ndarray:
        r"""
        Local percentage return volatility:
            σ_loc(S) = σ S^{γ - 1}.

        Parameters
        ----------
        S : float or np.ndarray
            Asset price level S ≥ 0.

        Returns
        -------
        float or np.ndarray
            Local volatility value(s).
        """
        s_arr = np.asarray(S, dtype=float)
        if np.any(s_arr < 0.0):
            raise ValueError("Asset price S must be non-negative")

        if self.gamma == 1.0:
            vol = np.full_like(s_arr, self.sigma)
        elif self.gamma < 1.0:
            with np.errstate(divide="ignore"):
                vol = np.where(s_arr == 0.0, np.inf, self.sigma * (s_arr ** (self.gamma - 1.0)))
        else:
            vol = self.sigma * (s_arr ** (self.gamma - 1.0))

        return float(vol) if np.isscalar(S) else vol

    def absorption_probability(
        self, T: float, r: float | None = None, q: float | None = None
    ) -> float:
        """Exact absorbing mass; noncentral-chi-square identity, Schroder (1989)."""
        validate_time(T)
        rate = self.r if r is None else float(r)
        dividend = self.q if q is None else float(q)
        if not np.all(np.isfinite([rate, dividend])):
            raise ValueError("Rates must be finite")
        if T == 0 or self.gamma >= 1:
            return 0.0
        a = 1 - self.gamma
        scale = self._transition_scale(T, self.mu if r is None and q is None else rate - dividend)
        return float(stats.chi2.sf(self.s0 ** (2 * a) / scale, df=1 / a))

    # ------------------------------------------------------------------
    # Analytical Option Pricing (Schroder 1989 / Cox 1975)
    # ------------------------------------------------------------------

    def call_price(
        self, K: float, T: float, r: float | None = None, q: float | None = None
    ) -> float:
        """Noncentral-chi-square European call (Schroder 1989).

        For gamma>1 this is the fully collateralized convention: discounted
        expected payoff plus the strict-local-martingale defect. It is not the
        minimal uncollateralized expected-payoff price. All evaluations are numerical.
        """
        if not np.isfinite(K) or K <= 0:
            raise ValueError("Strike K must be strictly positive and finite")
        if not np.isfinite(T) or T < 0:
            raise ValueError("Maturity T must be non-negative and finite")
        rate, dividend, drift = self._pricing_rates(r, q)
        if T == 0:
            return max(self.s0 - K, 0.0)
        spot = self.s0 * math.exp(-dividend * T)
        strike = K * math.exp(-rate * T)
        if self.gamma == 1:
            d1 = (math.log(self.s0 / K) + (drift + 0.5 * self.sigma**2) * T) / (
                self.sigma * math.sqrt(T)
            )
            return float(
                max(
                    spot * stats.norm.cdf(d1)
                    - strike * stats.norm.cdf(d1 - self.sigma * math.sqrt(T)),
                    0,
                )
            )
        a = 1 - self.gamma
        scale = self._transition_scale(T, drift)
        x = self.s0 ** (2 * a) / scale
        y = (K * math.exp(-drift * T)) ** (2 * a) / scale
        if a > 0:
            value = spot * stats.ncx2.sf(y, df=2 + 1 / a, nc=x) - strike * stats.ncx2.cdf(
                x, df=1 / a, nc=y
            )
        else:
            value = spot * stats.ncx2.sf(x, df=-1 / a, nc=y) - strike * stats.ncx2.cdf(
                y, df=2 - 1 / a, nc=x
            )
        return float(max(value, 0))

    def put_price(
        self, K: float, T: float, r: float | None = None, q: float | None = None
    ) -> float:
        """Discounted European expected put payoff.

        The gamma>1 call includes full collateralization; put-call parity refers
        to that call convention. For gamma<=1 both are ordinary expected payoffs.
        """
        if not np.isfinite(K) or K <= 0:
            raise ValueError("Strike K must be strictly positive and finite")
        if not np.isfinite(T) or T < 0:
            raise ValueError("Maturity T must be non-negative and finite")
        rate, dividend, drift = self._pricing_rates(r, q)
        if T == 0:
            return max(K - self.s0, 0.0)
        spot = self.s0 * math.exp(-dividend * T)
        strike = K * math.exp(-rate * T)
        if self.gamma == 1:
            d1 = (math.log(self.s0 / K) + (drift + 0.5 * self.sigma**2) * T) / (
                self.sigma * math.sqrt(T)
            )
            return float(
                max(
                    strike * stats.norm.cdf(-d1 + self.sigma * math.sqrt(T))
                    - spot * stats.norm.cdf(-d1),
                    0,
                )
            )
        a = 1 - self.gamma
        if a < 0:
            return float(max(self.call_price(K, T, r, q) - spot + strike, 0))
        scale = self._transition_scale(T, drift)
        x = self.s0 ** (2 * a) / scale
        y = (K * math.exp(-drift * T)) ** (2 * a) / scale
        value = strike * stats.ncx2.sf(x, df=1 / a, nc=y) - spot * stats.ncx2.cdf(
            y, df=2 + 1 / a, nc=x
        )
        return float(max(value, 0))

    def price(
        self,
        K: float,
        T: float,
        option_type: str = "call",
        r: float | None = None,
        q: float | None = None,
    ) -> float:
        r"""Unified European option pricer for 'call' or 'put'."""
        if option_type == "call":
            return self.call_price(K, T, r=r, q=q)
        elif option_type == "put":
            return self.put_price(K, T, r=r, q=q)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    # ------------------------------------------------------------------
    # Greeks (Delta, Gamma, Vega)
    # ------------------------------------------------------------------

    def delta(
        self,
        K: float,
        T: float,
        option_type: str = "call",
        r: float | None = None,
        q: float | None = None,
        eps: float = 1e-4,
    ) -> float:
        """Central finite-difference delta using immutable parameter-bump copies."""
        if not np.isfinite(eps) or not 0 < eps < 1:
            raise ValueError("eps must be finite and between zero and one")
        h = self.s0 * eps
        up = self._bumped(s0=self.s0 + h).price(K, T, option_type, r, q)
        down = self._bumped(s0=self.s0 - h).price(K, T, option_type, r, q)
        return float((up - down) / (2 * h))

    def gamma_greek(
        self, K: float, T: float, r: float | None = None, q: float | None = None, eps: float = 1e-4
    ) -> float:
        """Central finite-difference gamma; the source process is never changed."""
        if not np.isfinite(eps) or not 0 < eps < 1:
            raise ValueError("eps must be finite and between zero and one")
        h = self.s0 * eps
        up = self._bumped(s0=self.s0 + h).call_price(K, T, r, q)
        down = self._bumped(s0=self.s0 - h).call_price(K, T, r, q)
        return float((up - 2 * self.call_price(K, T, r, q) + down) / (h * h))

    def vega(
        self, K: float, T: float, r: float | None = None, q: float | None = None, eps: float = 1e-4
    ) -> float:
        """Central finite-difference vega using immutable parameter-bump copies."""
        if not np.isfinite(eps) or not 0 < eps < 1:
            raise ValueError("eps must be finite and between zero and one")
        h = self.sigma * eps
        up = self._bumped(sigma=self.sigma + h).call_price(K, T, r, q)
        down = self._bumped(sigma=self.sigma - h).call_price(K, T, r, q)
        return float((up - down) / (2 * h))

    # ------------------------------------------------------------------
    # Transition CDF & Conditional Moments
    # ------------------------------------------------------------------

    def transition_cdf(
        self, k: float, T: float, r: float | None = None, q: float | None = None
    ) -> float:
        """CDF of the absorbing law (gamma<1), GBM or reciprocal Bessel law.

        For gamma>1 this is the actual state distribution, not a numerical
        strike derivative of a collateralized call price. Revuz--Yor, Ch. XI;
        Schroder (1989). The initial state is deterministic at T=0.
        """
        if not np.isfinite(T) or T < 0:
            raise ValueError("Maturity T must be non-negative and finite")
        if np.isnan(k):
            raise ValueError("CDF threshold must not be NaN")
        rate = self.r if r is None else float(r)
        dividend = self.q if q is None else float(q)
        if not np.all(np.isfinite([rate, dividend])):
            raise ValueError("Rates must be finite")
        if k < 0:
            return 0.0
        if T == 0:
            return float(k >= self.s0)
        if np.isposinf(k):
            return 1.0
        if k == 0:
            return self.absorption_probability(T, r=r, q=q)
        drift = self.mu if r is None and q is None else rate - dividend
        if self.gamma == 1:
            z = (math.log(k / self.s0) - (drift - 0.5 * self.sigma**2) * T) / (
                self.sigma * math.sqrt(T)
            )
            return float(stats.norm.cdf(z))
        a = 1 - self.gamma
        scale = self._transition_scale(T, drift)
        initial = self.s0 ** (2 * a) / scale
        terminal = (k * math.exp(-drift * T)) ** (2 * a) / scale
        if a > 0:
            return float(stats.ncx2.sf(initial, df=1 / a, nc=terminal))
        return float(stats.ncx2.sf(terminal, df=2 - 1 / a, nc=initial))

    def conditional_mean(self, t: float, s0: float | None = None) -> float:
        """Mean of the configured state, including the strict-local-martingale defect.

        For gamma<=1 the discounted state is a true martingale. For gamma>1
        the reciprocal-Bessel h-transform gives a regularized incomplete-Gamma
        factor; simply returning s0*exp(mu*t) would be incorrect.
        """
        validate_time(t)
        initial = self.s0 if s0 is None else float(s0)
        if not np.isfinite(initial) or initial < 0:
            raise ValueError("s0 must be non-negative and finite")
        if initial == 0 or t == 0:
            return initial
        mean = initial * math.exp(self.mu * t)
        if self.gamma > 1:
            b = self.gamma - 1
            argument = initial ** (-2 * b) / (2 * self._transition_scale(t, self.mu))
            mean *= float(special.gammainc(1 / (2 * b), argument))
        return float(mean)

    def conditional_variance(self, t: float, s0: float | None = None) -> float:
        """True conditional variance, not a finite-tail-cutoff second moment.

        Closed forms cover absorbed linear diffusion, branching diffusion and
        GBM. Other finite cases use positive centered-tail quadrature over the
        full support. For 1<gamma<=3/2 the second moment is infinite.
        """
        validate_time(t)
        initial = self.s0 if s0 is None else float(s0)
        if not np.isfinite(initial) or initial < 0:
            raise ValueError("s0 must be non-negative and finite")
        if t == 0 or initial == 0:
            return 0.0
        if 1 < self.gamma <= 1.5:
            return math.inf
        if self.gamma == 1:
            return float(initial**2 * math.exp(2 * self.mu * t) * math.expm1(self.sigma**2 * t))
        if self.gamma == 0.5:
            return float(
                self.sigma**2 * initial * math.exp(self.mu * t) * t * special.exprel(self.mu * t)
            )
        if self.gamma == 0:
            variance = self.sigma**2 * t * float(special.exprel(-2 * self.mu * t))
            std = math.sqrt(variance)
            z = initial / std
            # The image density has an atom at zero; it is not a free Gaussian.
            value = (
                variance * special.erf(z / math.sqrt(2))
                - 2 * initial**2 * stats.norm.sf(z)
                + 2 * initial * std * stats.norm.pdf(z)
            )
            return float(max(value, 0) * math.exp(2 * self.mu * t))
        law = self if initial == self.s0 else self._bumped(s0=initial)
        mean = law.conditional_mean(t)
        width = min(mean, self.sigma * initial**self.gamma * math.sqrt(t))
        if width <= 0 or not np.isfinite(width):
            raise ValueError("CEV variance exceeds numerical conditioning range")
        atom = law.absorption_probability(t)
        a = 1 - self.gamma
        scale = law._transition_scale(t, self.mu)
        noncentrality = initial ** (2 * a) / scale
        discount = math.exp(-self.mu * t)

        def survival(s: float) -> float:
            terminal = (s * discount) ** (2 * a) / scale
            # Direct lower tails prevent cancellation in 1-CDF in the far tail.
            if a > 0:
                return float(stats.ncx2.cdf(noncentrality, df=1 / a, nc=terminal))
            return float(stats.ncx2.cdf(terminal, df=2 - 1 / a, nc=noncentrality))

        bound = mean / width

        def left(z: float) -> float:
            s = max(mean - width * z, 0)
            return z * max(law.transition_cdf(s, t) - atom, 0)

        middle = min(bound, 8.0)
        lower = integrate.quad(left, 0, middle, epsabs=1e-10, epsrel=1e-9, limit=300)[0]
        if bound > middle:
            lower += integrate.quad(left, middle, bound, epsabs=1e-10, epsrel=1e-9, limit=300)[0]
        upper = integrate.quad(
            lambda z: z * survival(mean + width * z),
            0,
            np.inf,
            epsabs=1e-10,
            epsrel=1e-9,
            limit=400,
        )[0]
        result = atom * mean * mean + 2 * width * width * (lower + upper)
        if not np.isfinite(result) or result < 0:
            raise ArithmeticError("CEV variance quadrature failed")
        return float(result)

    def conditional_std(self, t: float, s0: float | None = None) -> float:
        r"""Conditional standard deviation std(S_t | S_0 = s_0)."""
        return float(math.sqrt(self.conditional_variance(t, s0=s0)))

    # ------------------------------------------------------------------
    # Exactness, Triplet & Properties
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "CEVProcess has state-dependent local volatility σ(S) = σ S^{γ-1} "
                "and non-independent increments — it is not a Lévy process and has "
                "no Lévy–Khintchine triplet. Use .call_price(), .put_price(), "
                ".absorption_probability(), .conditional_mean(), "
                ".conditional_variance(), or .simulate()."
            ),
        )

    def _properties(self) -> ProcessProperties:
        regime_desc = (
            f"equity leverage regime (γ={self.gamma} < 1), 0 is absorbing"
            if self.gamma < 1.0
            else (
                "GBM regime (γ=1.0)"
                if self.gamma == 1.0
                else f"commodity regime (γ={self.gamma} > 1)"
            )
        )
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(abs(self.mu) < 1e-12 and self.gamma <= 1.0),
            has_finite_variance=(self.gamma <= 1 or self.gamma > 1.5),
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=(2 * self.gamma - 1 if self.gamma > 1 else None),
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "Constant Elasticity of Variance (CEV) process: dS_t = μ S_t dt + σ S_t^γ dW_t.",
                f"Parameters: μ={self.mu}, σ={self.sigma}, γ={self.gamma}, s0={self.s0}.",
                f"Local volatility: σ(S) = {self.sigma} * S^{self.gamma - 1.0:.3f}.",
                f"Regime: {regime_desc}.",
                "CEVProcess has state-dependent local volatility and non-independent increments.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        return self.cumulants_at(order, t=1.0)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_steps: int,
        dt: float | None = None,
        time_horizon: float | None = None,
        n_paths: int = 1,
        rng: np.random.Generator | None = None,
        method: str = "exact",
        *,
        T: float | None = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the CEV process.

        Parameters
        ----------
        n_steps : int
            Number of time steps.
        dt : float, optional
            Time step size Δt > 0.
        time_horizon : float, optional
            Terminal horizon T > 0.
        n_paths : int, default 1
            Number of independent sample paths.
        rng : np.random.Generator, optional
            Random number generator.
        method : {"exact", "milstein", "euler", "absorbed_euler"}, default "exact"
            Simulation scheme.

        Returns
        -------
        np.ndarray
            Path array of shape (n_paths, n_steps + 1) with paths[:, 0] == s0.
        """
        if T is not None and time_horizon is not None and not np.isclose(T, time_horizon):
            raise ValueError("T and time_horizon must agree")
        horizon = time_horizon if T is None else T
        return sample_cev(
            mu=self.mu,
            sigma=self.sigma,
            gamma=self.gamma,
            s0=self.s0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=horizon,
            rng=rng,
            method=method,
        )

    def _transition_scale(self, t: float, drift: float) -> float:
        """Squared-Bessel scale after discounting the linear drift (Cox/Schroder)."""
        a = 1 - self.gamma
        time = t * float(special.exprel(-2 * drift * a * t))
        scale = self.sigma**2 * a * a * time
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("CEV transition parameters exceed numerical range")
        return scale

    def _bumped(self, *, s0: float | None = None, sigma: float | None = None) -> CEVProcess:
        """Parameter bumps create a new law; never mutate an existing process or tree."""
        return CEVProcess(
            mu=self.mu,
            sigma=self.sigma if sigma is None else sigma,
            gamma=self.gamma,
            s0=self.s0 if s0 is None else s0,
            q=self.q,
        )

    def cumulants_at(self, order: int, t: float = 1.0) -> dict[int, float]:
        validate_order(order)
        validate_time(t)
        if t == 0:
            return {n: self.s0 if n == 1 else 0.0 for n in range(1, order + 1)}
        if order > 2:
            raise NotImplementedError("CEV conditional cumulants are implemented through order two")
        result = {1: self.conditional_mean(t)}
        if order == 2:
            result[2] = self.conditional_variance(t)
            if not np.isfinite(result[2]):
                raise ValueError("CEV second moment does not exist for 1 < gamma <= 3/2")
        return result

    def _pricing_rates(self, r: float | None, q: float | None) -> tuple[float, float, float]:
        rate = self.r if r is None else float(r)
        dividend = self.q if q is None else float(q)
        if not np.all(np.isfinite([rate, dividend])):
            raise ValueError("Pricing rates must be finite")
        drift = self.mu if r is None and q is None else rate - dividend
        return rate, dividend, drift


# Aliases
CEV = CEVProcess
ConstantElasticityOfVariance = CEVProcess
