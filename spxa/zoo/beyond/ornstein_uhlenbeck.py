r"""
Gaussian Ornstein–Uhlenbeck (OU) process / Vasicek process.

The Ornstein–Uhlenbeck process satisfies the linear stochastic differential equation:
    dX_t = θ (μ - X_t) dt + σ dW_t,   X_0 = x_0

where:
  θ > 0 : Speed of mean reversion (relaxation rate)
  μ ∈ ℝ : Long-term mean level
  σ > 0 : Volatility parameter (diffusion coefficient)
  W_t   : Standard 1-dimensional Brownian motion

Theoretical Properties
----------------------
1. Explicit Solution:
   By variation of parameters (Itô's formula applied to e^{θt} X_t):
       X_t = μ + (x_0 - μ) e^{-θt} + σ ∫_0^t e^{-θ(t-s)} dW_s

2. Exact Conditional Distribution:
   Conditioned on X_0 = x_0, X_t is strictly Gaussian:
       X_t | X_0 ~ N(m(t), v(t))
   where:
       m(t) = μ + (x_0 - μ) e^{-θt}
       v(t) = (σ² / (2θ)) (1 - e^{-2θt})

3. Stationary Distribution:
   As t → ∞, X_t converges in law to the invariant Gaussian distribution:
       X_∞ ~ N(μ, σ² / (2θ))
   with stationary mean μ and stationary variance σ² / (2θ). All cumulants
   of order n ≥ 3 vanish.

4. Temporal Dependence & Autocorrelation:
   The stationary autocovariance and autocorrelation functions are:
       Cov(X_t, X_{t+h}) = (σ² / (2θ)) e^{-θh}   for h ≥ 0
       Corr(X_t, X_{t+h}) = e^{-θh}
   The characteristic relaxation time is τ = 1/θ, and the half-life
   of mean reversion is t_{1/2} = ln(2) / θ.

5. Power Spectral Density (Wiener–Khinchin theorem):
   The two-sided power spectral density is a Lorentzian (Cauchy) profile:
       S(ω) = σ² / (π (θ² + ω²))

6. Vasicek Zero-Coupon Bond Pricing:
   Interpreting X_t as the short-term interest rate r_t gives the classical
   Vasicek (1977) affine term structure model:
       P(t, T) = E[exp(-∫_t^T r_s ds) | r_t] = A(τ) exp(-B(τ) r_t)
   where τ = T - t, B(τ) = (1 - e^{-θτ}) / θ, and ln A(τ) is known in closed form.

Exactness Level: MOMENT_PROPAGATION (mean-reverting drift and serially correlated
increments prevent a Lévy–Khintchine triplet, but the transition law is exact Gaussian).

References
----------
Uhlenbeck, G.E. & Ornstein, L.S. (1930). On the theory of the Brownian motion.
*Physical Review*, 36(5), 823–841.

Vasicek, O. (1977). An equilibrium characterization of the term structure.
*Journal of Financial Economics*, 5(2), 177–188.

Bibby, B.M., Skovgaard, I.M. & Sørensen, M. (2005). Diffusion-type models with
given marginal distributions and autocorrelation function. *Bernoulli*, 11(2), 191–220.
"""

from __future__ import annotations

import math

import numpy as np

from spxa.core._validation import validate_order, validate_time
from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class OrnsteinUhlenbeckProcess(Process):
    r"""
    Gaussian Ornstein–Uhlenbeck (OU) mean-reverting diffusion process.

    dX_t = θ (μ - X_t) dt + σ dW_t,   X_0 = x_0.

    Parameters
    ----------
    theta : float
        Speed of mean reversion θ > 0.
    mu : float
        Long-term equilibrium mean level μ ∈ ℝ.
    sigma : float
        Diffusion volatility coefficient σ > 0.
    x0 : float, optional
        Initial value X_0. Defaults to `mu` if not provided.

    Aliases
    -------
    OrnsteinUhlenbeck, VasicekProcess, Vasicek.

    References
    ----------
    Uhlenbeck & Ornstein (1930), *Physical Review*, 36(5), 823–841.
    Vasicek (1977), *Journal of Financial Economics*, 5(2), 177–188.
    """

    def __init__(
        self,
        theta: float,
        mu: float,
        sigma: float,
        x0: float | None = None,
    ) -> None:
        if not np.all(np.isfinite([theta, mu, sigma, mu if x0 is None else x0])):
            raise ValueError("Process parameters and initial state must be finite")
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")

        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.x0 = self.mu if x0 is None else float(x0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=(
                "OrnsteinUhlenbeck(theta="
                f"{self.theta}"
                ", mu="
                f"{self.mu}"
                ", sigma="
                f"{self.sigma}"
                ", x0="
                f"{self.x0}"
                ")"
            ),
            math_note=(
                "dX_t = θ(μ - X_t)dt + σ dW_t; θ="
                f"{self.theta}"
                ", μ="
                f"{self.mu}"
                ", σ="
                f"{self.sigma}"
                ", x0="
                f"{self.x0}"
                "; Stationary law: N(μ="
                f"{self.mu}"
                ", σ²/(2θ)="
                f"{self.stationary_variance:.6f}"
                "); Half-life = ln(2)/θ = "
                f"{self.half_life:.4f}"
            ),
            reference="Uhlenbeck & Ornstein (1930); Vasicek (1977)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Structural & Timescale Properties
    # ------------------------------------------------------------------

    @property
    def relaxation_time(self) -> float:
        r"""Characteristic relaxation time τ = 1 / θ."""
        return 1.0 / self.theta

    @property
    def half_life(self) -> float:
        r"""Half-life of mean reversion t_{1/2} = ln(2) / θ."""
        return math.log(2.0) / self.theta

    @property
    def stationary_mean(self) -> float:
        r"""Stationary expectation E[X_∞] = μ."""
        return self.mu

    @property
    def stationary_variance(self) -> float:
        r"""Stationary variance Var(X_∞) = σ² / (2θ)."""
        return float(self.sigma**2 / (2.0 * self.theta))

    @property
    def stationary_std(self) -> float:
        r"""Stationary standard deviation std(X_∞) = σ / √(2θ)."""
        return float(self.sigma / math.sqrt(2.0 * self.theta))

    @property
    def stationary_skewness(self) -> float:
        r"""Stationary skewness γ_1 = 0 (Gaussian)."""
        return 0.0

    @property
    def stationary_kurtosis(self) -> float:
        r"""Stationary excess kurtosis γ_2 = 0 (Gaussian)."""
        return 0.0

    @property
    def asymptotic_yield(self) -> float:
        r"""
        Long-term asymptotic zero-coupon bond yield as maturity τ → ∞:
        y_∞ = μ - σ² / (2θ²).
        Vasicek (1977), Section 3.
        """
        return float(self.mu - self.sigma**2 / (2.0 * (self.theta**2)))

    # ------------------------------------------------------------------
    # Analytical Moments & Cumulants
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float, x0: float | None = None) -> float:
        validate_time(t)
        initial = self.x0 if x0 is None else float(x0)
        if not np.isfinite(initial):
            raise ValueError("x0 must be finite")
        return float(initial * math.exp(-self.theta * t) + self.mu * (-math.expm1(-self.theta * t)))

    def conditional_variance(self, t: float) -> float:
        r"""
        Conditional variance Var(X_t | X_0) = (σ² / (2θ)) (1 - e^{-2θt}).

        Parameters
        ----------
        t : float
            Time horizon t ≥ 0.
        """
        if not np.isfinite(t) or t < 0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float((self.sigma**2 / (2.0 * self.theta)) * (-math.expm1(-2.0 * self.theta * t)))

    def conditional_std(self, t: float) -> float:
        r"""Conditional standard deviation std(X_t | X_0)."""
        return float(math.sqrt(self.conditional_variance(t)))

    def conditional_cumulant(
        self,
        n: int,
        t: float,
        x0: float | None = None,
    ) -> float:
        r"""
        Conditional cumulant κ_n(X_t | X_0).
        Because X_t | X_0 is Gaussian, all cumulants of order n ≥ 3 are zero.
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"Cumulant order must be an integer >= 1, got {n}")
        if not np.isfinite(t) or t < 0:
            raise ValueError(f"Time t must be non-negative, got {t}")

        if n == 1:
            return self.conditional_mean(t, x0)
        elif n == 2:
            return self.conditional_variance(t)
        else:
            return 0.0

    def conditional_cumulants(
        self,
        t: float,
        order: int = 4,
        x0: float | None = None,
    ) -> dict[int, float]:
        r"""Compute exact conditional cumulants up to specified order."""
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        return {n: self.conditional_cumulant(n, t, x0) for n in range(1, order + 1)}

    def stationary_cumulant(self, n: int) -> float:
        r"""
        Exact n-th cumulant of the stationary distribution:
        κ_1 = μ, κ_2 = σ² / (2θ), κ_n = 0 for n ≥ 3.
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"Cumulant order must be an integer >= 1, got {n}")
        if n == 1:
            return self.mu
        elif n == 2:
            return self.stationary_variance
        else:
            return 0.0

    def autocovariance(self, h: float) -> float:
        r"""
        Stationary autocovariance Cov(X_t, X_{t+h}) = Var(X_∞) e^{-θh}.

        Parameters
        ----------
        h : float
            Time lag h ≥ 0.
        """
        if not np.isfinite(h) or h < 0:
            raise ValueError(f"Lag h must be non-negative, got {h}")
        return float(self.stationary_variance * math.exp(-self.theta * h))

    def autocorrelation(self, h: float) -> float:
        r"""Stationary autocorrelation Corr(X_t, X_{t+h}) = e^{-θh}."""
        if not np.isfinite(h) or h < 0:
            raise ValueError(f"Lag h must be non-negative, got {h}")
        return float(math.exp(-self.theta * h))

    def two_time_covariance(self, s: float, t: float) -> float:
        r"""
        Non-stationary covariance Cov(X_s, X_t) for deterministic initial condition X_0 = x_0:
        Cov(X_s, X_t) = (σ² / (2θ)) [e^{-θ|t-s|} - e^{-θ(s+t)}].
        """
        if not np.all(np.isfinite([s, t])) or s < 0 or t < 0:
            raise ValueError("Time arguments s and t must be non-negative and finite")
        return float(self.conditional_variance(min(s, t)) * math.exp(-self.theta * abs(t - s)))

    def spectral_density(
        self,
        omega: float | np.ndarray,
    ) -> float | np.ndarray:
        r"""
        Two-sided spectral density S(ω) = σ² / (2π (θ² + ω²)).

        Convention: C(h) = integral exp(i ω h) S(ω) dω over the real line,
        so integral S(ω) dω = stationary_variance.

        Parameters
        ----------
        omega : float or array_like
            Angular frequency ω.
        """
        w_arr = np.asarray(omega, dtype=float)
        is_scalar = w_arr.ndim == 0
        w_arr = np.atleast_1d(w_arr)

        dens = (self.sigma**2 / (2 * math.pi)) / (self.theta**2 + w_arr**2)
        return float(dens[0]) if is_scalar else dens

    # ------------------------------------------------------------------
    # Characteristic Function & Laplace Transform
    # ------------------------------------------------------------------

    def char_func(
        self, u: complex | float | np.ndarray, t: float = 1.0, x0: float | None = None
    ) -> complex | np.ndarray:
        validate_time(t)
        if x0 is not None and not np.isfinite(x0):
            raise ValueError("x0 must be finite")
        z = np.asarray(u, dtype=complex)
        result = np.exp(
            1j * z * self.conditional_mean(t, x0) - 0.5 * z * z * self.conditional_variance(t)
        )
        return complex(result) if z.ndim == 0 else np.asarray(result)

    def laplace_transform(
        self,
        u: float,
        t: float = 1.0,
        x0: float | None = None,
    ) -> float:
        r"""
        Conditional Laplace transform E[exp(-u X_t) | X_0]:
        L(u) = exp(-u m(t) + (1/2) u² v(t)).
        """
        if not np.isfinite(t) or t < 0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        m = self.conditional_mean(t, x0)
        v = self.conditional_variance(t)
        return float(math.exp(-u * m + 0.5 * (u**2) * v))

    # ------------------------------------------------------------------
    # Transition & Stationary Probability Densities
    # ------------------------------------------------------------------

    def transition_density(
        self,
        y: float | np.ndarray,
        t: float,
        x0: float | None = None,
        s: float = 0.0,
    ) -> float | np.ndarray:
        r"""
        Exact transition Gaussian probability density p(y, t | x_s, s).

        p(y, t | x, s) = (1 / √(2π v)) exp(-(y - m)² / (2v))
        where Δt = t - s > 0, m = μ + (x - μ) e^{-θΔt}, and v = (σ²/(2θ))(1 - e^{-2θΔt}).
        """
        dt = t - s
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"t must be strictly greater than s (dt={dt})")

        val_x0 = self.x0 if x0 is None else float(x0)
        y_arr = np.asarray(y, dtype=float)
        is_scalar = y_arr.ndim == 0
        y_arr = np.atleast_1d(y_arr)

        m = self.conditional_mean(dt, x0=val_x0)
        v = self.conditional_variance(dt)
        std = math.sqrt(v)

        norm_factor = 1.0 / (math.sqrt(2.0 * math.pi) * std)
        dens = norm_factor * np.exp(-0.5 * ((y_arr - m) / std) ** 2)

        return float(dens[0]) if is_scalar else dens

    def stationary_density(
        self,
        y: float | np.ndarray,
    ) -> float | np.ndarray:
        r"""
        Exact stationary Gaussian probability density f_∞(y) ~ N(μ, σ²/(2θ)).
        """
        y_arr = np.asarray(y, dtype=float)
        is_scalar = y_arr.ndim == 0
        y_arr = np.atleast_1d(y_arr)

        std = self.stationary_std
        norm_factor = 1.0 / (math.sqrt(2.0 * math.pi) * std)
        dens = norm_factor * np.exp(-0.5 * ((y_arr - self.mu) / std) ** 2)

        return float(dens[0]) if is_scalar else dens

    # ------------------------------------------------------------------
    # Vasicek Term Structure & Zero-Coupon Bond Pricing
    # ------------------------------------------------------------------

    def bond_price(
        self,
        tau: float | np.ndarray,
        r0: float | None = None,
    ) -> float | np.ndarray:
        r"""
        Vasicek zero-coupon bond price P(τ) = A(τ) exp(-B(τ) r_0).

        Parameters
        ----------
        tau : float or array_like
            Time to maturity τ = T - t ≥ 0.
        r0 : float, optional
            Current spot short rate r_t. Defaults to `self.x0`.

        Returns
        -------
        float or np.ndarray
            Discount bond price P(τ).
        """
        tau_arr = np.asarray(tau, dtype=float)
        is_scalar = tau_arr.ndim == 0
        tau_arr = np.atleast_1d(tau_arr)

        if not np.all(np.isfinite(tau_arr)) or np.any(tau_arr < 0):
            raise ValueError("Maturity tau must be non-negative")

        val_r0 = self.x0 if r0 is None else float(r0)
        out = np.ones_like(tau_arr)
        pos = tau_arr > 0.0

        if np.any(pos):
            t_pos = tau_arr[pos]
            b_val = (-np.expm1(-self.theta * t_pos)) / self.theta
            gamma_term = self.mu - self.sigma**2 / (2.0 * (self.theta**2))
            ln_a = gamma_term * (b_val - t_pos) - (self.sigma**2 / (4.0 * self.theta)) * (b_val**2)
            out[pos] = np.exp(ln_a - b_val * val_r0)

        return float(out[0]) if is_scalar else out

    def bond_yield(
        self,
        tau: float | np.ndarray,
        r0: float | None = None,
    ) -> float | np.ndarray:
        r"""
        Continuously compounded zero-coupon bond yield y(τ) = -ln(P(τ)) / τ.

        For τ = 0, returns the instantaneous spot rate r_0.
        """
        tau_arr = np.asarray(tau, dtype=float)
        is_scalar = tau_arr.ndim == 0
        tau_arr = np.atleast_1d(tau_arr)

        if not np.all(np.isfinite(tau_arr)) or np.any(tau_arr < 0):
            raise ValueError("Maturity tau must be non-negative")

        val_r0 = self.x0 if r0 is None else float(r0)
        out = np.empty_like(tau_arr)
        zero_mask = tau_arr == 0.0
        pos_mask = ~zero_mask

        out[zero_mask] = val_r0
        if np.any(pos_mask):
            t_pos = tau_arr[pos_mask]
            b_val = (-np.expm1(-self.theta * t_pos)) / self.theta
            gamma_term = self.mu - self.sigma**2 / (2.0 * (self.theta**2))
            ln_a = gamma_term * (b_val - t_pos) - (self.sigma**2 / (4.0 * self.theta)) * (b_val**2)
            ln_p = ln_a - b_val * val_r0
            out[pos_mask] = -ln_p / t_pos

        return float(out[0]) if is_scalar else out

    # ------------------------------------------------------------------
    # Process Framework & Protocol Overrides
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "OrnsteinUhlenbeckProcess has mean-reverting drift and serially "
                "correlated increments — it is not a Lévy process and has no "
                "Lévy–Khintchine triplet. Use .conditional_mean(), "
                ".conditional_variance(), .stationary_mean, .bond_price(), or "
                ".simulate()."
            ),
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "Gaussian Ornstein–Uhlenbeck (Vasicek) process: dX_t = θ(μ - X_t)dt + σ dW_t.",
                f"Parameters: θ={self.theta}, μ={self.mu}, σ={self.sigma}, x0={self.x0}.",
                (
                    "Stationary distribution: N(mean="
                    f"{self.mu}"
                    ", variance="
                    f"{self.stationary_variance:.6f}"
                    ")."
                ),
                (
                    "Relaxation time τ="
                    f"{self.relaxation_time:.4f}"
                    ", half-life t_1/2="
                    f"{self.half_life:.4f}"
                    "."
                ),
                (
                    "OrnsteinUhlenbeckProcess has mean-reverting drift and serially "
                    "correlated increments."
                ),
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        """The common protocol denotes X_1, not the stationary distribution."""
        return self.conditional_cumulants(t=1.0, order=order)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
        method: str = "exact",
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the Ornstein–Uhlenbeck process.

        Parameters
        ----------
        n_steps : int
            Number of time steps.
        n_paths : int, default 1
            Number of independent sample paths.
        T : float, default 1.0
            Terminal time horizon.
        rng : np.random.Generator, optional
            Random number generator.
        method : {"exact", "euler"}, default "exact"
            Simulation scheme:
            - "exact": Exact Gaussian transition law (zero discretization error).
            - "euler": Standard Euler–Maruyama discretisation.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). paths[:, 0] == self.x0.
        """
        from spxa.sim.exact import sample_ou

        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

        dt = T / n_steps
        return sample_ou(
            theta=self.theta,
            mu=self.mu,
            sigma=self.sigma,
            x0=self.x0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=rng,
            method=method,
        )

    def __repr__(self) -> str:
        return (
            f"OrnsteinUhlenbeck(theta={self.theta}, mu={self.mu}, sigma={self.sigma}, x0={self.x0})"
        )

    def cumulants_at(self, order: int, t: float = 1.0) -> dict[int, float]:
        validate_order(order)
        validate_time(t)
        return self.conditional_cumulants(t=t, order=order)


# Standard quantitative aliases
OrnsteinUhlenbeck = OrnsteinUhlenbeckProcess
VasicekProcess = OrnsteinUhlenbeckProcess
Vasicek = OrnsteinUhlenbeckProcess
