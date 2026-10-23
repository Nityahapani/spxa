r"""
Cox–Ingersoll–Ross (CIR) square-root diffusion process.

The CIR process satisfies the stochastic differential equation (SDE):
    dr_t = κ(θ - r_t) dt + σ √r_t dW_t,   r_0 ≥ 0

where:
  κ > 0 : Speed of mean reversion
  θ > 0 : Long-term mean level
  σ > 0 : Volatility parameter
  W_t   : Standard 1-dimensional Brownian motion

Theoretical Properties
----------------------
1. Non-negativity & Feller Condition:
   - If 2κθ ≥ σ² (the Feller condition), the boundary at 0 is strictly
     inaccessible, so r_t > 0 almost surely for all t > 0.
   - If 2κθ < σ², the boundary at 0 is accessible and instantaneously
     reflecting.

2. Exact Transition Law (Non-Central Chi-Squared):
   Conditional on r_s at time s < t (with Δt = t - s), the scaled quantity
   2c r_t follows a non-central chi-squared distribution:
       2c r_t | r_s ~ χ²_d(λ)
   where:
       c = 2κ / (σ² (1 - e^{-κΔt}))
       d = 4κθ / σ²                 (effective degrees of freedom)
       λ = 2c r_s e^{-κΔt}          (non-centrality parameter)

3. Stationary Distribution:
   As t → ∞, r_t converges in law to a Gamma distribution:
       r_∞ ~ Gamma(α = 2κθ / σ², β = 2κ / σ²)
   with stationary mean θ and stationary variance θσ² / (2κ).

4. Affine Term Structure & Bond Pricing:
   The Laplace transform and zero-coupon bond pricing have exact closed forms:
       P(t, T) = E[exp(-∫_t^T r_s ds) | r_t] = A(τ) exp(-B(τ) r_t)
   where τ = T - t and A(τ), B(τ) solve the classical Riccati system.

Exactness Level: MOMENT_PROPAGATION (state-dependent diffusion σ√r_t prevents
a Lévy–Khintchine triplet, but exact transition law and moments are available).

References
----------
Cox, J.C., Ingersoll, J.E. & Ross, S.A. (1985). A theory of the term
structure of interest rates. *Econometrica*, 53(2), 385–407.

Feller, W. (1951). Two singular diffusion problems. *Annals of Mathematics*,
54(1), 173–182.

Lord, R., Koekkoek, R. & van Dijk, D. (2010). A comparison of biased simulation
schemes for stochastic volatility models. *Mathematical Finance*, 20(2), 177–213.
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


class CIRProcess(Process):
    r"""
    Cox–Ingersoll–Ross (CIR) square-root diffusion process.

    dr_t = κ(θ - r_t) dt + σ √r_t dW_t,   r_0 ≥ 0.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed κ > 0.
    theta : float
        Long-term mean level θ > 0.
    sigma : float
        Volatility parameter σ > 0.
    r0 : float, optional
        Initial value r_0 ≥ 0. Defaults to `theta` if not provided.

    Attributes
    ----------
    feller_ratio : float
        2κθ / σ². Values ≥ 1 indicate the Feller condition is satisfied.
    feller_condition : bool
        True if 2κθ ≥ σ² (zero boundary strictly inaccessible).

    References
    ----------
    Cox, Ingersoll & Ross (1985), *Econometrica*, 53(2), 385–407.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        r0: float | None = None,
    ) -> None:
        if not np.all(np.isfinite([kappa, theta, sigma, theta if r0 is None else r0])):
            raise ValueError("Process parameters and initial state must be finite")
        if kappa <= 0:
            raise ValueError(f"kappa must be strictly positive, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)

        if r0 is None:
            self.r0 = self.theta
        else:
            if r0 < 0:
                raise ValueError(f"r0 must be non-negative, got {r0}")
            self.r0 = float(r0)

        feller_ratio = 2.0 * self.kappa * self.theta / (self.sigma**2)
        feller_ok = feller_ratio >= 1.0

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=(
                "CIRProcess(kappa="
                f"{self.kappa}"
                ", theta="
                f"{self.theta}"
                ", sigma="
                f"{self.sigma}"
                ", r0="
                f"{self.r0}"
                ")"
            ),
            math_note=(
                f"dr_t = κ(θ - r_t)dt + σ√r_t dW_t; κ={self.kappa}, θ={self.theta}, "
                f"σ={self.sigma}, r0={self.r0}; Feller ratio 2κθ/σ² = {feller_ratio:.4f} "
                f"({'satisfied' if feller_ok else 'violated'})"
            ),
            reference="Cox, Ingersoll & Ross (1985), Econometrica 53(2), 385–407",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Core structural properties
    # ------------------------------------------------------------------

    @property
    def feller_ratio(self) -> float:
        r"""Feller ratio 2κθ / σ²."""
        return float(2.0 * self.kappa * self.theta / (self.sigma**2))

    @property
    def feller_condition(self) -> bool:
        r"""True if 2κθ ≥ σ² (the zero boundary is strictly inaccessible)."""
        return bool(self.feller_ratio >= 1.0)

    @property
    def degrees_of_freedom(self) -> float:
        r"""Effective non-central chi-squared degrees of freedom d = 4κθ / σ²."""
        return float(4.0 * self.kappa * self.theta / (self.sigma**2))

    @property
    def stationary_shape(self) -> float:
        r"""Shape parameter α = 2κθ / σ² of the stationary Gamma distribution."""
        return float(2.0 * self.kappa * self.theta / (self.sigma**2))

    @property
    def stationary_scale(self) -> float:
        r"""Scale parameter θ_scale = σ² / (2κ) of the stationary Gamma distribution."""
        return float(self.sigma**2 / (2.0 * self.kappa))

    @property
    def stationary_rate(self) -> float:
        r"""Rate parameter β = 2κ / σ² of the stationary Gamma distribution."""
        return float(2.0 * self.kappa / (self.sigma**2))

    @property
    def stationary_mean(self) -> float:
        r"""Stationary mean E[r_∞] = θ."""
        return self.theta

    @property
    def stationary_variance(self) -> float:
        r"""Stationary variance Var(r_∞) = θσ² / (2κ)."""
        return float(self.theta * self.sigma**2 / (2.0 * self.kappa))

    @property
    def stationary_std(self) -> float:
        r"""Stationary standard deviation std(r_∞)."""
        return float(np.sqrt(self.stationary_variance))

    @property
    def stationary_skewness(self) -> float:
        r"""Stationary skewness γ_1 = 2 / √α = 2σ / √(2κθ)."""
        return float(2.0 / np.sqrt(self.stationary_shape))

    @property
    def stationary_kurtosis(self) -> float:
        r"""Stationary excess kurtosis γ_2 = 6 / α = 3σ² / (κθ)."""
        return float(6.0 / self.stationary_shape)

    @property
    def asymptotic_yield(self) -> float:
        r"""
        Long-term asymptotic zero-coupon bond yield as maturity τ → ∞:
        y_∞ = 2κθ / (κ + √(κ² + 2σ²)).
        """
        gamma = math.sqrt(self.kappa**2 + 2.0 * self.sigma**2)
        return float(2.0 * self.kappa * self.theta / (self.kappa + gamma))

    # ------------------------------------------------------------------
    # Analytical Conditional Moments & Cumulants
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float, r0: float | None = None) -> float:
        return self.conditional_cumulant(1, t, r0)

    def conditional_variance(self, t: float, r0: float | None = None) -> float:
        return self.conditional_cumulant(2, t, r0)

    def conditional_std(self, t: float, r0: float | None = None) -> float:
        r"""Conditional standard deviation std(r_t | r_0)."""
        return float(np.sqrt(self.conditional_variance(t, r0)))

    def conditional_cumulant(self, n: int, t: float, r0: float | None = None) -> float:
        """Noncentral-chi-square cumulants, in a form stable as t tends to zero."""
        validate_order(n)
        validate_time(t)
        initial = self.r0 if r0 is None else float(r0)
        if not np.isfinite(initial) or initial < 0:
            raise ValueError("r0 must be non-negative and finite")
        if t == 0:
            return initial if n == 1 else 0.0
        scale = self.sigma**2 * (-math.expm1(-self.kappa * t)) / (2 * self.kappa)
        shape = 2 * self.kappa * self.theta / self.sigma**2
        return float(
            math.factorial(n - 1)
            * (shape * scale**n + n * initial * math.exp(-self.kappa * t) * scale ** (n - 1))
        )

    def conditional_cumulants(
        self,
        t: float,
        order: int = 4,
        r0: float | None = None,
    ) -> dict[int, float]:
        r"""Compute exact conditional cumulants up to specified order."""
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        return {n: self.conditional_cumulant(n, t, r0) for n in range(1, order + 1)}

    def stationary_cumulant(self, n: int) -> float:
        r"""
        Exact n-th cumulant of the stationary Gamma distribution:
        κ_n(r_∞) = (n - 1)! θ (σ² / (2κ))^{n - 1}.
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"Cumulant order must be an integer >= 1, got {n}")
        fact = math.factorial(n - 1)
        scale = self.sigma**2 / (2.0 * self.kappa)
        return float(fact * self.theta * (scale ** (n - 1)))

    def autocovariance(self, h: float) -> float:
        r"""
        Stationary autocovariance Cov(r_t, r_{t+h}) = Var(r_∞) e^{-κh}.

        Parameters
        ----------
        h : float
            Time lag h ≥ 0.
        """
        if not np.isfinite(h) or h < 0:
            raise ValueError(f"Lag h must be non-negative, got {h}")
        return float(self.stationary_variance * np.exp(-self.kappa * h))

    def autocorrelation(self, h: float) -> float:
        r"""Stationary autocorrelation Corr(r_t, r_{t+h}) = e^{-κh}."""
        if not np.isfinite(h) or h < 0:
            raise ValueError(f"Lag h must be non-negative, got {h}")
        return float(np.exp(-self.kappa * h))

    # ------------------------------------------------------------------
    # Characteristic Function & Laplace Transform
    # ------------------------------------------------------------------

    def laplace_transform(self, u: float, t: float, r0: float | None = None) -> float:
        if not np.isfinite(u) or u < 0:
            raise ValueError("Transform argument u must be non-negative and finite")
        return float(complex(self.char_func(1j * u, t, r0)).real)

    def char_func(
        self, u: complex | float | np.ndarray, t: float = 1.0, r0: float | None = None
    ) -> complex | np.ndarray:
        validate_time(t)
        initial = self.r0 if r0 is None else float(r0)
        if not np.isfinite(initial) or initial < 0:
            raise ValueError("r0 must be non-negative and finite")
        z = np.asarray(u, dtype=complex)
        scale = self.sigma**2 * (-math.expm1(-self.kappa * t)) / (2 * self.kappa)
        if np.any(1 + z.imag * scale <= 0):
            raise ValueError("CIR MGF is outside its finite domain")
        shape = 2 * self.kappa * self.theta / self.sigma**2
        log_cf = -shape * np.log1p(-1j * z * scale) + 1j * z * initial * math.exp(
            -self.kappa * t
        ) / (1 - 1j * z * scale)
        result = np.exp(log_cf)
        return complex(result) if z.ndim == 0 else np.asarray(result)

    # ------------------------------------------------------------------
    # Transition & Stationary Probability Densities
    # ------------------------------------------------------------------

    def transition_density(
        self, y: float | np.ndarray, t: float, r0: float | None = None, s: float = 0.0
    ) -> float | np.ndarray:
        """Scaled noncentral-chi-square density, avoiding unscaled Bessel overflow."""
        from scipy.stats import ncx2

        dt = t - s
        if not np.isfinite(s) or s < 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("t must be strictly greater than s, with finite nonnegative times")
        initial = self.r0 if r0 is None else float(r0)
        if not np.isfinite(initial) or initial < 0:
            raise ValueError("r0 must be non-negative and finite")
        scale = self.sigma**2 * (-math.expm1(-self.kappa * dt)) / (4 * self.kappa)
        values = np.asarray(y, dtype=float)
        result = np.asarray(
            ncx2.pdf(
                values / scale,
                self.degrees_of_freedom,
                initial * math.exp(-self.kappa * dt) / scale,
            )
            / scale
        )
        return float(result) if values.ndim == 0 else result

    def stationary_density(self, y: float | np.ndarray) -> float | np.ndarray:
        """Stationary Gamma density, evaluated in log-stable library form."""
        from scipy.stats import gamma

        values = np.asarray(y, dtype=float)
        result = np.asarray(gamma.pdf(values, self.stationary_shape, scale=self.stationary_scale))
        return float(result) if values.ndim == 0 else result

    # ------------------------------------------------------------------
    # Term Structure & Zero-Coupon Bond Pricing
    # ------------------------------------------------------------------

    def bond_price(
        self,
        tau: float | np.ndarray,
        r0: float | None = None,
    ) -> float | np.ndarray:
        r"""
        Cox–Ingersoll–Ross zero-coupon bond price P(τ) = A(τ) exp(-B(τ) r_0).

        Parameters
        ----------
        tau : float or array_like
            Time to maturity τ = T - t ≥ 0.
        r0 : float, optional
            Current spot short rate r_t ≥ 0. Defaults to `self.r0`.

        Returns
        -------
        float or np.ndarray
            Discount bond price P(τ) ∈ (0, 1].
        """
        tau_arr = np.asarray(tau, dtype=float)
        is_scalar = tau_arr.ndim == 0
        tau_arr = np.atleast_1d(tau_arr)

        if not np.all(np.isfinite(tau_arr)) or np.any(tau_arr < 0):
            raise ValueError("Maturity tau must be non-negative")

        val_r0 = self.r0 if r0 is None else float(r0)
        if not np.isfinite(val_r0) or val_r0 < 0:
            raise ValueError(f"r0 must be non-negative, got {val_r0}")

        gamma = math.sqrt(self.kappa**2 + 2.0 * self.sigma**2)
        power = 2.0 * self.kappa * self.theta / (self.sigma**2)

        out = np.ones_like(tau_arr)
        pos = tau_arr > 0.0

        if np.any(pos):
            t_pos = tau_arr[pos]
            exp_neg_g = np.exp(-gamma * t_pos)
            denom = (gamma + self.kappa) + (gamma - self.kappa) * exp_neg_g
            b_val = 2.0 * (1.0 - exp_neg_g) / denom
            ln_a = power * (
                math.log(2.0 * gamma) + 0.5 * (self.kappa - gamma) * t_pos - np.log(denom)
            )
            ln_p = ln_a - b_val * val_r0
            out[pos] = np.exp(ln_p)

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

        val_r0 = self.r0 if r0 is None else float(r0)
        if not np.isfinite(val_r0) or val_r0 < 0:
            raise ValueError(f"r0 must be non-negative, got {val_r0}")

        gamma = math.sqrt(self.kappa**2 + 2.0 * self.sigma**2)
        power = 2.0 * self.kappa * self.theta / (self.sigma**2)

        out = np.empty_like(tau_arr)
        zero_mask = tau_arr == 0.0
        pos_mask = ~zero_mask

        out[zero_mask] = val_r0
        if np.any(pos_mask):
            t_pos = tau_arr[pos_mask]
            exp_neg_g = np.exp(-gamma * t_pos)
            denom = (gamma + self.kappa) + (gamma - self.kappa) * exp_neg_g
            b_val = 2.0 * (1.0 - exp_neg_g) / denom
            ln_a = power * (
                math.log(2.0 * gamma) + 0.5 * (self.kappa - gamma) * t_pos - np.log(denom)
            )
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
                "CIRProcess has state-dependent diffusion σ√r_t and serially "
                "correlated increments — it is not a Lévy process and has no "
                "Lévy–Khintchine triplet. Use .conditional_mean(), "
                ".conditional_variance(), .stationary_mean, .bond_price(), or "
                ".simulate()."
            ),
        )

    def _properties(self) -> ProcessProperties:
        feller_status = (
            "strictly positive (Feller condition 2κθ ≥ σ² satisfied)"
            if self.feller_condition
            else "instantaneously reflecting at 0 (2κθ < σ²)"
        )
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
                "Cox–Ingersoll–Ross (CIR) square-root diffusion: dr_t = κ(θ - r_t)dt + σ√r_t dW_t.",
                f"Parameters: κ={self.kappa}, θ={self.theta}, σ={self.sigma}, r0={self.r0}.",
                f"Feller ratio 2κθ/σ² = {self.feller_ratio:.4f} ({feller_status}).",
                (
                    "Stationary distribution: Gamma(shape="
                    f"{self.stationary_shape:.4f}"
                    ", scale="
                    f"{self.stationary_scale:.4f}"
                    ")."
                ),
                (
                    "CIRProcess has state-dependent diffusion σ√r_t and serially "
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
        Simulate sample paths of the CIR process.

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
        method : {"exact", "truncated_euler", "reflected_euler"}, default "exact"
            Simulation scheme:
            - "exact": Exact non-central chi-squared transition distribution (zero
            discretization bias).
            - "truncated_euler": Higham–Mao full truncation Euler–Maruyama scheme.
            - "reflected_euler": Reflected Euler–Maruyama scheme.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). paths[:, 0] == self.r0.
        """
        from spxa.sim.exact import sample_cir

        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

        dt = T / n_steps
        return sample_cir(
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            r0=self.r0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=rng,
            method=method,
        )

    def __repr__(self) -> str:
        return (
            f"CIRProcess(kappa={self.kappa}, theta={self.theta}, sigma={self.sigma}, r0={self.r0})"
        )

    def cumulants_at(self, order: int, t: float = 1.0) -> dict[int, float]:
        validate_order(order)
        validate_time(t)
        return self.conditional_cumulants(t=t, order=order)


# Alias for readability and standard financial naming
CoxIngersollRoss = CIRProcess
