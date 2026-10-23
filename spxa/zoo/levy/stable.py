"""
α-stable Lévy process.

The only class of Lévy processes that is closed under addition and scaling
(stable distributions). Generalises the Gaussian (α=2) and Cauchy (α=1).
Power-law tails: moments of order ≥ α do not exist.

Lévy measure (Samorodnitsky & Taqqu 1994, Property 1.2.15):
  ν(dx) = (c₊ x^{-α-1} 𝟙_{x>0} + c₋ |x|^{-α-1} 𝟙_{x<0}) dx

where c₊, c₋ ≥ 0, not both zero, and α ∈ (0, 2).

The characteristic exponent (Zolotarev parametrisation, type S):
  ψ(u) = -σ^α |u|^α (1 - iβ sign(u) tan(πα/2)) + iμu   α ≠ 1
  ψ(u) = -σ|u|(1 + iβ (2/π) sign(u) log|u|) + iμu        α = 1

Parameters follow the S₁ convention of Nolan (2020).

References
----------
Samorodnitsky, G. & Taqqu, M.S. (1994). *Stable Non-Gaussian Random Processes*.
Chapman & Hall.

Nolan, J.P. (2020). *Univariate Stable Distributions*. Springer.

Chambers, J.M., Mallows, C.L. & Stuck, B.W. (1976). A method for simulating
stable random variables. *Journal of the American Statistical Association*,
71(354), 340–344.
"""

from __future__ import annotations

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def _stable_levy_measure(alpha: float, beta: float, sigma: float) -> DensityLevyMeasure:
    """
    Lévy measure of α-stable process.

    ν(dx) = (c₊ x^{-α-1} 𝟙_{x>0} + c₋ |x|^{-α-1} 𝟙_{x<0}) dx

    where c₊ = σ^α · (1+β) · Γ(α+1)sin(πα/2)/π
          c₋ = σ^α · (1-β) · Γ(α+1)sin(πα/2)/π

    Samorodnitsky & Taqqu (1994), Property 1.2.15.
    """
    from scipy.special import gamma as gamma_fn

    factor = sigma**alpha * gamma_fn(alpha + 1) * np.sin(np.pi * alpha / 2) / np.pi
    c_plus = factor * (1 + beta)
    c_minus = factor * (1 - beta)

    def density(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        pos = x > 0
        neg = x < 0
        if c_plus > 0:
            out[pos] = c_plus * x[pos] ** (-alpha - 1)
        if c_minus > 0:
            out[neg] = c_minus * np.abs(x[neg]) ** (-alpha - 1)
        return out

    def tail(xv: float) -> float:
        if c_plus <= 0 or xv <= 0:
            return 0.0
        return float(c_plus * xv ** (-alpha) / alpha)

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=np.inf,
        _tail_fn=tail,
        _moment_fns={},
        _tail_moment_test=lambda n: n < alpha,
        _minimum_moment_order=2 if alpha >= 1 else 1,
    )


class AlphaStable(Process):
    """
    α-stable Lévy process (Lévy–Khintchine parametrisation, S₁ convention).

    The only non-trivial class of Lévy processes stable under addition and
    scaling. For α < 2, moments of order ≥ α do not exist; for α = 2 this
    reduces to Brownian motion.

    Parameters
    ----------
    alpha :
        Stability index α ∈ (0, 2]. α=2 → Gaussian, α=1 → Cauchy.
        Controls tail heaviness: smaller α → heavier tails.
    beta :
        Skewness parameter β ∈ [-1, 1].
        β=0 → symmetric, β=1 → maximally right-skewed, β=-1 → maximally left-skewed.
    sigma :
        Scale parameter σ > 0.
    mu :
        Location (shift) parameter μ ∈ ℝ. Default 0.

    Notes
    -----
    Moments: E[|X_1|^p] < ∞ iff p < α.
    So for α ≤ 1, the mean does not exist; for α ≤ 2, the variance does not
    exist (except at α=2 exactly).

    Characteristic exponent (Nolan 2020, Definition 1.6):
      ψ(u) = -σ^α|u|^α(1 - iβ sign(u) tan(πα/2)) + iμu   (α ≠ 1)
      ψ(u) = -σ|u|(1 + iβ(2/π)sign(u)log|u|) + iμu        (α = 1)

    Simulation via Chambers–Mallows–Stuck (1976) algorithm — exact in
    distribution for unit-time increments.

    References
    ----------
    Samorodnitsky, G. & Taqqu, M.S. (1994). *Stable Non-Gaussian Random
    Processes*. Chapman & Hall. Property 1.2.15.

    Nolan, J.P. (2020). *Univariate Stable Distributions*. Springer.
    Definition 1.6.

    Chambers, J.M., Mallows, C.L. & Stuck, B.W. (1976). A method for
    simulating stable random variables. *Journal of the American Statistical
    Association*, 71(354), 340–344.
    """

    def __init__(
        self,
        alpha: float,
        beta: float = 0.0,
        sigma: float = 1.0,
        mu: float = 0.0,
    ) -> None:
        if not (0 < alpha <= 2):
            raise ValueError(f"alpha must be in (0, 2], got {alpha}")
        if not (-1 <= beta <= 1):
            raise ValueError(f"beta must be in [-1, 1], got {beta}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.mu = mu

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"AlphaStable(alpha={alpha}, beta={beta}, sigma={sigma}, mu={mu})",
            math_note=(
                f"ψ(u) = -σ^α|u|^α(1-iβ·sign(u)·tan(πα/2)) + iμu; "
                f"moments of order < α={alpha} exist"
            ),
            reference="Samorodnitsky & Taqqu (1994); Nolan (2020) Def. 1.6; CMS (1976)",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        if self.alpha == 2.0:
            from spxa.zoo.levy.brownian import _ZeroLevyMeasure

            return LevyTriplet(b=self.mu, sigma_sq=2 * self.sigma**2, nu=_ZeroLevyMeasure())

        nu = _stable_levy_measure(self.alpha, self.beta, self.sigma)
        from scipy.special import gamma as gamma_fn

        difference = (
            2
            * self.sigma**self.alpha
            * gamma_fn(self.alpha + 1)
            * np.sin(np.pi * self.alpha / 2)
            * self.beta
            / np.pi
        )
        correction = (
            difference / (1 - self.alpha) if self.alpha != 1 else difference * (np.euler_gamma - 1)
        )
        return LevyTriplet(b=self.mu + correction, sigma_sq=0.0, nu=nu)

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=False,
            has_finite_variance=(self.alpha == 2.0),
            has_finite_mean=(self.alpha > 1.0),
            self_similarity_index=1.0 / self.alpha
            if self.mu == 0 and (self.alpha != 1 or self.beta == 0)
            else None,
            tail_index=self.alpha if self.alpha < 2 else None,
            is_subordinator=False,
        )

    def char_func_exact(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """S1 characteristic function; complex arguments require finite exponential moments."""
        if not np.isfinite(t) or t < 0:
            raise ValueError("Time must be finite and nonnegative")
        return np.exp(t * self.characteristic_exponent(u))

    def char_func(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Exact simulation via Chambers–Mallows–Stuck (1976) algorithm.

        Each increment X_{t_{k+1}} - X_{t_k} is an independent α-stable
        random variable scaled to the interval length Δt^{1/α}.

        Algorithm (CMS 1976, equations 2.1–2.3):
          1. U ~ Uniform(-π/2, π/2)
          2. E ~ Exponential(1)
          3. For α ≠ 1:
             X = S · sin(α(U+B)) / cos(U)^{1/α} · (cos(U-α(U+B))/E)^{(1-α)/α}
          4. Scale by σ·Δt^{1/α} and shift by μ·Δt

        Parameters
        ----------
        n_steps, n_paths, T, rng :
            Standard simulation parameters.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1).
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        size = (n_paths, n_steps)

        U = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
        E = rng.exponential(1.0, size=size)

        if abs(self.alpha - 1.0) > 1e-12:
            B = np.arctan(self.beta * np.tan(np.pi * self.alpha / 2)) / self.alpha
            S = (1 + (self.beta * np.tan(np.pi * self.alpha / 2)) ** 2) ** (1 / (2 * self.alpha))
            increments = (
                S
                * np.sin(self.alpha * (U + B))
                / np.cos(U) ** (1 / self.alpha)
                * (np.cos(U - self.alpha * (U + B)) / E) ** ((1 - self.alpha) / self.alpha)
            )
        else:
            increments = (2 / np.pi) * (
                (np.pi / 2 + self.beta * U) * np.tan(U)
                - self.beta * np.log(np.pi / 2 * E * np.cos(U) / (np.pi / 2 + self.beta * U))
            )

        increments = self.sigma * (dt ** (1 / self.alpha)) * increments + self.mu * dt
        if self.alpha == 1:
            # S1 scaling adds -2*beta*sigma*dt*log(dt)/pi to the location.
            increments += 2 * self.beta * self.sigma * dt * np.log(self.sigma * dt) / np.pi

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return (
            f"AlphaStable(alpha={self.alpha}, beta={self.beta}, sigma={self.sigma}, mu={self.mu})"
        )

    def characteristic_exponent(self, u: complex | float | np.ndarray) -> np.ndarray:
        """S1 continuous log CF (Nolan, 2020, Definition 1.6)."""
        z = np.asarray(u, dtype=complex)
        if self.alpha == 2:
            return np.asarray(1j * self.mu * z - self.sigma**2 * z * z)
        if np.any(z.imag != 0):
            raise ValueError(
                "General stable laws do not have exponential moments; use real frequencies"
            )
        absolute = np.abs(z.real)
        if self.alpha == 1:
            logarithm = np.log(np.where(absolute == 0, 1.0, absolute))
            return np.asarray(
                -self.sigma
                * absolute
                * (1 + 1j * self.beta * 2 / np.pi * np.sign(z.real) * logarithm)
                + 1j * self.mu * z
            )
        return np.asarray(
            -(self.sigma**self.alpha)
            * absolute**self.alpha
            * (1 - 1j * self.beta * np.sign(z.real) * np.tan(np.pi * self.alpha / 2))
            + 1j * self.mu * z
        )

    def cumulants(self, order: int) -> dict[int, float]:
        """Return existing cumulants; infinite stable moments are not quadrature artifacts."""
        if order < 1:
            raise ValueError("Cumulant order must be positive")
        if self.alpha == 2:
            return {
                n: self.mu if n == 1 else 2 * self.sigma**2 if n == 2 else 0.0
                for n in range(1, order + 1)
            }
        if order >= self.alpha:
            raise ValueError("Requested stable cumulant does not exist (tail moment criterion)")
        return {1: self.mu}
