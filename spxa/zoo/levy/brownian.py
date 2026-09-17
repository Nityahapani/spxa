"""
Standard Brownian motion (Wiener process).

The canonical Lévy process. Triplet (0, σ², 0) — pure Gaussian diffusion,
no drift, no jumps.

Reference: Wiener, N. (1923). Differential space. *Journal of Mathematics
and Physics*, 2, 131–174.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class _ZeroLevyMeasure(DensityLevyMeasure):
    """The zero measure ν ≡ 0. Lévy measure of Brownian motion."""

    def __init__(self) -> None:
        def _zero_density(x: np.ndarray) -> np.ndarray:
            return np.zeros_like(x, dtype=float)

        super().__init__(
            _density_fn=_zero_density,
            _total_mass=0.0,
            _tail_fn=lambda x: 0.0,
            _moment_fns={n: (lambda: 0.0) for n in range(1, 9)},
        )


class BrownianMotion(Process):
    """
    Standard Brownian motion (Wiener process) with drift and diffusion.

    The process X_t = μt + σW_t where W is a standard Brownian motion.

    Lévy–Khintchine triplet: (μ, σ², 0).
    No jump component — ν is the zero measure.

    Parameters
    ----------
    mu :
        Drift coefficient μ. Default 0.
    sigma :
        Diffusion coefficient σ > 0. Default 1.

    Notes
    -----
    Cumulants (Sato 1999, Example 8.3):
      κ_1 = μ  (mean per unit time)
      κ_2 = σ² (variance per unit time)
      κ_n = 0  for n ≥ 3

    References
    ----------
    Wiener, N. (1923). Differential space. *Journal of Mathematics and Physics*,
    2, 131–174.

    Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*.
    Cambridge University Press. Example 8.3.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.mu = mu
        self.sigma = sigma
        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"BrownianMotion(mu={mu}, sigma={sigma})",
            math_note=f"Triplet: (b={mu}, σ²={sigma**2}, ν=0)",
            reference="Wiener (1923); Sato (1999) Ex. 8.3",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        return LevyTriplet(b=self.mu, sigma_sq=self.sigma**2, nu=_ZeroLevyMeasure())

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=(self.mu == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=0.5,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
        )

    def char_func_exact(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Exact characteristic function: E[e^{iuX_t}] = exp(iμtu - σ²tu²/2).

        Sato (1999), Example 8.3.
        """
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        result = np.exp(1j * self.mu * t * u_arr - 0.5 * self.sigma**2 * t * u_arr**2)
        return result if result.shape != (1,) else result[0]

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Exact simulation via Gaussian increments.

        Each increment X_{t_{k+1}} - X_{t_k} ~ N(μ·Δt, σ²·Δt) independently.

        Parameters
        ----------
        n_steps :
            Number of time steps.
        n_paths :
            Number of independent paths.
        T :
            Terminal time.
        rng :
            Random number generator.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). paths[:, 0] == 0.
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        increments = rng.normal(
            loc=self.mu * dt,
            scale=self.sigma * np.sqrt(dt),
            size=(n_paths, n_steps),
        )
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"BrownianMotion(mu={self.mu}, sigma={self.sigma})"
