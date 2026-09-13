"""
Poisson process.

A pure-jump Lévy process with rate λ. The canonical finite-activity process.
Triplet (0, 0, λ·δ_1) under the convention of unit jumps, or more generally
(0, 0, λ·δ_c) for jump size c.

Reference: Kingman, J.F.C. (1993). *Poisson Processes*. Oxford University Press.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import LevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class _PoissonLevyMeasure(LevyMeasure):
    """
    Lévy measure of a Poisson process: ν = λ · δ_{jump_size}.

    A discrete measure (point mass at jump_size), so it has no density
    in the usual sense. Moments are computed directly.
    """

    def __init__(self, rate: float, jump_size: float) -> None:
        self.rate = rate
        self.jump_size = jump_size

    def density(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "The Poisson Lévy measure λ·δ_c is a discrete measure and has no density. "
            "Use total_mass(), tail(), or moment() instead."
        )

    def total_mass(self) -> float:
        return self.rate

    def tail(self, x: float) -> float:
        if x < self.jump_size:
            return self.rate
        return 0.0

    def moment(self, n: int) -> float:
        return self.rate * (self.jump_size**n)


class PoissonProcess(Process):
    """
    Poisson process with rate λ and jump size c.

    X_t counts the number of jumps of size c up to time t.
    N(t) ~ Poisson(λt) so X_t = c · N(t).

    Lévy–Khintchine triplet: (0, 0, λ·δ_c) for c ≤ 1,
    or (-λc, 0, λ·δ_c) for c > 1 (truncation function adjustment).

    Parameters
    ----------
    rate :
        Jump rate λ > 0. Expected number of jumps per unit time.
    jump_size :
        Size c of each jump. Default 1.

    Notes
    -----
    Cumulants (all cumulants equal for the Poisson distribution):
      κ_n(X_t) = λ · c^n · t  for all n ≥ 1

    References
    ----------
    Kingman, J.F.C. (1993). *Poisson Processes*. Oxford University Press.
    Sato, K.-I. (1999). Example 8.4.
    """

    def __init__(self, rate: float, jump_size: float = 1.0) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self.jump_size = jump_size

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"PoissonProcess(rate={rate}, jump_size={jump_size})",
            math_note=f"Triplet: (b, 0, λ·δ_c) with λ={rate}, c={jump_size}",
            reference="Kingman (1993); Sato (1999) Ex. 8.4",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        c = self.jump_size
        nu = _PoissonLevyMeasure(rate=self.rate, jump_size=c)
        b = -self.rate * c if abs(c) > 1 else 0.0
        return LevyTriplet(b=b, sigma_sq=0.0, nu=nu)

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=(self.jump_size > 0),
        )

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Exact simulation: increments are Poisson(λ·Δt) independently.

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
        jump_counts = rng.poisson(lam=self.rate * dt, size=(n_paths, n_steps))
        increments = jump_counts * self.jump_size
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"PoissonProcess(rate={self.rate}, jump_size={self.jump_size})"
