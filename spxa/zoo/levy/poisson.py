"""
Poisson process.

A pure-jump Lévy process with rate λ. The canonical finite-activity process.
Triplet (λ, 0, λ·δ_1) under the convention of unit jumps, or more generally
(λc*1_{|c|<=1}, 0, λ·δ_c) for jump size c.

Reference: Kingman, J.F.C. (1993). *Poisson Processes*. Oxford University Press.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

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

    def density(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError(
            "The Poisson Lévy measure λ·δ_c is a discrete measure and has no density. "
            "Use total_mass(), tail(), or moment() instead."
        )

    def total_mass(self) -> float:
        return self.rate if self.jump_size != 0 else 0.0

    def tail(self, x: float) -> float:
        if self.jump_size != 0 and x < self.jump_size:
            return self.rate
        return 0.0

    def moment(self, n: int) -> float:
        return self.rate * (self.jump_size**n)

    def integrate(
        self, f: Callable[[float], float], lower: float = -np.inf, upper: float = np.inf
    ) -> float:
        if lower > upper:
            raise ValueError("Integration bounds must be ordered")
        return (
            float(self.rate * f(self.jump_size))
            if self.jump_size != 0 and lower <= self.jump_size <= upper
            else 0.0
        )


class PoissonProcess(Process):
    """
    Poisson process with rate λ and jump size c.

    X_t counts the number of jumps of size c up to time t.
    N(t) ~ Poisson(λt) so X_t = c · N(t).

    Lévy–Khintchine triplet: (λc*1_{|c|<=1}, 0, λ·δ_c).

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
        b = self.rate * c if abs(c) <= 1 else 0.0
        return LevyTriplet(b=b, sigma_sq=0.0, nu=nu)

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=(self.jump_size == 0),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=(self.jump_size >= 0),
        )

    def char_func_exact(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """φ(u;t) = exp(λt(e^{icu}-1)). Characteristic function of a Poisson process."""
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        result = np.exp(self.rate * t * (np.exp(1j * self.jump_size * u_arr) - 1))
        return result if result.shape != (1,) else result[0]

    def char_func(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def cumulants(self, order: int) -> dict[int, float]:
        """κ_n(X_1)=rate*jump_size**n; equal cumulants for unit jumps."""
        return {n: self.rate * (self.jump_size**n) for n in range(1, order + 1)}

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
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

    def characteristic_exponent(self, u: complex | float | np.ndarray) -> np.ndarray:
        """Unit-time log CF on the continuous branch at zero."""
        z = np.asarray(u, dtype=complex)
        return self.rate * np.expm1(1j * self.jump_size * z)
