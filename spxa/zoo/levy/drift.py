r"""
Deterministic drift, the degenerate Lévy process :math:`X_t = \mu t`.

Its Lévy–Khintchine triplet is :math:`(\mu, 0, 0)`. Adding it to another
process shifts the mean without touching any other cumulant, and compensates
a process into a martingale: :math:`X - \mathrm{Drift}(\kappa_1(X))`. With
:math:`\mu \ge 0` its paths are non-decreasing, so it is a subordinator, and
:math:`X \,@\, \mathrm{Drift}(c)` is the deterministic time change
:math:`X_{ct}`.

Reference: Sato (1999), Example 8.3 (the case :math:`\sigma = 0`, :math:`\nu = 0`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind
from spxa.zoo.levy.brownian import _ZeroLevyMeasure


class Drift(Process):
    r"""
    Deterministic linear drift :math:`X_t = \mu t`.

    Parameters
    ----------
    mu :
        Drift rate :math:`\mu \in \mathbb R`.

    Notes
    -----
    - Cumulants: :math:`\kappa_1 = \mu` and :math:`\kappa_n = 0` for :math:`n \ge 2`.
    - Characteristic function: :math:`E[e^{iuX_t}] = e^{i\mu t u}`.
    - It is a martingale only for :math:`\mu = 0`, and a subordinator for
      :math:`\mu \ge 0`.
    - It is self-similar with index 1, since :math:`X_{ct} = cX_t`.
    """

    def __init__(self, mu: float) -> None:
        if not np.isfinite(mu):
            raise ValueError(f"mu must be finite, got {mu}")
        self.mu = float(mu)
        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"Drift(mu={self.mu})",
            math_note=f"Triplet: (b={self.mu}, σ²=0, ν=0)",
            reference="Sato (1999) Ex. 8.3",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        return LevyTriplet(b=self.mu, sigma_sq=0.0, nu=_ZeroLevyMeasure())

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=(self.mu == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=1.0,
            tail_index=None,
            is_subordinator=(self.mu >= 0.0),
        )

    def cumulants(self, order: int) -> dict[int, float]:
        """Exact cumulants: κ_1 = μ, κ_n = 0 for n ≥ 2."""
        if order < 1:
            raise ValueError("order must be >= 1")
        return {n: (self.mu if n == 1 else 0.0) for n in range(1, order + 1)}

    def char_func_exact(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """E[e^{iuX_t}] = exp(iμtu)."""
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        result = np.exp(1j * self.mu * t * u_arr)
        return result if result.shape != (1,) else result[0]

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def bernstein_function(self, lam: float | np.ndarray) -> np.ndarray:
        """Laplace exponent φ(λ) = μλ (a subordinator only for μ ≥ 0)."""
        lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
        result = self.mu * lam_arr
        return result if result.shape != (1,) else result[0]

    _vector_sampler = True

    def sample_at(self, times: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Independent draws of X_t for each entry of ``times``: mu*t."""
        rng = rng or np.random.default_rng()
        t = np.asarray(times, dtype=float)
        return self.mu * t

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Deterministic paths μ t on the grid; ``rng`` is accepted and unused."""
        grid = np.linspace(0.0, T, n_steps + 1)
        return np.tile(self.mu * grid, (n_paths, 1))

    def __repr__(self) -> str:
        return f"Drift(mu={self.mu})"
