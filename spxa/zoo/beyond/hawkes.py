"""
Hawkes self-exciting point process.

Not a Lévy process — the conditional intensity depends on history,
so increments are neither independent nor stationary. spxa tracks the
mean intensity, variance, and Fano factor analytically; simulation uses
Ogata's thinning algorithm.

References
----------
Hawkes, A.G. (1971). Spectra of some self-exciting and mutually exciting
point processes. *Biometrika*, 58(1), 83–90.

Hawkes, A.G. & Oakes, D. (1974). A cluster process representation of a
self-exciting process. *Journal of Applied Probability*, 11(3), 493–503.

Ogata, Y. (1981). On Lewis' simulation method for point processes.
*IEEE Transactions on Information Theory*, 27(1), 23–31.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class HawkesProcess(Process):
    """
    Hawkes self-exciting point process with exponential kernel.

    The conditional intensity is:
      λ(t) = μ + α ∑_{t_i < t} exp(-β(t - t_i))

    where μ > 0 is the background rate, α > 0 is the excitation amplitude,
    and β > 0 is the decay rate. The process is subcritical when α/β < 1,
    ensuring stationarity.

    Parameters
    ----------
    mu :
        Background intensity μ > 0.
    alpha :
        Excitation amplitude α > 0. Controls how much each event
        increases the intensity.
    beta :
        Exponential decay rate β > 0. Controls how quickly excitement fades.

    Notes
    -----
    Subcriticality: requires α/β < 1 (i.e. α < β). If violated,
    the process is supercritical and the intensity diverges.

    Stationary mean intensity (Hawkes 1971):
      λ̄ = μ / (1 - α/β)

    Asymptotic variance (Hawkes & Oakes 1974):
      Var(N_t) / t → λ̄ · (1 + α/β) / (1 - α/β)²  as t → ∞

    ExactnessLevel: MOMENT_PROPAGATION.

    References
    ----------
    Hawkes, A.G. (1971). *Biometrika*, 58(1), 83–90.
    Hawkes, A.G. & Oakes, D. (1974). *Journal of Applied Probability*, 11(3), 493–503.
    Ogata, Y. (1981). *IEEE Transactions on Information Theory*, 27(1), 23–31.
    """

    def __init__(self, mu: float, alpha: float, beta: float) -> None:
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if alpha >= beta:
            raise ValueError(
                f"Process is supercritical: requires alpha < beta (α/β < 1), "
                f"got alpha={alpha}, beta={beta}, ratio={alpha/beta:.3f}"
            )

        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self._branching_ratio = alpha / beta

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"HawkesProcess(mu={mu}, alpha={alpha}, beta={beta})",
            math_note=(
                f"λ(t) = μ + α·∑_{{t_i<t}} exp(-β(t-t_i)); "
                f"branching ratio α/β={self._branching_ratio:.3f}; "
                f"λ̄ = μ/(1-α/β) = {self.mean_intensity:.4f}"
            ),
            reference="Hawkes (1971); Hawkes & Oakes (1974); Ogata (1981)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    @property
    def mean_intensity(self) -> float:
        """
        Stationary mean intensity λ̄ = μ / (1 - α/β).

        Hawkes (1971), equation (3.5).
        """
        return self.mu / (1 - self._branching_ratio)

    @property
    def asymptotic_variance_rate(self) -> float:
        """
        Asymptotic variance rate: Var(N_t)/t → λ̄·(1+α/β)/(1-α/β)² as t→∞.

        Hawkes & Oakes (1974), Theorem 1.
        """
        r = self._branching_ratio
        return self.mean_intensity * (1 + r) / (1 - r) ** 2

    @property
    def fano_factor(self) -> float:
        """
        Asymptotic Fano factor F = Var(N_t)/E[N_t] → (1+α/β)/(1-α/β)².

        Always ≥ 1 for Hawkes processes (they are always overdispersed).
        Hawkes & Oakes (1974).
        """
        r = self._branching_ratio
        return (1 + r) / (1 - r) ** 2

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "HawkesProcess has history-dependent intensity — increments are "
                "neither independent nor stationary. No Lévy–Khintchine triplet exists. "
                "Use .mean_intensity, .asymptotic_variance_rate, .fano_factor, or .simulate()."
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
            is_subordinator=True,
            hurst_index=None,
            notes=[
                f"Hawkes process: branching ratio α/β={self._branching_ratio:.4f} < 1 (subcritical).",
                "Increments are positively correlated — not a Lévy process.",
                f"Fano factor F={(1+self._branching_ratio)/(1-self._branching_ratio)**2:.4f} > 1 (overdispersed).",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        """
        Return the asymptotic cumulants of N_t/t as t→∞.
        Only the first two are tracked analytically.
        """
        result: dict[int, float] = {}
        if order >= 1:
            result[1] = self.mean_intensity
        if order >= 2:
            result[2] = self.asymptotic_variance_rate
        for n in range(3, order + 1):
            result[n] = float("nan")
        return result

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate Hawkes process paths via Ogata's thinning algorithm.

        The thinning algorithm (Ogata 1981):
          1. Propose next event time from a Poisson process with rate λ*(t)
             = current upper bound on λ(t).
          2. Accept with probability λ(t_proposed) / λ*(t_proposed).
          3. Update the intensity and repeat until T is reached.

        Returns the counting process N_t on a grid of n_steps+1 points.

        Parameters
        ----------
        n_steps :
            Number of output time grid points minus 1.
        n_paths :
            Number of independent realisations.
        T :
            Terminal time.
        rng :
            Random number generator.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). Integer-valued counting process.
        """
        rng = rng or np.random.default_rng()
        time_grid = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_paths, n_steps + 1))

        for path_idx in range(n_paths):
            event_times: list[float] = []
            t = 0.0
            lambda_star = self.mu

            while t < T:
                dt = rng.exponential(1.0 / lambda_star)
                t_proposed = t + dt

                if t_proposed > T:
                    break

                lambda_t = self.mu + self.alpha * sum(
                    np.exp(-self.beta * (t_proposed - s)) for s in event_times
                )

                if rng.uniform() < lambda_t / lambda_star:
                    event_times.append(t_proposed)
                    lambda_star = lambda_t + self.alpha

                t = t_proposed
                lambda_star = max(lambda_star, self.mu)

            counts = np.searchsorted(np.array(event_times), time_grid, side="right")
            paths[path_idx] = counts.astype(float)

        return paths

    def __repr__(self) -> str:
        return f"HawkesProcess(mu={self.mu}, alpha={self.alpha}, beta={self.beta})"
