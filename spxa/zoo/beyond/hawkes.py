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
        if alpha < 0:
            raise ValueError(f"alpha must be nonnegative, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if alpha >= beta:
            raise ValueError(
                f"Process is supercritical: requires alpha < beta (α/β < 1), "
                f"got alpha={alpha}, beta={beta}, ratio={alpha / beta:.3f}"
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
        Asymptotic variance rate: Var(N_t)/t → λ̄/(1-α/β)² as t→∞.

        Hawkes & Oakes (1974), Theorem 1.
        """
        r = self._branching_ratio
        return self.mean_intensity / (1 - r) ** 2

    @property
    def fano_factor(self) -> float:
        """
        Asymptotic Fano factor F = Var(N_t)/E[N_t] → 1/(1-α/β)².

        Always ≥ 1 for Hawkes processes (they are always overdispersed).
        Hawkes & Oakes (1974).
        """
        r = self._branching_ratio
        return 1 / (1 - r) ** 2

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
            is_subordinator=False,
            hurst_index=None,
            notes=[
                f"Hawkes branching ratio α/β={self._branching_ratio:.4f} < 1 (subcritical).",
                "Increments are positively correlated — not a Lévy process.",
                f"Asymptotic Fano factor F={self.fano_factor:.4f} (overdispersed).",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        return self.cumulants_at(order, 1.0)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
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

    def cumulants_at(self, order: int, t: float = 1.0) -> dict[int, float]:
        """Finite-horizon cumulants with empty history (lambda_0=mu).

        Apply the Markov generator to monomials N^p lambda^q:
        A f=-beta*(lambda-mu)*d_lambda f
            +lambda*(f(N+1,lambda+alpha)-f(N,lambda)).
        Polynomial moments close by total degree; solve by matrix exponential.
        References: Hawkes (1971); polynomial generator/Dynkin moment identity.
        """
        from math import comb

        from scipy.linalg import expm

        from spxa.ops.product import raw_moments_to_cumulants

        if not isinstance(order, int) or order < 1 or order > 8 or not np.isfinite(t) or t < 0:
            raise ValueError("Require order 1..8 and finite nonnegative time")
        basis = [
            (p, q) for degree in range(order + 1) for p in range(degree + 1) for q in [degree - p]
        ]
        index = {key: i for i, key in enumerate(basis)}
        matrix = np.zeros((len(basis), len(basis)))
        for row, (p, q) in enumerate(basis):
            matrix[row, row] -= self.beta * q
            if q:
                matrix[row, index[p, q - 1]] += self.beta * self.mu * q
            for i in range(p + 1):
                for j in range(q + 1):
                    if (i, j) != (p, q):
                        matrix[row, index[i, j + 1]] += (
                            comb(p, i) * comb(q, j) * self.alpha ** (q - j)
                        )
        initial = np.array([self.mu**q if p == 0 else 0.0 for p, q in basis])
        moments = expm(t * matrix) @ initial
        return raw_moments_to_cumulants({n: float(moments[index[n, 0]]) for n in range(order + 1)})
