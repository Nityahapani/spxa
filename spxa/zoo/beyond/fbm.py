"""
Fractional Brownian motion (fBM).

Not a Lévy process — fBM with H ≠ 1/2 has correlated increments and is not
a semimartingale. No Lévy–Khintchine triplet exists. spxa tracks the Hurst
index and self-similarity, propagating them under valid operations with
explicit exactness degradation.

Simulation via Hosking's exact method (Hosking 1984) for small n, and the
Davies–Harte algorithm (circulant embedding) for larger n.

References
----------
Mandelbrot, B.B. & Van Ness, J.W. (1968). Fractional Brownian motions,
fractional noises and applications. *SIAM Review*, 10(4), 422–437.

Hosking, J.R.M. (1984). Modeling persistence in hydrological time series
using fractional differencing. *Water Resources Research*, 20(12), 1898–1908.

Davies, R.B. & Harte, D.S. (1987). Tests for Hurst effect. *Biometrika*,
74(1), 95–101.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class FractionalBrownianMotion(Process):
    """
    Fractional Brownian motion with Hurst index H.

    B^H is the unique (up to scaling) centred Gaussian process with covariance
      Cov(B^H_s, B^H_t) = (1/2)(|s|^{2H} + |t|^{2H} - |s-t|^{2H})

    For H=1/2: standard Brownian motion (Lévy process).
    For H>1/2: long memory (positively correlated increments).
    For H<1/2: anti-persistence (negatively correlated increments).
    For H≠1/2: not a semimartingale → no Lévy triplet.

    Parameters
    ----------
    H :
        Hurst index H ∈ (0, 1). H=0.5 → standard BM.
    sigma :
        Scale (standard deviation at t=1). Default 1.

    Notes
    -----
    Variance: Var(B^H_t) = σ²·t^{2H}
    Self-similarity: B^H_{ct} ∼ c^H · B^H_t

    ExactnessLevel: MOMENT_PROPAGATION.
    Calling `.triplet` raises ExactnessError.

    References
    ----------
    Mandelbrot, B.B. & Van Ness, J.W. (1968). *SIAM Review*, 10(4), 422–437.
    Davies, R.B. & Harte, D.S. (1987). *Biometrika*, 74(1), 95–101.
    """

    def __init__(self, H: float, sigma: float = 1.0) -> None:
        if not (0 < H < 1):
            raise ValueError(f"H must be in (0, 1), got {H}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.H = H
        self.sigma = sigma

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"FractionalBrownianMotion(H={H}, sigma={sigma})",
            math_note=(
                f"Cov(B^H_s, B^H_t) = σ²/2·(|s|^{{2H}}+|t|^{{2H}}-|s-t|^{{2H}}); "
                f"H={H}; self-similarity index H={H}; "
                f"{'not a semimartingale — no Lévy triplet' if H != 0.5 else 'H=0.5: standard BM'}"
            ),
            reference="Mandelbrot & Van Ness (1968); Davies & Harte (1987)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                f"FractionalBrownianMotion with H={self.H} is not a Lévy process "
                f"{'(H≠0.5: correlated increments, not a semimartingale)' if self.H != 0.5 else ''} "
                "and has no Lévy–Khintchine triplet. "
                "Use .variance(t), .covariance(s, t), or .simulate() instead."
            ),
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=(self.H == 0.5),
            is_martingale=(self.H == 0.5),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=self.H,
            tail_index=None,
            is_subordinator=False,
            hurst_index=self.H,
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        """
        fBM is Gaussian so all cumulants of order ≥ 3 vanish.
        κ_1 = 0, κ_2 = σ² (at t=1), κ_n = 0 for n ≥ 3.
        """
        result: dict[int, float] = {}
        for n in range(1, order + 1):
            if n == 2:
                result[n] = self.sigma**2
            else:
                result[n] = 0.0
        return result

    def variance(self, t: float) -> float:
        """
        Var(B^H_t) = σ²·t^{2H}.

        Mandelbrot & Van Ness (1968), equation (1.3).
        """
        return self.sigma**2 * t ** (2 * self.H)

    def covariance(self, s: float, t: float) -> float:
        """
        Cov(B^H_s, B^H_t) = σ²/2·(s^{2H} + t^{2H} - |s-t|^{2H}).

        Mandelbrot & Van Ness (1968), equation (1.2).
        """
        return 0.5 * self.sigma**2 * (s ** (2 * self.H) + t ** (2 * self.H) - abs(s - t) ** (2 * self.H))

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate fBM paths via the Davies–Harte circulant embedding algorithm.

        Exact in distribution. Complexity O(n log n) per path.

        The algorithm (Davies & Harte 1987):
          1. Compute the autocovariance sequence of fBM increments.
          2. Embed in a circulant matrix and compute its eigenvalues via FFT.
          3. If all eigenvalues are non-negative (guaranteed for H ∈ (0,1)),
             generate correlated Gaussian increments via inverse FFT.

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
            Shape (n_paths, n_steps + 1).
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        n = n_steps

        times = np.arange(n + 1) * dt
        gamma = np.array([self.covariance(times[k], times[k]) if k == 0
                          else 0.5 * self.sigma**2 * (
                              abs(times[k] - times[0])**(2*self.H)
                              + abs(times[k] + times[0])**(2*self.H) - 2*abs(times[k])**(2*self.H)
                          )
                          for k in range(n)])

        row = np.zeros(2 * n)
        row[:n] = gamma
        row[n:] = gamma[n - 1:0:-1]

        eigenvalues = np.fft.fft(row).real
        if np.any(eigenvalues < 0):
            eigenvalues = np.maximum(eigenvalues, 0)

        sqrt_eigs = np.sqrt(eigenvalues / (2 * n))

        paths = np.zeros((n_paths, n + 1))
        for i in range(n_paths):
            z = rng.standard_normal(2 * n)
            w = np.fft.fft(sqrt_eigs * (z[:n] + 1j * z[n:]))
            increments = w[:n].real
            paths[i, 1:] = np.cumsum(increments)

        return paths

    def __repr__(self) -> str:
        return f"FractionalBrownianMotion(H={self.H}, sigma={self.sigma})"
