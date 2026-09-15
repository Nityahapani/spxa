"""
Gamma process.

A pure-jump subordinator (non-decreasing Lévy process) with independent
Gamma-distributed increments. Fundamental building block for subordination:
Brownian motion subordinated by a Gamma process yields the Variance Gamma model.

Lévy measure: ν(dx) = (a/x) exp(-bx) 𝟙_{x>0} dx
Triplet: (a/b - a∫_1^∞ (1/x)e^{-bx} dx·𝟙_{...}, 0, ν)  — simplified via
the known mean: b_effective = 0 when we use the mean-corrected form.

In spxa we parametrise by (a, b) matching the standard shape/rate convention
of the Lévy measure density k(x) = a·x^{-1}·e^{-bx} on (0,∞).

This gives:
  E[X_t]   = a·t / b
  Var(X_t) = a·t / b²
  Cumulant κ_n(X_t) = a·t·(n-1)! / b^n

Reference: Moran, P.A.P. (1968). An Introduction to Probability Theory.
Oxford University Press. Chapter 6.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def _gamma_levy_measure(a: float, b: float) -> DensityLevyMeasure:
    """
    Lévy measure of the Gamma process: ν(dx) = a·x^{-1}·e^{-bx} 𝟙_{x>0} dx.

    Moments: ∫_0^∞ x^n ν(dx) = a·Γ(n)/b^n  (Cont & Tankov 2004, Table 4.2).
    """
    from scipy.special import expi  # type: ignore[import-untyped]
    from scipy import integrate  # type: ignore[import-untyped]

    def density(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        pos = x > 0
        out[pos] = a * np.exp(-b * x[pos]) / x[pos]
        return out

    def tail(xv: float) -> float:
        if xv <= 0:
            return np.inf
        result, _ = integrate.quad(lambda t: a * np.exp(-b * t) / t, xv, np.inf)
        return result

    moment_fns = {
        n: (lambda n=n: a * float(gamma_fn(n)) / b**n)
        for n in range(1, 9)
    }

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=np.inf,
        _tail_fn=tail,
        _moment_fns=moment_fns,
    )


class GammaProcess(Process):
    """
    Gamma (subordinator) process with Lévy measure a·x^{-1}·e^{-bx} 𝟙_{x>0}.

    Increments: X_t - X_s ~ Gamma(a(t-s), b) in the shape/rate parametrisation,
    i.e. density f(x) ∝ x^{a(t-s)-1} e^{-bx}.

    This is a pure-jump subordinator: paths are non-decreasing,
    is_subordinator = True.

    Parameters
    ----------
    a :
        Shape rate parameter a > 0. Controls jump frequency.
    b :
        Inverse scale (rate) parameter b > 0. Controls jump size decay.

    Notes
    -----
    Cumulants (Cont & Tankov 2004, Table 4.2):
      κ_n(X_t) = a·t·(n-1)! / b^n

    The drift in the Lévy triplet is computed numerically from the Lévy measure
    to be consistent with the truncation function 𝟙_{|x|≤1}.

    References
    ----------
    Moran, P.A.P. (1968). *An Introduction to Probability Theory*.
    Oxford University Press.

    Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
    Chapman & Hall/CRC. Table 4.2.
    """

    def __init__(self, a: float, b: float) -> None:
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}")
        if b <= 0:
            raise ValueError(f"b must be positive, got {b}")
        self.a = a
        self.b = b

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"GammaProcess(a={a}, b={b})",
            math_note=f"Lévy density: k(x) = {a}·x^{{-1}}·exp(-{b}·x)·𝟙_{{x>0}}",
            reference="Moran (1968); Cont & Tankov (2004) Table 4.2",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        from scipy import integrate  # type: ignore[import-untyped]
        from scipy.special import expi  # type: ignore[import-untyped]

        nu = _gamma_levy_measure(self.a, self.b)
        # ∫_0^1 a·e^{-bx}/x dx = a · E_1(b) where E_1 is the exponential integral.
        # expi(-b) = -E_1(b) for b > 0, so E_1(b) = -expi(-b).
        b_drift = -self.a * float(expi(-self.b))
        return LevyTriplet(b=b_drift, sigma_sq=0.0, nu=nu)

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=True,
        )

    def bernstein_function(self, lam: float | np.ndarray) -> np.ndarray:
        """
        Exact Bernstein (Laplace exponent) function of the Gamma subordinator.

        φ(λ) = a · log(1 + λ/b)

        Derivation: E[e^{-λ G_1}] = (b/(b+λ))^a for G_1 ~ Gamma(a, b).
        So φ(λ) = -log E[e^{-λ G_1}] = a · log(1 + λ/b).

        Reference: Schilling, Song & Vondraček (2012), Example 3.9.

        Parameters
        ----------
        lam :
            Non-negative real argument(s).
        """
        lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
        result = self.a * np.log1p(lam_arr / self.b)
        return result if result.shape != (1,) else result[0]

    def cumulant_exact(self, n: int, t: float = 1.0) -> float:
        """
        Exact closed-form cumulant: κ_n(X_t) = a·t·(n-1)! / b^n.

        Reference: Cont & Tankov (2004), Table 4.2.

        Parameters
        ----------
        n :
            Cumulant order ≥ 1.
        t :
            Time. Default 1.
        """
        import math
        return self.a * t * math.factorial(n - 1) / self.b**n

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Exact simulation: increments Gamma(a·Δt, b) independently.

        Uses numpy's Gamma sampler with shape=a·Δt and scale=1/b.

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
        increments = rng.gamma(
            shape=self.a * dt,
            scale=1.0 / self.b,
            size=(n_paths, n_steps),
        )
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"GammaProcess(a={self.a}, b={self.b})"
