"""
Inverse Gaussian (Wald) process.

A subordinator with independent IG-distributed increments. Fundamental
in NIG construction (NIG = BM subordinated by IG process) and in
reliability and first-passage-time modelling.

Lévy density (Cont & Tankov 2004, Table 4.1):
  k(x) = (λ/(2π))^{1/2} · x^{-3/2} · exp(-λ(x-μ)²/(2μ²x))  for x > 0

This is the density of the IG(μ, λ) distribution scaled to unit time,
generalised to arbitrary t via the self-similarity of the IG process:
  X_t ~ IG(μt, λt²)

Bernstein function: φ(s) = λ/μ · (1 - sqrt(1 - 2μ²s/λ))

References
----------
Tweedie, M.C.K. (1957). Statistical properties of inverse Gaussian
distributions I. *Annals of Mathematical Statistics*, 28(2), 362–377.

Barndorff-Nielsen, O.E. & Shephard, N. (2001). Non-Gaussian OU-based
models. *Journal of the Royal Statistical Society B*, 63(2), 167–241.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapter 4, Table 4.1.
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


def _ig_levy_measure(mu: float, lam: float) -> DensityLevyMeasure:
    """
    Lévy measure of IG process.

    k(x) = sqrt(λ/(2π)) · x^{-3/2} · exp(-λ(x-μ)²/(2μ²x))  for x > 0

    Cont & Tankov (2004), Table 4.1.
    """
    prefactor = np.sqrt(lam / (2 * np.pi))

    def density(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        pos = x > 0
        xp = x[pos]
        out[pos] = prefactor * xp**(-1.5) * np.exp(-lam * (xp - mu)**2 / (2 * mu**2 * xp))
        return out

    def tail(xv: float) -> float:
        from scipy import integrate  # type: ignore[import-untyped]
        result, _ = integrate.quad(
            lambda t: prefactor * t**(-1.5) * np.exp(-lam * (t - mu)**2 / (2 * mu**2 * t)),
            xv, np.inf, limit=200
        )
        return result

    def moment_1() -> float:
        return mu

    def moment_2() -> float:
        return mu**3 / lam + mu**2

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=np.inf,
        _tail_fn=tail,
        _moment_fns={1: moment_1, 2: moment_2},
    )


class InverseGaussianProcess(Process):
    """
    Inverse Gaussian (Wald) process — a subordinator.

    X_t has independent increments distributed as IG(μt, λt²):
      X_t - X_s ~ IG(μ(t-s), λ(t-s)²)

    The IG(μ, λ) distribution has density:
      f(x; μ, λ) = sqrt(λ/(2πx³)) exp(-λ(x-μ)²/(2μ²x))  for x > 0

    Parameters
    ----------
    mu :
        Mean of X_1. μ > 0.
    lam :
        Shape parameter. λ > 0. Controls the variance: Var(X_1) = μ³/λ.

    Notes
    -----
    Cumulants (Tweedie 1957, equation 2.5):
      κ_n(X_t) = t · (2n-1)!! · μ^{2n-1} / λ^{n-1}
    where (2n-1)!! = 1·3·5·...·(2n-1).
    Explicitly:
      κ₁ = μ
      κ₂ = 3μ³/λ
      κ₃ = 15μ⁵/λ²
      κ₄ = 105μ⁷/λ³

    Bernstein function (Schilling et al. 2012):
      φ(s) = (λ/μ)(1 - sqrt(1 - 2μ²s/λ))

    is_subordinator = True (non-decreasing paths).

    References
    ----------
    Tweedie, M.C.K. (1957). *Annals of Mathematical Statistics*, 28(2),
    362–377.

    Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
    Table 4.1.

    Schilling, R., Song, R. & Vondraček, Z. (2012). *Bernstein Functions*,
    2nd ed. De Gruyter. Example 3.12.
    """

    def __init__(self, mu: float, lam: float) -> None:
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        self.mu = mu
        self.lam = lam

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"InverseGaussianProcess(mu={mu}, lam={lam})",
            math_note=(
                f"Lévy density: sqrt(λ/2π)·x^{{-3/2}}·exp(-λ(x-μ)²/(2μ²x)); "
                f"φ(s) = (λ/μ)(1-sqrt(1-2μ²s/λ)); "
                f"κ₁={mu}, κ₂={mu**3/lam:.4f}"
            ),
            reference="Tweedie (1957); Cont & Tankov (2004) Table 4.1; Schilling et al. (2012) Ex. 3.12",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        from scipy import integrate  # type: ignore[import-untyped]
        nu = _ig_levy_measure(self.mu, self.lam)
        prefactor = np.sqrt(self.lam / (2 * np.pi))
        b_drift, _ = integrate.quad(
            lambda x: prefactor * x**(-0.5) * np.exp(-self.lam * (x - self.mu)**2 / (2 * self.mu**2 * x)),
            0.0, 1.0, limit=200,
        )
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
        Exact Bernstein function: φ(s) = (λ/μ)(1 - sqrt(1 - 2μ²s/λ)).

        Schilling, Song & Vondraček (2012), Example 3.12.
        """
        s = np.atleast_1d(np.asarray(lam, dtype=float))
        inner = 1.0 - 2.0 * self.mu**2 * s / self.lam
        inner = np.maximum(inner, 0.0)
        result = (self.lam / self.mu) * (1.0 - np.sqrt(inner))
        return result if result.shape != (1,) else result[0]

    def cumulant_exact(self, n: int, t: float = 1.0) -> float:
        """
        Exact closed-form cumulant.

        κ_n(X_t) = t · (2n-1)!! · μ^{2n-1} / λ^{n-1}
        where (2n-1)!! = 1·3·5·...·(2n-1).

        Tweedie (1957), equation (2.5).
        """
        from math import prod
        double_factorial = prod(range(1, 2 * n, 2))
        return t * double_factorial * self.mu**(2 * n - 1) / self.lam**(n - 1)

    def cumulants(self, order: int) -> dict[int, float]:
        return {n: self.cumulant_exact(n) for n in range(1, order + 1)}

    def char_func_exact(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Exact characteristic function.

        φ(u; t) = exp(t · λ/μ · (1 - sqrt(1 - 2iuμ²/λ - u²μ²/λ)))
                = exp(t · φ_Bernstein(iu ... ))

        More precisely for the IG(μt, λt²) distribution:
          φ(u) = exp(λt/μ · (1 - sqrt(1 - 2iuμ²/λ + u²μ²... )))

        Using the known MGF result (Tweedie 1957):
          φ(u; t) = exp(tλ/μ · (1 - sqrt(1 - 2iuμ²/λ)))

        Reference: Barndorff-Nielsen & Shephard (2001), equation (A.2).
        """
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        inner = 1.0 - 2j * u_arr * self.mu**2 / self.lam
        exponent = t * (self.lam / self.mu) * (1.0 - np.sqrt(inner))
        result = np.exp(exponent)
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
        Exact simulation: increments X_{t_{k+1}} - X_{t_k} ~ IG(μΔt, λΔt²).

        Uses numpy's wald sampler: wald(mean=μΔt, scale=λΔt²).

        Devroye (1986), Chapter 4.
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        increments = rng.wald(
            mean=self.mu * dt,
            scale=self.lam * dt**2,
            size=(n_paths, n_steps),
        )
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"InverseGaussianProcess(mu={self.mu}, lam={self.lam})"
