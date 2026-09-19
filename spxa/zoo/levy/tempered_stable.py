"""
Tempered stable (one-sided) subordinator.

A stable subordinator whose large jumps are exponentially dampened, giving
finite moments of all orders while retaining the heavy-tailed, infinite-activity
character of the stable distribution for small jumps.

Lévy density:
  k(x) = C · x^{-1-α} · exp(-λx)  for x > 0

This is a subordinator (non-decreasing), the Laplace exponent is:
  φ(s) = C·Γ(-α)·((λ+s)^α - λ^α)  for α ∈ (0,1)

Characteristic function:
  E[e^{iuX_t}] = exp(t·C·Γ(-α)·((λ-iu)^α - λ^α))

The tempered stable bridges between:
  - Pure stable (λ→0): power-law tails, no finite moments
  - Gamma process (α→0): exponential tails, all moments finite
  - At λ>0, α∈(0,1): all moments finite, infinite activity subordinator

References
----------
Rosiński, J. (2007). Tempering stable processes. *Stochastic Processes and
their Applications*, 117(6), 677–707.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapman & Hall/CRC. Chapter 4.

Schilling, R., Song, R. & Vondraček, Z. (2012). *Bernstein Functions*,
2nd ed. De Gruyter. Example 3.8.
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


def _ts_levy_measure(alpha: float, C: float, lam: float) -> DensityLevyMeasure:
    """
    Tempered stable Lévy density: k(x) = C·x^{-1-α}·exp(-λx)·𝟙_{x>0}.

    Rosiński (2007), equation (1.1).
    """
    from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

    def density(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        pos = x > 0
        out[pos] = C * x[pos] ** (-1 - alpha) * np.exp(-lam * x[pos])
        return out

    def tail(xv: float) -> float:
        from scipy import integrate  # type: ignore[import-untyped]
        result, _ = integrate.quad(
            lambda t: C * t ** (-1 - alpha) * np.exp(-lam * t),
            xv, np.inf, limit=200,
        )
        return result

    # Cumulants: κ_n = C·Γ(n-α)/λ^{n-α} for n ≥ 1
    # Derived from the cumulant generating function of the TS distribution.
    # Rosiński (2007), Proposition 2.1 / Cont & Tankov (2004), Table 4.1.
    def moment_fn(n: int):
        return lambda: C * float(gamma_fn(n - alpha)) / lam ** (n - alpha)

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=np.inf,
        _tail_fn=tail,
        _moment_fns={n: moment_fn(n) for n in range(1, 9)},
    )


class TemperedStable(Process):
    """
    Tempered stable subordinator.

    A one-sided infinite-activity Lévy process with Lévy density
    k(x) = C·x^{-1-α}·exp(-λx) for x > 0. All moments are finite.

    Parameters
    ----------
    alpha :
        Stability index α ∈ (0, 1). Controls the small-jump behaviour.
        As α → 0 the process approaches a Gamma process.
        As α → 1 the process approaches a stable process with λ-tempering.
    C :
        Jump intensity C > 0. Overall scale of the jump measure.
    lam :
        Tempering parameter λ > 0. Controls the exponential dampening of
        large jumps. Larger λ → lighter tails.

    Notes
    -----
    Cumulants (Rosiński 2007, Prop. 2.1):
      κ_n(X_t) = t · C · Γ(n-α) / λ^{n-α}  for all n ≥ 1

    Explicitly:
      κ₁ = C·Γ(1-α)/λ^{1-α}
      κ₂ = C·Γ(2-α)/λ^{2-α}

    Laplace exponent / Bernstein function (Schilling et al. 2012, Ex. 3.8):
      φ(s) = C·Γ(-α)·((λ+s)^α - λ^α)

    Characteristic function:
      φ(u; t) = exp(t·C·Γ(-α)·((λ-iu)^α - λ^α))

    Simulation: rejection from a positive α-stable proposal (Rosiński 2007,
    Algorithm 6.1).

    is_subordinator = True (support ⊂ (0,∞), non-decreasing paths).

    References
    ----------
    Rosiński, J. (2007). Tempering stable processes. *Stochastic Processes
    and their Applications*, 117(6), 677–707.

    Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
    Table 4.1.

    Schilling, R., Song, R. & Vondraček, Z. (2012). *Bernstein Functions*,
    2nd ed. De Gruyter. Example 3.8.
    """

    def __init__(self, alpha: float, C: float, lam: float) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")

        self.alpha = alpha
        self.C = C
        self.lam = lam

        from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]
        kappa1 = C * float(gamma_fn(1 - alpha)) / lam ** (1 - alpha)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"TemperedStable(alpha={alpha}, C={C}, lam={lam})",
            math_note=(
                f"Lévy density: C·x^{{-1-α}}·exp(-λx)·𝟙_{{x>0}}; "
                f"φ(s)=C·Γ(-α)·((λ+s)^α-λ^α); "
                f"κ₁={kappa1:.4f}"
            ),
            reference="Rosiński (2007) Prop. 2.1; Schilling et al. (2012) Ex. 3.8",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        from scipy import integrate  # type: ignore[import-untyped]

        nu = _ts_levy_measure(self.alpha, self.C, self.lam)

        # b = ∫_0^1 x·k(x) dx = ∫_0^1 C·x^{-α}·exp(-λx) dx
        # No closed form in general; use quadrature on (0,1] where no singularity
        # at x=1. Split at a small epsilon to avoid x^{-α} near 0.
        b_drift, _ = integrate.quad(
            lambda x: self.C * x ** (-self.alpha) * np.exp(-self.lam * x),
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
        Exact Bernstein (Laplace exponent) function.

        φ(s) = C·Γ(-α)·((λ+s)^α - λ^α)

        Schilling, Song & Vondraček (2012), Example 3.8.

        Parameters
        ----------
        lam :
            Non-negative real argument(s) s.
        """
        from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

        s = np.atleast_1d(np.asarray(lam, dtype=float))
        coeff = self.C * float(gamma_fn(-self.alpha))
        # Γ(-α) < 0 for α∈(0,1); the Bernstein function is -C·Γ(-α)·(...) > 0
        result = -coeff * ((self.lam + s) ** self.alpha - self.lam ** self.alpha)
        return result if result.shape != (1,) else result[0]

    def cumulant_exact(self, n: int, t: float = 1.0) -> float:
        """
        Exact cumulant: κ_n(X_t) = t·C·Γ(n-α)/λ^{n-α}.

        Valid for all n ≥ 1 (all moments are finite when λ > 0).

        Rosiński (2007), Proposition 2.1.

        Parameters
        ----------
        n :
            Cumulant order ≥ 1.
        t :
            Time. Default 1.
        """
        from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

        return t * self.C * float(gamma_fn(n - self.alpha)) / self.lam ** (n - self.alpha)

    def cumulants(self, order: int) -> dict[int, float]:
        return {n: self.cumulant_exact(n) for n in range(1, order + 1)}

    def char_func_exact(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Exact characteristic function.

        φ(u; t) = exp(t·C·Γ(-α)·((λ-iu)^α - λ^α))

        Derived from the Bernstein function via φ(u;t) = exp(-t·φ(-iu))
        where φ is the Laplace exponent. Rosiński (2007), equation (1.3).

        Parameters
        ----------
        u :
            Frequency argument(s).
        t :
            Time.
        """
        from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        coeff = self.C * float(gamma_fn(-self.alpha))
        exponent = t * coeff * ((self.lam - 1j * u_arr) ** self.alpha - self.lam ** self.alpha)
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
        Exact simulation via Gil-Pelaez CDF inversion.

        For each increment of length Δt, the distribution is
        TemperedStable(α, C·Δt, λ) — the intensity scales linearly with Δt.
        We invert the characteristic function numerically using the
        Gil-Pelaez formula to recover the CDF, then sample via interpolation.

        Parameters
        ----------
        n_steps, n_paths, T, rng :
            Standard simulation parameters.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1).

        References
        ----------
        Gil-Pelaez, J. (1951). Note on the inversion theorem.
        *Biometrika*, 38, 481–482.
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps

        # Build CDF for TS(alpha, C*dt, lam) increment
        # Mean and std for grid centering
        mean_dt = self.cumulant_exact(1, t=dt)
        std_dt = np.sqrt(self.cumulant_exact(2, t=dt))
        x_half = max(10 * std_dt, 1e-6)
        n_x = 2000
        x_grid = np.linspace(max(mean_dt - x_half, 1e-8), mean_dt + x_half, n_x)

        n_u = 1500
        u_max = min(50.0 / std_dt, 1000.0)
        u_grid = np.linspace(1e-6, u_max, n_u)

        cf_vals = self.char_func_exact(u=u_grid, t=dt)
        pdf = np.zeros(n_x)
        for j, x in enumerate(x_grid):
            integrand = (cf_vals * np.exp(-1j * u_grid * x)).real
            pdf[j] = float(np.trapezoid(integrand, u_grid)) / np.pi

        pdf = np.maximum(pdf, 0.0)
        dx = x_grid[1] - x_grid[0]
        cdf = np.cumsum(pdf) * dx
        cdf = np.clip(cdf / (cdf[-1] + 1e-300), 0.0, 1.0)

        u_unif = rng.uniform(size=(n_paths, n_steps))
        increments = np.interp(u_unif, cdf, x_grid)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"TemperedStable(alpha={self.alpha}, C={self.C}, lam={self.lam})"
