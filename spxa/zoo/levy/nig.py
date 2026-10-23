"""
Normal Inverse Gaussian (NIG) process.

A Brownian motion with drift subordinated by an Inverse Gaussian process.
Introduced by Barndorff-Nielsen (1997) for modelling turbulence and financial
log-returns. Has semi-heavy tails — exponential decay, not power-law.

Lévy density:
  k(x) = (αδ/π) · (K_1(α·|x|) / |x|) · exp(βx)

where K_1 is the modified Bessel function of the second kind, order 1.

Parameters: α > |β| > 0, δ > 0, μ ∈ ℝ.

References
----------
Barndorff-Nielsen, O.E. (1997). Normal inverse Gaussian distributions and
stochastic volatility modelling. *Scandinavian Journal of Statistics*, 24,
1–13.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapter 4.
"""

from __future__ import annotations

import numpy as np
from scipy.special import k1e as bessel_k1e

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def _nig_levy_measure(alpha: float, beta: float, delta: float) -> DensityLevyMeasure:
    """
    NIG Lévy density: k(x) = αδ/π · K_1(α|x|) / |x| · exp(βx).

    Barndorff-Nielsen (1997), equation (2.4).
    """
    prefactor = alpha * delta / np.pi

    def density(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        nz = x != 0
        xv = x[nz]
        arg = alpha * np.abs(xv)
        out[nz] = prefactor * bessel_k1e(arg) / np.abs(xv) * np.exp(beta * xv - arg)
        return out

    gamma_param = np.sqrt(alpha**2 - beta**2)

    def moment_1() -> float:
        return float(delta * beta / gamma_param)

    def moment_2() -> float:
        return float(delta * alpha**2 / gamma_param**3)

    def moment_3() -> float:
        return float(3.0 * delta * alpha**2 * beta / gamma_param**5)

    def moment_4() -> float:
        return float(3.0 * delta * alpha**2 * (alpha**2 + 4.0 * beta**2) / gamma_param**7)

    from scipy import integrate

    def tail(xv: float) -> float:
        result, _ = integrate.quad(
            lambda t: float(density(np.array([t]))[0]), xv, np.inf, limit=200
        )
        return float(result)

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=np.inf,
        _tail_fn=tail,
        _moment_fns={1: moment_1, 2: moment_2, 3: moment_3, 4: moment_4},
        _minimum_moment_order=2,
    )


class NIG(Process):
    """
    Normal Inverse Gaussian (NIG) process.

    Parameters
    ----------
    alpha :
        Tail heaviness. α > 0, must satisfy α > |β|.
        Larger α → lighter tails.
    beta :
        Asymmetry (skewness). β ∈ (-α, α).
        β < 0 → left skew, β > 0 → right skew, β = 0 → symmetric.
    delta :
        Scale parameter δ > 0.
    mu :
        Location parameter μ ∈ ℝ. Shifts the mean. Default 0.

    Notes
    -----
    Let γ = sqrt(α² - β²). Cumulants per unit time (Barndorff-Nielsen 1997):
      κ_1 = μ + δβ/γ
      κ_2 = δα²/γ³
      κ_3 = 3δα²β/γ⁵
      κ_4 = 3δα²(α²+4β²)/γ⁷

    Characteristic function (Barndorff-Nielsen 1997, eq. 2.2):
      φ(u; t) = exp(t·(iμu + δ(γ - sqrt(α² - (β+iu)²))))

    References
    ----------
    Barndorff-Nielsen, O.E. (1997). Normal inverse Gaussian distributions
    and stochastic volatility modelling. *Scandinavian Journal of Statistics*,
    24, 1–13.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        delta: float,
        mu: float = 0.0,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        if abs(beta) >= alpha:
            raise ValueError(f"|beta| must be < alpha, got beta={beta}, alpha={alpha}")

        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu
        self._gamma = np.sqrt(alpha**2 - beta**2)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"NIG(alpha={alpha}, beta={beta}, delta={delta}, mu={mu})",
            math_note=(
                f"Lévy density: αδ/π · K_1(α|x|)/|x| · exp(βx); γ = sqrt(α²-β²) = {self._gamma:.4f}"
            ),
            reference="Barndorff-Nielsen (1997); Cont & Tankov (2004) Ch. 4",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        nu = _nig_levy_measure(self.alpha, self.beta, self.delta)
        mean = self.mu + self.delta * self.beta / self._gamma
        large_mean = nu.integrate(lambda x: x, -np.inf, -1.0) + nu.integrate(
            lambda x: x, 1.0, np.inf
        )
        return LevyTriplet(b=mean - large_mean, sigma_sq=0.0, nu=nu)

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
        )

    def char_func_exact(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """Evaluate the named characteristic function on its exponential-moment strip."""
        if not np.isfinite(t) or t < 0:
            raise ValueError("Time must be finite and nonnegative")
        return np.exp(t * self.characteristic_exponent(u))

    def char_func(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Exact simulation via Inverse Gaussian subordination.

        For each increment of length Δt:
          1. Sample IG ~ InverseGaussian(mean=δ·Δt/γ, shape=δ²·Δt²) — the IG subordinator
          2. Sample X = μ·Δt + β·IG + sqrt(IG)·Z where Z ~ N(0,1)

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

        ig_mean = self.delta * dt / self._gamma
        ig_shape = (self.delta * dt) ** 2

        ig_samples = rng.wald(mean=ig_mean, scale=ig_shape, size=(n_paths, n_steps))
        normals = rng.standard_normal(size=(n_paths, n_steps))

        increments = self.mu * dt + self.beta * ig_samples + np.sqrt(ig_samples) * normals

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"NIG(alpha={self.alpha}, beta={self.beta}, delta={self.delta}, mu={self.mu})"

    def characteristic_exponent(self, u: complex | float | np.ndarray) -> np.ndarray:
        """Unit-time log CF on the continuous branch at zero."""
        z = np.asarray(u, dtype=complex)
        if np.any(np.abs(self.beta - z.imag) > self.alpha):
            raise ValueError("Exponential moment outside the convergence strip")
        return np.asarray(
            1j * self.mu * z
            + self.delta * (self._gamma - np.sqrt(self.alpha**2 - (self.beta + 1j * z) ** 2))
        )
