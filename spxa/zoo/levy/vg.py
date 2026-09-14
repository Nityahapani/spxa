"""
Variance Gamma (VG) process.

A Brownian motion with drift subordinated by an independent Gamma process.
One of the most widely used models in quantitative finance for asset returns.

Construction: X_t = θ·G_t + σ·W_{G_t} where G is Gamma(1/ν, 1/ν) and W is
standard BM, with G and W independent.

Lévy density:
  k(x) = (C/|x|) · exp(A·x - B·|x|)
where:
  C = 1/ν
  A = θ/σ²
  B = sqrt(θ²/σ⁴ + 2/(σ²ν))

This is an infinite-activity, finite-variation process (paths have finite
variation but infinitely many small jumps).

References
----------
Madan, D.B., Carr, P. & Chang, E.C. (1998). The Variance Gamma process and
option pricing. *European Finance Review*, 2, 79–105.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapter 4, Table 4.2.
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


def _vg_levy_measure(sigma: float, nu: float, theta: float) -> DensityLevyMeasure:
    """
    VG Lévy density: k(x) = C/|x| * exp(A*x - B*|x|).

    Madan, Carr & Chang (1998), equation (10).
    Cont & Tankov (2004), Table 4.2.
    """
    C = 1.0 / nu
    A = theta / sigma**2
    B = np.sqrt(theta**2 / sigma**4 + 2.0 / (sigma**2 * nu))

    def density(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        nz = x != 0
        xv = x[nz]
        out[nz] = (C / np.abs(xv)) * np.exp(A * xv - B * np.abs(xv))
        return out

    def tail(xv: float) -> float:
        from scipy import integrate  # type: ignore[import-untyped]
        result, _ = integrate.quad(
            lambda t: (C / t) * np.exp(A * t - B * t),
            xv, np.inf, limit=200
        )
        return result

    def moment_1() -> float:
        return theta

    def moment_2() -> float:
        return sigma**2 + theta**2 * nu

    def moment_3() -> float:
        return 2 * theta**3 * nu**2 + 3 * sigma**2 * theta * nu

    def moment_4() -> float:
        return (3 * sigma**4 * nu + 12 * sigma**2 * theta**2 * nu**2
                + 6 * theta**4 * nu**3)

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=np.inf,
        _tail_fn=tail,
        _moment_fns={1: moment_1, 2: moment_2, 3: moment_3, 4: moment_4},
    )


class VarianceGamma(Process):
    """
    Variance Gamma process.

    Parametrised by (σ, ν, θ) following Madan, Carr & Chang (1998).

    Parameters
    ----------
    sigma :
        Volatility of the Brownian component. σ > 0.
    nu :
        Variance rate of the Gamma subordinator. ν > 0.
        Controls the kurtosis: larger ν → heavier tails.
    theta :
        Drift of the Brownian component. θ ∈ ℝ.
        Controls skewness: θ < 0 gives left skew (typical for equity returns).

    Notes
    -----
    Cumulants (Madan, Carr & Chang 1998, Section 2):
      κ_1 = θ
      κ_2 = σ² + θ²ν
      κ_3 = 2θ³ν² + 3σ²θν
      κ_4 = 3σ⁴ν + 12σ²θ²ν² + 6θ⁴ν³

    The characteristic function is (Madan et al. 1998, eq. 4):
      φ(u) = (1 - iuθν + σ²νu²/2)^{-t/ν}

    References
    ----------
    Madan, D.B., Carr, P. & Chang, E.C. (1998). The Variance Gamma process and
    option pricing. *European Finance Review*, 2, 79–105.
    """

    def __init__(self, sigma: float, nu: float, theta: float = 0.0) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        self.sigma = sigma
        self.nu = nu
        self.theta = theta

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"VarianceGamma(sigma={sigma}, nu={nu}, theta={theta})",
            math_note=(
                f"Lévy density: C/|x|·exp(Ax - B|x|) with "
                f"C=1/ν={1/nu:.4f}, A=θ/σ²={theta/sigma**2:.4f}, "
                f"B=sqrt(θ²/σ⁴+2/(σ²ν))={np.sqrt(theta**2/sigma**4+2/(sigma**2*nu)):.4f}"
            ),
            reference="Madan, Carr & Chang (1998); Cont & Tankov (2004) Table 4.2",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        # The VG Lévy density k(x) = C/|x| * exp(Ax - B|x|) has a non-integrable
        # singularity at 0 that makes generic quadrature unreliable.
        # We use the known closed-form mean E[X_1] = θ as the drift anchor,
        # with the truncation-function correction absorbed analytically.
        # The drift in the Lévy–Khintchine formula with h(x)=1_{|x|≤1} is:
        #   b = E[X_1] - ∫_{|x|>1} x k(x) dx
        # The second term is computed numerically on [1,∞), where the integrand
        # has no singularity and converges exponentially (exponential tails).
        from scipy import integrate  # type: ignore[import-untyped]

        nu_measure = _vg_levy_measure(self.sigma, self.nu, self.theta)

        C = 1.0 / self.nu
        A = self.theta / self.sigma**2
        B = np.sqrt(self.theta**2 / self.sigma**4 + 2.0 / (self.sigma**2 * self.nu))

        # ∫_1^∞ x * k(x) dx  (positive side, no singularity at origin)
        tail_pos, _ = integrate.quad(
            lambda x: (C / x) * np.exp(A * x - B * x) * x,
            1.0, np.inf, limit=200
        )
        # ∫_{-∞}^{-1} x * k(x) dx = -∫_1^∞ x * k(-x) dx
        tail_neg, _ = integrate.quad(
            lambda x: (C / x) * np.exp(-A * x - B * x) * (-x),
            1.0, np.inf, limit=200
        )
        # b = mean - tail_correction
        b_drift = self.theta - tail_pos - tail_neg

        return LevyTriplet(b=b_drift, sigma_sq=0.0, nu=nu_measure)

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """Use the exact closed-form characteristic function for VG."""
        return self.char_func_exact(u=u, t=t)

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

    def cumulants(self, order: int) -> dict[int, float]:
        """
        Exact closed-form cumulants for VG.

        κ_1 = θ
        κ_2 = σ² + θ²ν
        κ_3 = 2θ³ν² + 3σ²θν
        κ_4 = 3σ⁴ν + 12σ²θ²ν² + 6θ⁴ν³

        Reference: Madan, Carr & Chang (1998), Section 2.
        """
        s, n, t_ = self.sigma, self.nu, self.theta
        result: dict[int, float] = {}
        if order >= 1:
            result[1] = t_
        if order >= 2:
            result[2] = s**2 + t_**2 * n
        if order >= 3:
            result[3] = 2 * t_**3 * n**2 + 3 * s**2 * t_ * n
        if order >= 4:
            result[4] = 3 * s**4 * n + 12 * s**2 * t_**2 * n**2 + 6 * t_**4 * n**3
        for k in range(5, order + 1):
            result[k] = self._triplet().cumulant(k)
        return result

    def char_func_exact(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Exact closed-form characteristic function.

        φ(u; t) = (1 - iuθν + σ²νu²/2)^{-t/ν}

        Reference: Madan, Carr & Chang (1998), equation (4).

        Parameters
        ----------
        u :
            Frequency argument(s).
        t :
            Time.
        """
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        base = 1.0 - 1j * u_arr * self.theta * self.nu + 0.5 * self.sigma**2 * self.nu * u_arr**2
        result = base ** (-t / self.nu)
        return result if result.shape != (1,) else result[0]

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Exact simulation via Gamma subordination.

        For each increment:
          1. Sample G ~ Gamma(Δt/ν, 1/ν)  (Gamma subordinator increment)
          2. Sample X = θ·G + σ·sqrt(G)·Z where Z ~ N(0,1)

        This is exact because VG = BM-with-drift subordinated by Gamma.

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

        gamma_increments = rng.gamma(
            shape=dt / self.nu,
            scale=self.nu,
            size=(n_paths, n_steps),
        )
        normals = rng.standard_normal(size=(n_paths, n_steps))
        increments = self.theta * gamma_increments + self.sigma * np.sqrt(gamma_increments) * normals

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"VarianceGamma(sigma={self.sigma}, nu={self.nu}, theta={self.theta})"
