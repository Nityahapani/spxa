"""
Meixner process.

An infinite-activity pure-jump Lévy process with hypergeometric characteristic
function, rich moment structure, and a natural connection to orthogonal
polynomial systems (Meixner polynomials). Introduced to finance by Schoutens
& Teugels (1998).

Characteristic function (Schoutens & Teugels 1998, equation 3):
  φ(u; t) = (cos(β/2) / cosh((αu - iβ)/2))^{2δt}

where α > 0, β ∈ (-π, π), δ > 0, m ∈ ℝ.

Lévy density (Schoutens 2002, p. 53):
  k(x) = δ · exp(βx/α) / (x · sinh(πx/α))  for x ≠ 0

Parameters
----------
alpha : scale (α > 0)
beta  : asymmetry (β ∈ (-π, π))
delta : shape (δ > 0)
m     : location shift (m ∈ ℝ, default 0)

References
----------
Schoutens, W. & Teugels, J.L. (1998). Lévy processes, polynomials and
martingales. *Communications in Statistics — Stochastic Models*, 14(1–2),
335–349.

Schoutens, W. (2002). *Meixner Processes in Finance*. Research Report.
K.U. Leuven.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapter 4.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def _meixner_levy_measure(alpha: float, beta: float, delta: float) -> DensityLevyMeasure:
    """
    Meixner Lévy density: k(x) = δ·exp(βx/α) / (x·sinh(πx/α)) for x ≠ 0.

    Schoutens (2002), p. 53.
    """

    def density(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        nz = x != 0
        xv = x[nz]
        z = np.pi * np.abs(xv) / alpha
        out[nz] = 2 * delta * np.exp(beta * xv / alpha - z) / (np.abs(xv) * (-np.expm1(-2 * z)))
        return out

    def tail(xv: float) -> float:
        from scipy import integrate

        result, _ = integrate.quad(
            lambda t: float(density(np.array([t]))[0]), xv, np.inf, limit=200
        )
        return float(result)

    # Cumulants via the known closed-form formulas (Schoutens & Teugels 1998)
    def moment_1() -> float:
        return float(delta * alpha * np.tan(beta / 2))

    def moment_2() -> float:
        cos2 = np.cos(beta / 2) ** 2
        return float(delta * alpha**2 / (2 * cos2))

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=np.inf,
        _tail_fn=tail,
        _moment_fns={1: moment_1, 2: moment_2},
        _minimum_moment_order=2,
    )


class MeixnerProcess(Process):
    """
    Meixner process.

    Unique among Lévy models for having a characteristic function expressible
    entirely in terms of elementary trigonometric/hyperbolic functions, and for
    its orthogonal polynomial structure (Meixner–Pollaczek polynomials are
    eigenfunctions of the generator).

    Parameters
    ----------
    alpha :
        Scale parameter α > 0.
    beta :
        Asymmetry parameter β ∈ (-π, π).
        β = 0 gives a symmetric process.
    delta :
        Shape parameter δ > 0. Controls jump intensity.
    m :
        Location shift m ∈ ℝ. Default 0.

    Notes
    -----
    Cumulants (Schoutens & Teugels 1998, equations 5–8):
      κ₁ = δα tan(β/2) + m
      κ₂ = δα²/(2cos²(β/2))
      κ₃ = δα³ sin(β/2)/(2cos³(β/2))   [= δα³ tan(β/2)/(2cos²(β/2))]
      κ₄ = δα⁴(2 - cos β)/(4cos⁴(β/2))

    Characteristic function (Schoutens & Teugels 1998, eq. 3):
      φ(u; t) = exp(imu) · (cos(β/2)/cosh((αu - iβ)/2))^{2δt}

    Approximate simulation via finite-grid Fourier/CDF inversion
    (Schoutens 2002, Algorithm 2.1).

    References
    ----------
    Schoutens, W. & Teugels, J.L. (1998). *Communications in Statistics —
    Stochastic Models*, 14(1–2), 335–349.

    Schoutens, W. (2002). *Meixner Processes in Finance*. K.U. Leuven.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        delta: float,
        m: float = 0.0,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not (-np.pi < beta < np.pi):
            raise ValueError(f"beta must be in (-π, π), got {beta}")
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")

        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.m = m

        cos_half = np.cos(beta / 2)
        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"MeixnerProcess(alpha={alpha}, beta={beta}, delta={delta}, m={m})",
            math_note=(
                f"φ(u;t) = exp(imu)·(cos(β/2)/cosh((αu-iβ)/2))^{{2δt}}; "
                f"Lévy density: δ·exp(βx/α)/(x·sinh(πx/α)); "
                f"κ₁={delta * alpha * np.tan(beta / 2) + m:.4f}, "
                f"κ₂={delta * alpha**2 / (2 * cos_half**2):.4f}"
            ),
            reference="Schoutens & Teugels (1998); Schoutens (2002); Cont & Tankov (2004) Ch. 4",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        nu = _meixner_levy_measure(self.alpha, self.beta, self.delta)
        large_mean = nu.integrate(lambda x: x, -np.inf, -1) + nu.integrate(lambda x: x, 1, np.inf)
        return LevyTriplet(b=self.cumulant_exact(1) - large_mean, sigma_sq=0.0, nu=nu)

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
        """Evaluate the Meixner CF with time-scaled location and stable log-cosh."""
        if not np.isfinite(t) or t < 0:
            raise ValueError("Time must be finite and nonnegative")
        return np.exp(t * self.characteristic_exponent(u))

    def char_func(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def cumulant_exact(self, n: int, t: float = 1.0) -> float:
        """Cumulants from K'(s)=m+delta*alpha*tan((beta+alpha*s)/2).

        Derivative polynomials satisfy P_0(z)=z and
        P_{k+1}(z)=(1+z*z) P'_k(z). No finite differences are used.

        References
        ----------
        Schoutens & Teugels (1998), Meixner processes and polynomials.
        """
        from numpy.polynomial import Polynomial

        if n < 1:
            raise ValueError("Cumulant order must be positive")
        polynomial = Polynomial([0.0, 1.0])
        for _ in range(n - 1):
            polynomial = polynomial.deriv() * Polynomial([1.0, 0.0, 1.0])
        value = (
            self.delta
            * self.alpha
            * (self.alpha / 2) ** (n - 1)
            * polynomial(np.tan(self.beta / 2))
        )
        return float(t * (value + (self.m if n == 1 else 0.0)))

    def cumulants(self, order: int) -> dict[int, float]:
        return {n: self.cumulant_exact(n) for n in range(1, order + 1)}

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulation via FFT-based CDF inversion (Gil-Pelaez).

        For each increment of length Δt, samples from the Meixner(α,β,δΔt,mΔt)
        distribution by:
          1. Computing the CDF via the exact characteristic function on a grid.
          2. Sampling uniform variates and inverting the CDF via interpolation.

        This is a numerical approximation (up to grid resolution) and always
        terminates. Accuracy improves with n_fft.

        Parameters
        ----------
        n_steps, n_paths, T, rng :
            Standard simulation parameters.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1).
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps

        # Build CDF via Gil-Pelaez inversion on a fine x-grid
        std_dt = max(np.sqrt(self.cumulant_exact(2, t=dt)), 1e-6)
        mean_dt = self.cumulant_exact(1, t=dt)
        n_x = 2000
        x_half = max(12 * std_dt, 1e-4)
        x_grid = np.linspace(mean_dt - x_half, mean_dt + x_half, n_x)

        # Integrate the imaginary part of CF to get PDF via Gil-Pelaez:
        # f(x) = 1/π ∫_0^∞ Re[e^{-iux} φ(u)] du
        n_u = 2000
        u_max = min(200.0 / self.alpha, 800.0)
        u_grid = np.linspace(1e-6, u_max, n_u)

        cf_vals = self.char_func_exact(u=u_grid, t=dt)
        pdf = np.zeros(n_x)
        for j, x in enumerate(x_grid):
            integrand = (cf_vals * np.exp(-1j * u_grid * x)).real
            pdf[j] = float(trapezoid(integrand, u_grid)) / np.pi

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
        return (
            f"MeixnerProcess(alpha={self.alpha}, beta={self.beta}, delta={self.delta}, m={self.m})"
        )

    def characteristic_exponent(self, u: complex | float | np.ndarray) -> np.ndarray:
        """Stable log-cosh form on the MGF strip |beta-alpha*Im(u)|<pi."""
        z = np.asarray(u, dtype=complex)
        if np.any(np.abs(self.beta - self.alpha * z.imag) >= np.pi):
            raise ValueError("Meixner exponential moment lies outside its convergence strip")
        w = (self.alpha * z - 1j * self.beta) / 2
        w = np.where(w.real >= 0, w, -w)
        log_cosh = w + np.log1p(np.exp(-2 * w)) - np.log(2.0)
        return np.asarray(
            1j * self.m * z + 2 * self.delta * (np.log(np.cos(self.beta / 2)) - log_cosh)
        )
