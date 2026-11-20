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

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def _meixner_levy_measure(alpha: float, beta: float, delta: float) -> DensityLevyMeasure:
    r"""
    Meixner Lévy density :math:`k(x) = \delta e^{\beta x/\alpha} / (x \sinh(\pi x/\alpha))`, :math:`x \ne 0`.

    It is evaluated in the overflow-free form

    .. math:: k(x) = \frac{2\delta\, e^{(\beta x - \pi|x|)/\alpha}}{|x|\,(1 - e^{-2\pi|x|/\alpha})},

    because the naive quotient is ``inf/inf`` once :math:`\pi|x|/\alpha > 710`.
    Moments of every order are the cumulants
    :math:`\int x^n k = 2\delta(\alpha/2)^n P_{n-1}(\tan(\beta/2))`
    (see :meth:`MeixnerProcess.cumulant_exact`).

    Schoutens (2002), p. 53.
    """
    from numpy.polynomial import polynomial as P

    def density(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        out = np.zeros_like(x_arr)
        nz = x_arr != 0
        ax = np.abs(x_arr[nz])
        out[nz] = (
            2
            * delta
            * np.exp((beta * x_arr[nz] - np.pi * ax) / alpha)
            / (-ax * np.expm1(-2 * np.pi * ax / alpha))
        )
        return out

    def tail(xv: float) -> float:
        from scipy import integrate  # type: ignore[import-untyped]

        result, _ = integrate.quad(
            lambda t: float(density(np.array([t]))[0]), xv, np.inf, limit=200
        )
        return float(result)

    def make_moment(n: int):  # type: ignore[no-untyped-def]
        poly = np.array([0.0, 1.0])
        for _ in range(n - 1):
            poly = P.polymul(P.polyder(poly), [1.0, 0.0, 1.0])
        value = 2 * delta * (alpha / 2) ** n * float(P.polyval(np.tan(beta / 2), poly))
        return lambda: value

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=np.inf,
        _tail_fn=tail,
        _moment_fns={n: make_moment(n) for n in range(1, 21)},
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
      κₙ = 2δ(α/2)ⁿ · dⁿ⁻¹/dyⁿ⁻¹ tan y at y = β/2, for every n ≥ 2

    Characteristic function (Schoutens & Teugels 1998, eq. 3):
      φ(u; t) = exp(imu) · (cos(β/2)/cosh((αu - iβ)/2))^{2δt}

    Simulation via acceptance-rejection from a Cauchy proposal
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
        from scipy import integrate  # type: ignore[import-untyped]

        nu = _meixner_levy_measure(self.alpha, self.beta, self.delta)
        a, b = self.alpha, self.beta

        # b = m + ∫_{|x|≤1} x k(x) dx. Pairing x with −x removes the 1/x
        # singularity: x k(x) − x k(−x) = 2δ sinh(βx/α)/sinh(πx/α), which is
        # evaluated as sign(β) e^{(|β|−π)x/α} (1 − e^{−2|β|x/α})/(1 − e^{−2πx/α}).
        def paired(x: float) -> float:
            return float(
                2
                * self.delta
                * np.sign(b)
                * np.exp((abs(b) - np.pi) * x / a)
                * np.expm1(-2 * abs(b) * x / a)
                / np.expm1(-2 * np.pi * x / a)
            )

        small, _ = integrate.quad(paired, 0.0, 1.0, limit=200)
        return LevyTriplet(b=self.m + small, sigma_sq=0.0, nu=nu)

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

    def char_func_exact(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Exact characteristic function.

        φ(u; t) = exp(imu) · (cos(β/2) / cosh((αu - iβ)/2))^{2δt}

        Schoutens & Teugels (1998), equation (3).
        """
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        cos_half = np.cos(self.beta / 2)
        z = (self.alpha * u_arr - 1j * self.beta) / 2.0
        # log cosh(z) via the identity log cosh(z) = log(e^z + e^{-z}) - log 2
        # Use logsumexp-style: for Re(z)>=0, factor out e^z:
        #   log cosh(z) = z + log(1 + e^{-2z}) - log 2
        # The issue is e^{-2z} may overflow when Re(z) << 0.
        # Robust: compute log(e^z + e^{-z}) - log 2 via the real-part stable form.
        # Since cosh(z) = cosh(x)cos(y) + i sinh(x)sin(y) for z=x+iy,
        # and |cosh(z)|² = cosh²(x) - sin²(y), we can compute directly for
        # the moderate values we encounter (|Re(z)| = α|u|/2 ≤ α·u_max/2).
        # For the FFT grid u_max is chosen so that α·u_max/2 ≈ 20σ which is safe.
        cosh_z = np.cosh(z)
        log_cosh = np.log(np.where(np.abs(cosh_z) > 1e-300, cosh_z, 1e-300 + 0j))
        log_ratio = np.log(cos_half + 0j) - log_cosh
        exponent = 1j * self.m * t * u_arr + 2 * self.delta * t * log_ratio
        result = np.exp(exponent)
        return result if result.shape != (1,) else result[0]

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def cumulant_exact(self, n: int, t: float = 1.0) -> float:
        """
        Exact cumulants of every order.

        On its strip the cumulant exponent is
        K(z) = mz + 2δ[log cos(β/2) − log cos((αz + β)/2)], so

        κ₁ = δα tan(β/2) + m
        κₙ = 2δ(α/2)ⁿ Pₙ₋₁(tan(β/2))   for n ≥ 2,

        where dᵏ/dyᵏ tan y = Pₖ(tan y), P₀(T) = T, Pₖ₊₁ = Pₖ′(T)(1 + T²). In closed form:

        κ₂ = δα²/(2cos²(β/2))
        κ₃ = δα³ sin(β/2)/(2cos³(β/2))
        κ₄ = δα⁴(2 − cos β)/(4cos⁴(β/2))

        The excess kurtosis is therefore κ₄/κ₂² = (2 − cos β)/δ
        (Schoutens 2003, Table 5.5).
        """
        if n < 1:
            raise ValueError(f"cumulant order must be >= 1, got {n}")
        from numpy.polynomial import polynomial as P

        a, b, d = self.alpha, self.beta, self.delta
        poly = np.array([0.0, 1.0])
        for _ in range(n - 1):
            poly = P.polymul(P.polyder(poly), [1.0, 0.0, 1.0])
        kappa = 2.0 * d * (a / 2.0) ** n * float(P.polyval(np.tan(b / 2.0), poly))
        if n == 1:
            kappa += self.m
        return float(t * kappa)

    def cumulants(self, order: int) -> dict[int, float]:
        return {n: self.cumulant_exact(n) for n in range(1, order + 1)}

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulation via FFT-based CDF inversion (Gil-Pelaez).

        For each increment of length Δt, samples from the Meixner(α,β,δΔt,mΔt)
        distribution by:
          1. Computing the CDF via the exact characteristic function on a grid.
          2. Sampling uniform variates and inverting the CDF via interpolation.

        This is exact in distribution (up to grid resolution) and always
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
        return (
            f"MeixnerProcess(alpha={self.alpha}, beta={self.beta}, delta={self.delta}, m={self.m})"
        )
