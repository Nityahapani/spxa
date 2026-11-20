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
from spxa.sim.cf_sampling import sample_from_cf
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
        r"""
        Exact characteristic function.

        φ(u; t) = exp(imu) · (cos(β/2) / cosh((αu - iβ)/2))^{2δt}

        Schoutens & Teugels (1998), equation (3). Evaluated as
        :math:`e^{t\psi(u)}` with the overflow-free exponent of
        :meth:`_char_exponent`.
        """
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        # the overflow-free exponent: np.cosh overflows once alpha |u| / 2 > 710
        result = np.exp(t * self._char_exponent(u_arr))
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

    def _char_exponent(self, u: np.ndarray) -> np.ndarray:
        r"""
        :math:`\psi(u) = 2\delta\big[\log\cos(\beta/2) - \log\cosh\big(\tfrac{\alpha u - i\beta}{2}\big)\big] + imu`.

        With :math:`w = (\alpha u - i\beta)/2 = a + ib`,
        :math:`\operatorname{Re}\cosh w = \cosh a\cos b > 0` whenever
        :math:`|b| < \pi/2`. That covers the real axis (:math:`|\beta| < \pi`)
        and the whole convergence strip, so the principal logarithm is
        analytic there. It is evaluated as
        :math:`\log\cosh w = s w + \log(1 + e^{-2sw}) - \log 2`, with
        :math:`s = \operatorname{sign}(a)`, which never overflows. Both
        imaginary parts lie in :math:`(-\pi, \pi)` and agree modulo
        :math:`2\pi`, so they are equal.
        """
        u = np.asarray(u, dtype=complex)
        w = (self.alpha * u - 1j * self.beta) / 2.0
        s = np.where(w.real >= 0, 1.0, -1.0)
        log_cosh = s * w + np.log1p(np.exp(-2.0 * s * w)) - np.log(2.0)
        return 2.0 * self.delta * (np.log(np.cos(self.beta / 2.0)) - log_cosh) + 1j * self.m * u

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulation by FFT inversion of the characteristic function.

        Each increment of length Δt is drawn from Meixner(α, β, δΔt, mΔt):
        the CDF is recovered from the exact characteristic function on a grid
        covering the slower exponential tail, rate (π − |β|)/α (see
        :mod:`spxa.sim.cf_sampling`), then inverted at uniform variates.
        Accurate up to the grid discretisation.

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
        k = self.cumulants(2)
        increments = sample_from_cf(
            lambda u: self.char_func(u, dt),
            mean=k[1] * dt,
            std=float(np.sqrt(k[2] * dt)),
            size=(n_paths, n_steps),
            rng=rng,
            tail_decay=(np.pi - abs(self.beta)) / self.alpha,
        )
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return (
            f"MeixnerProcess(alpha={self.alpha}, beta={self.beta}, delta={self.delta}, m={self.m})"
        )
