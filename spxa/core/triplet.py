"""
Lévy–Khintchine triplet.

Every Lévy process X on ℝ is uniquely associated with a characteristic triplet
(b, σ², ν) where:
  b  ∈ ℝ          — drift (relative to the truncation function 𝟙_{|x|≤1})
  σ² ≥ 0          — Gaussian variance component
  ν              — Lévy measure on ℝ \ {0}, ∫ (1 ∧ x²) ν(dx) < ∞

The characteristic exponent is:
  ψ(u) = ibu - ½σ²u² + ∫ (e^{iux} - 1 - iux·𝟙_{|x|≤1}) ν(dx)

Reference: Sato (1999), Theorem 8.1.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from spxa.core.levy_measure import CompoundLevyMeasure, LevyMeasure, ScaledLevyMeasure


@dataclass(frozen=True)
class LevyTriplet:
    """
    Characteristic triplet (b, sigma_sq, nu) of a Lévy process.

    Immutable. All arithmetic returns a new LevyTriplet.

    Parameters
    ----------
    b :
        Drift coefficient. Defined relative to the truncation function
        h(x) = 𝟙_{|x|≤1}. Real number.
    sigma_sq :
        Gaussian variance (σ²). Must be non-negative.
    nu :
        Lévy measure. A LevyMeasure instance satisfying ∫ (1 ∧ x²) ν(dx) < ∞.

    References
    ----------
    Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*.
    Cambridge University Press. Theorem 8.1.
    """

    b: float
    sigma_sq: float
    nu: LevyMeasure

    def __post_init__(self) -> None:
        if self.sigma_sq < 0:
            raise ValueError(f"sigma_sq must be non-negative, got {self.sigma_sq}")

    def char_exp(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Evaluate the characteristic function E[e^{iuX_t}] = exp(t·ψ(u)).

        Uses the Lévy–Khintchine formula numerically. For u far from 0,
        the integral is split at ±1 to handle the singularity at x=0.

        Parameters
        ----------
        u :
            Frequency argument(s).
        t :
            Time. Default 1.
        """
        from scipy import integrate  # type: ignore[import-untyped]

        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        results = np.zeros_like(u_arr, dtype=complex)

        for i, ui in enumerate(u_arr.flat):
            gaussian_term = -0.5 * self.sigma_sq * ui**2
            drift_term = 1j * self.b * ui

            def integrand(x: float) -> complex:
                if x == 0.0:
                    return complex(0.0)
                k = float(self.nu.density(np.array([x]))[0])
                exp_term = np.exp(1j * ui * x) - 1 - 1j * ui * x * (abs(x) <= 1)
                return exp_term * k

            jump_re, _ = integrate.quad(lambda x: integrand(x).real, -np.inf, np.inf,
                                        limit=200, points=[-1.0, 0.0, 1.0])
            jump_im, _ = integrate.quad(lambda x: integrand(x).imag, -np.inf, np.inf,
                                        limit=200, points=[-1.0, 0.0, 1.0])

            psi = drift_term + gaussian_term + complex(jump_re, jump_im)
            results.flat[i] = np.exp(t * psi)

        return results if results.shape != (1,) else results[0]

    def cumulant(self, n: int) -> float:
        """
        Return the n-th cumulant of X_1 per unit time, κ_n / t.

        For a Lévy process, κ_n(X_t) = t · κ_n(X_1).

        Exact formulas (Sato 1999, Theorem 25.3):
          κ_1 = b + ∫_{|x|>1} x ν(dx)
          κ_2 = σ² + ∫ x² ν(dx)
          κ_n = ∫ xⁿ ν(dx),  n ≥ 3

        The n-th cumulant exists iff ∫_{|x|>1} |x|ⁿ ν(dx) < ∞.

        Parameters
        ----------
        n :
            Cumulant order. Must be ≥ 1.

        References
        ----------
        Sato, K.-I. (1999). Theorem 25.3.
        """
        if n < 1:
            raise ValueError("Cumulant order must be >= 1")
        if n == 1:
            from scipy import integrate  # type: ignore[import-untyped]
            tail_correction, _ = integrate.quad(
                lambda x: float(self.nu.density(np.array([x]))[0]) * x,
                -np.inf, -1.0
            )
            tail_correction2, _ = integrate.quad(
                lambda x: float(self.nu.density(np.array([x]))[0]) * x,
                1.0, np.inf
            )
            return self.b + tail_correction + tail_correction2
        if n == 2:
            return self.sigma_sq + self.nu.moment(2)
        return self.nu.moment(n)

    def __add__(self, other: LevyTriplet) -> LevyTriplet:
        """
        Add two triplets. Corresponds to adding independent Lévy processes.

        (b_X, σ²_X, ν_X) + (b_Y, σ²_Y, ν_Y) = (b_X+b_Y, σ²_X+σ²_Y, ν_X+ν_Y)

        Reference: Sato (1999), Proposition 11.10.
        """
        return LevyTriplet(
            b=self.b + other.b,
            sigma_sq=self.sigma_sq + other.sigma_sq,
            nu=self.nu + other.nu,
        )

    def scale(self, c: float) -> LevyTriplet:
        """
        Return the triplet of cX.

        The drift transforms to account for the change in the truncation
        region: the image measure ν_Z(B) = ν_X(B/c) may shift mass across
        the unit ball boundary, requiring a drift correction.

        Reference: Sato (1999), proof of Proposition 11.10.

        Parameters
        ----------
        c :
            Non-zero scalar.
        """
        if c == 0:
            raise ValueError("Scale factor must be non-zero")

        from scipy import integrate  # type: ignore[import-untyped]

        scaled_nu = ScaledLevyMeasure(base=self.nu, scale=c)

        def original_density(x: float) -> float:
            return float(self.nu.density(np.array([x]))[0])

        if abs(c) < 1:
            lo, hi = abs(c), 1.0
            sign = np.sign(c)
            correction_pos, _ = integrate.quad(lambda x: original_density(x) * x, lo, hi)
            correction_neg, _ = integrate.quad(lambda x: original_density(x) * x, -hi, -lo)
            drift_correction = sign * (correction_pos + correction_neg)
        elif abs(c) > 1:
            lo, hi = 1.0, abs(c)
            sign = np.sign(c)
            correction_pos, _ = integrate.quad(lambda x: original_density(x) * x, lo, hi)
            correction_neg, _ = integrate.quad(lambda x: original_density(x) * x, -hi, -lo)
            drift_correction = -sign * (correction_pos + correction_neg)
        else:
            drift_correction = 0.0

        return LevyTriplet(
            b=c * self.b + drift_correction,
            sigma_sq=c**2 * self.sigma_sq,
            nu=scaled_nu,
        )

    def __repr__(self) -> str:
        return (
            f"LevyTriplet(b={self.b!r}, sigma_sq={self.sigma_sq!r}, "
            f"nu={self.nu.__class__.__name__}(...))"
        )
