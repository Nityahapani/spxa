"""Lévy–Khintchine triplets with truncation x*1_{|x|<=1}.

References: Sato (1999), Theorem 8.1, Proposition 11.10, Theorem 25.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spxa.core.levy_measure import LevyMeasure, ScaledLevyMeasure


@dataclass(frozen=True)
class LevyTriplet:
    """Immutable drift b, Gaussian variance sigma_sq, and jump measure nu."""

    b: float
    sigma_sq: float
    nu: LevyMeasure

    def __post_init__(self) -> None:
        if not np.isfinite(self.b) or not np.isfinite(self.sigma_sq) or self.sigma_sq < 0:
            raise ValueError("Drift must be finite and Gaussian variance finite and nonnegative")

    def characteristic_exponent(self, u: complex | float | np.ndarray) -> np.ndarray:
        """Return the continuous unit-time log CF, without principal-log wrapping.

        Numerical integration respects atomic measures and splits at the
        truncation boundaries. The compensated exponential is evaluated by
        a Taylor series near zero to avoid subtractive cancellation.
        """
        frequencies = np.asarray(u, dtype=complex)
        result = np.empty(frequencies.shape, dtype=complex)
        for index in np.ndindex(frequencies.shape):
            frequency = complex(frequencies[index])
            if frequency == 0:
                result[index] = 0
                continue
            integral = self.nu._levy_integral(frequency)
            result[index] = 1j * self.b * frequency - 0.5 * self.sigma_sq * frequency**2 + integral
        return result

    def char_exp(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """Evaluate exp(t*psi(u)), the characteristic function, not its logarithm."""
        if not np.isfinite(t) or t < 0:
            raise ValueError("Time must be finite and nonnegative")
        if t == 0:
            return np.ones_like(np.asarray(u), dtype=complex)
        return np.exp(t * self.characteristic_exponent(u))

    def cumulant(self, n: int) -> float:
        """Unit-time cumulant, provided the required tail moment exists (Sato 25.3)."""
        if n < 1:
            raise ValueError("Cumulant order must be positive")
        if self.nu._tail_moment_finite(n) is False:
            raise ValueError("Requested cumulant does not exist (tail moment criterion)")
        if n == 1:
            # Restrict the quadrature domains explicitly. A tiny far-tail
            # interval can be missed if its indicator is integrated over all
            # probability space. Keep the indicator to exclude boundary atoms.
            def large_jump(x: float) -> float:
                return x if abs(x) > 1 else 0.0

            return (
                self.b
                + self.nu.integrate(large_jump, -np.inf, -1)
                + self.nu.integrate(large_jump, 1, np.inf)
            )
        return float(self.nu.moment(n) + (self.sigma_sq if n == 2 else 0.0))

    def __add__(self, other: LevyTriplet) -> LevyTriplet:
        """Add the triplets of independent processes (Sato, Proposition 11.10)."""
        return LevyTriplet(self.b + other.b, self.sigma_sq + other.sigma_sq, self.nu + other.nu)

    def scale(self, c: float) -> LevyTriplet:
        """Triplet of c*X, with the full truncation correction.

        b_new = c*b + integral c*x*(1_{|c*x|<=1}-1_{|x|<=1}) nu(dx).
        Atom contributions at either truncation boundary are included.
        """
        if not np.isfinite(c) or c == 0:
            raise ValueError("Scale factor must be finite and nonzero")
        lo, hi = sorted((1.0, 1 / abs(c)))

        def integrand(x: float) -> float:
            return c * x * (int(abs(c * x) <= 1) - int(abs(x) <= 1))

        correction = 0.0
        if lo < hi:
            correction = self.nu.integrate(integrand, lo, hi) + self.nu.integrate(
                integrand, -hi, -lo
            )
        return LevyTriplet(
            c * self.b + correction, c * c * self.sigma_sq, ScaledLevyMeasure(self.nu, c)
        )
