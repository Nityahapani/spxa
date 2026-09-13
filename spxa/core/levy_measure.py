"""
Lévy measure representation.

The Lévy measure ν is a measure on ℝ \ {0} satisfying ∫ (1 ∧ x²) ν(dx) < ∞.
It encodes the jump structure of a Lévy process: ν(B) equals the expected
number of jumps per unit time with size in B.

Rather than representing ν as an opaque callable, spxa makes it a typed object
with subtypes that carry analytical properties. This allows:
- Addition of Lévy measures by combining components
- Exact computation of moments ∫ xⁿ ν(dx) when the density is known
- Tail queries ν((x, ∞)) for risk and regularity analysis
- Activity classification (finite vs infinite activity)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


class LevyMeasure(abc.ABC):
    """
    Abstract base for Lévy measures.

    Subclasses represent specific parametric families. The key design invariant:
    every subclass must be able to answer moment queries analytically where the
    closed form is known, and fall back to numerical quadrature otherwise.
    """

    @abc.abstractmethod
    def density(self, x: ArrayLike) -> np.ndarray:
        """
        Return the density k(x) such that ν(dx) = k(x) dx, evaluated at x.

        Not all Lévy measures have a density (e.g. compound Poisson with
        discrete jump distribution). Subclasses without a density should raise
        NotImplementedError.

        Parameters
        ----------
        x :
            Points at which to evaluate the density. Must not include 0.
        """

    @abc.abstractmethod
    def total_mass(self) -> float:
        """
        Return ν(ℝ \ {0}).

        Finite total mass corresponds to finite activity (compound Poisson).
        Returns np.inf for infinite activity processes.
        """

    @abc.abstractmethod
    def tail(self, x: float) -> float:
        """
        Return ν((x, ∞)) for x > 0.

        The tail of the Lévy measure governs large-jump frequency and
        the existence of moments: ∫_{|y|>1} |y|ⁿ ν(dy) < ∞ iff the
        n-th moment exists.

        Parameters
        ----------
        x :
            Must be positive.
        """

    def moment(self, n: int) -> float:
        """
        Return ∫ xⁿ ν(dx) via numerical quadrature.

        Subclasses should override with exact formulas where available.
        The integral is split at ±1 to handle the singularity at 0.

        Parameters
        ----------
        n :
            Order. Must be ≥ 1.
        """
        from scipy import integrate  # type: ignore[import-untyped]

        if n < 1:
            raise ValueError("Moment order must be >= 1")

        def integrand(x: float) -> float:
            if x == 0:
                return 0.0
            k = float(self.density(np.array([x]))[0])
            return (x**n) * k

        val_neg, _ = integrate.quad(integrand, -np.inf, -1e-10)
        val_pos, _ = integrate.quad(integrand, 1e-10, np.inf)
        return val_neg + val_pos

    def is_finite_activity(self) -> bool:
        """Return True iff ν(ℝ \ {0}) < ∞ (compound Poisson case)."""
        return np.isfinite(self.total_mass())

    def __add__(self, other: LevyMeasure) -> CompoundLevyMeasure:
        """Add two Lévy measures. Used when adding independent Lévy processes."""
        return CompoundLevyMeasure([self, other])


@dataclass
class CompoundLevyMeasure(LevyMeasure):
    """
    Sum of multiple Lévy measures.

    Arises from adding independent Lévy processes: ν_{X+Y} = ν_X + ν_Y.
    Stores components and delegates to each.
    """

    components: list[LevyMeasure] = field(default_factory=list)

    def density(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return sum(c.density(x_arr) for c in self.components)  # type: ignore[return-value]

    def total_mass(self) -> float:
        masses = [c.total_mass() for c in self.components]
        if any(np.isinf(m) for m in masses):
            return np.inf
        return sum(masses)

    def tail(self, x: float) -> float:
        return sum(c.tail(x) for c in self.components)

    def moment(self, n: int) -> float:
        return sum(c.moment(n) for c in self.components)

    def __add__(self, other: LevyMeasure) -> CompoundLevyMeasure:
        if isinstance(other, CompoundLevyMeasure):
            return CompoundLevyMeasure(self.components + other.components)
        return CompoundLevyMeasure(self.components + [other])


@dataclass
class DensityLevyMeasure(LevyMeasure):
    """
    Lévy measure specified by a density function k(x) on ℝ \ {0}.

    Used for named processes whose Lévy density is known in closed form
    (Gamma, NIG, VG, CGMY, etc.).

    Parameters
    ----------
    _density_fn :
        Callable mapping an array of non-zero reals to the density values.
    _total_mass :
        Total mass ν(ℝ \ {0}). Pass np.inf for infinite activity.
    _tail_fn :
        Callable mapping x > 0 to ν((x, ∞)). If None, computed numerically.
    _moment_fns :
        Optional dict mapping moment order n to an exact formula for ∫ xⁿ ν(dx).
    """

    _density_fn: Callable[[np.ndarray], np.ndarray]
    _total_mass: float
    _tail_fn: Callable[[float], float] | None = None
    _moment_fns: dict[int, Callable[[], float]] = field(default_factory=dict)

    def density(self, x: ArrayLike) -> np.ndarray:
        return self._density_fn(np.asarray(x, dtype=float))

    def total_mass(self) -> float:
        return self._total_mass

    def tail(self, x: float) -> float:
        if self._tail_fn is not None:
            return self._tail_fn(x)
        from scipy import integrate  # type: ignore[import-untyped]
        result, _ = integrate.quad(lambda t: float(self.density(np.array([t]))[0]), x, np.inf)
        return result

    def moment(self, n: int) -> float:
        if n in self._moment_fns:
            return self._moment_fns[n]()
        return super().moment(n)


@dataclass
class ScaledLevyMeasure(LevyMeasure):
    """
    Image measure ν_Z where Z = cX: ν_Z(B) = ν_X(B/c).

    Arises from scalar multiplication of a Lévy process.
    Sato (1999), proof of Proposition 11.10.

    Parameters
    ----------
    base :
        The Lévy measure of the original process X.
    scale :
        The scalar c. Must be non-zero.
    """

    base: LevyMeasure
    scale: float

    def __post_init__(self) -> None:
        if self.scale == 0:
            raise ValueError("Scale factor must be non-zero")

    def density(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return self.base.density(x_arr / self.scale) / abs(self.scale)

    def total_mass(self) -> float:
        return self.base.total_mass()

    def tail(self, x: float) -> float:
        return self.base.tail(x / abs(self.scale))

    def moment(self, n: int) -> float:
        return (self.scale**n) * self.base.moment(n)
