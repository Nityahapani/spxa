"""Lévy measures, including density-backed and atomic image measures.

Integration is with respect to the measure, not necessarily a Lebesgue density.
References: Sato (1999), Chapters 8, 11 and 25.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from spxa.core._immutable import Immutable, ImmutableMeta


class LevyMeasure(Immutable, abc.ABC, metaclass=ImmutableMeta):
    """Abstract Lévy measure on the punctured real line."""

    @abc.abstractmethod
    def density(self, x: ArrayLike) -> np.ndarray:
        """Return the Lebesgue density, or raise for atomic/singular measures."""

    @abc.abstractmethod
    def total_mass(self) -> float:
        """Return total jump intensity, possibly infinite."""

    @abc.abstractmethod
    def tail(self, x: float) -> float:
        """Return the strict right tail nu((x, infinity)), for positive x."""

    def integrate(
        self, f: Callable[[float], float], lower: float = -np.inf, upper: float = np.inf
    ) -> float:
        """Integrate a real function against nu over [lower, upper].

        Density-backed measures use adaptive quadrature, split at zero and
        the truncation boundaries. Atomic subclasses override this method.
        Integrands must cancel the singularity sufficiently for convergence.
        """
        from scipy.integrate import quad

        if lower > upper:
            raise ValueError("Integration bounds must be ordered")
        points = [lower, *(x for x in (-1.0, 0.0, 1.0) if lower < x < upper), upper]

        def integrand(x: float) -> float:
            value = f(x)
            if value == 0 or x == 0:
                return 0.0
            return float(value * self.density(np.array([x]))[0])

        return float(
            sum(
                quad(integrand, a, b, epsabs=1e-9, limit=300)[0]
                for a, b in zip(points[:-1], points[1:], strict=True)
                if a < b
            )
        )

    def moment(self, n: int) -> float:
        """Return integral x**n nu(dx), only when the integral exists."""
        if n < 1:
            raise ValueError("Moment order must be positive")
        return self.integrate(lambda x: x**n)

    def is_finite_activity(self) -> bool:
        """Whether the total jump intensity is finite."""
        return bool(np.isfinite(self.total_mass()))

    def __add__(self, other: LevyMeasure) -> LevyMeasure:
        """Sum of independent jump measures (Sato, Proposition 11.10)."""
        return CompoundLevyMeasure([self, other])

    def _levy_integral(self, u: complex) -> complex:
        """Compensated LK jump integral; weighted tails avoid oscillatory truncation."""
        from scipy.integrate import quad

        def kernel(x: float) -> complex:
            z = 1j * u * x
            if abs(x) <= 1:
                return (
                    complex(z * z * (0.5 + z * (1 / 6 + z * (1 / 24 + z / 120))))
                    if abs(z) < 1e-3
                    else complex(np.expm1(z) - z)
                )
            return complex(np.expm1(z))

        if self.total_mass() == 0:
            return 0j
        try:
            self.density(np.array([1.0]))
        except NotImplementedError:
            return complex(
                self.integrate(lambda x: kernel(x).real), self.integrate(lambda x: kernel(x).imag)
            )
        if u.imag != 0:
            return complex(
                self.integrate(lambda x: kernel(x).real), self.integrate(lambda x: kernel(x).imag)
            )
        small = complex(
            self.integrate(lambda x: kernel(x).real, -1, 1),
            self.integrate(lambda x: kernel(x).imag, -1, 1),
        )
        tail = 0j
        for sign in (-1.0, 1.0):

            def density(x: float, sign: float = sign) -> float:
                return float(self.density(np.array([sign * x]))[0])

            mass = quad(density, 1, np.inf, epsabs=1e-10, limit=300)[0]
            real = (
                quad(density, 1, np.inf, weight="cos", wvar=float(u.real), epsabs=1e-9, limlst=300)[
                    0
                ]
                - mass
            )
            imag = (
                sign
                * quad(
                    density, 1, np.inf, weight="sin", wvar=float(u.real), epsabs=1e-9, limlst=300
                )[0]
            )
            tail += complex(real, imag)
        return small + tail

    def _tail_moment_finite(self, n: int) -> bool | None:
        """Known tail-moment eligibility, or None if a generic measure cannot decide."""
        return None


@dataclass(frozen=True)
class CompoundLevyMeasure(LevyMeasure):
    """Sum of component measures; atoms are preserved during integration."""

    components: Sequence[LevyMeasure] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))

    def density(self, x: ArrayLike) -> np.ndarray:
        return sum(
            (c.density(x) for c in self.components), np.zeros_like(np.asarray(x, dtype=float))
        )

    def total_mass(self) -> float:
        return float(sum(c.total_mass() for c in self.components))

    def tail(self, x: float) -> float:
        return float(sum(c.tail(x) for c in self.components))

    def integrate(
        self, f: Callable[[float], float], lower: float = -np.inf, upper: float = np.inf
    ) -> float:
        return float(sum(c.integrate(f, lower, upper) for c in self.components))

    def moment(self, n: int) -> float:
        return float(sum(c.moment(n) for c in self.components))

    def __add__(self, other: LevyMeasure) -> CompoundLevyMeasure:
        components = other.components if isinstance(other, CompoundLevyMeasure) else (other,)
        return CompoundLevyMeasure((*self.components, *components))

    def _levy_integral(self, u: complex) -> complex:
        return sum((m._levy_integral(u) for m in self.components), 0j)

    def _tail_moment_finite(self, n: int) -> bool | None:
        values = [m._tail_moment_finite(n) for m in self.components]
        return False if False in values else True if all(v is True for v in values) else None


@dataclass(frozen=True)
class DensityLevyMeasure(LevyMeasure):
    """Measure specified by its density, optional tail and exact moments."""

    _density_fn: Callable[[np.ndarray], np.ndarray]
    _total_mass: float
    _tail_fn: Callable[[float], float] | None = None
    _moment_fns: Mapping[int, Callable[[], float]] = field(default_factory=dict)
    _tail_moment_test: Callable[[int], bool] | None = None
    _minimum_moment_order: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "_moment_fns", MappingProxyType(dict(self._moment_fns)))

    def density(self, x: ArrayLike) -> np.ndarray:
        return np.asarray(self._density_fn(np.asarray(x, dtype=float)))

    def total_mass(self) -> float:
        return self._total_mass

    def tail(self, x: float) -> float:
        if x <= 0:
            raise ValueError("tail requires a positive threshold")
        if self._tail_fn is not None:
            return float(self._tail_fn(x))
        return self.integrate(lambda y: float(y > x), x, np.inf)

    def moment(self, n: int) -> float:
        if n < self._minimum_moment_order or self._tail_moment_finite(n) is False:
            raise ValueError("Requested Levy-measure moment does not exist")
        if n in self._moment_fns:
            return float(self._moment_fns[n]())
        return super().moment(n)

    def _tail_moment_finite(self, n: int) -> bool | None:
        return self._tail_moment_test(n) if self._tail_moment_test is not None else None


@dataclass(frozen=True)
class ScaledLevyMeasure(LevyMeasure):
    """Image of a measure by x -> scale*x, including reflection (Sato, Ch. 11)."""

    base: LevyMeasure
    scale: float
    _truncation_correction: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale == 0:
            raise ValueError("Scale must be finite and nonzero")
        lo, hi = sorted((1.0, 1 / abs(self.scale)))

        def correction(x: float) -> float:
            return self.scale * x * (int(abs(self.scale * x) <= 1) - int(abs(x) <= 1))

        value = (
            0.0
            if lo == hi
            else (
                self.base.integrate(correction, lo, hi) + self.base.integrate(correction, -hi, -lo)
            )
        )
        object.__setattr__(self, "_truncation_correction", float(value))

    def density(self, x: ArrayLike) -> np.ndarray:
        return np.asarray(
            self.base.density(np.asarray(x, dtype=float) / self.scale) / abs(self.scale)
        )

    def total_mass(self) -> float:
        return self.base.total_mass()

    def integrate(
        self, f: Callable[[float], float], lower: float = -np.inf, upper: float = np.inf
    ) -> float:
        if lower > upper:
            raise ValueError("Integration bounds must be ordered")
        if lower == upper:
            return self.base.integrate(
                lambda x: f(self.scale * x), lower / self.scale, upper / self.scale
            )
        points = [lower, *(x for x in (-1.0, 0.0, 1.0) if lower < x < upper), upper]
        total = 0.0
        for left, right in zip(points[:-1], points[1:], strict=True):
            if left == right:
                continue
            a, b = sorted((left / self.scale, right / self.scale))
            total += self.base.integrate(lambda x: f(self.scale * x), a, b)
        # Splitting closed intervals would count atoms at internal breakpoints
        # twice; integrate singletons once and remove their repeated contribution.
        for point in points[1:-1]:
            total -= self.base.integrate(
                lambda x: f(self.scale * x), point / self.scale, point / self.scale
            )
        return float(total)

    def tail(self, x: float) -> float:
        if x <= 0:
            raise ValueError("tail requires a positive threshold")
        if self.scale > 0:
            return self.base.tail(x / self.scale)
        return self.base.integrate(lambda y: float(self.scale * y > x), -np.inf, x / self.scale)

    def moment(self, n: int) -> float:
        return float(self.scale**n * self.base.moment(n))

    def _tail_moment_finite(self, n: int) -> bool | None:
        return self.base._tail_moment_finite(n)

    def _levy_integral(self, u: complex) -> complex:
        """Use the original integral; do not rediscover atoms/narrow peaks from a density."""
        return self.base._levy_integral(self.scale * u) - 1j * u * self._truncation_correction
