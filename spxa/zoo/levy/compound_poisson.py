r"""
Compound Poisson process.

A pure-jump Lévy process with jump arrivals governed by a homogeneous Poisson
process and independent, identically distributed jump sizes.

Formal specification:
  X_t = \sum_{i=1}^{N_t} Y_i

where N_t ~ Poisson(λt) is a Poisson counter with rate λ > 0, and {Y_i} are
i.i.d. jump sizes drawn from probability law F_Y.

Lévy measure:
  ν(dx) = λ · F_Y(dx)
with finite total mass ν(ℝ \ {0}) = λ P(Y≠0) (finite activity). Zero-size events are excluded.

Characteristic exponent:
  ψ(u) = λ · (E[e^{iuY}] - 1)
yielding characteristic function:
  E[e^{iuX_t}] = exp(λt · (φ_Y(u) - 1))

Exact Lévy–Khintchine triplet (b, σ², ν) under canonical truncation h(x) = x·𝟙_{|x|≤1}:
  b = λ · E[Y · 𝟙_{|Y|≤1}]
  σ² = 0
  ν(dx) = λ · F_Y(dx)

Cumulants:
  κ_n(X_t) = λ · t · E[Y^n]  for all n ≥ 1

References
----------
Kingman, J.F.C. (1993). *Poisson Processes*. Oxford University Press.
Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapman & Hall/CRC. Section 3.4.
Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*.
Cambridge University Press. Example 8.4.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats

from spxa.core._immutable import Immutable, ImmutableMeta
from spxa.core._validation import simulation_grid, validate_order, validate_time
from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import LevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.sim.exact import sample_compound_poisson
from spxa.story.narrator import CompositionNode, NodeKind


class JumpDistribution(Immutable, abc.ABC, metaclass=ImmutableMeta):
    r"""
    Abstract base class for jump size distributions in compound Poisson processes.

    Subclasses define the probability distribution of individual jump shocks Y,
    providing closed-form characteristic functions, moments, tails, and truncation
    integrals whenever possible.
    """

    @abc.abstractmethod
    def char_func(self, u: complex | float | np.ndarray) -> np.ndarray:
        r"""Characteristic function of the jump size: \phi_Y(u) = E[e^{iuY}]."""

    @abc.abstractmethod
    def moment(self, n: int) -> float:
        r"""Raw moment of jump size: E[Y^n] for n \ge 1."""

    @abc.abstractmethod
    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        """Draw independent random jump sizes."""

    @abc.abstractmethod
    def truncation_mean(self, cutoff: float = 1.0) -> float:
        r"""Truncation expectation: E[Y · 𝟙_{|Y| \le cutoff}]."""

    def density(self, x: ArrayLike) -> np.ndarray:
        """Probability density f_Y(x), or raise NotImplementedError for discrete laws."""
        raise NotImplementedError(
            f"{type(self).__name__} does not have a continuous probability density."
        )

    def tail(self, x: float) -> float:
        r"""Right tail probability P(Y > x) for x > 0."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement analytical tail evaluation."
        )

    @property
    def is_positive(self) -> bool:
        """Return True if jump support is nonnegative (yielding a subordinator)."""
        return False

    @property
    def has_finite_mean(self) -> bool:
        """Return True if E[|Y|] < inf."""
        return True

    @property
    def has_finite_variance(self) -> bool:
        """Return True if Var(Y) < inf."""
        return True

    @property
    def tail_index(self) -> float | None:
        """Heavy-tail index alpha if P(Y > x) ~ x^{-alpha}, else None."""
        return None

    def _cf_minus_one(self, u: complex | float | np.ndarray) -> np.ndarray:
        """Stable increment of the jump CF; custom laws may override the fallback."""
        return np.asarray(self.char_func(u) - 1)


class NormalJump(JumpDistribution):
    r"""
    Gaussian jump distribution: Y ~ N(\mu, \sigma^2).

    Parameters
    ----------
    mean :
        Mean jump size \mu. Default 0.0.
    std :
        Jump size standard deviation \sigma > 0. Default 1.0.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        if not np.isfinite(mean):
            raise ValueError("mean must be finite")
        self.mean = float(mean)
        self.std = float(std)

    def char_func(self, u: complex | float | np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=complex)
        if not np.all(np.isfinite(u_arr)):
            raise ValueError("Frequencies must be finite")
        return np.exp(1j * self.mean * u_arr - 0.5 * (self.std**2) * (u_arr**2))

    def moment(self, n: int) -> float:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("Moment order must be >= 1")
        # E[Y^n] for Y = mu + sigma*Z, Z ~ N(0,1)
        res = 0.0
        for k in range(0, n + 1, 2):
            double_fact = 1 if k == 0 else math.prod(range(1, k, 2))
            res += math.comb(n, k) * (self.mean ** (n - k)) * (self.std**k) * double_fact
        return float(res)

    def truncation_mean(self, cutoff: float = 1.0) -> float:
        c = float(cutoff)
        if np.isnan(c):
            raise ValueError("cutoff must not be NaN")
        if c <= 0:
            return 0.0
        z1 = (-c - self.mean) / self.std
        z2 = (c - self.mean) / self.std
        phi1 = stats.norm.pdf(z1)
        phi2 = stats.norm.pdf(z2)
        cdf1 = special.ndtr(z1)
        cdf2 = special.ndtr(z2)
        return float(self.mean * (cdf2 - cdf1) - self.std * (phi2 - phi1))

    def density(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return np.asarray(stats.norm.pdf(x_arr, loc=self.mean, scale=self.std))

    def tail(self, x: float) -> float:
        # P(Y > x) = 1 - Phi((x - mu) / sigma) = Phi((mu - x) / sigma)
        return float(special.ndtr((self.mean - x) / self.std))

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        return rng.normal(loc=self.mean, scale=self.std, size=size)

    @property
    def is_positive(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"NormalJump(mean={self.mean}, std={self.std})"

    def _cf_minus_one(self, u: complex | float | np.ndarray) -> np.ndarray:
        self.char_func(u)  # Validate the same frequency domain as the public method.
        z = np.asarray(u, dtype=complex)
        return np.asarray(np.expm1(1j * self.mean * z - 0.5 * self.std**2 * z * z))


class ExponentialJump(JumpDistribution):
    r"""
    Exponential jump distribution: Y ~ Exp(\beta) with density \beta e^{-\beta x} 𝟙_{x>0}.

    Parameters
    ----------
    rate :
        Rate parameter \beta > 0 (mean jump is 1/\beta). Default 1.0.
    """

    def __init__(self, rate: float = 1.0) -> None:
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = float(rate)

    def char_func(self, u: complex | float | np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=complex)
        if not np.all(np.isfinite(u_arr)):
            raise ValueError("Frequencies must be finite")
        if np.any(u_arr.imag <= -self.rate):
            raise ValueError("Exponential jump MGF requires p < rate")
        return np.asarray(self.rate / (self.rate - 1j * u_arr))

    def moment(self, n: int) -> float:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("Moment order must be >= 1")
        return float(math.factorial(n) / (self.rate**n))

    def truncation_mean(self, cutoff: float = 1.0) -> float:
        c = float(cutoff)
        if c <= 0:
            return 0.0
        # \int_0^c x \beta e^{-\beta x} dx = (1/\beta)[1 - (1 + \beta c) e^{-\beta c}]
        return float((1.0 / self.rate) * (1.0 - (1.0 + self.rate * c) * np.exp(-self.rate * c)))

    def density(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        out = np.zeros_like(x_arr)
        pos = x_arr > 0
        out[pos] = self.rate * np.exp(-self.rate * x_arr[pos])
        return out

    def tail(self, x: float) -> float:
        if x <= 0:
            return 1.0
        return float(np.exp(-self.rate * x))

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        return rng.exponential(scale=1.0 / self.rate, size=size)

    @property
    def is_positive(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"ExponentialJump(rate={self.rate})"

    def _cf_minus_one(self, u: complex | float | np.ndarray) -> np.ndarray:
        value = self.char_func(u)
        return np.asarray(1j * np.asarray(u, dtype=complex) * value / self.rate)


class DoubleExponentialJump(JumpDistribution):
    r"""
    Asymmetric double-exponential (Laplace-like) jump distribution.

    f_Y(y) = p \eta_1 e^{-\eta_1 y} 𝟙_{y > 0} + (1-p) \eta_2 e^{\eta_2 y} 𝟙_{y < 0}

    Parameters
    ----------
    p :
        Probability of positive jump p \in [0, 1]. Default 0.5.
    eta1 :
        Positive jump decay rate \eta_1 > 0. Default 1.0.
    eta2 :
        Negative jump decay rate \eta_2 > 0. Default 1.0.
    """

    def __init__(self, p: float = 0.5, eta1: float = 1.0, eta2: float = 1.0) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        if not np.isfinite(eta1) or eta1 <= 0:
            raise ValueError(f"eta1 must be positive, got {eta1}")
        if not np.isfinite(eta2) or eta2 <= 0:
            raise ValueError(f"eta2 must be positive, got {eta2}")
        self.p = float(p)
        self.eta1 = float(eta1)
        self.eta2 = float(eta2)

    def char_func(self, u: complex | float | np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=complex)
        if not np.all(np.isfinite(u_arr)):
            raise ValueError("Frequencies must be finite")
        if (self.p > 0 and np.any(u_arr.imag <= -self.eta1)) or (
            self.p < 1 and np.any(u_arr.imag >= self.eta2)
        ):
            raise ValueError("Double-exponential jump MGF is outside its finite strip")
        result = np.zeros_like(u_arr)
        if self.p > 0:
            result += self.p * self.eta1 / (self.eta1 - 1j * u_arr)
        if self.p < 1:
            result += (1 - self.p) * self.eta2 / (self.eta2 + 1j * u_arr)
        return result

    def moment(self, n: int) -> float:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("Moment order must be >= 1")
        fact = math.factorial(n)
        pos = self.p / (self.eta1**n)
        neg = ((-1) ** n) * (1.0 - self.p) / (self.eta2**n)
        return float(fact * (pos + neg))

    def truncation_mean(self, cutoff: float = 1.0) -> float:
        c = float(cutoff)
        if c <= 0:
            return 0.0
        pos_part = (self.p / self.eta1) * (1.0 - (1.0 + self.eta1 * c) * np.exp(-self.eta1 * c))
        neg_part = ((1.0 - self.p) / self.eta2) * (
            1.0 - (1.0 + self.eta2 * c) * np.exp(-self.eta2 * c)
        )
        return float(pos_part - neg_part)

    def density(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        out = np.zeros_like(x_arr)
        pos = x_arr > 0
        neg = x_arr < 0
        out[pos] = self.p * self.eta1 * np.exp(-self.eta1 * x_arr[pos])
        out[neg] = (1.0 - self.p) * self.eta2 * np.exp(self.eta2 * x_arr[neg])
        return out

    def tail(self, x: float) -> float:
        if x > 0:
            return float(self.p * np.exp(-self.eta1 * x))
        elif x < 0:
            return float(1.0 - (1.0 - self.p) * np.exp(self.eta2 * x))
        return float(self.p)

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        n = int(np.prod(size)) if isinstance(size, tuple) else int(size)
        is_up = rng.random(n) < self.p
        up = rng.exponential(1.0 / self.eta1, size=n)
        down = -rng.exponential(1.0 / self.eta2, size=n)
        samples = np.where(is_up, up, down)
        return samples.reshape(size) if isinstance(size, tuple) else samples

    @property
    def is_positive(self) -> bool:
        return self.p == 1.0

    def __repr__(self) -> str:
        return f"DoubleExponentialJump(p={self.p}, eta1={self.eta1}, eta2={self.eta2})"

    def _cf_minus_one(self, u: complex | float | np.ndarray) -> np.ndarray:
        self.char_func(u)
        z = np.asarray(u, dtype=complex)
        result = np.zeros_like(z)
        if self.p > 0:
            result += self.p * 1j * z / (self.eta1 - 1j * z)
        if self.p < 1:
            result -= (1 - self.p) * 1j * z / (self.eta2 + 1j * z)
        return result


class UniformJump(JumpDistribution):
    r"""
    Continuous uniform jump distribution: Y ~ Uniform(low, high).

    Parameters
    ----------
    low :
        Lower support bound. Default -1.0.
    high :
        Upper support bound (must be > low). Default 1.0.
    """

    def __init__(self, low: float = -1.0, high: float = 1.0) -> None:
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(f"low must be < high, got low={low}, high={high}")
        self.low = float(low)
        self.high = float(high)

    def char_func(self, u: complex | float | np.ndarray) -> np.ndarray:
        """Centered sinc avoids declaring small nonzero frequencies to be zero."""
        u_arr = np.asarray(u, dtype=complex)
        if not np.all(np.isfinite(u_arr)):
            raise ValueError("Frequencies must be finite")
        width = self.high - self.low
        midpoint = self.low + width / 2
        return np.asarray(np.exp(1j * u_arr * midpoint) * np.sinc(u_arr * width / (2 * np.pi)))

    def moment(self, n: int) -> float:
        validate_order(n)
        half = (self.high - self.low) / 2
        midpoint = self.low + half
        return float(
            sum(
                math.comb(n, k) * midpoint ** (n - k) * half**k / (k + 1)
                for k in range(0, n + 1, 2)
            )
        )

    def truncation_mean(self, cutoff: float = 1.0) -> float:
        c = float(cutoff)
        left_bound = max(self.low, -c)
        right_bound = min(self.high, c)
        if left_bound >= right_bound:
            return 0.0
        diff = self.high - self.low
        return float((right_bound - left_bound) * (right_bound + left_bound) / (2.0 * diff))

    def density(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        out = np.zeros_like(x_arr)
        inside = (x_arr >= self.low) & (x_arr <= self.high)
        out[inside] = 1.0 / (self.high - self.low)
        return out

    def tail(self, x: float) -> float:
        if x <= self.low:
            return 1.0
        if x >= self.high:
            return 0.0
        return float((self.high - x) / (self.high - self.low))

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        return rng.uniform(low=self.low, high=self.high, size=size)

    @property
    def is_positive(self) -> bool:
        return self.low >= 0.0

    def __repr__(self) -> str:
        return f"UniformJump(low={self.low}, high={self.high})"

    def _cf_minus_one(self, u: complex | float | np.ndarray) -> np.ndarray:
        self.char_func(u)
        z = np.asarray(u, dtype=complex)
        half = (self.high - self.low) / 2
        w = z * half
        difference = np.asarray(np.sinc(w / np.pi) - 1)
        small = np.abs(w) < 0.01
        v = w[small]
        difference[small] = -v * v / 6 + v**4 / 120 - v**6 / 5040 + v**8 / 362880
        phase = np.expm1(1j * z * (self.low + half))
        return np.asarray(phase * (1 + difference) + difference)


class DiscreteJump(JumpDistribution):
    """
    Discrete jump size distribution with finite support {v_1, ..., v_k}.

    Parameters
    ----------
    values :
        Sequence of jump size locations.
    probs :
        Optional probability mass vector. If None, assigns uniform probability 1/k.
    """

    def __init__(self, values: Sequence[float], probs: Sequence[float] | None = None) -> None:
        val_arr = np.asarray(values, dtype=float)
        if val_arr.ndim != 1 or not np.all(np.isfinite(val_arr)):
            raise ValueError("values must be a finite one-dimensional sequence")
        if val_arr.size == 0:
            raise ValueError("values cannot be empty")
        if probs is None:
            prob_arr = np.full(val_arr.size, 1.0 / val_arr.size)
        else:
            prob_arr = np.asarray(probs, dtype=float)
            if prob_arr.shape != val_arr.shape:
                raise ValueError("probs and values must have the same length")
            if not np.all(np.isfinite(prob_arr)) or np.any(prob_arr < 0):
                raise ValueError("probabilities must be non-negative")
            total = float(prob_arr.sum())
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(f"probabilities must sum to 1.0, got {total}")
            prob_arr = prob_arr / total

        self.values = val_arr
        self.probs = prob_arr

    def char_func(self, u: complex | float | np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=complex)
        if not np.all(np.isfinite(u_arr)):
            raise ValueError("Frequencies must be finite")
        # sum p_k * exp(i u v_k)
        # shape broadcasting: u_arr.shape + (len(values),)
        u_expanded = np.expand_dims(u_arr, axis=-1)
        phase = np.exp(1j * u_expanded * self.values)
        return np.asarray(np.sum(phase * self.probs, axis=-1))

    def moment(self, n: int) -> float:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("Moment order must be >= 1")
        return float(np.sum(self.probs * (self.values**n)))

    def truncation_mean(self, cutoff: float = 1.0) -> float:
        c = float(cutoff)
        mask = np.abs(self.values) <= c
        return float(np.sum(self.probs[mask] * self.values[mask]))

    def density(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError(
            "Discrete jump distributions have discrete point masses and no continuous density."
        )

    def tail(self, x: float) -> float:
        mask = self.values > x
        return float(np.sum(self.probs[mask]))

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        return rng.choice(self.values, p=self.probs, size=size)

    @property
    def is_positive(self) -> bool:
        return bool(np.all(self.values[self.probs > 0] >= 0.0))

    def __repr__(self) -> str:
        vals = list(np.round(self.values, 4))
        probs = list(np.round(self.probs, 4))
        return f"DiscreteJump(values={vals}, probs={probs})"

    def _cf_minus_one(self, u: complex | float | np.ndarray) -> np.ndarray:
        self.char_func(u)
        z = np.asarray(u, dtype=complex)
        return np.asarray(
            np.sum(self.probs * np.expm1(1j * np.expand_dims(z, -1) * self.values), axis=-1)
        )


class _CompoundPoissonLevyMeasure(LevyMeasure):
    r"""
    Lévy measure of a Compound Poisson process: ν(dx) = λ · F_Y(dx).
    """

    def __init__(self, rate: float, jump_dist: JumpDistribution) -> None:
        self.rate = float(rate)
        self.jump_dist = jump_dist

    def density(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return self.rate * self.jump_dist.density(x_arr)

    def total_mass(self) -> float:
        if isinstance(self.jump_dist, DiscreteJump):
            return float(self.rate * self.jump_dist.probs[self.jump_dist.values != 0].sum())
        return self.rate

    def tail(self, x: float) -> float:
        if isinstance(self.jump_dist, DiscreteJump):
            keep = (self.jump_dist.values > x) & (self.jump_dist.values != 0)
            return float(self.rate * self.jump_dist.probs[keep].sum())
        return self.rate * self.jump_dist.tail(x)

    def moment(self, n: int) -> float:
        return self.rate * self.jump_dist.moment(n)

    def is_finite_activity(self) -> bool:
        return True

    def integrate(
        self, f: Callable[[float], float], lower: float = -np.inf, upper: float = np.inf
    ) -> float:
        """Preserve atoms; integrate continuous built-ins in probability space.

        CDF/PPF and SF/ISF halves avoid missing a narrow shifted density or
        cancelling a small right-tail probability. Bounds are closed for atoms.
        Zero-size events are not part of a Levy measure on the punctured line.
        """
        from scipy.integrate import quad

        if lower > upper:
            raise ValueError("Integration bounds must be ordered")
        jump = self.jump_dist
        if isinstance(jump, DiscreteJump):
            mask = (jump.values >= lower) & (jump.values <= upper) & (jump.values != 0)
            return float(
                self.rate
                * sum(
                    p * f(float(x))
                    for x, p in zip(jump.values[mask], jump.probs[mask], strict=True)
                )
            )

        def continuous(distribution: Any, sign: float = 1.0) -> float:
            a, b = sorted((sign * lower, sign * upper))
            total = 0.0
            for tail in (False, True):
                probability = distribution.sf if tail else distribution.cdf
                quantile = distribution.isf if tail else distribution.ppf
                qlo, qhi = sorted((float(probability(a)), float(probability(b))))
                qhi = min(qhi, 0.5)
                if qlo >= qhi:
                    continue
                points = sorted(
                    {
                        qlo,
                        qhi,
                        *(
                            float(probability(sign * x))
                            for x in (-1, 0, 1)
                            if qlo < float(probability(sign * x)) < qhi
                        ),
                    }
                )
                total += sum(
                    quad(
                        lambda q, quantile=quantile: f(sign * float(quantile(q))),
                        x,
                        y,
                        epsabs=1e-10,
                        limit=300,
                    )[0]
                    for x, y in zip(points[:-1], points[1:], strict=True)
                )
            return float(total)

        if isinstance(jump, NormalJump):
            result = continuous(stats.norm(loc=jump.mean, scale=jump.std))
        elif isinstance(jump, ExponentialJump):
            result = continuous(stats.expon(scale=1 / jump.rate))
        elif isinstance(jump, UniformJump):
            result = continuous(stats.uniform(loc=jump.low, scale=jump.high - jump.low))
        elif isinstance(jump, DoubleExponentialJump):
            result = jump.p * continuous(stats.expon(scale=1 / jump.eta1)) + (
                1 - jump.p
            ) * continuous(stats.expon(scale=1 / jump.eta2), -1)
        else:
            return super().integrate(f, lower, upper)
        return float(self.rate * result)

    def _levy_integral(self, u: complex) -> complex:
        return self.rate * (
            complex(self.jump_dist._cf_minus_one(u)) - 1j * u * self.jump_dist.truncation_mean()
        )

    def _tail_moment_finite(self, n: int) -> bool | None:
        if n == 1:
            return self.jump_dist.has_finite_mean
        if n == 2:
            return self.jump_dist.has_finite_variance
        if isinstance(
            self.jump_dist,
            (NormalJump, ExponentialJump, DoubleExponentialJump, UniformJump, DiscreteJump),
        ):
            return True
        return None


class CompoundPoisson(Process):
    r"""
    Compound Poisson process with arrival rate λ and jump distribution F_Y.

    X_t = \sum_{i=1}^{N_t} Y_i

    where N_t ~ Poisson(λt) is a homogeneous Poisson process and {Y_i} are
    independent, identically distributed jump sizes.

    Parameters
    ----------
    rate :
        Jump arrival rate λ > 0. Expected number of jumps per unit time.
    jump_dist :
        Jump distribution representation. Accepts:
        - An instance of `JumpDistribution` (`NormalJump`, `ExponentialJump`,
          `DoubleExponentialJump`, `UniformJump`, `DiscreteJump`, etc.)
        - A distribution name string: "normal" / "gaussian", "exponential",
          "double_exponential" / "kou", "uniform"
        - A single number (float/int) for deterministic jump sizes of magnitude c
        - Default: `NormalJump(mean=0.0, std=1.0)`
    **kwargs :
        Additional keyword arguments forwarded to the jump distribution constructor
        when `jump_dist` is passed as a string name.

    Notes
    -----
    Lévy–Khintchine triplet (b, σ², ν):
      b = λ · E[Y · 𝟙_{|Y|≤1}]
      σ² = 0
      ν(dx) = λ · F_Y(dx)

    Cumulants:
      κ_n(X_t) = λ · t · E[Y^n]

    Characteristic function:
      φ_{X_t}(u) = exp(λt · (φ_Y(u) - 1))

    References
    ----------
    Kingman, J.F.C. (1993). *Poisson Processes*. Oxford University Press.
    Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
    Chapman & Hall/CRC. Section 3.4.
    Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*.
    Cambridge University Press. Example 8.4.
    """

    def __init__(
        self,
        rate: float,
        jump_dist: JumpDistribution | str | float | int | None = None,
        **kwargs: Any,
    ) -> None:
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = float(rate)

        # Parse jump distribution
        if jump_dist is None:
            self.jump_dist: JumpDistribution = NormalJump(**kwargs)
        elif isinstance(jump_dist, JumpDistribution):
            if kwargs:
                raise TypeError("JumpDistribution instances do not accept extra parameters")
            self.jump_dist = jump_dist
        elif isinstance(jump_dist, (int, float)):
            if kwargs:
                raise TypeError("A constant jump does not accept extra parameters")
            self.jump_dist = DiscreteJump(values=[float(jump_dist)], probs=[1.0])
        elif isinstance(jump_dist, str):
            name = jump_dist.lower()
            if name in ("normal", "gaussian"):
                self.jump_dist = NormalJump(**kwargs)
            elif name == "exponential":
                if "jump_rate" in kwargs and "beta" in kwargs:
                    raise TypeError("Specify only one of jump_rate and beta")
                jump_r = kwargs.pop("jump_rate", kwargs.pop("beta", 1.0))
                self.jump_dist = ExponentialJump(rate=jump_r, **kwargs)
            elif name in ("double_exponential", "kou"):
                self.jump_dist = DoubleExponentialJump(**kwargs)
            elif name == "uniform":
                self.jump_dist = UniformJump(**kwargs)
            else:
                raise ValueError(
                    f"Unknown jump distribution name: '{jump_dist}'. "
                    "Use one of: 'normal', 'exponential', 'double_exponential', 'uniform', "
                    "or pass a JumpDistribution instance."
                )
        else:
            raise TypeError(
                f"Unsupported jump_dist type: {type(jump_dist)}. "
                "Expected JumpDistribution instance, distribution name string, or float."
            )

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CompoundPoisson(rate={self.rate}, jump_dist={self.jump_dist!r})",
            math_note=(
                f"Compound Poisson with λ={self.rate}, jump law {self.jump_dist!r}; "
                f"triplet (b={self._triplet_drift():.4f}, σ²=0, ν=λ·F_Y)"
            ),
            reference="Kingman (1993); Cont & Tankov (2004) Sec. 3.4; Sato (1999) Ex. 8.4",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet_drift(self) -> float:
        return self.rate * self.jump_dist.truncation_mean(cutoff=1.0)

    def _triplet(self) -> LevyTriplet:
        nu = _CompoundPoissonLevyMeasure(self.rate, self.jump_dist)
        b = self._triplet_drift()
        return LevyTriplet(b=b, sigma_sq=0.0, nu=nu)

    def _properties(self) -> ProcessProperties:
        # A compound Poisson process is a martingale iff E[X_1] = 0 <=> E[Y] = 0
        mean_jump = self.jump_dist.moment(1) if self.jump_dist.has_finite_mean else np.nan
        is_martingale = np.isclose(mean_jump, 0.0, atol=1e-12)

        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=bool(is_martingale),
            has_finite_variance=self.jump_dist.has_finite_variance,
            has_finite_mean=self.jump_dist.has_finite_mean,
            self_similarity_index=None,
            tail_index=self.jump_dist.tail_index,
            is_subordinator=self.jump_dist.is_positive,
        )

    def char_func_exact(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        validate_time(t)
        if t == 0:
            return np.ones_like(np.asarray(u), dtype=complex)
        return np.asarray(np.exp(t * self.char_exponent(u)))

    def char_func(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def cumulant_exact(self, n: int, t: float = 1.0) -> float:
        validate_order(n)
        validate_time(t)
        if t == 0:
            return 0.0
        if self.triplet.nu._tail_moment_finite(n) is False:
            raise ValueError("Requested jump moment does not exist")
        value = float(self.rate * t * self.jump_dist.moment(n))
        if not np.isfinite(value):
            raise ValueError("Requested jump cumulant is not finite")
        return value

    def cumulants(self, order: int) -> dict[int, float]:
        validate_order(order)
        return {n: self.cumulant_exact(n) for n in range(1, order + 1)}

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,  # noqa: N803
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Exact simulation via Poisson increment sampling and jump compounding.

        Parameters
        ----------
        n_steps :
            Number of time steps.
        n_paths :
            Number of independent sample paths.
        T :
            Terminal horizon.
        rng :
            Random number generator instance.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). paths[:, 0] == 0.0.
        """
        rng = rng or np.random.default_rng()
        dt = simulation_grid(n_steps, n_paths, T=T)
        increments = sample_compound_poisson(
            rate=self.rate,
            jump_sampler=lambda g, sz: self.jump_dist.sample(g, sz),
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=rng,
        )
        paths = np.zeros((n_paths, n_steps + 1), dtype=float)
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"CompoundPoisson(rate={self.rate}, jump_dist={self.jump_dist!r})"

    def char_exponent(self, u: complex | float | np.ndarray) -> np.ndarray:
        """Continuous unit-time exponent; do not take a principal logarithm."""
        return np.asarray(self.rate * self.jump_dist._cf_minus_one(u))
