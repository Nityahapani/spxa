"""Marginal distribution adapters and explicitly approximate Fourier density grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from scipy.integrate import trapezoid

from spxa.core.process import Process


@dataclass
class Marginal:
    """Known scalar marginal; lattice scale converts a count to a jump size."""

    distribution: Any
    discrete: bool = False
    scale: float = 1.0
    atom: float | None = None

    def density(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x)
        if self.atom is not None:
            return np.asarray(x == self.atom, dtype=float)
        if self.discrete:
            points = x / self.scale
            return np.where(
                np.isclose(points, np.round(points), atol=1e-10, rtol=0),
                self.distribution.pmf(np.round(points)),
                0.0,
            )
        return np.asarray(self.distribution.pdf(x))

    def quantile(self, p: np.ndarray | float) -> np.ndarray:
        p = np.asarray(p)
        if self.atom is not None:
            return np.full_like(p, self.atom, dtype=float)
        if self.discrete and self.scale < 0:
            return np.asarray(self.scale * self.distribution.ppf(1 - p))
        return np.asarray(self.scale * self.distribution.ppf(p))

    def support_grid(self) -> np.ndarray:
        if self.atom is not None:
            return np.array([self.atom])
        upper = int(self.distribution.ppf(1 - 1e-13))
        if upper > 1_000_000:
            raise ValueError("Discrete divergence support exceeds numerical budget")
        return self.scale * np.arange(upper + 1, dtype=float)


def known_marginal(process: Process, t: float) -> Marginal | None:
    """Return a SciPy reference distribution where a matching parameterization exists."""
    from spxa.zoo.beyond.cev import CEVProcess
    from spxa.zoo.beyond.cir import CIRProcess
    from spxa.zoo.beyond.heston import HestonProcess
    from spxa.zoo.beyond.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess
    from spxa.zoo.levy.brownian import BrownianMotion
    from spxa.zoo.levy.compound_poisson import CompoundPoisson, DiscreteJump
    from spxa.zoo.levy.gamma import GammaProcess
    from spxa.zoo.levy.inverse_gaussian import InverseGaussianProcess
    from spxa.zoo.levy.merton import MertonJumpDiffusion
    from spxa.zoo.levy.negative_binomial import NegativeBinomialProcess
    from spxa.zoo.levy.nig import NIG
    from spxa.zoo.levy.poisson import PoissonProcess
    from spxa.zoo.levy.stable import AlphaStable

    if not np.isfinite(t) or t < 0:
        raise ValueError("Time must be finite and nonnegative")
    if t == 0:
        return (
            Marginal(
                None,
                True,
                atom=float(
                    process.r0
                    if isinstance(process, CIRProcess)
                    else process.x0
                    if isinstance(process, OrnsteinUhlenbeckProcess)
                    else process.s0
                    if isinstance(process, CEVProcess)
                    else 0.0
                ),
            )
            if isinstance(
                process,
                (
                    BrownianMotion,
                    GammaProcess,
                    InverseGaussianProcess,
                    NIG,
                    PoissonProcess,
                    NegativeBinomialProcess,
                    AlphaStable,
                    CIRProcess,
                    OrnsteinUhlenbeckProcess,
                    HestonProcess,
                    CEVProcess,
                    CompoundPoisson,
                    MertonJumpDiffusion,
                ),
            )
            else None
        )
    if isinstance(process, OrnsteinUhlenbeckProcess):
        return Marginal(
            stats.norm(
                loc=process.conditional_mean(t), scale=np.sqrt(process.conditional_variance(t))
            )
        )
    if isinstance(process, CIRProcess):
        scale = process.sigma**2 * (-np.expm1(-process.kappa * t)) / (4 * process.kappa)
        return Marginal(
            stats.ncx2(
                df=process.degrees_of_freedom,
                nc=process.r0 * np.exp(-process.kappa * t) / scale,
                scale=scale,
            )
        )
    if isinstance(process, CompoundPoisson) and isinstance(process.jump_dist, DiscreteJump):
        jump = process.jump_dist
        values = np.unique(jump.values[(jump.probs > 0) & (jump.values != 0)])
        if len(values) == 0:
            return Marginal(None, True, atom=0.0)
        if len(values) == 1:
            probability = float(jump.probs[jump.values == values[0]].sum())
            return Marginal(stats.poisson(process.rate * t * probability), True, float(values[0]))
    if isinstance(process, BrownianMotion):
        if process.sigma == 0:
            return Marginal(None, True, atom=process.mu * t)
        return Marginal(stats.norm(loc=process.mu * t, scale=process.sigma * np.sqrt(t)))
    if isinstance(process, GammaProcess):
        return Marginal(stats.gamma(a=process.a * t, scale=1 / process.b))
    if isinstance(process, InverseGaussianProcess):
        return Marginal(
            stats.invgauss(mu=process.mu / (process.lam * t), scale=process.lam * t * t)
        )
    if isinstance(process, NIG):
        return Marginal(
            stats.norminvgauss(
                a=process.alpha * process.delta * t,
                b=process.beta * process.delta * t,
                loc=process.mu * t,
                scale=process.delta * t,
            )
        )
    if isinstance(process, PoissonProcess):
        return (
            Marginal(stats.poisson(process.rate * t), True, process.jump_size)
            if process.jump_size
            else Marginal(None, True, atom=0.0)
        )
    if isinstance(process, NegativeBinomialProcess):
        return Marginal(stats.nbinom(process.r * t, 1 - process.p), True)
    if isinstance(process, AlphaStable):
        from copy import deepcopy

        reference = deepcopy(stats.levy_stable)
        reference.parameterization = "S1"
        return Marginal(
            reference(
                process.alpha,
                process.beta,
                loc=process.mu * t,
                scale=process.sigma * t ** (1 / process.alpha),
            )
        )
    return None


def density_grid(
    process: Process, t: float, x: np.ndarray, u_max: float, n_points: int
) -> np.ndarray:
    """Fourier-invert on a supplied grid; reject unresolved/atomic marginals.

    This is a truncated numerical inversion, not an exact distribution adapter.
    A finite-activity pure-jump marginal has an atom and must not be treated as
    an ordinary density. Known discrete families have separate adapters.
    """
    known = known_marginal(process, t)
    if known is not None:
        if known.discrete:
            raise ValueError("A discrete marginal has no Lebesgue density")
        return known.density(x)
    if process.exactness.name == "EXACT":
        triplet = process.triplet
        if triplet.sigma_sq == 0 and triplet.nu.is_finite_activity():
            raise ValueError("An atomic/continuous mixture requires a dedicated marginal adapter")
    u = np.linspace(-u_max, u_max, n_points)
    cf = process.char_func(u, t)
    density = np.empty(len(x))
    for start in range(0, len(x), 128):
        values = (np.exp(-1j * x[start : start + 128, None] * u) * cf).real
        density[start : start + 128] = trapezoid(values, u, axis=1) / (2 * np.pi)
    if np.min(density) < -1e-3 * max(float(np.max(density)), 1.0):
        raise ValueError("Fourier density unresolved; increase u_max/n_points")
    return np.maximum(density, 0.0)


def common_grid(X: Process, Y: Process, t: float, n_points: int) -> np.ndarray:
    """Moment-scaled spatial range; unlike a fixed domain this is translation/scale aware."""
    if n_points < 128:
        raise ValueError("At least 128 density points required")
    limits = []
    for process in (X, Y):
        k = process.cumulants_at(2, t)
        if not np.isfinite(k[1]) or not np.isfinite(k[2]) or k[2] <= 0:
            raise ValueError("Numerical density inversion needs a finite positive variance")
        limits.extend([k[1] - 12 * np.sqrt(k[2]), k[1] + 12 * np.sqrt(k[2])])
    return np.linspace(min(limits), max(limits), n_points)
