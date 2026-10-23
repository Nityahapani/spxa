"""Marginal divergences, with atomic laws kept distinct from continuous densities.

Hellinger uses square-root density/PMF overlap, not products of characteristic
functions. Generic Fourier inversions are numerical approximations and reject
unresolved or mixed marginals instead of silently inventing a density.
References: Cont & Tankov (2004), Section 4.6; Villani (2003), Chapter 2.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad, trapezoid

from spxa.analytics._marginals import common_grid, density_grid, known_marginal
from spxa.core.process import Process


def l2_char_func_distance(
    X: Process, Y: Process, t: float = 1.0, u_max: float = 50.0, n_points: int = 2000
) -> float:
    """Truncated squared L2 characteristic-function distance (Plancherel identity)."""
    if u_max <= 0 or n_points < 2 or not np.isfinite(t) or t < 0:
        raise ValueError("Positive frequency range/resolution and nonnegative time required")
    u = np.linspace(-u_max, u_max, n_points)
    return float(trapezoid(np.abs(X.char_func(u, t) - Y.char_func(u, t)) ** 2, u) / (2 * np.pi))


def hellinger_distance(
    X: Process, Y: Process, t: float = 1.0, u_max: float = 50.0, n_points: int = 2000
) -> float:
    """Squared Hellinger distance H²=1-integral sqrt(p*q).

    Known marginals use SciPy PDF/PMF quadrature (discrete tails truncated at
    probability 1e-13). Pure discrete and continuous laws are singular to one
    another. Other continuous laws use moment-scaled, truncated Fourier grids;
    u_max/n_points control that approximation. Mixed atomic/continuous laws
    without an adapter raise ValueError. No CF-product surrogate is used.

    References
    ----------
    Cont & Tankov (2004), Section 4.6 (Hellinger affinity).
    """
    from spxa.zoo.levy.brownian import BrownianMotion

    if u_max <= 0 or n_points < 2:
        raise ValueError("Positive frequency range and at least two points required")
    a, b = known_marginal(X, t), known_marginal(Y, t)
    if X is Y:
        return 0.0
    if (
        isinstance(X, BrownianMotion)
        and isinstance(Y, BrownianMotion)
        and X.sigma > 0
        and Y.sigma > 0
        and t > 0
    ):
        variance = (X.sigma**2 + Y.sigma**2) * t
        affinity = np.sqrt(2 * X.sigma * Y.sigma * t / variance) * np.exp(
            -((X.mu - Y.mu) ** 2) * t * t / (4 * variance)
        )
        return float(max(0.0, 1 - affinity))
    if a is not None and b is not None:
        if a.discrete != b.discrete:
            return 1.0
        if a.discrete:
            grid = np.union1d(a.support_grid(), b.support_grid())
            overlap = float(np.sum(np.sqrt(a.density(grid) * b.density(grid))))
        else:
            lower = min(float(a.quantile(1e-12)), float(b.quantile(1e-12)))
            upper = max(float(a.quantile(1 - 1e-12)), float(b.quantile(1 - 1e-12)))
            overlap = quad(
                lambda x: float(np.sqrt(a.density(x) * b.density(x))),
                lower,
                upper,
                epsabs=1e-9,
                limit=400,
            )[0]
    else:
        grid = common_grid(X, Y, t, n_points)
        p, q = density_grid(X, t, grid, u_max, n_points), density_grid(Y, t, grid, u_max, n_points)
        masses = [float(trapezoid(v, grid)) for v in (p, q)]
        if not all(abs(m - 1) < 0.01 for m in masses):
            raise ValueError(
                "Density grid captures insufficient mass; increase inversion resolution"
            )
        overlap = float(trapezoid(np.sqrt(p * q), grid))
    if overlap > 1 + 1e-6 or overlap < 0:
        raise ArithmeticError("Hellinger affinity outside [0,1]; numerical quadrature unresolved")
    return float(np.clip(1 - overlap, 0, 1))


def wasserstein2_distance(
    X: Process, Y: Process, t: float = 1.0, u_max: float = 100.0, n_points: int = 4000
) -> float:
    """W2 via the complete quantile identity integral_0^1 (Qx-Qy)^2 dp.

    Both marginals must have finite second moments. Known distributions use
    reference quantiles over the full probability interval, with exact Gaussian
    handling. The generic continuous-law fallback uses a moment-scaled spatial
    grid and Gauss-Legendre probability quadrature; it remains approximate and
    rejects an inversion with more than 1% missing/erroneous mass.

    References
    ----------
    Villani (2003), Topics in Optimal Transportation, Chapter 2.
    """
    from spxa.zoo.levy.brownian import BrownianMotion

    if t < 0 or not np.isfinite(t) or u_max <= 0 or n_points < 2:
        raise ValueError("Nonnegative time and positive inversion controls required")
    if not X.properties.has_finite_variance or not Y.properties.has_finite_variance:
        raise ValueError("W2 requires finite second moments")
    a, b = known_marginal(X, t), known_marginal(Y, t)
    if X is Y:
        return 0.0
    if isinstance(X, BrownianMotion) and isinstance(Y, BrownianMotion):
        return float(np.sqrt(((X.mu - Y.mu) * t) ** 2 + (X.sigma - Y.sigma) ** 2 * t))
    if a is not None and b is not None:
        if a.discrete and b.discrete:
            values = np.union1d(a.support_grid(), b.support_grid())
            masses_a = a.density(values)
            masses_b = b.density(values)
            probabilities = np.unique(
                np.clip(np.r_[0.0, np.cumsum(masses_a), np.cumsum(masses_b), 1.0], 0, 1)
            )
            middle = np.clip(
                (probabilities[1:] + probabilities[:-1]) / 2,
                np.nextafter(0.0, 1.0),
                np.nextafter(1.0, 0.0),
            )
            value = np.sum(np.diff(probabilities) * (a.quantile(middle) - b.quantile(middle)) ** 2)
        else:
            value = quad(
                lambda p: float((a.quantile(p) - b.quantile(p)) ** 2), 0, 1, epsabs=1e-7, limit=400
            )[0]
        return float(np.sqrt(max(0.0, value)))
    grid = common_grid(X, Y, t, n_points)
    quantiles = []
    nodes, weights = np.polynomial.legendre.leggauss(500)
    p = (nodes + 1) / 2
    for process in (X, Y):
        density = density_grid(process, t, grid, u_max, n_points)
        cdf = cumulative_trapezoid(density, grid, initial=0)
        if abs(cdf[-1] - 1) > 0.01:
            raise ValueError("Density grid captures insufficient mass")
        quantiles.append(np.interp(p, cdf, grid))
    return float(np.sqrt(np.dot(weights / 2, (quantiles[0] - quantiles[1]) ** 2)))


def kl_divergence_mc(
    X: Process,
    Y: Process,
    t: float = 1.0,
    n_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo marginal KL estimate E_P[log(p/q)].

    Uses known PDF/PMFs where available, otherwise approximate Fourier densities.
    Singularity yields infinity. Negative estimates due to sampling noise are
    returned, not clamped to manufacture positivity. Generic mixtures without
    density adapters are explicitly unsupported.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    a, b = known_marginal(X, t), known_marginal(Y, t)
    if a is not None and b is not None and a.discrete != b.discrete:
        return float("inf")
    samples = X.simulate(1, n_paths=n_samples, T=t, rng=rng or np.random.default_rng())[:, -1]
    p = a.density(samples) if a is not None else density_grid(X, t, samples, 100.0, 4000)
    q = b.density(samples) if b is not None else density_grid(Y, t, samples, 100.0, 4000)
    if np.any(q == 0):
        return float("inf")
    if np.any(p <= 0):
        raise ValueError(
            "Density evaluation is zero at a sampled point; numerical inversion unresolved"
        )
    return float(np.mean(np.log(p) - np.log(q)))
