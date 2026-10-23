"""
Bridge sampling for stochastic processes.

A bridge is a process conditioned on hitting a fixed value at a fixed time.
The Brownian bridge B(t | B_T = b) is the most fundamental example and
serves as a building block for more general bridges.

Implementations
---------------
1. Brownian bridge: exact conditional Gaussian simulation.
2. Gamma bridge: exact via the beta distribution identity.
3. Lévy bridge: approximate via sequential bisection (Devroye 1986).
4. Pinned simulation: simulate a process conditioned on its terminal value
   by accept-reject or conditioning on the increments.

Applications
------------
- Path space Monte Carlo with fixed endpoints (barrier options, first
  passage times).
- Brownian bridge interpolation for computing path functionals.
- Quasi-Monte Carlo path construction (principal component analysis).

References
----------
Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*.
Springer. Section 3.2.

Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer.
Chapter 11.

Imai, J. & Tan, K.S. (2006). A general dimension reduction technique for
derivative pricing. *Journal of Computational Finance*, 10(2), 79–99.
"""

from __future__ import annotations

import numpy as np


def brownian_bridge(
    t_start: float = 0.0,
    t_end: float = 1.0,
    x_start: float = 0.0,
    x_end: float = 0.0,
    sigma: float = 1.0,
    n_points: int = 100,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample paths of a Brownian bridge conditioned on X(t_start)=x_start,
    X(t_end)=x_end.

    The conditional distribution is:
        X(t) | X(t_start)=x_start, X(t_end)=x_end
        ~ N(μ(t), σ²·v(t))
    where:
        μ(t) = x_start + (x_end - x_start)·(t - t_start)/(t_end - t_start)
        v(t) = (t - t_start)·(t_end - t)/(t_end - t_start)

    The simulation proceeds sequentially by conditioning at each interior
    point. This is the exact algorithm.

    Parameters
    ----------
    t_start, t_end :
        Start and end times.
    x_start, x_end :
        Pinned values at t_start and t_end.
    sigma :
        Diffusion coefficient.
    n_points :
        Number of interior time points (excluding endpoints).
    n_paths :
        Number of independent bridge paths.
    rng :
        Random number generator.

    Returns
    -------
    (time_grid, paths) :
        time_grid : shape (n_points + 2,)  including t_start, t_end
        paths     : shape (n_paths, n_points + 2)

    References
    ----------
    Glasserman (2003), Section 3.2.1.
    """
    if (
        t_end <= t_start
        or sigma < 0
        or n_points < 0
        or n_paths < 1
        or not all(np.isfinite(v) for v in (t_start, t_end, x_start, x_end, sigma))
    ):
        raise ValueError(
            "Ordered finite times/endpoints, nonnegative sigma/points "
            "and positive paths are required"
        )
    rng = rng or np.random.default_rng()
    time_grid = np.linspace(t_start, t_end, n_points + 2)
    paths = np.zeros((n_paths, n_points + 2))
    paths[:, 0] = x_start
    paths[:, -1] = x_end

    for i in range(1, n_points + 1):
        t = time_grid[i]
        t_prev = time_grid[i - 1]
        t_end_local = time_grid[-1]
        x_prev = paths[:, i - 1]
        x_end_val = paths[:, -1]  # always the pinned endpoint

        mu_cond = x_prev + (x_end_val - x_prev) * (t - t_prev) / (t_end_local - t_prev)
        var_cond = sigma**2 * (t - t_prev) * (t_end_local - t) / (t_end_local - t_prev)

        paths[:, i] = mu_cond + np.sqrt(var_cond) * rng.standard_normal(n_paths)

    return time_grid, paths


def gamma_bridge(
    a: float,
    b: float,
    T: float = 1.0,
    x_end: float | None = None,
    n_points: int = 100,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample paths of a Gamma process bridge conditioned on G(T) = x_end.

    Uses the identity: for a Gamma process G with G(T) = g_T,
    the bridge G(t)/G(T) | G(T)=g_T has the same distribution as
    the order statistics of a Beta(a·t, a·(T-t)) distribution scaled
    by g_T.

    More precisely, for the Gamma bridge at time t:
        G(t) | G(T) = g_T  ~  g_T · Beta(a·t/b, a·(T-t)/b)

    Parameters
    ----------
    a, b :
        GammaProcess parameters (shape rate, inverse scale).
    T :
        Terminal time.
    x_end :
        Terminal value G(T). If None, sampled from Gamma(a·T, b).
    n_points :
        Number of interior time points.
    n_paths :
        Number of bridge paths.
    rng :
        Random number generator.

    Returns
    -------
    (time_grid, paths) :
        time_grid : shape (n_points + 2,)
        paths     : shape (n_paths, n_points + 2)

    References
    ----------
    Glasserman (2003), Section 3.2.2.
    """
    rng = rng or np.random.default_rng()
    time_grid = np.linspace(0, T, n_points + 2)

    if x_end is None:
        g_T = rng.gamma(shape=a * T, scale=1.0 / b, size=n_paths)
    else:
        g_T = np.full(n_paths, x_end)

    paths = np.zeros((n_paths, n_points + 2))
    paths[:, 0] = 0.0
    paths[:, -1] = g_T

    if a <= 0 or b <= 0 or T <= 0 or n_points < 0 or np.any(g_T < 0):
        raise ValueError("Positive Gamma parameters/horizon and nonnegative endpoint required")
    weights = rng.dirichlet(a * np.diff(time_grid), size=n_paths)
    paths[:, 1:] = g_T[:, None] * np.cumsum(weights, axis=1)
    paths[:, -1] = g_T

    return time_grid, paths


def brownian_bridge_interpolate(
    coarse_paths: np.ndarray,
    coarse_times: np.ndarray,
    fine_times: np.ndarray,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Refine coarse Brownian paths to a finer time grid using bridge interpolation.

    Between each pair of adjacent coarse points (t_k, t_{k+1}) with values
    (X_k, X_{k+1}), inserts fine-grid points sampled from the Brownian bridge
    conditional distribution.

    Parameters
    ----------
    coarse_paths :
        Shape (n_paths, n_coarse + 1). Coarse-grid paths.
    coarse_times :
        Shape (n_coarse + 1,). Times for coarse paths.
    fine_times :
        Shape (n_fine + 1,). Target fine-grid times. Must contain all
        coarse_times as a subset.
    sigma :
        Diffusion coefficient.
    rng :
        Random number generator.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_fine + 1). Refined paths.
    """
    rng = rng or np.random.default_rng()
    coarse_times = np.asarray(coarse_times, dtype=float)
    fine_times = np.asarray(fine_times, dtype=float)
    coarse_paths = np.asarray(coarse_paths, dtype=float)
    if (
        coarse_times.ndim != 1
        or fine_times.ndim != 1
        or len(coarse_times) < 2
        or np.any(np.diff(coarse_times) <= 0)
        or np.any(np.diff(fine_times) <= 0)
        or coarse_paths.ndim != 2
        or coarse_paths.shape[1] != len(coarse_times)
        or not np.all(np.isfinite(coarse_times))
        or not np.all(np.isfinite(fine_times))
        or not np.all(np.isfinite(coarse_paths))
        or sigma < 0
        or not np.all(np.isin(coarse_times, fine_times))
        or fine_times[0] != coarse_times[0]
        or fine_times[-1] != coarse_times[-1]
    ):
        raise ValueError(
            "Ordered finite grids with all coarse knots and matching path shape required"
        )
    fine_paths = np.empty((len(coarse_paths), len(fine_times)))
    for interval in range(len(coarse_times) - 1):
        left, right = coarse_times[interval : interval + 2]
        previous = coarse_paths[:, interval].copy()
        previous_t = left
        end = coarse_paths[:, interval + 1]
        fine_paths[:, np.searchsorted(fine_times, left)] = previous
        for k in np.flatnonzero((fine_times > left) & (fine_times < right)):
            t = fine_times[k]
            mean = previous + (end - previous) * (t - previous_t) / (right - previous_t)
            variance = sigma * sigma * (t - previous_t) * (right - t) / (right - previous_t)
            previous = mean + np.sqrt(variance) * rng.standard_normal(len(coarse_paths))
            previous_t = t
            fine_paths[:, k] = previous
        fine_paths[:, np.searchsorted(fine_times, right)] = end
    return fine_paths


def first_passage_time_bm(
    level: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    T_max: float = 10.0,
    n_steps: int = 10000,
    n_paths: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Estimate first passage times of BM(μ, σ) to a level via simulation.

    τ = inf{t ≥ 0 : X_t ≥ level}

    Returns the first passage time for each path (np.inf if not reached
    by T_max).

    Parameters
    ----------
    level :
        The barrier level.
    mu, sigma :
        BM drift and diffusion.
    T_max :
        Maximum simulation time.
    n_steps :
        Time discretisation steps.
    n_paths :
        Number of paths.
    rng :
        Generator.

    Returns
    -------
    np.ndarray
        Shape (n_paths,). First passage times.

    Notes
    -----
    For BM with μ=0 and level > 0, the exact distribution is the Lévy
    distribution (one-sided stable with α=1/2):
        P(τ ≤ t) = 2·Φ(-level/(σ√t))
    where Φ is the standard normal CDF.
    """
    from spxa.zoo.levy import BrownianMotion

    rng = rng or np.random.default_rng()
    bm = BrownianMotion(mu=mu, sigma=sigma)
    paths = bm.simulate(n_steps=n_steps, n_paths=n_paths, T=T_max, rng=rng)
    time_grid = np.linspace(0, T_max, n_steps + 1)

    first_passage = np.full(n_paths, np.inf)
    for i in range(n_paths):
        crossings = np.where(paths[i] >= level)[0]
        if len(crossings) > 0:
            first_passage[i] = time_grid[crossings[0]]

    return first_passage


def first_passage_time_exact_bm(
    level: float,
    sigma: float = 1.0,
    n_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Exact samples of the first passage time of standard BM to a level.

    For μ=0, τ_level has the Lévy (one-sided stable α=1/2) distribution:
        τ ~ level²/σ² · InverseGaussian(1, 1)  [Lévy distribution]

    Concretely: P(τ ≤ t) = 2·Φ(-level/(σ√t)) for level > 0.
    Samples via the quantile transform of this CDF.

    Parameters
    ----------
    level :
        Barrier level > 0.
    sigma :
        BM diffusion.
    n_samples :
        Number of exact samples.
    rng :
        Generator.

    Returns
    -------
    np.ndarray
        Shape (n_samples,). Exact first passage time samples.

    References
    ----------
    Karatzas, I. & Shreve, S.E. (1991). *Brownian Motion and Stochastic
    Calculus*, 2nd ed. Springer. Proposition 2.8.2.
    """
    from scipy.stats import norm

    if (
        not np.isfinite(level)
        or not np.isfinite(sigma)
        or level <= 0
        or sigma <= 0
        or n_samples < 1
    ):
        raise ValueError(f"level must be positive, got {level}")

    rng = rng or np.random.default_rng()
    u = rng.uniform(size=n_samples)
    # τ ~ Lévy(0, c) with c = level²/σ²
    # CDF: F(t) = 2·Φ(-level/(σ√t))
    # Quantile: solve 2·Φ(-level/(σ√τ)) = u
    # → Φ⁻¹(u/2) = -level/(σ√τ)
    # → τ = (level / (σ · |Φ⁻¹(u/2)|))²
    z = norm.ppf(u / 2)
    tau = (level / (sigma * np.abs(z))) ** 2
    return np.asarray(tau)
