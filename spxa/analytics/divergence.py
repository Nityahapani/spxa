"""
Divergences between Lévy processes.

For two Lévy processes X and Y with characteristic exponents ψ_X and ψ_Y,
divergence measures can be computed using:

1. The Plancherel identity: for L² functions of the characteristic functions,
   distances in frequency space translate to distances in distribution space.

2. Direct Lévy measure comparison: for processes with known densities,
   f-divergences between the Lévy measures give divergences between
   finite-dimensional distributions.

3. KL divergence between Lévy processes is infinite in general for processes
   with different Lévy measures (Hellinger distance between path measures).
   spxa computes the more tractable KL between marginal distributions at time t,
   approximated via Gil-Pelaez inversion.

This module implements:
  - Squared Hellinger distance between marginal distributions at time t
  - L² (Plancherel) distance between characteristic functions
  - KL divergence via importance sampling (Monte Carlo, for EXACT processes)
  - Wasserstein-2 distance between marginals (via quantile function comparison)

References
----------
Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapter 4, Section 4.6.

Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*.
Chapter 14.
"""

from __future__ import annotations

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process


def l2_char_func_distance(
    X: Process,
    Y: Process,
    t: float = 1.0,
    u_max: float = 50.0,
    n_points: int = 2000,
) -> float:
    """
    L² distance between characteristic functions at time t.

    d²_L²(X_t, Y_t) = (1/2π) ∫ |φ_X(u;t) - φ_Y(u;t)|² du

    By the Plancherel theorem this equals the L² distance between the
    corresponding densities (when they exist):
      = ∫ |f_X(x;t) - f_Y(x;t)|² dx

    Computed by numerical integration on [-u_max, u_max].

    Parameters
    ----------
    X, Y :
        Processes with ExactnessLevel.EXACT.
    t :
        Time. Default 1.
    u_max :
        Truncation of the frequency domain. Should be large enough that
        |φ(u)| is negligible beyond u_max.
    n_points :
        Number of quadrature points.

    Returns
    -------
    float
        Non-negative L² distance squared.

    Raises
    ------
    ExactnessError
        If either process does not support char_func.

    References
    ----------
    Plancherel theorem. See e.g. Rudin (1987), Theorem 9.13.
    """
    u_grid = np.linspace(-u_max, u_max, n_points)
    du = u_grid[1] - u_grid[0]

    cf_X = X.char_func(u=u_grid, t=t)
    cf_Y = Y.char_func(u=u_grid, t=t)

    diff_sq = np.abs(cf_X - cf_Y) ** 2
    return float(np.trapezoid(diff_sq, u_grid) / (2 * np.pi))


def hellinger_distance(
    X: Process,
    Y: Process,
    t: float = 1.0,
    u_max: float = 50.0,
    n_points: int = 2000,
) -> float:
    """
    Squared Hellinger distance between marginal distributions at time t.

    H²(X_t, Y_t) = 1 - Re[∫ φ_X(u/2; t) · conj(φ_Y(u/2; t)) · (1/2π) du]
                 = 1 - Re[φ_{X_t - Y_t'}(0)] ... (characteristic function route)

    Computed via the identity:
      2H²(P, Q) = ∫ (√p - √q)² dx = 2 - 2∫ √(pq) dx

    and using the fact that ∫ √(pq) dx can be expressed via the characteristic
    function of (X - Y') where Y' is an independent copy of Y, evaluated at 0:
      ∫ √(pq) dx = Re[(1/2π) ∫ φ_X(u/2; t) · conj(φ_Y(u/2; t)) du]

    Parameters
    ----------
    X, Y :
        Processes supporting char_func.
    t :
        Time. Default 1.
    u_max, n_points :
        Frequency domain truncation and resolution.

    Returns
    -------
    float
        Squared Hellinger distance H² ∈ [0, 1].

    References
    ----------
    Cont & Tankov (2004), Section 4.6.
    """
    u_grid = np.linspace(-u_max, u_max, n_points)

    cf_X_half = X.char_func(u=u_grid / 2, t=t)
    cf_Y_half = Y.char_func(u=u_grid / 2, t=t)

    integrand = (cf_X_half * np.conj(cf_Y_half)).real
    overlap = float(np.trapezoid(integrand, u_grid) / (2 * np.pi))
    overlap = np.clip(overlap, 0.0, 1.0)

    return 1.0 - overlap


def wasserstein2_distance(
    X: Process,
    Y: Process,
    t: float = 1.0,
    u_max: float = 100.0,
    n_points: int = 4000,
) -> float:
    """
    Wasserstein-2 distance between marginal distributions at time t.

    W_2²(X_t, Y_t) = ∫_0^1 (F_X^{-1}(p) - F_Y^{-1}(p))² dp

    where F^{-1} is the quantile function. Computed by:
    1. Recovering CDFs from characteristic functions via Gil-Pelaez inversion.
    2. Computing quantile functions numerically.
    3. Integrating the squared difference of quantile functions.

    Parameters
    ----------
    X, Y :
        Processes supporting char_func.
    t :
        Time. Default 1.
    u_max, n_points :
        Frequency domain and x-grid resolution.

    Returns
    -------
    float
        W₂ distance (not squared).

    References
    ----------
    Villani, C. (2003). *Topics in Optimal Transportation*. AMS. Chapter 2.
    Gil-Pelaez, J. (1951). Note on the inversion theorem. *Biometrika*, 38, 481–482.
    """
    from scipy import integrate as sci_integrate  # type: ignore[import-untyped]

    def get_quantile_fn(proc: Process) -> tuple[np.ndarray, np.ndarray]:
        u_grid = np.linspace(1e-6, u_max, n_points // 2)

        cf_vals = proc.char_func(u=u_grid, t=t)

        x_vals = np.linspace(-20.0, 20.0, n_points)
        dx = x_vals[1] - x_vals[0]

        pdf = np.zeros(len(x_vals))
        for i, x in enumerate(x_vals):
            integrand = (cf_vals * np.exp(-1j * u_grid * x)).real
            pdf[i] = float(np.trapezoid(integrand, u_grid)) / np.pi

        pdf = np.maximum(pdf, 0)
        cdf = np.cumsum(pdf) * dx
        cdf = np.clip(cdf / cdf[-1], 0, 1)

        return x_vals, cdf

    x_X, cdf_X = get_quantile_fn(X)
    x_Y, cdf_Y = get_quantile_fn(Y)

    p_grid = np.linspace(0.01, 0.99, 500)
    q_X = np.interp(p_grid, cdf_X, x_X)
    q_Y = np.interp(p_grid, cdf_Y, x_Y)

    w2_sq = float(np.trapezoid((q_X - q_Y) ** 2, p_grid))
    return float(np.sqrt(max(w2_sq, 0.0)))


def kl_divergence_mc(
    X: Process,
    Y: Process,
    t: float = 1.0,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Monte Carlo estimate of KL divergence KL(X_t || Y_t).

    KL(P||Q) = E_P[log(p(X)/q(X))]

    Estimated using importance sampling: draw samples from X_t,
    recover densities via Gil-Pelaez inversion evaluated at the samples,
    and compute the log ratio.

    This is an approximate method. The density inversion introduces
    numerical error. Use l2_char_func_distance or hellinger_distance
    for more reliable analytical divergences.

    Parameters
    ----------
    X, Y :
        Processes supporting char_func and simulate.
    t :
        Time.
    n_samples :
        Number of Monte Carlo samples.
    rng :
        Random number generator.

    Returns
    -------
    float
        Estimated KL(X_t || Y_t). May be negative due to numerical error
        in density estimation — clamp to 0 if this occurs.
    """
    rng = rng or np.random.default_rng()

    paths = X.simulate(n_steps=1, n_paths=n_samples, T=t, rng=rng)
    samples = paths[:, -1]

    u_grid = np.linspace(-100.0, 100.0, 2000)
    du = u_grid[1] - u_grid[0]

    def density_at(proc: Process, x_vals: np.ndarray) -> np.ndarray:
        cf = proc.char_func(u=u_grid, t=t)
        densities = np.zeros(len(x_vals))
        for i, x in enumerate(x_vals):
            integrand = (cf * np.exp(-1j * u_grid * x)).real
            densities[i] = float(np.trapezoid(integrand, u_grid)) / (2 * np.pi)
        return np.maximum(densities, 1e-300)

    p_vals = density_at(X, samples)
    q_vals = density_at(Y, samples)

    log_ratios = np.log(p_vals / q_vals)
    return float(np.maximum(np.mean(log_ratios), 0.0))
