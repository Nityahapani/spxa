"""
Euler–Maruyama discretisation for SDEs driven by Lévy processes.

The Euler–Maruyama scheme for a general SDE

    dZ_t = a(Z_t, t) dt + b(Z_t, t) dX_t

where X is a Lévy process is:

    Z_{t_{k+1}} = Z_{t_k} + a(Z_{t_k}, t_k)·Δt + b(Z_{t_k}, t_k)·ΔX_k

with ΔX_k = X_{t_{k+1}} - X_{t_k} sampled exactly from the Lévy process.

For SDEs driven by Brownian motion this has strong order 0.5 and weak order
1.0 (Kloeden & Platen 1992). For general Lévy drivers the strong convergence
rate depends on the moment structure of the Lévy measure (Jacod et al. 2005).

References
----------
Kloeden, P.E. & Platen, E. (1992). *Numerical Solution of Stochastic
Differential Equations*. Springer. Chapter 9.

Jacod, J., Kurtz, T.G., Méléard, S. & Protter, P. (2005). The approximate
Euler method for Lévy driven stochastic differential equations.
*Annales de l'Institut Henri Poincaré*, 41(3), 523–558.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from spxa.core.process import Process


def euler_maruyama(
    driver: Process,
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    z0: float = 0.0,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Euler–Maruyama scheme for dZ_t = a(Z_t,t) dt + b(Z_t,t) dX_t.

    Parameters
    ----------
    driver :
        Lévy process X providing the increments ΔX_k.
    drift :
        Drift coefficient a(z, t) : ℝ × ℝ → ℝ.
    diffusion :
        Diffusion coefficient b(z, t) : ℝ × ℝ → ℝ.
    z0 :
        Initial value Z_0. Default 0.
    T :
        Terminal time. Default 1.
    n_steps :
        Number of time steps. Larger → more accurate.
    n_paths :
        Number of independent paths.
    rng :
        Random number generator.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps + 1). paths[:, 0] = z0.

    Notes
    -----
    Strong convergence order: 0.5 for Brownian driver; depends on jump
    moment structure for general Lévy drivers.

    References
    ----------
    Kloeden & Platen (1992), Chapter 9.
    Jacod et al. (2005), Theorem 3.1.
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)

    driver_paths = driver.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
    dX = np.diff(driver_paths, axis=1)

    paths = np.full((n_paths, n_steps + 1), z0, dtype=float)
    for k in range(n_steps):
        t_k = time_grid[k]
        z_k = paths[:, k]
        a_k = np.array([drift(z, t_k) for z in z_k])
        b_k = np.array([diffusion(z, t_k) for z in z_k])
        paths[:, k + 1] = z_k + a_k * dt + b_k * dX[:, k]

    return paths


def euler_maruyama_autonomous(
    driver: Process,
    drift: float,
    diffusion: float,
    z0: float = 0.0,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Euler–Maruyama for the constant-coefficient SDE dZ_t = a dt + b dX_t.

    This is vectorised and significantly faster than the general case.
    The solution is:
        Z_t = z0 + a·t + b·X_t

    which is exact (no discretisation error) since the coefficients are
    constant. The simulation reduces to simulating X directly.

    Parameters
    ----------
    driver :
        Lévy process X.
    drift :
        Constant drift a.
    diffusion :
        Constant diffusion coefficient b.
    z0, T, n_steps, n_paths, rng :
        Standard parameters.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps + 1).
    """
    rng = rng or np.random.default_rng()
    time_grid = np.linspace(0, T, n_steps + 1)
    driver_paths = driver.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
    return z0 + drift * time_grid[np.newaxis, :] + diffusion * driver_paths


def geometric_levy(
    driver: Process,
    mu: float,
    sigma: float,
    s0: float = 1.0,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate the geometric Lévy process S_t = S_0 · exp((μ - σ²/2)t + σ·X_t).

    This is the standard model for log-returns in jump-diffusion finance.
    When X is Brownian motion this is geometric Brownian motion.
    When X is a Lévy process this gives exponential Lévy models (VG, NIG, etc.).

    The simulation is exact: simulate log(S_t/S_0) = (μ-σ²/2)t + σ·X_t,
    then exponentiate. No discretisation error.

    Parameters
    ----------
    driver :
        Log-return Lévy process X.
    mu :
        Drift of log-price.
    sigma :
        Volatility scaling of X.
    s0 :
        Initial price. Default 1.
    T, n_steps, n_paths, rng :
        Standard parameters.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps + 1). Asset prices (not log-prices).
    """
    rng = rng or np.random.default_rng()
    time_grid = np.linspace(0, T, n_steps + 1)
    driver_paths = driver.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)

    # Risk-neutral drift correction: ensures E[S_t] = S_0 * exp(mu * t)
    # For a general Lévy process X, E[exp(sigma*X_1)] = phi_X(-i*sigma; 1)
    # The log of this is the cumulant generating function at sigma.
    # Normalize the exponential Levy noise: S_t = S0*exp((mu+drift)*t + sigma*X_t).
    # So drift_correction = mu - log(phi_X(-i*sigma; 1))
    mgf_val = complex(driver.char_func(u=-1j * sigma, t=1.0))
    if (
        not np.isfinite(mgf_val)
        or mgf_val.real <= 0
        or abs(mgf_val.imag) > 1e-8 * max(1.0, mgf_val.real)
    ):
        raise ValueError("A finite positive exponential moment is required")
    drift_correction = mu - np.log(mgf_val.real)

    log_paths = drift_correction * time_grid[np.newaxis, :] + sigma * driver_paths
    return np.asarray(s0 * np.exp(log_paths))


def convergence_order_estimate(
    driver: Process,
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    z0: float = 0.0,
    T: float = 1.0,
    n_paths_ref: int = 5000,
    step_sizes: tuple[int, ...] = (50, 100, 200, 400),
    rng_seed: int = 0,
) -> dict[str, object]:
    """
    Empirically estimate the strong convergence order of Euler–Maruyama
    by comparing to a fine-grid reference solution.

    Computes E[|Z_T^{(n)} - Z_T^{ref}|] for increasing n and fits a log-log
    regression to estimate the order.

    Parameters
    ----------
    driver :
        Lévy process driver.
    drift, diffusion :
        SDE coefficients.
    z0, T :
        Initial condition and terminal time.
    n_paths_ref :
        Number of paths for the reference solution.
    step_sizes :
        Tuple of n_steps values to test (excluding the reference).
    rng_seed :
        Seed for reproducibility.

    Returns
    -------
    dict with keys: 'errors' (list of mean errors), 'dt_values',
    'estimated_order' (float), 'log_log_slope'.
    """
    from math import lcm

    if len(step_sizes) < 2 or any(n < 1 for n in step_sizes) or T <= 0 or n_paths_ref < 1:
        raise ValueError("At least two positive grids, positive horizon/path count required")
    n_ref = lcm(*step_sizes) * 8
    if n_ref > 1_000_000:
        raise ValueError("Grid least common multiple exceeds reference-step budget")
    driving = driver.simulate(n_ref, n_paths=n_paths_ref, T=T, rng=np.random.default_rng(rng_seed))
    fine_increments = np.diff(driving, axis=1)

    def terminal(increments: np.ndarray) -> np.ndarray:
        n = increments.shape[1]
        values = np.full(n_paths_ref, z0, dtype=float)
        for k in range(n):
            d = np.array([drift(float(v), k * T / n) for v in values])
            g = np.array([diffusion(float(v), k * T / n) for v in values])
            values += d * T / n + g * increments[:, k]
        return values

    reference = terminal(fine_increments)
    errors = []
    dt_values = []
    for n in step_sizes:
        increments = fine_increments.reshape(n_paths_ref, n, n_ref // n).sum(axis=2)
        errors.append(float(np.mean(np.abs(terminal(increments) - reference))))
        dt_values.append(T / n)

    log_dt = np.log(dt_values)
    log_err = np.log(np.maximum(errors, 1e-15))
    slope = float(np.polyfit(log_dt, log_err, 1)[0])

    return {
        "errors": errors,
        "dt_values": dt_values,
        "estimated_order": slope,
        "log_log_slope": slope,
        "step_sizes": list(step_sizes),
    }
