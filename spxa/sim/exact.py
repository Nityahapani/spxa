"""
Exact simulation algorithms for specific Lévy process families.

Each function here produces samples that are exact in distribution — no
discretisation error. The algorithms are chosen to be the most efficient
known exact method for each family.

Algorithms implemented
----------------------
1. Brownian motion: Gaussian increments (trivial, O(n)).
2. Gamma process: Gamma(a·Δt, b) increments (O(n)).
3. Inverse Gaussian process: Wald distribution increments (O(n)).
4. Variance Gamma: Gamma subordination of BM (O(n)).
5. NIG: Inverse Gaussian subordination of BM (O(n)).
6. α-stable: Chambers–Mallows–Stuck (1976) (O(n)).
7. Compound Poisson: Poisson arrival times + jump sizes (O(N) where N is
   the (random) number of jumps; fast for low-intensity processes).
8. Tempered stable: Rejection from stable (Rosiński 2007).

References
----------
Chambers, J.M., Mallows, C.L. & Stuck, B.W. (1976). A method for simulating
stable random variables. *Journal of the American Statistical Association*,
71(354), 340–344.

Rosiński, J. (2007). Tempering stable processes. *Stochastic Processes and
their Applications*, 117(6), 677–707.

Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer.
Chapter 9 (Gamma), Chapter 4 (Inverse Gaussian).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from spxa.core._validation import simulation_grid


def sample_brownian(
    mu: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exact increments of BM(μ, σ): ΔX ~ N(μ·Δt, σ²·Δt).

    Parameters
    ----------
    mu, sigma :
        Drift and diffusion of BM.
    dt :
        Time step size.
    n_steps, n_paths :
        Grid and path count.
    rng :
        Generator.

    Returns
    -------
    np.ndarray
        Increments, shape (n_paths, n_steps).
    """
    return rng.normal(loc=mu * dt, scale=sigma * np.sqrt(dt), size=(n_paths, n_steps))


def sample_gamma(
    a: float,
    b: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exact increments of Gamma process: ΔX ~ Gamma(a·Δt, 1/b).

    Shape/rate parametrisation matching GammaProcess(a, b).
    Mean = a·Δt/b, Variance = a·Δt/b².

    Devroye (1986), Chapter 9.
    """
    return rng.gamma(shape=a * dt, scale=1.0 / b, size=(n_paths, n_steps))


def sample_inverse_gaussian(
    mu_ig: float,
    lambda_ig: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exact increments of Inverse Gaussian process via Wald distribution.

    The IG(μ, λ) distribution has density:
        f(x) = sqrt(λ/(2πx³)) exp(-λ(x-μ)²/(2μ²x))

    numpy's `wald(mean=μ, scale=λ)` samples IG(μ, λ) directly.

    For the IG process with mean rate μ_ig and shape λ_ig per unit time:
        ΔX ~ IG(μ_ig·Δt, λ_ig·Δt²)

    Devroye (1986), Chapter 4.
    """
    return rng.wald(
        mean=mu_ig * dt,
        scale=lambda_ig * dt**2,
        size=(n_paths, n_steps),
    )


def sample_vg(
    sigma: float,
    nu: float,
    theta: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exact increments of Variance Gamma via Gamma subordination.

    Algorithm (Madan, Carr & Chang 1998):
      1. G ~ Gamma(Δt/ν, ν)
      2. ΔX = θ·G + σ·√G·Z,  Z ~ N(0,1)

    Parameters match VarianceGamma(sigma, nu, theta).
    """
    G = rng.gamma(shape=dt / nu, scale=nu, size=(n_paths, n_steps))
    Z = rng.standard_normal(size=(n_paths, n_steps))
    return theta * G + sigma * np.sqrt(G) * Z


def sample_nig(
    alpha: float,
    beta: float,
    delta: float,
    mu: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exact increments of NIG via Inverse Gaussian subordination.

    Algorithm (Barndorff-Nielsen 1997):
      γ = sqrt(α² - β²)
      1. IG ~ Wald(mean=δ·Δt/γ, scale=(δ·Δt)²)
      2. ΔX = μ·Δt + β·IG + √IG·Z,  Z ~ N(0,1)
    """
    gamma = np.sqrt(alpha**2 - beta**2)
    IG = rng.wald(mean=delta * dt / gamma, scale=(delta * dt) ** 2, size=(n_paths, n_steps))
    Z = rng.standard_normal(size=(n_paths, n_steps))
    return mu * dt + beta * IG + np.sqrt(IG) * Z


def sample_alpha_stable(
    alpha: float,
    beta: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exact increments of α-stable process via Chambers–Mallows–Stuck (1976).

    Each increment is distributed as σ·Δt^{1/α}·S where S ~ S_α(1, β, 0)
    is a standard stable.

    CMS algorithm (Chambers et al. 1976, equations 2.1–2.3):
      U ~ Uniform(-π/2, π/2),  E ~ Exp(1)
      For α ≠ 1:
        B = arctan(β·tan(πα/2)) / α
        S_α = (1 + β²·tan²(πα/2))^{1/(2α)}
            · sin(α(U+B)) / cos(U)^{1/α}
            · (cos(U - α(U+B)) / E)^{(1-α)/α}
    """
    size = (n_paths, n_steps)
    U = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
    E = rng.exponential(1.0, size=size)

    if abs(alpha - 1.0) > 1e-10:
        B = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
        S_scale = (1 + (beta * np.tan(np.pi * alpha / 2)) ** 2) ** (1 / (2 * alpha))
        X = (
            S_scale
            * np.sin(alpha * (U + B))
            / np.cos(U) ** (1 / alpha)
            * (np.cos(U - alpha * (U + B)) / E) ** ((1 - alpha) / alpha)
        )
    else:
        X = (2 / np.pi) * (
            (np.pi / 2 + beta * U) * np.tan(U)
            - beta * np.log(np.pi / 2 * E * np.cos(U) / (np.pi / 2 + beta * U))
        )

    increments = sigma * (dt ** (1 / alpha)) * X
    if alpha == 1:
        increments += 2 * beta * sigma * dt * np.log(sigma * dt) / np.pi
    return np.asarray(increments)


def sample_compound_poisson(
    rate: float,
    jump_sampler: Callable[[np.random.Generator, int], np.ndarray],
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized Poisson compounding; sampler errors are never swallowed/retried.

    The callable receives an integer count and must produce that many finite
    real jumps. Retrying after an exception would both mask a bug and consume
    a different random stream. This utility returns increments, not paths.
    """
    dt = simulation_grid(n_steps, n_paths, dt=dt)
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("rate must be nonnegative and finite")
    counts = rng.poisson(rate * dt, size=(n_paths, n_steps)).ravel()
    total = int(counts.sum())
    if total == 0:
        return np.zeros((n_paths, n_steps))
    jumps = np.asarray(jump_sampler(rng, total))
    if np.iscomplexobj(jumps):
        raise ValueError("Jump samples must be real")
    jumps = np.asarray(jumps, dtype=float).ravel()
    if jumps.size != total or not np.all(np.isfinite(jumps)):
        raise ValueError("Jump sampler must return the requested number of finite values")
    cells = np.repeat(np.arange(counts.size), counts)
    return np.bincount(cells, weights=jumps, minlength=counts.size).reshape(n_paths, n_steps)


def sample_tempered_stable(
    alpha: float,
    C: float,
    lam: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    n_rejection_attempts: int = 50,
) -> np.ndarray:
    """
    Exact increments of a one-sided tempered stable process via rejection
    from an α-stable proposal (exponential tilting; see Rosiński 2007).

    The one-sided tempered stable (also called a CGMY subordinator) has
    Lévy density k(x) = C·x^{-1-α}·exp(-λx) for x > 0.

    Algorithm: use Laplace exponent a*s**alpha with a=C*Gamma(1-alpha)*dt/alpha.
    Divide into m=ceil(a*lam**alpha) blocks so the acceptance probability in each
    block is at least exp(-1). Exponentially tilt each positive-stable block
    with exp(-lam*X), then sum the independent accepted blocks.

    Parameters
    ----------
    alpha :
        Stability index α ∈ (0, 1). Must be < 1 for a subordinator.
    C :
        Jump intensity C > 0.
    lam :
        Tempering parameter λ > 0.
    n_rejection_attempts :
        Proposal budget per block: n_rejection_attempts*n_paths*n_steps.
        Exhaustion raises RuntimeError; there is no approximate fallback.

    Returns
    -------
    np.ndarray
        Increments, shape (n_paths, n_steps).

    References
    ----------
    Rosiński, J. (2007). *Stochastic Processes and their Applications*,
    117(6), 677–707.
    """
    from scipy.special import gamma

    if not (0 < alpha < 1) or not all(np.isfinite(v) and v > 0 for v in (C, lam, dt)):
        raise ValueError("Require 0<alpha<1 and finite positive C, lam and dt")
    if n_steps < 1 or n_paths < 1 or n_rejection_attempts < 1:
        raise ValueError("Positive sizes and proposal budget required")
    # Untempered Laplace exponent a*s**alpha, a=C*Gamma(1-alpha)*dt/alpha.
    a = C * float(gamma(1 - alpha)) * dt / alpha
    pieces = max(1, int(np.ceil(a * lam**alpha)))
    if pieces > 10000:
        raise ValueError("Time subdivision exceeds sampling budget; use a smaller dt")
    scale = (a / pieces) ** (1 / alpha)
    total = n_paths * n_steps
    output = np.zeros(total)
    for _ in range(pieces):
        accepted_parts = []
        remaining = total
        budget = n_rejection_attempts * total
        while remaining:
            if budget <= 0:
                raise RuntimeError(
                    "Tempered-stable proposal budget exhausted; no approximate fallback used"
                )
            size = min(budget, max(remaining, min(100000, 3 * remaining)))
            angle = rng.uniform(np.nextafter(0.0, 1.0), np.pi, size)
            exponential = rng.exponential(size=size)
            log_value = (
                np.log(scale)
                + np.log(np.sin(alpha * angle))
                - np.log(np.sin(angle)) / alpha
                + (1 - alpha) / alpha * (np.log(np.sin((1 - alpha) * angle)) - np.log(exponential))
            )
            proposed = np.exp(log_value)
            selected = proposed[np.log(rng.uniform(size=size)) < -lam * proposed][:remaining]
            accepted_parts.append(selected)
            remaining -= len(selected)
            budget -= size
        output += np.concatenate(accepted_parts)
    return output.reshape(n_paths, n_steps)


def increments_to_paths(
    increments: np.ndarray,
    x0: float = 0.0,
) -> np.ndarray:
    """
    Convert an increment array to path array by prepending x0 and cumsum.

    Parameters
    ----------
    increments :
        Shape (n_paths, n_steps).
    x0 :
        Initial value. Default 0.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps + 1).
    """
    n_paths, n_steps = increments.shape
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    paths[:, 1:] = x0 + np.cumsum(increments, axis=1)
    return paths


def sample_cir(
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    dt: float,
    n_steps: int,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
    method: str = "exact",
) -> np.ndarray:
    r"""
    Simulate Cox–Ingersoll–Ross (CIR) square-root diffusion paths.

    dr_t = κ(θ - r_t)dt + σ√r_t dW_t,  r_0 = r0.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed κ > 0.
    theta : float
        Long-term mean level θ > 0.
    sigma : float
        Volatility parameter σ > 0.
    r0 : float
        Initial value r_0 ≥ 0.
    dt : float
        Time step size Δt > 0.
    n_steps : int
        Number of simulation time steps.
    n_paths : int, default 1
        Number of independent sample paths.
    rng : np.random.Generator, optional
        Random number generator.
    method : {"exact", "truncated_euler", "reflected_euler"}, default "exact"
        Simulation scheme:
        - "exact": Non-central chi-squared transition distribution (exact in law).
        - "truncated_euler": Higham–Mao full-truncation Euler–Maruyama scheme.
        - "reflected_euler": Reflected Euler–Maruyama scheme.

    Returns
    -------
    np.ndarray
        Path array of shape (n_paths, n_steps + 1), with paths[:, 0] == r0.
    """
    dt = simulation_grid(n_steps, n_paths, dt=dt)
    if not np.all(np.isfinite([kappa, theta, sigma, r0])):
        raise ValueError("Simulation parameters must be finite")
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        raise ValueError("kappa, theta, and sigma must be strictly positive")
    if r0 < 0:
        raise ValueError("r0 must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be strictly positive")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    rng = rng or np.random.default_rng()
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(r0)

    if n_steps == 0:
        return paths

    curr = np.full(n_paths, float(r0), dtype=float)

    if method == "exact":
        exp_k = math.exp(-kappa * dt)
        c = 2.0 * kappa / (sigma**2 * (-math.expm1(-kappa * dt)))
        d = 4.0 * kappa * theta / (sigma**2)
        inv_2c = 1.0 / (2.0 * c)
        two_c_exp_k = 2.0 * c * exp_k

        for k in range(n_steps):
            nonc = two_c_exp_k * np.maximum(curr, 0.0)
            curr = rng.noncentral_chisquare(df=d, nonc=nonc) * inv_2c
            paths[:, k + 1] = curr

    elif method == "truncated_euler":
        # Higham & Mao (2005) / Lord, Koekkoek, van Dijk (2010) full truncation
        sqrt_dt = math.sqrt(dt)
        for k in range(n_steps):
            pos = np.maximum(curr, 0.0)
            z = rng.standard_normal(size=n_paths)
            curr = curr + kappa * (theta - pos) * dt + sigma * np.sqrt(pos) * sqrt_dt * z
            curr = np.maximum(curr, 0.0)
            paths[:, k + 1] = curr

    elif method == "reflected_euler":
        sqrt_dt = math.sqrt(dt)
        for k in range(n_steps):
            pos = np.maximum(curr, 0.0)
            z = rng.standard_normal(size=n_paths)
            curr = np.abs(curr + kappa * (theta - curr) * dt + sigma * np.sqrt(pos) * sqrt_dt * z)
            paths[:, k + 1] = curr

    else:
        raise ValueError(
            f"Unknown simulation method '{method}'. "
            "Supported methods: 'exact', 'truncated_euler', 'reflected_euler'."
        )

    return paths


def sample_ou(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    dt: float,
    n_steps: int,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
    method: str = "exact",
) -> np.ndarray:
    r"""
    Simulate Gaussian Ornstein–Uhlenbeck (Vasicek) process paths.

    dX_t = θ(μ - X_t)dt + σ dW_t,  X_0 = x0.

    Parameters
    ----------
    theta : float
        Mean-reversion speed θ > 0.
    mu : float
        Long-term mean level μ ∈ ℝ.
    sigma : float
        Diffusion volatility coefficient σ > 0.
    x0 : float
        Initial position X_0.
    dt : float
        Time step size Δt > 0.
    n_steps : int
        Number of time steps.
    n_paths : int, default 1
        Number of independent sample paths.
    rng : np.random.Generator, optional
        Random number generator.
    method : {"exact", "euler"}, default "exact"
        Simulation algorithm:
        - "exact": Exact Gaussian transition law (zero discretization error).
        - "euler": Standard Euler–Maruyama scheme.

    Returns
    -------
    np.ndarray
        Path array of shape (n_paths, n_steps + 1), with paths[:, 0] == x0.
    """
    dt = simulation_grid(n_steps, n_paths, dt=dt)
    if not np.all(np.isfinite([theta, mu, sigma, x0])):
        raise ValueError("Simulation parameters must be finite")
    if theta <= 0 or sigma <= 0:
        raise ValueError("theta and sigma must be strictly positive")
    if dt <= 0:
        raise ValueError("dt must be strictly positive")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    rng = rng or np.random.default_rng()
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(x0)

    if n_steps == 0:
        return paths

    curr = np.full(n_paths, float(x0), dtype=float)

    if method == "exact":
        exp_th = math.exp(-theta * dt)
        cond_var = (sigma**2 / (2.0 * theta)) * (-math.expm1(-2.0 * theta * dt))
        std_inc = math.sqrt(max(cond_var, 0.0))

        for k in range(n_steps):
            z = rng.standard_normal(size=n_paths)
            curr = mu + (curr - mu) * exp_th + std_inc * z
            paths[:, k + 1] = curr

    elif method == "euler":
        sqrt_dt = math.sqrt(dt)
        for k in range(n_steps):
            z = rng.standard_normal(size=n_paths)
            curr = curr + theta * (mu - curr) * dt + sigma * sqrt_dt * z
            paths[:, k + 1] = curr

    else:
        raise ValueError(
            f"Unknown simulation method '{method}'. Supported methods: 'exact', 'euler'."
        )

    return paths


def sample_heston(
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    mu: float = 0.0,
    s0: float = 1.0,
    dt: float | None = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: float | None = None,
    rng: np.random.Generator | None = None,
    method: str = "full_truncation",
    return_variance: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    r"""
    Simulate sample paths of the Heston stochastic volatility model.

    .. math::

        dX_t = \left(\mu - \frac{1}{2} V_t\right) dt + \sqrt{V_t} \, dW_t^S, \quad X_0 = 0 \\
        dV_t = \kappa (\theta - V_t) dt + \sigma_v \sqrt{V_t} \, dW_t^V, \quad V_0 = v_0

    with :math:`\mathbb{E}[dW_t^S dW_t^V] = \rho \, dt`.

    Parameters
    ----------
    v0 : float
        Initial variance :math:`V_0 > 0`.
    kappa : float
        Mean-reversion rate :math:`\kappa > 0`.
    theta : float
        Long-term variance :math:`\theta > 0`.
    sigma_v : float
        Volatility of variance :math:`\sigma_v > 0`.
    rho : float
        Correlation :math:`\rho \in [-1, 1]`.
    mu : float, default 0.0
        Log-return continuous drift :math:`\mu`.
    s0 : float, default 1.0
        Initial asset spot price :math:`S_0 > 0`.
    dt : float, optional
        Time step size :math:`\Delta t > 0`.
    n_steps : int, default 100
        Number of time steps.
    n_paths : int, default 1
        Number of sample paths.
    T : float, optional
        Terminal horizon :math:`T > 0`.
    rng : np.random.Generator, optional
        Random number generator.
    method : {"full_truncation", "euler"}, default "full_truncation"
        Discretisation algorithm:
        - "full_truncation": Lord et al. (2010) Full Truncation scheme.
        - "euler": Truncated Euler–Maruyama scheme.
    return_variance : bool, default False
        If True, returns tuple ``(paths_x, paths_v)``.
        If False, returns ``paths_x``.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Path array of shape ``(n_paths, n_steps + 1)``.
    """
    dt = simulation_grid(n_steps, n_paths, dt=dt, T=T)
    if not np.all(np.isfinite([v0, kappa, theta, sigma_v, rho, mu, s0])):
        raise ValueError("Simulation parameters must be finite")
    if v0 <= 0:
        raise ValueError("v0 must be strictly positive")
    if kappa <= 0 or theta <= 0 or sigma_v <= 0:
        raise ValueError("kappa, theta, and sigma_v must be strictly positive")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1]")
    if s0 <= 0:
        raise ValueError("s0 must be strictly positive")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    if dt is None:
        if T is not None:
            if T <= 0:
                raise ValueError("T must be strictly positive")
            dt = T / n_steps if n_steps > 0 else 1.0
        else:
            dt = 1.0 / n_steps if n_steps > 0 else 1.0
    else:
        if dt <= 0:
            raise ValueError("dt must be strictly positive")

    rng = rng or np.random.default_rng()
    paths_x = np.zeros((n_paths, n_steps + 1), dtype=float)
    paths_v = np.empty((n_paths, n_steps + 1), dtype=float)
    paths_v[:, 0] = float(v0)

    if n_steps == 0:
        if return_variance:
            return paths_x, paths_v
        return paths_x

    sqrt_dt = math.sqrt(dt)
    sqrt_1_minus_rho2 = math.sqrt(max(0.0, 1.0 - rho**2))

    curr_x = np.zeros(n_paths, dtype=float)
    curr_v = np.full(n_paths, float(v0), dtype=float)

    if method == "full_truncation":
        for k in range(n_steps):
            z1 = rng.standard_normal(size=n_paths)
            z_indep = rng.standard_normal(size=n_paths)
            z2 = rho * z1 + sqrt_1_minus_rho2 * z_indep

            v_pos = np.maximum(curr_v, 0.0)
            sqrt_v_pos = np.sqrt(v_pos)

            v_next = curr_v + kappa * (theta - v_pos) * dt + sigma_v * sqrt_v_pos * sqrt_dt * z1
            curr_x = curr_x + (mu - 0.5 * v_pos) * dt + sqrt_v_pos * sqrt_dt * z2
            curr_v = np.maximum(v_next, 0.0)

            paths_x[:, k + 1] = curr_x
            paths_v[:, k + 1] = curr_v

    elif method == "euler":
        for k in range(n_steps):
            z1 = rng.standard_normal(size=n_paths)
            z_indep = rng.standard_normal(size=n_paths)
            z2 = rho * z1 + sqrt_1_minus_rho2 * z_indep

            v_pos = np.maximum(curr_v, 0.0)
            sqrt_v_pos = np.sqrt(v_pos)

            curr_v = np.maximum(
                0.0, curr_v + kappa * (theta - v_pos) * dt + sigma_v * sqrt_v_pos * sqrt_dt * z1
            )
            curr_x = curr_x + (mu - 0.5 * v_pos) * dt + sqrt_v_pos * sqrt_dt * z2

            paths_x[:, k + 1] = curr_x
            paths_v[:, k + 1] = curr_v

    else:
        raise ValueError(
            f"Unknown simulation method '{method}'. Supported methods: 'full_truncation', 'euler'."
        )

    if return_variance:
        return paths_x, paths_v
    return paths_x


def sample_merton(
    drift: float,
    sigma: float,
    rate: float,
    jump_mean: float,
    jump_std: float,
    dt: float | None = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""
    Simulate exact sample paths of the Merton (1976) jump-diffusion process.

    .. math::

        X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} Y_i, \quad Y_i \sim \mathcal{N}(\mu_J,
        \sigma_J^2)

    Exact simulation with zero time discretization error for any step size :math:`\Delta t`.

    Parameters
    ----------
    drift : float
        Continuous drift :math:`\mu`.
    sigma : float
        Diffusion volatility :math:`\sigma > 0`.
    rate : float
        Jump arrival rate :math:`\lambda > 0`.
    jump_mean : float
        Mean jump size :math:`\mu_J`.
    jump_std : float
        Jump size standard deviation :math:`\sigma_J > 0`.
    dt : float, optional
        Time step size :math:`\Delta t > 0`.
    n_steps : int, default 100
        Number of time steps.
    n_paths : int, default 1
        Number of independent sample paths.
    T : float, optional
        Terminal time horizon :math:`T > 0`.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Path array of shape ``(n_paths, n_steps + 1)`` with ``paths[:, 0] == 0.0``.
    """
    dt = simulation_grid(n_steps, n_paths, dt=dt, T=T)
    if not np.all(np.isfinite([drift, sigma, rate, jump_mean, jump_std])):
        raise ValueError("Simulation parameters must be finite")
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive")
    if rate <= 0:
        raise ValueError("rate must be strictly positive")
    if jump_std <= 0:
        raise ValueError("jump_std must be strictly positive")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    if dt is None:
        if T is not None:
            if T <= 0:
                raise ValueError("T must be strictly positive")
            dt = T / n_steps if n_steps > 0 else 1.0
        else:
            dt = 1.0 / n_steps if n_steps > 0 else 1.0
    else:
        if dt <= 0:
            raise ValueError("dt must be strictly positive")

    rng = rng or np.random.default_rng()
    paths = np.zeros((n_paths, n_steps + 1), dtype=float)

    if n_steps == 0:
        return paths

    k_jumps = rng.poisson(lam=rate * dt, size=(n_paths, n_steps))
    z0 = rng.standard_normal(size=(n_paths, n_steps))
    z1 = rng.standard_normal(size=(n_paths, n_steps))

    sqrt_dt = math.sqrt(dt)
    jump_part = k_jumps * jump_mean + np.sqrt(k_jumps) * jump_std * z1
    increments = drift * dt + sigma * sqrt_dt * z0 + jump_part

    paths[:, 1:] = np.cumsum(increments, axis=1)
    return paths


def sample_cev(
    mu: float,
    sigma: float,
    gamma: float,
    s0: float = 100.0,
    dt: float | None = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: float | None = None,
    rng: np.random.Generator | None = None,
    method: str = "exact",
) -> np.ndarray:
    """CEV paths: exact Markov transitions or explicitly approximate Euler/Milstein.

    Set a=1-gamma, tau=integral_0^dt exp(-2*mu*a*s)ds and c=sigma²*a²*tau.
    For a>0, the absorbing squared-Bessel kernel has gamma-mixture weights
      w_k = exp(-q)*q**(k+nu)/Gamma(k+nu+1), q=S**(2a)/(2c), nu=1/(2a).
    These weights result by drawing H~Gamma(nu,1), absorbing if H>=q,
    and otherwise drawing K~Poisson(q-H), Z~Gamma(K+1,1). Then
      S_next=exp(mu*dt)*(2c*Z)**(1/(2a)).
    This directly samples the absorbing atom as well as the continuous law.
    For a<0, the reciprocal-Bessel transition is a standard noncentral chi-square;
    gamma=1 uses the exact lognormal law. No Milstein fallback is labeled exact.

    Kernel: Revuz & Yor (1999), Ch. XI; Schroder (1989). The mixture follows
    by expanding I_nu and a beta-integral; derivation in theory/refresh contracts.
    NumPy floating-point/random-generator parameter limits still apply.
    """
    dt = simulation_grid(n_steps, n_paths, dt=dt, T=T)
    if not np.all(np.isfinite([mu, sigma, gamma, s0])):
        raise ValueError("CEV simulation parameters must be finite")
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if s0 < 0:
        raise ValueError("s0 must be non-negative")
    valid = ("exact", "milstein", "euler", "absorbed_euler")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got {method}")
    rng = rng or np.random.default_rng()
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    if n_steps == 0 or s0 == 0:
        return paths
    if method == "exact" and gamma == 1:
        increments = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * rng.standard_normal(
            (n_paths, n_steps)
        )
        paths[:, 1:] = s0 * np.exp(np.cumsum(increments, axis=1))
        return paths
    current = np.full(n_paths, float(s0), dtype=float)
    if method == "exact":
        a = 1 - gamma
        rate = -2 * mu * a
        tau = math.expm1(rate * dt) / rate if rate != 0 else dt
        scale = sigma * sigma * a * a * tau
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("CEV exact transition exceeds numerical range")
        for step in range(1, n_steps + 1):
            alive = current > 0
            if not np.any(alive):
                break
            previous = current[alive]
            power = previous ** (2 * a) / scale
            if not np.all(np.isfinite(power)):
                raise ValueError("CEV transition noncentrality exceeds numerical range")
            if a > 0:
                q = power / 2
                hitting = rng.gamma(shape=1 / (2 * a), scale=1, size=previous.size)
                survives = hitting < q
                next_state = np.zeros_like(previous)
                if np.any(survives):
                    count = rng.poisson(q[survives] - hitting[survives])
                    z = rng.gamma(shape=count + 1, scale=1)
                    next_state[survives] = math.exp(mu * dt) * (2 * scale * z) ** (1 / (2 * a))
            else:
                z = rng.noncentral_chisquare(df=2 - 1 / a, nonc=power)
                next_state = math.exp(mu * dt) * (scale * z) ** (1 / (2 * a))
            if not np.all(np.isfinite(next_state)):
                raise ValueError("CEV transition state exceeds numerical range")
            current[alive] = next_state
            paths[:, step] = current
        return paths
    # These schemes have discretization/boundary bias. Zero remains absorbing.
    for step in range(1, n_steps + 1):
        alive = current > 0
        if not np.any(alive):
            break
        previous = current[alive]
        z = rng.standard_normal(previous.size)
        next_state = previous + mu * previous * dt + sigma * previous**gamma * math.sqrt(dt) * z
        if method == "milstein" and gamma > 0:
            next_state += (
                0.5 * gamma * sigma * sigma * previous ** (2 * gamma - 1) * dt * (z * z - 1)
            )
        current[alive] = np.maximum(next_state, 0)
        paths[:, step] = current
    return paths
