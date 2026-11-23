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
from typing import Optional, Sequence, Union, Tuple

import numpy as np
from scipy import linalg as la


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

    return sigma * (dt ** (1 / alpha)) * X


def sample_compound_poisson(
    rate: float,
    jump_sampler: callable,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exact increments of a compound Poisson process via thinning.

    For each interval of length Δt:
      N ~ Poisson(rate·Δt)
      ΔX = ∑_{i=1}^{N} J_i  where J_i ~ jump_sampler(rng, size)

    Parameters
    ----------
    rate :
        Jump rate λ (arrivals per unit time).
    jump_sampler :
        Callable (rng, size) → np.ndarray of jump sizes.
    dt, n_steps, n_paths :
        Standard parameters.
    rng :
        Generator.

    Returns
    -------
    np.ndarray
        Increments, shape (n_paths, n_steps).
    """
    jump_counts = rng.poisson(lam=rate * dt, size=(n_paths, n_steps))
    flat_counts = jump_counts.ravel()
    total_jumps = int(flat_counts.sum())

    if total_jumps > 0:
        try:
            cell_ids = np.repeat(np.arange(flat_counts.size), flat_counts)
            all_jumps = jump_sampler(rng, total_jumps)
            all_jumps = np.asarray(all_jumps, dtype=float).ravel()
            increments = np.bincount(
                cell_ids, weights=all_jumps, minlength=flat_counts.size
            ).reshape(n_paths, n_steps)
            return increments
        except Exception:
            pass

    increments = np.zeros((n_paths, n_steps), dtype=float)
    for i in range(n_paths):
        for k in range(n_steps):
            n_jumps = jump_counts[i, k]
            if n_jumps > 0:
                increments[i, k] = np.sum(jump_sampler(rng, n_jumps))
    return increments


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
    from an α-stable proposal (Rosiński 2007, Algorithm 6.1).

    The one-sided tempered stable (also called a CGMY subordinator) has
    Lévy density k(x) = C·x^{-1-α}·exp(-λx) for x > 0.

    Algorithm (Rosiński 2007):
      1. Sample X from positive α-stable with scale σ = (CΓ(1-α))^{1/α}.
      2. Accept with probability exp(-λ·X).
      3. Repeat until n_steps·n_paths samples accepted.

    Parameters
    ----------
    alpha :
        Stability index α ∈ (0, 1). Must be < 1 for a subordinator.
    C :
        Jump intensity C > 0.
    lam :
        Tempering parameter λ > 0.
    n_rejection_attempts :
        Max attempts per sample before falling back to Gamma approximation.

    Returns
    -------
    np.ndarray
        Increments, shape (n_paths, n_steps).

    References
    ----------
    Rosiński, J. (2007). *Stochastic Processes and their Applications*,
    117(6), 677–707. Algorithm 6.1.
    """
    if alpha >= 1:
        raise ValueError(
            f"Tempered stable subordinator requires alpha < 1, got {alpha}. "
            "For alpha >= 1 use CGMY directly."
        )
    from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

    sigma_stable = (C * float(gamma_fn(1 - alpha))) ** (1 / alpha)
    total = n_paths * n_steps
    samples = np.zeros(total)
    idx = 0

    while idx < total:
        needed = total - idx
        # Propose from positive α-stable (one-sided, β=1)
        U = rng.uniform(-np.pi / 2, np.pi / 2, size=needed * n_rejection_attempts)
        E = rng.exponential(1.0, size=needed * n_rejection_attempts)
        B = np.pi / (2 * alpha)
        S_alpha = (
            sigma_stable
            * (dt ** (1 / alpha))
            * np.sin(alpha * (U + B))
            / np.cos(U) ** (1 / alpha)
            * (np.cos(U - alpha * (U + B)) / E) ** ((1 - alpha) / alpha)
        )
        positive = S_alpha > 0
        S_alpha = S_alpha[positive]

        accept_prob = np.exp(-lam * S_alpha)
        accept_mask = rng.uniform(size=len(S_alpha)) < accept_prob
        accepted = S_alpha[accept_mask]

        take = min(len(accepted), needed)
        samples[idx : idx + take] = accepted[:take]
        idx += take

    return samples.reshape(n_paths, n_steps)


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
    rng: Optional[np.random.Generator] = None,
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
        c = 2.0 * kappa / (sigma**2 * (1.0 - exp_k))
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
    rng: Optional[np.random.Generator] = None,
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
        cond_var = (sigma**2 / (2.0 * theta)) * (1.0 - math.exp(-2.0 * theta * dt))
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
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    method: str = "full_truncation",
    return_variance: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Simulate exact sample paths of the Merton (1976) jump-diffusion process.

    .. math::

        X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} Y_i, \quad Y_i \sim \mathcal{N}(\mu_J, \sigma_J^2)

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
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    method: str = "exact",
) -> np.ndarray:
    r"""
    Simulate Constant Elasticity of Variance (CEV) diffusion paths.

    .. math::

        dS_t = \mu S_t dt + \sigma S_t^\gamma dW_t, \quad S_0 = s_0 \ge 0

    For :math:`\gamma < 1`, zero is an absorbing boundary: once a path reaches
    zero, it remains at zero indefinitely.

    Parameters
    ----------
    mu : float
        Drift rate :math:`\mu`.
    sigma : float
        Volatility scale parameter :math:`\sigma > 0`.
    gamma : float
        Elasticity power exponent :math:`\gamma \ge 0`.
    s0 : float, default 100.0
        Initial asset price :math:`s_0 \ge 0`.
    dt : float, optional
        Time step size :math:`\Delta t > 0`.
    n_steps : int, default 100
        Number of simulation time steps.
    n_paths : int, default 1
        Number of independent sample paths.
    T : float, optional
        Terminal time horizon :math:`T > 0`.
    rng : np.random.Generator, optional
        Random number generator.
    method : {"exact", "milstein", "euler", "absorbed_euler"}, default "exact"
        Simulation scheme:
        - "exact": Exact geometric Brownian motion when :math:`\gamma = 1`;
          second-order Milstein scheme with exact absorbing zero boundary when :math:`\gamma \ne 1`.
        - "milstein": Milstein scheme with absorbing zero boundary.
        - "euler" / "absorbed_euler": Euler–Maruyama scheme with absorbing zero boundary.

    Returns
    -------
    np.ndarray
        Path array of shape ``(n_paths, n_steps + 1)`` with ``paths[:, 0] == s0``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if s0 < 0:
        raise ValueError("s0 must be non-negative")
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

    valid_methods = ("exact", "milstein", "euler", "absorbed_euler")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method}")

    rng = rng or np.random.default_rng()
    paths = np.zeros((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(s0)

    if n_steps == 0 or s0 == 0.0:
        return paths

    # Case 1: Exact geometric Brownian motion when gamma == 1
    if abs(gamma - 1.0) < 1e-7 or (method == "exact" and abs(gamma - 1.0) < 1e-7):
        z = rng.standard_normal((n_paths, n_steps))
        log_inc = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
        paths[:, 1:] = s0 * np.exp(np.cumsum(log_inc, axis=1))
        return paths

    # Step-by-step simulation with absorption at 0
    curr = np.full(n_paths, float(s0), dtype=float)
    sqrt_dt = math.sqrt(dt)
    use_milstein = method in ("exact", "milstein")

    for step in range(1, n_steps + 1):
        pos = curr > 0.0
        if not np.any(pos):
            # All paths absorbed at 0
            break

        z = rng.standard_normal(n_paths)
        s_pos = curr[pos]
        z_pos = z[pos]

        drift = mu * s_pos * dt
        diff = sigma * (s_pos**gamma) * sqrt_dt * z_pos

        if use_milstein and gamma > 0.0:
            mil = 0.5 * gamma * (sigma**2) * (s_pos ** (2.0 * gamma - 1.0)) * dt * (z_pos**2 - 1.0)
            s_next = s_pos + drift + diff + mil
        else:
            s_next = s_pos + drift + diff

        # Absorbing boundary at 0
        curr[pos] = np.maximum(s_next, 0.0)
        paths[:, step] = curr

    return paths


def sample_kou(
    drift: float,
    sigma: float,
    rate: float,
    p: float,
    eta1: float,
    eta2: float,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Simulate exact sample paths of the Kou (2002) jump-diffusion process.

    .. math::

        X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} Y_i

    where :math:`Y_i` has an asymmetric double exponential density:
    with probability :math:`p`, :math:`Y \sim \mathrm{Exp}(\eta_1)` (upward jump);
    with probability :math:`1 - p`, :math:`-Y \sim \mathrm{Exp}(\eta_2)` (downward jump).

    Exact in law: zero time discretization error.

    Parameters
    ----------
    drift : float
        Continuous drift parameter :math:`\mu`.
    sigma : float
        Continuous diffusion volatility :math:`\sigma > 0`.
    rate : float
        Jump arrival intensity :math:`\lambda > 0`.
    p : float
        Probability of upward jump :math:`p \in [0, 1]`.
    eta1 : float
        Upward jump decay rate :math:`\eta_1 > 1`.
    eta2 : float
        Downward jump decay rate :math:`\eta_2 > 0`.
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
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive")
    if rate <= 0:
        raise ValueError("rate must be strictly positive")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if eta1 <= 1.0:
        raise ValueError(f"eta1 must be > 1.0, got {eta1}")
    if eta2 <= 0.0:
        raise ValueError(f"eta2 must be > 0.0, got {eta2}")
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
    total_jumps = int(np.sum(k_jumps))

    jump_increments = np.zeros((n_paths, n_steps), dtype=float)
    if total_jumps > 0:
        is_up = rng.uniform(0.0, 1.0, size=total_jumps) < p
        jump_vals = np.where(
            is_up,
            rng.exponential(1.0 / eta1, size=total_jumps),
            -rng.exponential(1.0 / eta2, size=total_jumps),
        )
        rows, cols = np.where(k_jumps > 0)
        counts = k_jumps[rows, cols]
        repeated_rows = np.repeat(rows, counts)
        repeated_cols = np.repeat(cols, counts)
        np.add.at(jump_increments, (repeated_rows, repeated_cols), jump_vals)

    z = rng.standard_normal((n_paths, n_steps))
    continuous_inc = drift * dt + sigma * math.sqrt(dt) * z
    paths[:, 1:] = np.cumsum(continuous_inc + jump_increments, axis=1)
    return paths


def sample_bates(
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    rate: float = 1.0,
    jump_mean: float = -0.05,
    jump_std: float = 0.1,
    mu: float = 0.0,
    s0: float = 1.0,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    method: str = "full_truncation",
    return_variance: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""
    Simulate sample paths of the Bates (1996) stochastic volatility with jumps model.

    .. math::

        dX_t = \left(\mu - \lambda \bar{k} - \frac{1}{2} V_t\right) dt + \sqrt{V_t} \, dW_t^S + Y \, dN_t, \quad X_0 = 0 \\
        dV_t = \kappa (\theta - V_t) dt + \sigma_v \sqrt{V_t} \, dW_t^V, \quad V_0 = v_0

    with :math:`\mathbb{E}[dW_t^S dW_t^V] = \rho \, dt`, Poisson intensity :math:`\lambda`,
    and :math:`Y \sim \mathcal{N}(\mu_J, \sigma_J^2)`.

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
    rate : float
        Poisson jump intensity :math:`\lambda \ge 0`. Default 1.0.
    jump_mean : float
        Mean log-jump size :math:`\mu_J`. Default -0.05.
    jump_std : float
        Log-jump standard deviation :math:`\sigma_J > 0`. Default 0.10.
    mu : float
        Continuous drift rate :math:`\mu`. Default 0.0.
    s0 : float
        Initial spot price :math:`S_0 > 0`. Default 1.0.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps. Default 100.
    n_paths : int
        Number of independent paths. Default 1.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.
    method : str
        Simulation method for variance: ``'full_truncation'`` (Lord et al. 2010) or ``'euler'``.
    return_variance : bool
        If True, returns tuple ``(paths_x, paths_v)``. Default False.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Path array of shape ``(n_paths, n_steps + 1)``.
    """
    if v0 <= 0:
        raise ValueError("v0 must be strictly positive")
    if kappa <= 0 or theta <= 0 or sigma_v <= 0:
        raise ValueError("kappa, theta, and sigma_v must be strictly positive")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1]")
    if rate < 0:
        raise ValueError("rate must be non-negative")
    if jump_std <= 0:
        raise ValueError("jump_std must be strictly positive")
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
    k_bar = math.exp(jump_mean + 0.5 * jump_std**2) - 1.0

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

            if rate > 0.0:
                n_jumps = rng.poisson(lam=rate * dt, size=n_paths)
                jump_shock = np.where(
                    n_jumps > 0,
                    n_jumps * jump_mean
                    + np.sqrt(n_jumps) * jump_std * rng.standard_normal(size=n_paths),
                    0.0,
                )
                curr_x = (
                    curr_x
                    + (mu - rate * k_bar - 0.5 * v_pos) * dt
                    + sqrt_v_pos * sqrt_dt * z2
                    + jump_shock
                )
            else:
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

            v_next = np.maximum(
                0.0, curr_v + kappa * (theta - v_pos) * dt + sigma_v * sqrt_v_pos * sqrt_dt * z1
            )

            if rate > 0.0:
                n_jumps = rng.poisson(lam=rate * dt, size=n_paths)
                jump_shock = np.where(
                    n_jumps > 0,
                    n_jumps * jump_mean
                    + np.sqrt(n_jumps) * jump_std * rng.standard_normal(size=n_paths),
                    0.0,
                )
                curr_x = (
                    curr_x
                    + (mu - rate * k_bar - 0.5 * v_pos) * dt
                    + sqrt_v_pos * sqrt_dt * z2
                    + jump_shock
                )
            else:
                curr_x = curr_x + (mu - 0.5 * v_pos) * dt + sqrt_v_pos * sqrt_dt * z2

            curr_v = v_next

            paths_x[:, k + 1] = curr_x
            paths_v[:, k + 1] = curr_v

    else:
        raise ValueError(
            f"Unknown simulation method '{method}'. Supported methods: 'full_truncation', 'euler'."
        )

    if return_variance:
        return paths_x, paths_v
    return paths_x


def sample_hull_white(
    a: float,
    sigma: float,
    b: float = 0.03,
    r0: float = 0.03,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    method: str = "exact",
) -> np.ndarray:
    r"""
    Simulate sample paths of the Hull-White (1990) Gaussian short rate process.

    .. math::

        dr_t = a(b - r_t) dt + \sigma dW_t, \quad r_0 \in \mathbb{R}

    Parameters
    ----------
    a : float
        Mean reversion speed :math:`a > 0`.
    sigma : float
        Short-rate volatility :math:`\sigma > 0`.
    b : float
        Long-term rate level :math:`b \in \mathbb{R}`. Default 0.03.
    r0 : float
        Initial rate :math:`r_0`. Default 0.03.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of steps. Default 100.
    n_paths : int
        Number of paths. Default 1.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.
    method : str
        Simulation method: ``'exact'`` (Gaussian conditional transition with zero
        discretization error) or ``'euler'``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n_steps + 1)``.
    """
    if a <= 0:
        raise ValueError("Mean-reversion speed 'a' must be strictly positive")
    if sigma <= 0:
        raise ValueError("Volatility 'sigma' must be strictly positive")
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
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(r0)

    if n_steps == 0:
        return paths

    if method == "exact":
        exp_adt = math.exp(-a * dt)
        std_step = math.sqrt((sigma**2 / (2.0 * a)) * (1.0 - math.exp(-2.0 * a * dt)))
        for k in range(n_steps):
            z = rng.standard_normal(size=n_paths)
            paths[:, k + 1] = b + (paths[:, k] - b) * exp_adt + std_step * z

    elif method == "euler":
        sqrt_dt = math.sqrt(dt)
        for k in range(n_steps):
            z = rng.standard_normal(size=n_paths)
            paths[:, k + 1] = paths[:, k] + a * (b - paths[:, k]) * dt + sigma * sqrt_dt * z

    else:
        raise ValueError(
            f"Unknown simulation method '{method}'. Supported methods: 'exact', 'euler'."
        )

    return paths


def sample_sabr(
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
    f0: float = 100.0,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    method: str = "log_euler",
    return_volatility: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""
    Simulate sample paths of the SABR stochastic volatility model.

    .. math::

        dF_t = \alpha_t F_t^\beta dW_t^F, \quad F_0 = f_0 > 0 \\
        d\alpha_t = \nu \alpha_t dW_t^\alpha, \quad \alpha_0 = \alpha > 0

    with :math:`\mathbb{E}[dW_t^F dW_t^\alpha] = \rho dt`.

    Parameters
    ----------
    alpha : float
        Initial volatility :math:`\alpha > 0`.
    beta : float
        Elasticity power exponent :math:`\beta \in [0, 1]`.
    nu : float
        Vol-of-vol :math:`\nu \ge 0`.
    rho : float
        Correlation :math:`\rho \in [-1, 1]`.
    f0 : float
        Initial forward :math:`F_0 > 0`. Default 100.0.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps. Default 100.
    n_paths : int
        Number of paths. Default 1.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.
    method : str
        Simulation method: ``'log_euler'`` (exact geometric stepping for :math:`\alpha_t`)
        or ``'euler'``.
    return_volatility : bool
        If True, returns tuple ``(paths_f, paths_alpha)``. Default False.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Path array of shape ``(n_paths, n_steps + 1)``.
    """
    if alpha <= 0:
        raise ValueError("Initial volatility alpha must be strictly positive")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("Elasticity exponent beta must lie in [0, 1]")
    if nu < 0:
        raise ValueError("Vol-of-vol nu must be non-negative")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("Correlation rho must lie in [-1, 1]")
    if f0 <= 0:
        raise ValueError("Forward f0 must be strictly positive")
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
    paths_f = np.empty((n_paths, n_steps + 1), dtype=float)
    paths_a = np.empty((n_paths, n_steps + 1), dtype=float)
    paths_f[:, 0] = float(f0)
    paths_a[:, 0] = float(alpha)

    if n_steps == 0:
        if return_volatility:
            return paths_f, paths_a
        return paths_f

    sqrt_dt = math.sqrt(dt)
    sqrt_1_minus_rho2 = math.sqrt(max(0.0, 1.0 - rho**2))

    curr_f = np.full(n_paths, float(f0), dtype=float)
    curr_a = np.full(n_paths, float(alpha), dtype=float)

    if method == "log_euler":
        for k in range(n_steps):
            z_vol = rng.standard_normal(size=n_paths)
            z_indep = rng.standard_normal(size=n_paths)
            z_fwd = rho * z_vol + sqrt_1_minus_rho2 * z_indep

            next_a = curr_a * np.exp(-0.5 * (nu**2) * dt + nu * sqrt_dt * z_vol)
            f_pos = np.maximum(curr_f, 0.0)
            next_f = curr_f + curr_a * (f_pos**beta) * sqrt_dt * z_fwd

            if beta < 1.0:
                next_f = np.maximum(0.0, next_f)

            curr_f = next_f
            curr_a = next_a
            paths_f[:, k + 1] = curr_f
            paths_a[:, k + 1] = curr_a

    elif method == "euler":
        for k in range(n_steps):
            z_vol = rng.standard_normal(size=n_paths)
            z_indep = rng.standard_normal(size=n_paths)
            z_fwd = rho * z_vol + sqrt_1_minus_rho2 * z_indep

            next_a = np.maximum(1e-10, curr_a + nu * curr_a * sqrt_dt * z_vol)
            f_pos = np.maximum(curr_f, 0.0)
            next_f = curr_f + curr_a * (f_pos**beta) * sqrt_dt * z_fwd

            if beta < 1.0:
                next_f = np.maximum(0.0, next_f)

            curr_f = next_f
            curr_a = next_a
            paths_f[:, k + 1] = curr_f
            paths_a[:, k + 1] = curr_a

    else:
        raise ValueError(
            f"Unknown simulation method '{method}'. Supported methods: 'log_euler', 'euler'."
        )

    if return_volatility:
        return paths_f, paths_a
    return paths_f


def sample_gh(
    lambda_param: float = 1.0,
    alpha: float = 1.5,
    beta: float = 0.0,
    delta: float = 1.0,
    mu: float = 0.0,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Simulate sample paths of the Generalized Hyperbolic (GH) Lévy process.

    Parameters
    ----------
    lambda_param : float
        Shape parameter :math:`\lambda \in \mathbb{R}`.
    alpha : float
        Tail parameter :math:`\alpha > 0`.
    beta : float
        Asymmetry parameter :math:`\beta \in (-\alpha, \alpha)`.
    delta : float
        Scale parameter :math:`\delta > 0`.
    mu : float
        Drift parameter :math:`\mu \in \mathbb{R}`.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Terminal time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Simulated paths of shape ``(n_paths, n_steps + 1)`` with :math:`X_0 = 0`.
    """
    if alpha <= 0.0:
        raise ValueError("Parameter alpha must be strictly positive")
    if delta <= 0.0:
        raise ValueError("Parameter delta must be strictly positive")
    if abs(beta) >= alpha:
        raise ValueError("|beta| must be strictly less than alpha")
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

    paths = np.zeros((n_paths, n_steps + 1), dtype=float)
    if n_steps == 0:
        return paths

    rng = rng or np.random.default_rng()
    gamma = math.sqrt(alpha**2 - beta**2)

    # Fast exact NIG subordination when lambda = -0.5
    if math.isclose(lambda_param, -0.5, abs_tol=1e-7):
        ig_mean = delta * dt / gamma
        ig_shape = (delta * dt) ** 2
        ig_samples = rng.wald(mean=ig_mean, scale=ig_shape, size=(n_paths, n_steps))
        normals = rng.standard_normal(size=(n_paths, n_steps))
        increments = mu * dt + beta * ig_samples + np.sqrt(ig_samples) * normals
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    # General GH: FFT inversion of the characteristic function. The Bessel
    # term uses kve, so its logarithm is continuous (see GH.char_exponent).
    from scipy.special import kv as bessel_kv  # type: ignore[import-untyped]
    from scipy.special import kve as bessel_kve  # type: ignore[import-untyped]

    from spxa.sim.cf_sampling import sample_from_cf

    dg = delta * gamma
    k0 = float(bessel_kv(lambda_param, dg))
    scale = delta / gamma
    m1_z = scale * float(bessel_kv(lambda_param + 1.0, dg)) / k0
    m2_z = (scale**2) * float(bessel_kv(lambda_param + 2.0, dg)) / k0
    k1 = mu + beta * m1_z
    k2 = m1_z + (beta**2) * (m2_z - m1_z**2)
    k0e = float(bessel_kve(lambda_param, dg))

    def cf(u: np.ndarray) -> np.ndarray:
        zeta = np.sqrt(alpha**2 - (beta + 1j * u) ** 2)
        psi = (
            1j * mu * u
            + lambda_param * np.log(gamma / zeta)
            + np.log(bessel_kve(lambda_param, delta * zeta) / k0e)
            - delta * (zeta - gamma)
        )
        return np.exp(dt * psi)

    increments = sample_from_cf(
        cf,
        mean=dt * k1,
        std=math.sqrt(dt * k2),
        size=(n_paths, n_steps),
        rng=rng,
        tail_decay=alpha - abs(beta),
    )

    paths[:, 1:] = np.cumsum(increments, axis=1)
    return paths


def sample_black_karasinski(
    a: float,
    sigma: float,
    b: Optional[float] = None,
    r0: float = 0.03,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    return_log_rate: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""
    Simulate sample paths of the Black-Karasinski lognormal short-rate process.

    .. math::

        d(\ln r_t) = a (b - \ln r_t) dt + \sigma dW_t, \quad r_0 > 0
        r_t = \exp(x_t), \quad x_t = \ln(r_t)

    The log-rate :math:`x_t` is sampled via exact Gaussian Ornstein-Uhlenbeck transitions,
    guaranteeing zero time-discretization error for :math:`x_t` and strictly positive rates
    :math:`r_t > 0` almost surely.

    Parameters
    ----------
    a : float
        Mean-reversion speed :math:`a > 0`.
    sigma : float
        Volatility of log-rate :math:`\sigma > 0`.
    b : float, optional
        Long-term mean level of log-rate. Default :math:`\ln(r_0)`.
    r0 : float
        Initial short rate :math:`r_0 > 0`. Default 0.03.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.
    return_log_rate : bool
        If True, returns tuple ``(r_paths, x_paths)``. Default False.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Short-rate paths array of shape ``(n_paths, n_steps + 1)``.
    """
    if a <= 0.0:
        raise ValueError("Mean-reversion speed 'a' must be strictly positive")
    if sigma <= 0.0:
        raise ValueError("Volatility 'sigma' must be strictly positive")
    if r0 <= 0.0:
        raise ValueError("Initial short rate 'r0' must be strictly positive")
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
    x0 = math.log(r0)
    b_val = float(b) if b is not None else x0

    x_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    x_paths[:, 0] = x0

    if n_steps == 0:
        r_paths = np.full((n_paths, 1), float(r0), dtype=float)
        if return_log_rate:
            return r_paths, x_paths
        return r_paths

    exp_adt = math.exp(-a * dt)
    std_step = math.sqrt((sigma**2 / (2.0 * a)) * (1.0 - math.exp(-2.0 * a * dt)))

    for k in range(n_steps):
        z = rng.standard_normal(size=n_paths)
        x_paths[:, k + 1] = b_val + (x_paths[:, k] - b_val) * exp_adt + std_step * z

    r_paths = np.exp(x_paths)
    r_paths[:, 0] = float(r0)
    if return_log_rate:
        return r_paths, x_paths
    return r_paths


def sample_schobel_zhu(
    sigma0: float = 0.20,
    kappa: float = 1.5,
    theta: float = 0.20,
    tau: float = 0.30,
    rho: float = -0.5,
    s0: float = 100.0,
    rate: float = 0.0,
    dividend: float = 0.0,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    return_volatility: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""
    Simulate sample paths of the Schöbel-Zhu stochastic volatility model.

    .. math::

        dS_t = (r - q) S_t dt + \sigma_t S_t dW_t^S \\
        d\sigma_t = \kappa (\theta - \sigma_t) dt + \tau dW_t^\sigma

    with :math:`\mathbb{E}[dW_t^S dW_t^\sigma] = \rho dt`.

    Parameters
    ----------
    sigma0 : float
        Initial volatility :math:`\sigma_0`.
    kappa : float
        Mean-reversion speed :math:`\kappa > 0`.
    theta : float
        Long-term mean volatility :math:`\theta`.
    tau : float
        Vol-of-vol :math:`\tau > 0`.
    rho : float
        Correlation :math:`\rho \in [-1, 1]`.
    s0 : float
        Initial asset price :math:`S_0 > 0`.
    rate : float
        Risk-free rate.
    dividend : float
        Dividend yield.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random generator.
    return_volatility : bool
        If True, returns tuple ``(paths_s, paths_sigma)``. Default False.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Asset price paths array of shape ``(n_paths, n_steps + 1)``.
    """
    if kappa <= 0.0:
        raise ValueError("Mean-reversion speed 'kappa' must be strictly positive")
    if tau <= 0.0:
        raise ValueError("Vol-of-vol 'tau' must be strictly positive")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("Correlation 'rho' must lie in [-1, 1]")
    if s0 <= 0.0:
        raise ValueError("Initial price 's0' must be strictly positive")
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

    paths_s = np.empty((n_paths, n_steps + 1), dtype=float)
    paths_sigma = np.empty((n_paths, n_steps + 1), dtype=float)
    paths_s[:, 0] = float(s0)
    paths_sigma[:, 0] = float(sigma0)

    if n_steps == 0:
        if return_volatility:
            return paths_s, paths_sigma
        return paths_s

    exp_kdt = math.exp(-kappa * dt)
    std_step = math.sqrt((tau**2 / (2.0 * kappa)) * (1.0 - math.exp(-2.0 * kappa * dt)))
    sqrt_dt = math.sqrt(dt)
    sqrt_1_minus_rho2 = math.sqrt(max(0.0, 1.0 - rho**2))
    drift_const = (rate - dividend) * dt

    curr_s = np.full(n_paths, float(s0), dtype=float)
    curr_sigma = np.full(n_paths, float(sigma0), dtype=float)

    for k in range(n_steps):
        z_vol = rng.standard_normal(size=n_paths)
        z_indep = rng.standard_normal(size=n_paths)
        z_s = rho * z_vol + sqrt_1_minus_rho2 * z_indep

        next_sigma = theta + (curr_sigma - theta) * exp_kdt + std_step * z_vol
        next_s = curr_s * np.exp(
            drift_const - 0.5 * (curr_sigma**2) * dt + curr_sigma * sqrt_dt * z_s
        )

        curr_s = next_s
        curr_sigma = next_sigma
        paths_s[:, k + 1] = curr_s
        paths_sigma[:, k + 1] = curr_sigma

    if return_volatility:
        return paths_s, paths_sigma
    return paths_s


def sample_besq(
    dim: float = 2.0,
    x0: float = 1.0,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Simulate exact sample paths of the Squared Bessel process :math:`\mathrm{BESQ}^\delta(x_0)`.

    .. math::

        dX_t = \delta dt + 2 \sqrt{X_t} dW_t, \quad X_0 = x_0 \ge 0

    Transitions follow exact scaled non-central chi-squared increments with zero
    time discretization error:

    .. math::

        \frac{X_{t+\Delta t}}{\Delta t} \sim \chi^2\left(\delta, \frac{X_t}{\Delta t}\right)

    Parameters
    ----------
    dim : float
        Dimension parameter :math:`\delta \ge 0`.
    x0 : float
        Initial value :math:`X_0 \ge 0`.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Sample paths of shape ``(n_paths, n_steps + 1)``.
    """
    if dim < 0.0:
        raise ValueError("Dimension 'dim' must be non-negative")
    if x0 < 0.0:
        raise ValueError("Initial value 'x0' must be non-negative")
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

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(x0)

    if n_steps == 0:
        return paths

    rng = rng or np.random.default_rng()

    if dim > 0.0:
        for k in range(n_steps):
            curr_x = np.maximum(paths[:, k], 0.0)
            nc = curr_x / dt
            paths[:, k + 1] = rng.noncentral_chisquare(df=dim, nonc=nc) * dt
    else:
        # dim == 0 (absorbing boundary at 0)
        for k in range(n_steps):
            curr_x = paths[:, k]
            pos = curr_x > 0.0
            next_x = np.zeros(n_paths, dtype=float)
            if np.any(pos):
                nc = curr_x[pos] / dt
                # df=0 non-central chi-square is compound Poisson: N ~ Pois(nc/2), chisq(2N)
                poiss = rng.poisson(lam=0.5 * nc)
                pos_poiss = poiss > 0
                res = np.zeros_like(nc)
                if np.any(pos_poiss):
                    res[pos_poiss] = rng.chisquare(df=2 * poiss[pos_poiss]) * dt
                next_x[pos] = res
            paths[:, k + 1] = next_x

    return paths


def sample_bessel(
    dim: float = 2.0,
    r0: float = 1.0,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Simulate exact sample paths of the Bessel process :math:`\mathrm{BES}^\delta(r_0)`.

    .. math::

        R_t = \sqrt{X_t}, \quad X_t \sim \mathrm{BESQ}^\delta(r_0^2)

    Parameters
    ----------
    dim : float
        Dimension parameter :math:`\delta \ge 0`.
    r0 : float
        Initial value :math:`R_0 \ge 0`.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Sample paths of shape ``(n_paths, n_steps + 1)``.
    """
    if r0 < 0.0:
        raise ValueError("Initial value 'r0' must be non-negative")
    besq_paths = sample_besq(
        dim=dim,
        x0=r0**2,
        dt=dt,
        n_steps=n_steps,
        n_paths=n_paths,
        T=T,
        rng=rng,
    )
    return np.sqrt(np.maximum(besq_paths, 0.0))


def sample_jcir(
    kappa: float = 1.0,
    theta: float = 0.05,
    sigma: float = 0.15,
    lambda_j: float = 0.5,
    mu_j: float = 0.02,
    r0: float = 0.04,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Simulate sample paths of the Jump-Extended CIR (JCIR) process.

    .. math::

        dr_t = \kappa (\theta - r_t) dt + \sigma \sqrt{r_t} dW_t + dJ_t, \quad r_0 \ge 0

    Transitions follow exact scaled non-central chi-squared increments for the CIR
    diffusion component, compounded with Poisson exponential jump arrivals.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed :math:`\kappa > 0`.
    theta : float
        Diffusion long-term mean :math:`\theta > 0`.
    sigma : float
        Volatility-of-rate :math:`\sigma > 0`.
    lambda_j : float
        Jump arrival intensity :math:`\lambda_J \ge 0`.
    mu_j : float
        Mean jump size :math:`\mu_J > 0`.
    r0 : float
        Initial short rate :math:`r_0 \ge 0`.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Sample paths of shape ``(n_paths, n_steps + 1)``.
    """
    if kappa <= 0:
        raise ValueError("kappa must be strictly positive")
    if theta <= 0:
        raise ValueError("theta must be strictly positive")
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive")
    if lambda_j < 0:
        raise ValueError("lambda_j must be non-negative")
    if mu_j <= 0:
        raise ValueError("mu_j must be strictly positive")
    if r0 < 0:
        raise ValueError("r0 must be non-negative")
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

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(r0)

    if n_steps == 0:
        return paths

    rng = rng or np.random.default_rng()

    exp_kdt = math.exp(-kappa * dt)
    c = 2.0 * kappa / (sigma**2 * (1.0 - exp_kdt))
    df = 4.0 * kappa * theta / (sigma**2)

    for k in range(n_steps):
        curr_r = np.maximum(paths[:, k], 0.0)
        nc = 2.0 * c * curr_r * exp_kdt
        cir_next = rng.noncentral_chisquare(df=df, nonc=nc) / (2.0 * c)

        if lambda_j > 0.0:
            num_jumps = rng.poisson(lam=lambda_j * dt, size=n_paths)
            has_jumps = num_jumps > 0
            if np.any(has_jumps):
                jump_shocks = rng.gamma(shape=num_jumps[has_jumps], scale=mu_j)
                cir_next[has_jumps] += jump_shocks

        paths[:, k + 1] = cir_next

    return paths


def sample_schwartz_two_factor(
    s0: float = 20.0,
    delta0: float = 0.05,
    kappa: float = 1.2,
    alpha_hat: float = 0.08,
    sigma1: float = 0.30,
    sigma2: float = 0.20,
    rho: float = 0.60,
    rate: float = 0.05,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    return_yield: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""
    Simulate sample paths of the Schwartz Two-Factor commodity model.

    .. math::

        dX_t = (r - \delta_t - \frac{1}{2}\sigma_1^2) dt + \sigma_1 dW_1^Q(t), \quad S_t = \exp(X_t) \\
        d\delta_t = \kappa (\hat{\alpha} - \delta_t) dt + \sigma_2 dW_2^Q(t) \\
        \mathbb{E}[dW_1^Q dW_2^Q] = \rho dt

    Parameters
    ----------
    s0 : float
        Initial commodity spot price :math:`S_0 > 0`.
    delta0 : float
        Initial convenience yield :math:`\delta_0 \in \mathbb{R}`.
    kappa : float
        Mean-reversion speed of convenience yield :math:`\kappa > 0`.
    alpha_hat : float
        Risk-neutral long-term convenience yield level :math:`\hat{\alpha}`.
    sigma1 : float
        Commodity spot price volatility :math:`\sigma_1 > 0`.
    sigma2 : float
        Convenience yield volatility :math:`\sigma_2 > 0`.
    rho : float
        Spot-yield correlation :math:`\rho \in [-1, 1]`.
    rate : float
        Risk-free interest rate :math:`r`.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.
    return_yield : bool, optional
        If True, returns tuple ``(spot_paths, yield_paths)``. Default False.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        Spot price paths of shape ``(n_paths, n_steps + 1)``, or tuple with yield paths.
    """
    if s0 <= 0:
        raise ValueError("Initial spot price 's0' must be strictly positive")
    if kappa <= 0:
        raise ValueError("Mean-reversion speed 'kappa' must be strictly positive")
    if sigma1 <= 0:
        raise ValueError("Spot volatility 'sigma1' must be strictly positive")
    if sigma2 <= 0:
        raise ValueError("Yield volatility 'sigma2' must be strictly positive")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("Correlation 'rho' must be in [-1, 1]")
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

    paths_x = np.empty((n_paths, n_steps + 1), dtype=float)
    paths_delta = np.empty((n_paths, n_steps + 1), dtype=float)

    paths_x[:, 0] = math.log(s0)
    paths_delta[:, 0] = float(delta0)

    if n_steps == 0:
        paths_s = np.full((n_paths, 1), float(s0))
        if return_yield:
            return paths_s, paths_delta
        return paths_s

    rng = rng or np.random.default_rng()
    sq_dt = math.sqrt(dt)
    sqrt_1_rho2 = math.sqrt(max(0.0, 1.0 - rho**2))

    for k in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2_ind = rng.standard_normal(n_paths)
        z2 = rho * z1 + sqrt_1_rho2 * z2_ind

        curr_x = paths_x[:, k]
        curr_delta = paths_delta[:, k]

        next_x = curr_x + (rate - curr_delta - 0.5 * (sigma1**2)) * dt + sigma1 * sq_dt * z1
        next_delta = curr_delta + kappa * (alpha_hat - curr_delta) * dt + sigma2 * sq_dt * z2

        paths_x[:, k + 1] = next_x
        paths_delta[:, k + 1] = next_delta

    paths_s = np.exp(paths_x)
    paths_s[:, 0] = float(s0)

    if return_yield:
        return paths_s, paths_delta
    return paths_s


def sample_bns(
    lambda_param: float = 1.5,
    alpha: float = 2.0,
    beta_bdlp: float = 25.0,
    rho: float = -0.50,
    beta: float = -0.50,
    sigma0_sq: float = 0.04,
    s0: float = 100.0,
    rate: float = 0.05,
    dividend: float = 0.0,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    return_variance: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""
    Simulate sample paths of the Barndorff-Nielsen & Shephard (BNS) model.

    .. math::

        dX_t = (\mu + \beta \sigma_t^2) dt + \sigma_t dW_t + \rho dZ_{\lambda t}, \quad S_t = S_0 \exp(X_t) \\
        d\sigma_t^2 = -\lambda \sigma_t^2 dt + dZ_{\lambda t}

    Parameters
    ----------
    lambda_param : float
        Variance mean-reversion rate :math:`\lambda > 0`.
    alpha : float
        BDLP Gamma shape :math:`\alpha > 0`.
    beta_bdlp : float
        BDLP Gamma rate :math:`\beta_{bdlp} > 0`.
    rho : float
        Leverage parameter :math:`\rho \le 0`.
    beta : float
        Drift risk parameter :math:`\beta`.
    sigma0_sq : float
        Initial variance :math:`\sigma_0^2 > 0`.
    s0 : float
        Initial asset price :math:`S_0 > 0`.
    rate : float
        Risk-free interest rate :math:`r`.
    dividend : float
        Continuous dividend yield :math:`q`.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.
    return_variance : bool, optional
        If True, returns tuple ``(spot_paths, variance_paths)``. Default False.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        Asset price paths of shape ``(n_paths, n_steps + 1)``, or tuple with variance paths.
    """
    if lambda_param <= 0:
        raise ValueError("Mean-reversion rate 'lambda_param' must be strictly positive")
    if alpha <= 0:
        raise ValueError("BDLP parameter 'alpha' must be strictly positive")
    if beta_bdlp <= 0:
        raise ValueError("BDLP parameter 'beta_bdlp' must be strictly positive")
    if rho > 0:
        raise ValueError("Leverage parameter 'rho' must be non-positive")
    if sigma0_sq <= 0:
        raise ValueError("Initial variance 'sigma0_sq' must be strictly positive")
    if s0 <= 0:
        raise ValueError("Initial asset price 's0' must be strictly positive")
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

    paths_x = np.empty((n_paths, n_steps + 1), dtype=float)
    paths_var = np.empty((n_paths, n_steps + 1), dtype=float)

    paths_x[:, 0] = 0.0
    paths_var[:, 0] = float(sigma0_sq)

    if n_steps == 0:
        paths_s = np.full((n_paths, 1), float(s0))
        if return_variance:
            return paths_s, paths_var
        return paths_s

    rng = rng or np.random.default_rng()
    sq_dt = math.sqrt(dt)
    exp_ldt = math.exp(-lambda_param * dt)
    jump_rate = alpha * lambda_param * dt

    k_rho = (alpha * rho) / (beta_bdlp - rho)
    mu = (rate - dividend) - lambda_param * k_rho

    for k in range(n_steps):
        curr_x = paths_x[:, k]
        curr_var = np.maximum(paths_var[:, k], 0.0)

        dw = rng.standard_normal(n_paths) * sq_dt
        num_jumps = rng.poisson(lam=jump_rate, size=n_paths)

        has_jumps = num_jumps > 0
        jumps = np.zeros(n_paths, dtype=float)
        if np.any(has_jumps):
            jumps[has_jumps] = rng.gamma(shape=num_jumps[has_jumps], scale=1.0 / beta_bdlp)

        next_x = curr_x + (mu + beta * curr_var) * dt + np.sqrt(curr_var) * dw + rho * jumps
        next_var = curr_var * exp_ldt + jumps

        paths_x[:, k + 1] = next_x
        paths_var[:, k + 1] = next_var

    paths_s = s0 * np.exp(paths_x)
    paths_s[:, 0] = float(s0)

    if return_variance:
        return paths_s, paths_var
    return paths_s


def sample_jacobi(
    kappa: float = 1.5,
    theta: float = 0.4,
    sigma: float = 0.5,
    y_min: float = 0.0,
    y_max: float = 1.0,
    y0: float = 0.3,
    dt: Optional[float] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Simulate sample paths of the Jacobi diffusion on :math:`[y_{\min}, y_{\max}]`.

    .. math::

        dY_t = \kappa (\theta - Y_t) dt + \sigma \sqrt{\frac{(Y_t - y_{\min})(y_{\max} - Y_t)}{y_{\max} - y_{\min}}} dW_t

    Integrated via the Milstein scheme on the normalized unit interval :math:`[0, 1]`:

    .. math::

        X_{t+\Delta t} = X_t + \kappa(\theta_{\mathrm{norm}} - X_t)\Delta t
        + \sigma \sqrt{X_t(1 - X_t)}\Delta W + \frac{1}{4}\sigma^2(1 - 2X_t)(\Delta W^2 - \Delta t)

    Parameters
    ----------
    kappa : float
        Mean-reversion speed :math:`\kappa > 0`.
    theta : float
        Long-term mean :math:`\theta \in [y_{\min}, y_{\max}]`.
    sigma : float
        Diffusion parameter :math:`\sigma > 0`.
    y_min : float
        Lower bound.
    y_max : float
        Upper bound.
    y0 : float
        Initial value :math:`y_0 \in [y_{\min}, y_{\max}]`.
    dt : float, optional
        Time step size.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Simulated sample paths of shape ``(n_paths, n_steps + 1)``.
    """
    if kappa <= 0:
        raise ValueError("Mean-reversion speed 'kappa' must be strictly positive")
    if sigma <= 0:
        raise ValueError("Diffusion parameter 'sigma' must be strictly positive")
    if y_max <= y_min:
        raise ValueError("Upper bound 'y_max' must be strictly greater than 'y_min'")
    if not (y_min <= theta <= y_max):
        raise ValueError(f"Long-term mean 'theta' must be in [{y_min}, {y_max}]")
    if not (y_min <= y0 <= y_max):
        raise ValueError(f"Initial value 'y0' must be in [{y_min}, {y_max}]")
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

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(y0)

    if n_steps == 0:
        return paths

    rng = rng or np.random.default_rng()
    scale = y_max - y_min
    theta_norm = (theta - y_min) / scale
    x = np.full(n_paths, (y0 - y_min) / scale, dtype=float)

    sq_dt = math.sqrt(dt)

    for k in range(n_steps):
        dw = rng.standard_normal(n_paths) * sq_dt
        x_clip = np.clip(x, 1e-8, 1.0 - 1e-8)
        diff = sigma * np.sqrt(x_clip * (1.0 - x_clip))
        drift = kappa * (theta_norm - x)
        milstein = 0.25 * (sigma**2) * (1.0 - 2.0 * x_clip) * (dw**2 - dt)

        x = np.clip(x + drift * dt + diff * dw + milstein, 0.0, 1.0)
        paths[:, k + 1] = y_min + scale * x

    return paths


def sample_regime_switching_gbm(
    mu: Sequence[float],
    sigma: Sequence[float],
    transition_rates: Union[Sequence[Sequence[float]], np.ndarray],
    s0: float,
    initial_regime: int = 0,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    return_regimes: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""
    Sample paths of Regime-Switching Geometric Brownian Motion (RS-GBM).

    Simulates the continuous-time Markov chain regimes using exact transition
    probabilities :math:`P(\Delta t) = \exp(Q \Delta t)` and exact conditional log-GBM
    increments.

    Parameters
    ----------
    mu : Sequence[float]
        Regime drifts, length :math:`K`.
    sigma : Sequence[float]
        Regime volatilities, length :math:`K`.
    transition_rates : Sequence[Sequence[float]] or np.ndarray
        Continuous-time Markov chain generator matrix :math:`Q` of size :math:`K \times K`.
    s0 : float
        Initial spot price :math:`S_0 > 0`.
    initial_regime : int
        Initial regime index :math:`Y_0 \in \{0, \dots, K-1\}`.
    dt : float, optional
        Time step.
    n_steps : int, optional
        Number of steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        RNG instance.
    return_regimes : bool
        If True, returns ``(spot_paths, regime_paths)``.
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    q_mat = np.asarray(transition_rates, dtype=float)
    k_states = len(mu_arr)

    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")
    if s0 <= 0.0:
        raise ValueError(f"s0 must be > 0, got {s0}")
    if not (0 <= initial_regime < k_states):
        raise ValueError(f"initial_regime must be in [0, {k_states - 1}], got {initial_regime}")

    if n_steps is None:
        n_steps = 100
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

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

    s_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    s_paths[:, 0] = float(s0)
    reg_paths = np.empty((n_paths, n_steps + 1), dtype=int)
    reg_paths[:, 0] = int(initial_regime)

    if n_steps == 0:
        return (s_paths, reg_paths) if return_regimes else s_paths

    rng = rng or np.random.default_rng()

    # Exact transition probability matrix over time step dt
    p_dt = la.expm(q_mat * dt)
    p_dt = np.maximum(p_dt, 0.0)
    p_dt = p_dt / np.sum(p_dt, axis=1, keepdims=True)
    cum_p = np.cumsum(p_dt, axis=1)
    cum_p[:, -1] = 1.0  # Ensure numerical closure

    current_regimes = np.full(n_paths, initial_regime, dtype=int)
    current_s = np.full(n_paths, s0, dtype=float)
    sq_dt = math.sqrt(dt)

    for k in range(n_steps):
        # 1. Diffusion step under current regime
        m_curr = mu_arr[current_regimes]
        sig_curr = sigma_arr[current_regimes]
        z = rng.standard_normal(n_paths)
        log_inc = (m_curr - 0.5 * (sig_curr**2)) * dt + sig_curr * sq_dt * z
        current_s *= np.exp(log_inc)
        s_paths[:, k + 1] = current_s

        # 2. Vectorized Markov chain transition for the next step
        u = rng.uniform(0.0, 1.0, size=n_paths)
        thresholds = cum_p[current_regimes, :-1]
        current_regimes = np.sum(u[:, None] > thresholds, axis=1)
        reg_paths[:, k + 1] = current_regimes

    return (s_paths, reg_paths) if return_regimes else s_paths


def sample_chen(
    k: float = 1.5,
    theta_bar: float = 0.05,
    kappa: float = 0.8,
    alpha: float = 0.04,
    zeta_bar: float = 0.02,
    lambda_param: float = 1.0,
    beta: float = 0.05,
    r0: float = 0.04,
    theta0: Optional[float] = None,
    zeta0: Optional[float] = None,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    return_factors: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""
    Sample paths of the Chen Three-Factor Short Rate Model.

    Parameters
    ----------
    k : float
        Reversion rate of short rate towards stochastic mean.
    theta_bar : float
        Long-term central tendency level.
    kappa : float
        Reversion rate of central tendency.
    alpha : float
        Diffusion coefficient of central tendency.
    zeta_bar : float
        Long-term variance level.
    lambda_param : float
        Reversion rate of variance factor.
    beta : float
        Diffusion coefficient of variance factor.
    r0 : float
        Initial short rate.
    theta0 : float, optional
        Initial central tendency (defaults to theta_bar).
    zeta0 : float, optional
        Initial variance factor (defaults to zeta_bar).
    dt : float, optional
        Time step size.
    n_steps : int, optional
        Number of time steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        Random number generator.
    return_factors : bool
        If True, returns ``(r_paths, theta_paths, zeta_paths)``.
    """
    if k <= 0:
        raise ValueError("k must be strictly positive")
    if theta_bar <= 0:
        raise ValueError("theta_bar must be strictly positive")
    if kappa <= 0:
        raise ValueError("kappa must be strictly positive")
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive")
    if zeta_bar <= 0:
        raise ValueError("zeta_bar must be strictly positive")
    if lambda_param <= 0:
        raise ValueError("lambda_param must be strictly positive")
    if beta <= 0:
        raise ValueError("beta must be strictly positive")
    if r0 <= 0:
        raise ValueError("r0 must be strictly positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    th0 = theta_bar if theta0 is None else float(theta0)
    z0 = zeta_bar if zeta0 is None else float(zeta0)
    if th0 <= 0:
        raise ValueError("theta0 must be strictly positive")
    if z0 <= 0:
        raise ValueError("zeta0 must be strictly positive")

    if n_steps is None:
        n_steps = 100
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

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

    r_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    r_paths[:, 0] = float(r0)
    th_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    th_paths[:, 0] = float(th0)
    z_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    z_paths[:, 0] = float(z0)

    if n_steps == 0:
        return (r_paths, th_paths, z_paths) if return_factors else r_paths

    rng = rng or np.random.default_rng()
    sq_dt = math.sqrt(dt)

    curr_r = np.full(n_paths, r0, dtype=float)
    curr_th = np.full(n_paths, th0, dtype=float)
    curr_z = np.full(n_paths, z0, dtype=float)

    for step in range(n_steps):
        dw1 = rng.standard_normal(n_paths) * sq_dt
        dw2 = rng.standard_normal(n_paths) * sq_dt
        dw3 = rng.standard_normal(n_paths) * sq_dt

        # Factor 3: Variance factor zeta (CIR)
        z_pos = np.maximum(curr_z, 0.0)
        curr_z = np.maximum(
            curr_z + lambda_param * (zeta_bar - curr_z) * dt + beta * np.sqrt(z_pos) * dw3,
            0.0,
        )
        z_paths[:, step + 1] = curr_z

        # Factor 2: Central tendency theta (CIR)
        th_pos = np.maximum(curr_th, 0.0)
        curr_th = np.maximum(
            curr_th + kappa * (theta_bar - curr_th) * dt + alpha * np.sqrt(th_pos) * dw2,
            0.0,
        )
        th_paths[:, step + 1] = curr_th

        # Factor 1: Short rate r
        r_pos = np.maximum(curr_r, 0.0)
        curr_r = np.maximum(
            curr_r + k * (th_pos - curr_r) * dt + np.sqrt(z_pos * r_pos) * dw1,
            0.0,
        )
        r_paths[:, step + 1] = curr_r

    return (r_paths, th_paths, z_paths) if return_factors else r_paths


def sample_cbi(
    a: float = 0.05,
    b: float = 0.8,
    c: float = 0.1,
    lambda_j: float = 0.0,
    eta: float = 20.0,
    x0: float = 0.1,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Sample paths of Continuous-State Branching Process with Immigration (CBI).

    Simulates the Dawson-Watanabe diffusion with quadratic branching and optional
    compound Poisson jump immigration via high-order Milstein scheme with non-negative
    reflection.

    Parameters
    ----------
    a : float
        Continuous immigration rate.
    b : float
        Branching drift / decay parameter.
    c : float
        Branching diffusion coefficient.
    lambda_j : float
        Jump immigration rate.
    eta : float
        Exponential jump scale parameter.
    x0 : float
        Initial state.
    dt : float, optional
        Time step size.
    n_steps : int, optional
        Number of steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        Random number generator.
    """
    if a < 0:
        raise ValueError("a must be >= 0")
    if b < 0:
        raise ValueError("b must be >= 0")
    if c <= 0:
        raise ValueError("c must be > 0")
    if lambda_j < 0:
        raise ValueError("lambda_j must be >= 0")
    if eta <= 0:
        raise ValueError("eta must be > 0")
    if x0 < 0:
        raise ValueError("x0 must be >= 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    if n_steps is None:
        n_steps = 100
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

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

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(x0)

    if n_steps == 0:
        return paths

    rng = rng or np.random.default_rng()
    sq_dt = math.sqrt(dt)
    curr_x = np.full(n_paths, x0, dtype=float)

    for step in range(n_steps):
        dw = rng.standard_normal(n_paths) * sq_dt

        # Jump immigration
        jumps = np.zeros(n_paths, dtype=float)
        if lambda_j > 0.0:
            n_jumps = rng.poisson(lambda_j * dt, size=n_paths)
            has_j = n_jumps > 0
            if np.any(has_j):
                jumps[has_j] = rng.gamma(n_jumps[has_j], 1.0 / eta)

        # Milstein step for sqrt(c X_t)
        x_pos = np.maximum(curr_x, 0.0)
        diff = np.sqrt(c * x_pos)
        milstein = 0.25 * c * (dw**2 - dt)
        curr_x = np.maximum(curr_x + (a - b * curr_x) * dt + diff * dw + milstein + jumps, 0.0)
        paths[:, step + 1] = curr_x

    return paths
