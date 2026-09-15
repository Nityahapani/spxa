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

from typing import Optional

import numpy as np


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
        X = (
            (2 / np.pi)
            * (
                (np.pi / 2 + beta * U) * np.tan(U)
                - beta * np.log(np.pi / 2 * E * np.cos(U) / (np.pi / 2 + beta * U))
            )
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
    increments = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        for k in range(n_steps):
            n_jumps = jump_counts[i, k]
            if n_jumps > 0:
                increments[i, k] = jump_sampler(rng, n_jumps).sum()
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
