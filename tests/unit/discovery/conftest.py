"""Exact samplers for discovery tests."""

from __future__ import annotations

import numpy as np


def vg_increments(
    sigma: float, nu: float, theta: float, dt: float, n: int, seed: int, mu: float = 0.0
) -> np.ndarray:
    """VG increments as θG + σ√G Z with G ~ Gamma(dt/ν, ν) (Madan, Carr & Chang 1998)."""
    rng = np.random.default_rng(seed)
    g = rng.gamma(dt / nu, nu, size=n)
    return mu * dt + theta * g + sigma * np.sqrt(g) * rng.normal(size=n)


def gamma_increments(a: float, b: float, dt: float, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).gamma(a * dt, 1.0 / b, size=n)


def merton_increments(
    mu: float, sigma: float, lam: float, jm: float, js: float, dt: float, n: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = rng.poisson(lam * dt, size=n)
    return (
        mu * dt
        + sigma * np.sqrt(dt) * rng.normal(size=n)
        + jm * k
        + js * np.sqrt(k) * rng.normal(size=n)
    )
