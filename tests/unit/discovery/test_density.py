"""COS densities and distribution functions against closed forms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats

from spxa.discovery.density import (
    COSDistribution,
    cos_cdf,
    cos_pdf,
    model_distribution,
    truncation_range,
)
from spxa.discovery.expr import Model, leaf
from spxa.discovery.families import BROWNIAN, DRIFT, GAMMA, NIG_FAMILY, NORMAL_JUMPS, VARIANCE_GAMMA


def test_gaussian_pdf_and_cdf() -> None:
    mu, s = 0.1, 0.3

    def cf(u: np.ndarray) -> np.ndarray:
        return np.exp(1j * mu * u - 0.5 * s**2 * u**2)

    a, b = truncation_range(np.array([mu, s**2, 0.0, 0.0]))
    x = np.linspace(-0.8, 1.0, 37)
    np.testing.assert_allclose(cos_pdf(cf, x, a, b, 256), stats.norm.pdf(x, mu, s), atol=1e-12)
    np.testing.assert_allclose(cos_cdf(cf, x, a, b, 256), stats.norm.cdf(x, mu, s), atol=1e-12)
    assert cos_cdf(cf, np.array([a - 1, b + 1]), a, b, 64).tolist() == [0.0, 1.0]
    with pytest.raises(ValueError):
        truncation_range(np.zeros(4))


def vg_density(x: np.ndarray, sigma: float, nu: float, theta: float, t: float) -> np.ndarray:
    # Madan, Carr & Chang (1998), eq. (23)
    a = t / nu
    r = np.sqrt(2 * sigma**2 / nu + theta**2)
    ax = np.abs(x)
    return (
        2
        * np.exp(theta * x / sigma**2)
        / (nu**a * np.sqrt(2 * np.pi) * sigma * special.gamma(a))
        * (ax**2 / (2 * sigma**2 / nu + theta**2)) ** (a / 2 - 0.25)
        * special.kv(a - 0.5, ax * r / sigma**2)
    )


def test_variance_gamma_density_matches_madan_carr_chang() -> None:
    p, t = (0.2, 0.25, -0.15), 2.0  # t/ν = 8: bounded smooth density
    m = Model(leaf(VARIANCE_GAMMA), [p])
    dist = model_distribution(m, t=t, n_terms=2048)
    x = np.array([-0.9, -0.4, -0.05, 0.1, 0.35, 0.7])
    np.testing.assert_allclose(dist.pdf(x), vg_density(x, *p, t), rtol=1e-8)


def test_nig_density_and_cdf() -> None:
    alpha, beta, delta, t = 8.0, -2.0, 0.4, 0.5
    m = Model(leaf(NIG_FAMILY), [(alpha, beta, delta)])
    dist = model_distribution(m, t=t)
    ref = stats.norminvgauss(a=alpha * delta * t, b=beta * delta * t, scale=delta * t)
    x = np.linspace(-0.6, 0.4, 21)
    np.testing.assert_allclose(dist.pdf(x), ref.pdf(x), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(dist.cdf(x), ref.cdf(x), atol=1e-9)


def test_gamma_cdf() -> None:
    a, b, t = 3.0, 4.0, 1.5
    dist = model_distribution(Model(leaf(GAMMA), [(a, b)]), t=t, n_terms=4096)
    x = np.array([0.2, 0.8, 1.1, 2.5])
    np.testing.assert_allclose(dist.cdf(x), stats.gamma.cdf(x, a * t, scale=1 / b), atol=1e-6)


def test_merton_density_is_a_poisson_mixture() -> None:
    sigma, lam, m_j, s_j, mu, t = 0.15, 2.0, -0.08, 0.1, 0.03, 0.25
    m = Model(leaf(DRIFT) + leaf(BROWNIAN) + leaf(NORMAL_JUMPS), [(mu,), (sigma,), (lam, m_j, s_j)])
    dist = model_distribution(m, t=t)
    x = np.linspace(-0.4, 0.3, 15)
    ref = sum(
        stats.poisson.pmf(n, lam * t)
        * stats.norm.pdf(x, mu * t + n * m_j, np.sqrt(sigma**2 * t + n * s_j**2))
        for n in range(40)
    )
    np.testing.assert_allclose(dist.pdf(x), ref, rtol=1e-9, atol=1e-12)


def test_bin_probabilities() -> None:
    m = Model(leaf(BROWNIAN), [(1.0,)])
    dist = model_distribution(m, t=1.0)
    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    p = dist.bin_probabilities(edges)
    np.testing.assert_allclose(p, np.diff(stats.norm.cdf(edges)), atol=1e-12)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    custom = COSDistribution(
        lambda u: np.exp(-0.5 * u**2), np.array([0, 1, 0, 0]), bounds=(-12.0, 12.0)
    )
    assert custom.cdf(np.array([0.0]))[0] == pytest.approx(0.5, abs=1e-12)
