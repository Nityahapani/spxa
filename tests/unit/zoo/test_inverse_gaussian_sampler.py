"""The inverse Gaussian sampler behind the IG and NIG processes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from spxa.zoo.levy import NIG, GammaProcess, InverseGaussianProcess
from spxa.zoo.levy.inverse_gaussian import sample_inverse_gaussian


@pytest.mark.parametrize("mean, shape", [(1.0, 2.0), (0.01, 1e-4), (3.0, 0.05), (2.0, 400.0)])
def test_law_matches_scipy(mean: float, shape: float) -> None:
    rng = np.random.default_rng(20261023)
    x = sample_inverse_gaussian(np.full(100_000, mean), shape, rng)
    law = stats.invgauss(mu=mean / shape, scale=shape)
    assert stats.kstest(x, law.cdf).pvalue > 1e-3
    # mean m and variance m^3/s, to five standard errors
    se_mean = np.sqrt(mean**3 / shape / x.size)
    assert abs(x.mean() - mean) < 5 * se_mean


def test_only_the_ratio_matters() -> None:
    # x(m, s) = m * x(1, s/m) draw for draw, so tiny parameters lose nothing
    m, s = 1e-150, 1e-300
    a = sample_inverse_gaussian(np.full(1000, m), s, np.random.default_rng(3))
    b = sample_inverse_gaussian(np.full(1000, 1.0), s / m, np.random.default_rng(3))
    np.testing.assert_allclose(a, m * b, rtol=1e-14, atol=0)
    assert np.all(np.isfinite(a)) and np.all(a >= 0)


def test_degenerate_entries() -> None:
    rng = np.random.default_rng(5)
    out = sample_inverse_gaussian(np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.0, 1e-320]), rng)
    assert out[0] == 0.0
    assert np.all(np.isfinite(out[1:])) and np.all(out[1:] >= 0)
    assert sample_inverse_gaussian(np.zeros((2, 3)), 1.0, rng).shape == (2, 3)


def test_subordinated_nig_with_tiny_clock_increments() -> None:
    # Gamma(5, 5) increments over 1/252 are often below 1e-150; (delta t)^2 then
    # underflowed and Generator.wald raised "scale <= 0"
    proc = NIG(10.0, -2.0, 0.5) @ GammaProcess(5.0, 5.0)
    paths = proc.simulate(252, 4000, 1.0, np.random.default_rng(11))
    assert paths.shape == (4000, 253) and np.all(np.isfinite(paths))
    k = proc.cumulants(2)
    final = paths[:, -1]
    assert abs(final.mean() - k[1]) < 5 * np.sqrt(k[2] / final.size)
    assert final.var() == pytest.approx(k[2], rel=0.15)


def test_inverse_gaussian_process_sample_at() -> None:
    proc = InverseGaussianProcess(1.0, 2.0)
    t = np.array([0.0, 1e-200, 0.5, 2.0])
    draws = proc.sample_at(np.tile(t, (20_000, 1)), np.random.default_rng(13))
    assert np.all(draws[:, 0] == 0.0) and np.all(draws >= 0)
    # X_t ~ IG(mu t, lam t^2): mean mu t, variance mu^3 t / lam
    np.testing.assert_allclose(draws[:, 2:].mean(axis=0), [0.5, 2.0], rtol=0.03)
    np.testing.assert_allclose(draws[:, 2:].var(axis=0), [0.25, 1.0], rtol=0.1)
