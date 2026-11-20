"""Cumulant generating function series: sums, scaling and subordination."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from spxa.discovery import CGFSeries, cumulants_to_moments, moments_to_cumulants


def normal_moments(mu: float, s2: float, n: int) -> list[float]:
    # E[Y^k] for Y ~ N(mu, s2): sum_j C(k, 2j) mu^{k-2j} s2^j (2j-1)!!
    def dfact(m: int) -> int:
        return 1 if m <= 0 else m * dfact(m - 2)

    return [
        sum(
            comb(k, 2 * j) * mu ** (k - 2 * j) * s2**j * dfact(2 * j - 1) for j in range(k // 2 + 1)
        )
        for k in range(1, n + 1)
    ]


def test_moment_cumulant_round_trip() -> None:
    k = np.array([0.3, 1.7, -0.4, 2.2, 0.9, -1.1])
    np.testing.assert_allclose(moments_to_cumulants(cumulants_to_moments(k)), k, rtol=1e-13)
    # a Gaussian has kappa_n = 0 beyond 2
    np.testing.assert_allclose(
        moments_to_cumulants(normal_moments(0.5, 2.0, 6)), [0.5, 2.0, 0, 0, 0, 0], atol=1e-12
    )


def test_sum_scale_time() -> None:
    a, b = CGFSeries([1.0, 2.0, 3.0, 4.0]), CGFSeries([0.5, 0.25, 0.0])
    assert (a + b).order == 3
    np.testing.assert_allclose((a + b).cumulants, [1.5, 2.25, 3.0])
    np.testing.assert_allclose(a.scale(-2.0).cumulants, [-2.0, 8.0, -24.0, 64.0])
    np.testing.assert_allclose(a.at_time(0.5).cumulants, [0.5, 1.0, 1.5, 2.0])
    assert a.mean == 1.0 and a.variance == 2.0
    assert a.skewness == pytest.approx(3.0 / 2.0**1.5) and a.excess_kurtosis == 1.0
    with pytest.raises(ValueError):
        a.cumulant(5)
    with pytest.raises(ValueError):
        CGFSeries([])


def test_brownian_motion_in_gamma_time_is_variance_gamma() -> None:
    sigma, nu, theta, N = 0.3, 0.4, -0.2, 8
    bm = CGFSeries([theta, sigma**2] + [0.0] * (N - 2))
    # Gamma(a = 1/nu, b = 1/nu): kappa_n = a (n-1)! / b^n
    from math import factorial

    gamma = CGFSeries([(1 / nu) * factorial(n - 1) * nu**n for n in range(1, N + 1)])
    vg = gamma.compose(bm)
    s, n, t = sigma, nu, theta
    expected = [
        t,
        s**2 + t**2 * n,
        2 * t**3 * n**2 + 3 * s**2 * t * n,
        3 * s**4 * n + 12 * s**2 * t**2 * n**2 + 6 * t**4 * n**3,
    ]
    np.testing.assert_allclose(vg.cumulants[:4], expected, rtol=1e-13)
    # higher orders: VG cumulants are (1/nu) * kappa_n of log-ratio, check via -log(1 - theta nu x - s^2 nu x^2/2)/nu
    x = np.polynomial.Polynomial([0.0, t * n, 0.5 * s**2 * n])
    # -log(1 - x) = sum x^m / m
    series = sum((x**m) / m for m in range(1, N + 1)) / n
    coef = series.coef[1 : N + 1]
    np.testing.assert_allclose(vg.coefficients()[1:], coef, rtol=1e-12)


def test_poisson_time_gives_compound_poisson() -> None:
    lam, mu, s2, N = 1.3, 0.2, 0.5, 6
    poisson = CGFSeries([lam] * N)  # every cumulant of Poisson(lam) is lam
    bm = CGFSeries([mu, s2] + [0.0] * (N - 2))
    cp = poisson.compose(bm)
    # compound Poisson with N(mu, s2) jumps: kappa_n = lam E[Y^n]
    np.testing.assert_allclose(cp.cumulants, lam * np.array(normal_moments(mu, s2, N)), rtol=1e-13)


def test_deterministic_time_change_and_identity() -> None:
    x = CGFSeries([0.1, 0.4, -0.3, 0.8])
    np.testing.assert_allclose(CGFSeries([2.5, 0, 0, 0]).compose(x).cumulants, 2.5 * x.cumulants)
    np.testing.assert_allclose(x.compose(CGFSeries([1.0, 0, 0, 0])).cumulants, x.cumulants)
    np.testing.assert_allclose(
        CGFSeries.from_moments(x.moments()).cumulants, x.cumulants, rtol=1e-13
    )
    # evaluation near zero matches the truncated polynomial
    assert x(0.0) == 0
    assert abs(x(0.1) - (0.01 + 0.4 * 0.005 - 0.3 * 0.001 / 6 + 0.8 * 1e-4 / 24)) < 1e-15
    with pytest.raises(ValueError):
        CGFSeries.from_coefficients(np.array([1.0, 0.0]))
