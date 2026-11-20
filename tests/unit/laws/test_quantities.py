"""Quantities read off processes, against closed forms."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.laws.quantities import (
    char_exponent,
    char_func,
    cumulants,
    laplace_exponent,
    mean,
    mgf,
    variance,
)
from spxa.zoo.levy import BrownianMotion, GammaProcess, PoissonProcess, VarianceGamma


def test_moments_scale_with_time() -> None:
    X = BrownianMotion(mu=0.3, sigma=0.4)
    assert mean(X) == pytest.approx(0.3) and mean(X, t=2.5) == pytest.approx(0.75)
    assert variance(X, t=2.0) == pytest.approx(0.32)
    np.testing.assert_allclose(cumulants(X, 4, t=3.0), [0.9, 0.48, 0.0, 0.0], atol=1e-15)
    G = GammaProcess(a=2.0, b=4.0)  # κ_n = a (n-1)! / b^n
    np.testing.assert_allclose(cumulants(G, 4), [0.5, 0.125, 0.0625, 0.046875], rtol=1e-10)


def test_char_func_and_exponent() -> None:
    X = BrownianMotion(mu=0.1, sigma=0.5)
    u = np.array([0.0, 0.7, -2.0])
    psi = 1j * 0.1 * u - 0.5 * 0.25 * u**2
    np.testing.assert_allclose(char_exponent(X, u), psi, rtol=1e-14)
    np.testing.assert_allclose(char_func(X, u, t=1.5), np.exp(1.5 * psi), rtol=1e-12)
    assert char_func(X, 0.3).shape == (1,)
    # Poisson with jump size h: ψ(u) = λ(e^{iuh} - 1), complex u allowed
    P = PoissonProcess(rate=2.0, jump_size=0.5)
    w = np.array([1.0, 0.5 - 0.25j])
    np.testing.assert_allclose(char_exponent(P, w), 2.0 * (np.exp(1j * w * 0.5) - 1), rtol=1e-10)


def test_laplace_exponent_and_mgf() -> None:
    G = GammaProcess(a=1.5, b=3.0)  # K(z) = -a log(1 - z/b) for Re z < b
    z = np.array([-5.0, -0.5, 0.0, 1.0, -2.0 + 1.0j])
    # quadrature (the gamma measure has no closed-form exponent here)
    exact = -1.5 * np.log(1 - z / 3.0)
    np.testing.assert_allclose(laplace_exponent(G, z), exact, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(mgf(G, 1.0, t=2.0), (1 - 1 / 3.0) ** (-3.0), rtol=1e-7)
    V = VarianceGamma(sigma=0.2, nu=0.3, theta=-0.1)
    # K(z) = -log(1 - θνz - σ²νz²/2)/ν
    zz = np.array([0.5, -1.0])
    k = -np.log(1 - (-0.1) * 0.3 * zz - 0.04 * 0.3 * zz**2 / 2) / 0.3
    np.testing.assert_allclose(laplace_exponent(V, zz), k, rtol=1e-7)


def test_exponential_moment_strips() -> None:
    from spxa.laws.quantities import exponential_moment_strip as strip
    from spxa.zoo.levy import AlphaStable, InverseGaussianProcess, NegativeBinomialProcess

    inf = float("inf")
    assert strip(BrownianMotion(0.1, 0.3)) == (-inf, inf)
    assert strip(GammaProcess(a=2.0, b=3.0)) == (-inf, 3.0)
    assert strip(InverseGaussianProcess(mu=2.0, lam=4.0)) == (-inf, 0.5)
    assert strip(NegativeBinomialProcess(r=1.0, p=0.5))[1] == pytest.approx(np.log(2))
    V = VarianceGamma(sigma=0.2, nu=0.3, theta=-0.1)
    lo, hi = strip(V)
    for z in (lo, hi):  # roots of 1 - θνz - σ²νz²/2
        assert 1 - (-0.1) * 0.3 * z - 0.04 * 0.3 * z**2 / 2 == pytest.approx(0, abs=1e-12)
    assert strip(-2.0 * V) == pytest.approx((-hi / 2, -lo / 2))
    assert strip(V + GammaProcess(a=1.0, b=4.0)) == (lo, 4.0)
    assert strip(AlphaStable(1.5, 0.0, 1.0, 0.0)) == (0.0, 0.0)
    # BM(0.1, 0.3) in Gamma(2, 3) time: 0.1 z + 0.045 z^2 < 3
    zl, zh = strip(BrownianMotion(0.1, 0.3) @ GammaProcess(a=2.0, b=3.0))
    np.testing.assert_allclose(sorted(np.roots([0.045, 0.1, -3.0])), [zl, zh], rtol=1e-10)
    from spxa.zoo.levy import NormalTemperedStable

    with pytest.raises(NotImplementedError):
        strip(NormalTemperedStable(alpha=0.5, lambd=1.0, delta=1.0, theta=0.0, sigma=0.2))
