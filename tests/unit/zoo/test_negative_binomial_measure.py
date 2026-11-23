"""The negative binomial Lévy measure supports composite processes."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.zoo.levy import BrownianMotion, NegativeBinomialProcess


@pytest.fixture
def nb() -> NegativeBinomialProcess:
    return NegativeBinomialProcess(r=2.0, p=0.3)


def test_triplet_cf_matches_closed_form(nb: NegativeBinomialProcess) -> None:
    u = np.array([0.0, 0.7, -2.5, 4.0])
    np.testing.assert_allclose(nb.triplet.char_exp(u, 1.7), nb.char_func(u, 1.7), atol=1e-14)


def test_first_moment_between(nb: NegativeBinomialProcess) -> None:
    nu = nb.triplet.nu
    r, p = 2.0, 0.3
    assert nu.first_moment_between(1.0, np.inf) == pytest.approx(r * p**2 / (1 - p), rel=1e-14)
    assert nu.first_moment_between(0.5, 3.0) == pytest.approx(r * (p + p**2 + p**3), rel=1e-14)
    assert nu.first_moment_between(1.2, 1.9) == 0.0
    with pytest.raises(ValueError):
        nu.first_moment_between(2.0, 1.0)
    # kappa_1 = b + large-jump mean = rp/(1-p)
    assert nb.triplet.cumulant(1) == pytest.approx(r * p / (1 - p), rel=1e-14)


def test_jump_exponent_off_the_real_axis(nb: NegativeBinomialProcess) -> None:
    # Laplace exponent K(z) = r log((1-p)/(1-p e^z)) for z < -log p
    nu, b = nb.triplet.nu, nb.triplet.b
    for z in (-3.0, 0.5, 1.0 + 0.4j):
        got = b * z + nu.jump_exponent(-1j * z)
        assert got == pytest.approx(2.0 * np.log(0.7 / (1 - 0.3 * np.exp(z))), rel=1e-13)


def test_scaled_and_summed_processes(nb: NegativeBinomialProcess) -> None:
    for c in (0.4, 3.0, -1.0):
        Z = c * nb
        k = Z.cumulants(4)
        base = nb.cumulants(4)
        for n in range(1, 5):
            assert k[n] == pytest.approx(c**n * base[n], rel=1e-12)
        u = np.array([0.3, -1.1])
        np.testing.assert_allclose(Z.char_func(u), nb.char_func(c * u), atol=1e-14)
    S = nb + BrownianMotion(mu=0.1, sigma=0.2)
    assert S.cumulants(2)[2] == pytest.approx(nb.cumulants(2)[2] + 0.04, rel=1e-12)
