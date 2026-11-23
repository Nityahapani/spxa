"""Every Lévy primitive simulates with the positional order (n_steps, n_paths, T, rng).

Composites call ``simulate`` positionally. Kou and generalised hyperbolic took
``dt`` second, and NTS took ``t_grid`` first, so ``Kou + CompoundPoisson``
raised a TypeError when simulated.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spxa.zoo.levy import (
    CGMY,
    NIG,
    AlphaStable,
    BrownianMotion,
    CompoundPoisson,
    Drift,
    GammaProcess,
    GeneralizedHyperbolic,
    InverseGaussianProcess,
    KouJumpDiffusion,
    MeixnerProcess,
    MertonJumpDiffusion,
    NegativeBinomialProcess,
    NormalTemperedStable,
    PoissonProcess,
    TemperedStable,
    VarianceGamma,
)
from spxa.zoo.levy.compound_poisson import NormalJump

PRIMITIVES = [
    AlphaStable(1.5, 0.2, 0.5),
    BrownianMotion(0.1, 0.2),
    CGMY(1.0, 3.0, 4.0, 0.5),
    CompoundPoisson(2.0, NormalJump(0.0, 0.1)),
    Drift(0.3),
    GammaProcess(2.0, 3.0),
    GeneralizedHyperbolic(lambda_param=1.0, alpha=3.0, beta=0.5, delta=1.0, mu=0.0),
    InverseGaussianProcess(1.0, 2.0),
    KouJumpDiffusion(0.0, 0.2, 1.0, 0.4, 10.0, 5.0),
    MeixnerProcess(0.5, 0.2, 1.0, 0.0),
    MertonJumpDiffusion(0.0, 0.2, 1.0, -0.1, 0.1),
    NIG(3.0, 0.5, 1.0, 0.0),
    NegativeBinomialProcess(2.0, 0.5),
    NormalTemperedStable(alpha=0.6, lambd=1.5, delta=0.8, theta=0.1, sigma=0.3),
    PoissonProcess(2.0),
    TemperedStable(0.5, 1.0, 2.0),
    VarianceGamma(0.2, 0.3, -0.1),
]


@pytest.mark.parametrize("X", PRIMITIVES, ids=lambda X: type(X).__name__)
def test_positional_simulate(X) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = X.simulate(5, 3, 2.0, np.random.default_rng(0))
    assert paths.shape == (3, 6)
    assert np.all(paths[:, 0] == 0.0)
    assert np.all(np.isfinite(paths))


@pytest.mark.parametrize("X", PRIMITIVES, ids=lambda X: type(X).__name__)
def test_seeded_simulation_is_reproducible(X) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = X.simulate(4, 2, 1.0, np.random.default_rng(7))
        b = X.simulate(4, 2, 1.0, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_kou_plus_compound_poisson_simulates() -> None:
    Z = KouJumpDiffusion(0.0, 0.2, 1.0, 0.4, 10.0, 5.0) + CompoundPoisson(2.0, NormalJump(0.0, 0.1))
    assert Z.simulate(10, 4, 1.0, np.random.default_rng(1)).shape == (4, 11)


def test_keyword_spellings_still_work() -> None:
    kou = KouJumpDiffusion(0.0, 0.2, 1.0, 0.4, 10.0, 5.0)
    assert kou.simulate(n_steps=4, time_horizon=1.0, n_paths=2).shape == (2, 5)
    nts = NormalTemperedStable(alpha=0.6, lambd=1.5, delta=0.8, theta=0.1, sigma=0.3)
    grid = np.linspace(0.0, 1.0, 6)
    assert nts.simulate(grid, n_paths=2).shape == (2, 6)
    assert nts.simulate(t_grid=grid, n_paths=2).shape == (2, 6)


def test_nts_increments_follow_the_nts_law() -> None:
    """The clock is TemperedStable(alpha, delta, lambda), not a moment-matched Gamma."""
    X = NormalTemperedStable(alpha=0.6, lambd=1.5, delta=0.8, theta=0.4, sigma=0.3)
    x = X.simulate(1, 100_000, 0.5, np.random.default_rng(5))[:, -1]
    for u in (1.0, 3.0, 8.0):
        p = complex(X.char_func(u, 0.5))
        p2 = complex(X.char_func(2 * u, 0.5))
        m = np.mean(np.exp(1j * u * x))
        z_re = (m.real - p.real) / np.sqrt(((1 + p2.real) / 2 - p.real**2) / len(x))
        z_im = (m.imag - p.imag) / np.sqrt(((1 - p2.real) / 2 - p.imag**2) / len(x))
        assert max(abs(z_re), abs(z_im)) < 4.5, (u, z_re, z_im)
