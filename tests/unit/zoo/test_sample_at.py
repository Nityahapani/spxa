"""Vectorised draws of X_t at given times match the characteristic function.

``sample_at(times)`` returns independent copies of X_t, one per entry. Each
family is checked at two horizons with empirical-cf z-scores (|z| < 4.5, fixed
seeds) and with asymmetric parameters, so a swapped convention shows up. For
example, the negative binomial's numpy success probability is 1 - p.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxa.zoo.levy import (
    NIG,
    CGMY,
    AlphaStable,
    BrownianMotion,
    CompoundPoisson,
    Drift,
    GammaProcess,
    InverseGaussianProcess,
    KouJumpDiffusion,
    MertonJumpDiffusion,
    NegativeBinomialProcess,
    NormalTemperedStable,
    PoissonProcess,
    TemperedStable,
    VarianceGamma,
)
from spxa.zoo.levy.compound_poisson import NormalJump

FAMILIES = [
    BrownianMotion(0.1, 0.3),
    GammaProcess(2.0, 3.0),
    InverseGaussianProcess(1.2, 2.0),
    PoissonProcess(2.0, 0.5),
    NegativeBinomialProcess(0.89, 0.86),
    VarianceGamma(0.3, 0.4, -0.2),
    NIG(5.0, -1.5, 0.7, 0.1),
    CompoundPoisson(2.0, NormalJump(0.1, 0.2)),
    MertonJumpDiffusion(0.05, 0.2, 1.0, -0.1, 0.1),
    KouJumpDiffusion(0.0, 0.2, 1.0, 0.4, 10.0, 5.0),
    AlphaStable(1.5, 0.3, 0.4, 0.1),
    AlphaStable(1.0, 0.4, 0.5, 0.2),
    AlphaStable(0.7, -0.5, 0.5, 0.0),
    TemperedStable(0.6, 1.0, 1.0),
    CGMY(1.0, 3.0, 6.0, 0.6),
    NormalTemperedStable(0.6, 1.5, 0.8, 0.1, 0.3),
]


def _ecf_z(x: np.ndarray, phi, u: float) -> float:
    p, p2 = complex(phi(u)), complex(phi(2 * u))
    m = np.mean(np.exp(1j * u * x))
    n = len(x)
    z_re = (m.real - p.real) / np.sqrt(max((1 + p2.real) / 2 - p.real**2, 1e-30) / n)
    z_im = (m.imag - p.imag) / np.sqrt(max((1 - p2.real) / 2 - p.imag**2, 1e-30) / n)
    return float(max(abs(z_re), abs(z_im)))


@pytest.mark.parametrize("X", FAMILIES, ids=repr)
def test_draws_match_the_cf(X) -> None:
    assert X._vector_sampler
    rng = np.random.default_rng(2)
    for t in (0.37, 2.0):
        x = X.sample_at(np.full(40_000, t), rng)
        for u in (0.3, 1.0, 3.0):
            assert _ecf_z(x, lambda v: X.char_func(v, t), u) < 4.5, (t, u)


@pytest.mark.parametrize("X", FAMILIES, ids=repr)
def test_zero_time_is_zero_and_shape_is_kept(X) -> None:
    out = X.sample_at(np.array([[0.0, 0.5], [0.0, 1.5]]), np.random.default_rng(0))
    assert out.shape == (2, 2)
    assert np.all(out[:, 0] == 0.0)


def test_drift_is_deterministic() -> None:
    np.testing.assert_allclose(Drift(0.3).sample_at(np.array([0.0, 1.0, 2.5])), [0.0, 0.3, 0.75])


def test_mixed_times_in_one_call() -> None:
    """Each entry has its own horizon, so t = 4 draws have 4 times the variance of t = 1 draws."""
    X = BrownianMotion(0.0, 1.0)
    times = np.tile([1.0, 4.0], 50_000)
    x = X.sample_at(times, np.random.default_rng(3))
    assert np.var(x[times == 1.0]) == pytest.approx(1.0, rel=0.03)
    assert np.var(x[times == 4.0]) == pytest.approx(4.0, rel=0.03)


def test_default_sample_at_groups_equal_times() -> None:
    """Families without a direct sampler fall back to one simulate call per distinct time."""
    from spxa.zoo.levy import MeixnerProcess

    X = MeixnerProcess(0.3, -0.5, 1.0, 0.0)
    assert not X._vector_sampler
    x = X.sample_at(np.repeat([0.0, 0.5, 1.0], 20_000), np.random.default_rng(4))
    assert np.all(x[:20_000] == 0.0)
    for u in (1.0, 4.0):
        assert _ecf_z(x[40_000:], lambda v: X.char_func(v, 1.0), u) < 4.5


def test_default_sample_at_with_many_distinct_times() -> None:
    """Over 64 distinct times, the dyadic decomposition draws X_t with the right mixture cf."""
    X = CGMY(1.0, 5.0, 10.0, 1.5)
    assert not X._vector_sampler
    times = np.random.default_rng(5).uniform(0.2, 1.5, 20_000)
    x = X.sample_at(times, np.random.default_rng(6))
    for u in (0.5, 2.0):
        # conditionally on the times, E e^{iuX} averages phi(u, t_k)
        def mix(v: float) -> complex:
            return complex(np.mean(np.exp(times * X._char_exponent(np.array([v]))[0])))

        assert _ecf_z(x, mix, u) < 4.5
