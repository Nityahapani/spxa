"""Simulated X @ T paths follow the law of the subordinated process.

The old simulator interpolated one parent path linearly at the random clock
times. That shrinks the Gaussian variance between grid points and smears jumps.
For VG @ TS the empirical cf missed by z = 1.3e-9 in p-value terms. Every route
of the new simulator is covered here: vectorised parents, cf inversion,
integer-valued clocks and lattice parents.
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
    GammaProcess,
    InverseGaussianProcess,
    MeixnerProcess,
    NegativeBinomialProcess,
    PoissonProcess,
    TemperedStable,
    VarianceGamma,
)

CASES = {
    "vector parent, continuous clock": BrownianMotion(0.1, 0.3) @ GammaProcess(2.0, 2.0),
    "nested time change": (BrownianMotion(0.1, 0.3) @ GammaProcess(2.0, 2.0))
    @ InverseGaussianProcess(1.0, 2.0),
    "lattice parent": NegativeBinomialProcess(0.89, 0.86) @ InverseGaussianProcess(0.77, 1.46),
    "heavy-tailed parent": AlphaStable(1.0, 0.5, 0.4, 0.1) @ GammaProcess(2.0, 2.0),
    "sum parent, integer clock": (-0.7 * VarianceGamma(0.2, 0.3, 0.1) + NIG(5.0, 1.0, 1.0, 0.0))
    @ NegativeBinomialProcess(2.0, 0.5),
    "clock with an atom at zero": BrownianMotion(0.1, 0.3)
    @ (PoissonProcess(2.0) + GammaProcess(1.0, 3.0)),
    "cf inversion, TS clock": VarianceGamma(0.32, 0.79, 0.22) @ TemperedStable(0.49, 0.22, 0.55),
    "cf inversion, Meixner parent": MeixnerProcess(1.2, 0.7, 1.8, 0.0)
    @ InverseGaussianProcess(1.0, 2.0),
    "grouped, Poisson clock": CGMY(1.0, 3.0, 4.0, 0.5) @ PoissonProcess(1.5),
}


def _ecf_z(x: np.ndarray, Z, u: float, t: float) -> float:
    p, p2 = complex(Z.char_func(u, t)), complex(Z.char_func(2 * u, t))
    m = np.mean(np.exp(1j * u * x))
    n = len(x)
    z_re = (m.real - p.real) / np.sqrt(max((1 + p2.real) / 2 - p.real**2, 1e-30) / n)
    z_im = (m.imag - p.imag) / np.sqrt(max((1 - p2.real) / 2 - p.imag**2, 1e-30) / n)
    return float(max(abs(z_re), abs(z_im)))


@pytest.mark.parametrize("name", list(CASES))
def test_paths_follow_the_subordinated_law(name: str) -> None:
    Z = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = Z.simulate(4, 20_000, 1.0, np.random.default_rng(0))
    assert x.shape == (20_000, 5) and np.all(x[:, 0] == 0.0)
    for u in (0.3, 1.0, 3.0):
        assert _ecf_z(x[:, -1], Z, u, 1.0) < 4.5, ("X_T", u)
        assert _ecf_z(x[:, 1], Z, u, 0.25) < 4.5, ("first increment", u)


def test_increments_are_uncorrelated() -> None:
    Z = BrownianMotion(0.1, 0.3) @ GammaProcess(2.0, 2.0)
    x = Z.simulate(2, 40_000, 1.0, np.random.default_rng(5))
    d = np.diff(x, axis=1)
    assert abs(np.corrcoef(d[:, 0], d[:, 1])[0, 1]) < 4.5 / np.sqrt(40_000)


def test_composite_sample_at_matches_simulate_law() -> None:
    Z = BrownianMotion(0.1, 0.3) @ GammaProcess(2.0, 2.0)
    assert Z._vector_sampler
    x = Z.sample_at(np.full(40_000, 0.6), np.random.default_rng(6))
    for u in (0.3, 1.0, 3.0):
        assert _ecf_z(x, Z, u, 0.6) < 4.5
