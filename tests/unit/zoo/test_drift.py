"""Deterministic drift: the Lévy process with triplet (μ, 0, 0)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spxa import Drift
from spxa.zoo.levy import BrownianMotion, GammaProcess, VarianceGamma


def test_triplet_cumulants_and_char_func() -> None:
    d = Drift(0.3)
    t = d.triplet
    assert (t.b, t.sigma_sq, t.nu.total_mass()) == (0.3, 0.0, 0.0)
    assert d.cumulants(4) == {1: 0.3, 2: 0.0, 3: 0.0, 4: 0.0}
    u = np.array([-2.0, 0.0, 1.5])
    np.testing.assert_allclose(d.char_func(u, t=2.0), np.exp(0.6j * u), rtol=0, atol=1e-15)
    # the generic Lévy–Khintchine route agrees
    assert abs(t.char_exp(1.5, 2.0) - np.exp(0.9j)) < 1e-12


def test_properties() -> None:
    assert Drift(0.0).properties.is_martingale and Drift(0.0).properties.is_subordinator
    assert not Drift(-0.1).properties.is_martingale
    assert not Drift(-0.1).properties.is_subordinator
    assert Drift(2.0).properties.is_subordinator
    assert Drift(2.0).properties.self_similarity_index == 1.0
    with pytest.raises(ValueError):
        Drift(float("nan"))


def test_paths_are_deterministic_lines() -> None:
    paths = Drift(-0.5).simulate(4, n_paths=3, T=2.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(paths, np.tile([0.0, -0.25, -0.5, -0.75, -1.0], (3, 1)))


def test_compensates_a_process_into_a_martingale() -> None:
    vg = VarianceGamma(sigma=0.2, nu=0.3, theta=-0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        z = vg + Drift(0.1)
    c = z.cumulants(4)
    assert abs(c[1]) < 1e-12
    for n in (2, 3, 4):
        assert c[n] == pytest.approx(vg.cumulants(4)[n], rel=1e-9)


def test_time_change_by_a_positive_drift() -> None:
    # X @ Drift(c) is X run at speed c: X_{T_t} with T_t = c t
    x = BrownianMotion(mu=0.0, sigma=1.0) @ Drift(4.0)
    paths = x.simulate(1, n_paths=20_000, T=1.0, rng=np.random.default_rng(1))
    assert np.var(paths[:, -1]) == pytest.approx(4.0, rel=0.05)
    with pytest.raises(ValueError):
        BrownianMotion() @ Drift(-1.0)
    assert Drift(1.0).bernstein_function(2.0) == 2.0
    assert GammaProcess(1.0, 1.0).properties.is_subordinator
