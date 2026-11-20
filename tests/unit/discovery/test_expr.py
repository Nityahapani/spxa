"""Composition structures and bound models."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.discovery.expr import Leaf, Model, Scale, Subordinate, Sum, leaf
from spxa.discovery.families import (
    BROWNIAN,
    DRIFT,
    GAMMA,
    INVERSE_GAUSSIAN,
    NIG_FAMILY,
    NORMAL_JUMPS,
    POISSON,
    VARIANCE_GAMMA,
)

U = np.array([-4.0, -1.0, 0.3, 2.5])


def assert_same_law(a: Model, b_cumulants: np.ndarray, b_cf: np.ndarray, order: int = 8) -> None:
    np.testing.assert_allclose(a.cumulants(order), b_cumulants, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(a.char_func(U, t=1.5), b_cf, rtol=1e-12, atol=1e-14)


def test_brownian_motion_in_gamma_time_is_variance_gamma() -> None:
    sigma, nu, theta = 0.25, 0.4, -0.2
    sub = Subordinate(leaf(DRIFT) + leaf(BROWNIAN), leaf(GAMMA))
    m = Model(sub, [(theta,), (sigma,), (1 / nu, 1 / nu)])
    vg = (sigma, nu, theta)
    assert_same_law(
        m,
        VARIANCE_GAMMA.cumulants(vg, 8),
        np.exp(1.5 * VARIANCE_GAMMA.char_exponent(U, vg)),
    )
    np.testing.assert_allclose(m.strip(), VARIANCE_GAMMA.strip(vg), rtol=1e-10)


def test_brownian_motion_in_inverse_gaussian_time_is_nig() -> None:
    alpha, beta, delta = 5.0, -1.5, 0.4
    gamma = np.sqrt(alpha**2 - beta**2)
    sub = (leaf(DRIFT) + leaf(BROWNIAN)) @ leaf(INVERSE_GAUSSIAN)
    m = Model(sub, [(beta,), (1.0,), (delta / gamma, delta**2)])
    nig = (alpha, beta, delta)
    assert_same_law(m, NIG_FAMILY.cumulants(nig, 8), np.exp(1.5 * NIG_FAMILY.char_exponent(U, nig)))
    np.testing.assert_allclose(m.strip(), (-alpha - beta, alpha - beta), rtol=1e-9)


def test_poisson_time_gives_compound_poisson() -> None:
    lam, mean, std = 1.7, -0.05, 0.2
    m = Model((leaf(DRIFT) + leaf(BROWNIAN)) @ leaf(POISSON), [(mean,), (std,), (lam, 1.0)])
    cp = (lam, mean, std)
    assert_same_law(
        m, NORMAL_JUMPS.cumulants(cp, 8), np.exp(1.5 * NORMAL_JUMPS.char_exponent(U, cp))
    )
    assert m.strip() == (-np.inf, np.inf)


def test_variance_gamma_as_a_difference_of_gammas() -> None:
    sigma, nu, theta = 0.25, 0.4, -0.2
    lo, hi = VARIANCE_GAMMA.strip((sigma, nu, theta))
    diff = leaf(GAMMA) + -leaf(GAMMA)
    assert diff.key() == "Gamma + -Gamma"
    m = Model(diff, [(1 / nu, hi), (1 / nu, -lo)])
    vg = (sigma, nu, theta)
    assert_same_law(
        m, VARIANCE_GAMMA.cumulants(vg, 8), np.exp(1.5 * VARIANCE_GAMMA.char_exponent(U, vg))
    )
    np.testing.assert_allclose(m.strip(), (lo, hi), rtol=1e-12)
    assert not diff.is_subordinator
    assert (leaf(GAMMA) + leaf(POISSON)).is_subordinator


def test_scaling_and_time() -> None:
    m = Model(2.0 * leaf(GAMMA), [(3.0, 2.0)])
    np.testing.assert_allclose(m.cumulants(5), GAMMA.cumulants((3.0, 1.0), 5), rtol=1e-13)
    np.testing.assert_allclose(m.cumulants(3, t=0.25), 0.25 * GAMMA.cumulants((3.0, 1.0), 3))
    assert m.strip() == (-np.inf, 1.0)
    neg = Model(-leaf(GAMMA), [(3.0, 2.0)])
    assert neg.strip() == (-2.0, np.inf)
    assert neg.cumulants(3)[0] == -1.5
    assert not (-leaf(GAMMA)).is_subordinator


def test_free_round_trip_and_validation() -> None:
    s = leaf(VARIANCE_GAMMA) + -leaf(GAMMA) + leaf(DRIFT)
    assert isinstance(s, Sum) and s.size == 3 and s.n_params == 6
    m = Model(s, [(0.2, 0.3, -0.1), (2.0, 15.0), (0.01,)])
    m2 = Model.from_free(s, m.to_free())
    np.testing.assert_allclose(np.concatenate(m2.params), np.concatenate(m.params), rtol=1e-12)
    with pytest.raises(ValueError):
        Model(s, [(0.2, 0.3, -0.1)])
    with pytest.raises(ValueError):
        Model.from_free(s, np.zeros(5))
    with pytest.raises(ValueError):
        Subordinate(leaf(BROWNIAN), leaf(VARIANCE_GAMMA))
    with pytest.raises(ValueError):
        Scale(0.0, leaf(GAMMA))
    with pytest.raises(ValueError):
        Sum([leaf(GAMMA)])


def test_structural_identity() -> None:
    a = leaf(VARIANCE_GAMMA) + -leaf(GAMMA)
    b = Sum([Leaf(VARIANCE_GAMMA), Scale(-1.0, Leaf(GAMMA))])
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1
    assert repr(a) == "<structure VG + -Gamma>"
    assert (0.5 * (leaf(BROWNIAN) @ leaf(GAMMA))).key() == "0.5*((BM) @ (Gamma))"


def test_rendering() -> None:
    m = Model(
        leaf(VARIANCE_GAMMA) + -leaf(GAMMA) + leaf(DRIFT), [(0.2, 0.3, -0.1), (2.0, 4.0), (0.01,)]
    )
    assert m.render(3) == (
        "VarianceGamma(sigma=0.2, nu=0.3, theta=-0.1) - GammaProcess(a=2, b=4) + Drift(mu=0.01)"
    )
    assert m.render(3, weighted=True) == (
        "0.2 * VarianceGamma(sigma=1, nu=0.3, theta=-0.5) - (0.25 * GammaProcess(a=2, b=1)) + Drift(mu=0.01)"
    )
    sub = Model(leaf(BROWNIAN) @ leaf(GAMMA), [(0.3,), (2.0, 2.0)])
    assert str(sub) == "(BrownianMotion(sigma=0.3)) @ (GammaProcess(a=2, b=2))"
    assert repr(sub).startswith("Model((BrownianMotion")


def test_to_process_matches_the_model() -> None:
    m = Model(
        leaf(VARIANCE_GAMMA) + -leaf(GAMMA) + leaf(DRIFT), [(0.2, 0.3, -0.1), (2.0, 4.0), (0.6,)]
    )
    proc = m.to_process()
    zoo = proc.cumulants(4)
    np.testing.assert_allclose(
        [float(zoo[n]) for n in range(1, 5)], m.cumulants(4), rtol=1e-6, atol=1e-10
    )
    # the core cannot give a characteristic function for X @ T; the model can, and simulation agrees
    sub = Model(leaf(BROWNIAN) @ leaf(GAMMA), [(0.3,), (2.0, 2.0)])
    paths = sub.to_process().simulate(1, n_paths=40_000, T=1.0, rng=np.random.default_rng(7))
    x = paths[:, -1]
    k = sub.cumulants(4)
    assert np.var(x) == pytest.approx(k[1], rel=0.03)
    emp_cf = np.mean(np.exp(1j * 2.0 * x))
    assert abs(emp_cf - sub.char_func(2.0)) < 0.01
