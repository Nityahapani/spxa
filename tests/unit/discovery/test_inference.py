"""Sandwich and GMM standard errors; the Vuong test."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.discovery.empirical import from_statistics, summarize
from spxa.discovery.expr import Leaf, Model
from spxa.discovery.families import BROWNIAN, DRIFT, GAMMA, NIG_FAMILY, VARIANCE_GAMMA
from spxa.discovery.fit import BinnedData, fit_structure
from spxa.discovery.inference import (
    _sandwich,
    parameter_names,
    parameter_uncertainty,
    vuong_test,
)

from .conftest import vg_increments

DT = 1 / 52


@pytest.fixture(scope="module")
def data():
    s = summarize(vg_increments(0.2, 0.25, -0.3, DT, 2_000, seed=501, mu=0.1), dt=DT)
    return s, BinnedData.from_sample(s.sample, DT)


def test_sandwich_algebra() -> None:
    rng = np.random.default_rng(0)
    L = rng.normal(size=(6, 3))
    omega = np.eye(6)
    # Ω = I: the sandwich is (LᵀL)⁻¹
    np.testing.assert_allclose(_sandwich(L, omega), np.linalg.inv(L.T @ L), rtol=1e-12)
    # square L: L⁻¹ Ω L⁻ᵀ, whatever the scale of a row
    G = rng.normal(size=(4, 4))
    G[0] *= 1e4
    S = np.cov(rng.normal(size=(4, 50)))
    Gi = np.linalg.inv(G)
    np.testing.assert_allclose(_sandwich(G, S), Gi @ S @ Gi.T, rtol=1e-9)
    # rank deficiency and short systems give NaN
    L[:, 2] = L[:, 0] + L[:, 1]
    assert np.isnan(_sandwich(L, omega)).all()
    assert np.isnan(_sandwich(rng.normal(size=(2, 3)), np.eye(2))).all()


def test_parameter_names() -> None:
    m = Model(Leaf(GAMMA) + -Leaf(GAMMA) + Leaf(DRIFT), [(1.0, 2.0), (1.0, 3.0), (0.0,)])
    assert parameter_names(m) == [
        "Gamma[1].a",
        "Gamma[1].b",
        "Gamma[2].a",
        "Gamma[2].b",
        "Drift.mu",
    ]


def test_likelihood_standard_errors_match_the_sampling_spread(data) -> None:
    s, b = data
    r = fit_structure(Leaf(VARIANCE_GAMMA), s, binned=b, n_starts=1)
    u = parameter_uncertainty(r, s, binned=b)
    assert u.method == "sandwich"
    assert u.names == ["VG.sigma", "VG.nu", "VG.theta", "Drift.mu"]
    np.testing.assert_allclose(u.estimates, [v for p in r.model.params for v in p])
    # Monte Carlo standard deviations of the estimates over 100 samples of this
    # design (n = 2000 weekly VG(0.2, 0.25, -0.3), drift 0.1): the median
    # sandwich SEs were 0.0130, 0.0151, 0.0413, 0.0054
    mc = np.array([0.0132, 0.0146, 0.0363, 0.0055])
    ratio = u.standard_errors / mc
    assert np.all((ratio[:3] > 0.6) & (ratio[:3] < 1.6)), ratio
    # the drift's SE varies more between samples: at short horizons it is
    # pinned by the near-atom of the law, whose sharpness is itself random
    assert 0.3 < ratio[3] < 3.0
    ci = u.confidence_intervals(0.95)
    np.testing.assert_allclose(ci[:, 1] - ci[:, 0], 2 * 1.959964 * u.standard_errors, rtol=1e-6)
    assert "VG.sigma" in u.table() and str(u) == u.table()
    # rebuilding the bins from the sample gives the same answer
    again = parameter_uncertainty(r, s)
    np.testing.assert_allclose(again.standard_errors, u.standard_errors, rtol=1e-10)


def test_gmm_standard_errors(data) -> None:
    s, _ = data
    r = fit_structure(Leaf(VARIANCE_GAMMA), s, method="cumulants")
    u = parameter_uncertainty(r, s)
    assert u.method == "gmm"
    # exactly identified: equal to G⁻¹ Σ G⁻ᵀ mapped to natural parameters
    se = u.standard_errors
    assert np.all(np.isfinite(se)) and np.all(se > 0)
    # BM from Gaussian moments: σ̂² = k2, Var(k2) ≈ 2κ2²/n, so se(σ) ≈ σ/√(2n)
    g = from_statistics(mean=0.0, variance=0.04 * DT, dt=DT, n=4_000)
    bm = fit_structure(Leaf(BROWNIAN), g, method="cumulants")
    ubm = parameter_uncertainty(bm, g)
    assert ubm.standard_errors[0] == pytest.approx(0.2 / np.sqrt(2 * 4_000), rel=1e-4)
    # six parameters against four cumulants: not identified
    over = fit_structure(Leaf(VARIANCE_GAMMA) + Leaf(GAMMA), s, method="cumulants")
    assert np.isnan(parameter_uncertainty(over, s).standard_errors).all()
    with pytest.raises(ValueError, match="covariance"):
        bare = from_statistics(mean=0.0, variance=1.0)
        parameter_uncertainty(fit_structure(Leaf(BROWNIAN), bare, method="cumulants"), bare)


def test_vuong(data) -> None:
    s, b = data
    vg = fit_structure(Leaf(VARIANCE_GAMMA), s, binned=b, n_starts=1)
    bm = fit_structure(Leaf(BROWNIAN), s, binned=b, n_starts=1)
    nig = fit_structure(Leaf(NIG_FAMILY), s, binned=b, n_starts=1)
    v = vuong_test(vg, bm, b)
    assert v.preferred == "first" and v.z > 5 and v.p_value < 1e-6
    assert v.log_likelihood_ratio == pytest.approx(vg.loglik - bm.loglik, rel=1e-9)
    w = vuong_test(bm, vg, b)
    assert w.preferred == "second" and w.z == pytest.approx(-v.z, rel=1e-12)
    # VG over a week with ν = 0.25 has a density pole at the drift (Δt/ν < 1/2),
    # which NIG cannot reproduce: the test tells the two families apart
    close = vuong_test(vg, nig, b)
    assert close.preferred == "first" and close.z > 3
    raw = vuong_test(vg, bm, b, correction="none")
    assert raw.z > v.z  # the BIC correction penalises VG's two extra parameters
    with pytest.raises(ValueError, match="coincide"):
        vuong_test(vg, vg, b)
