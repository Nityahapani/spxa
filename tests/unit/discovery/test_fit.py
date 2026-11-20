"""Fitting a single structure: likelihood and cumulant objectives."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.discovery.empirical import from_statistics, summarize
from spxa.discovery.expr import Model, leaf
from spxa.discovery.families import BROWNIAN, DRIFT, GAMMA, NIG_FAMILY, VARIANCE_GAMMA
from spxa.discovery.fit import BinnedData, fit_structure, initial_params, with_location

from .conftest import gamma_increments, vg_increments

SIGMA, NU, THETA, MU = 0.2, 0.25, -0.3, 0.1
DT = 1 / 52


@pytest.fixture(scope="module")
def vg_data():
    return summarize(vg_increments(SIGMA, NU, THETA, DT, 6_000, seed=11, mu=MU), dt=DT)


def test_variance_gamma_parameters_are_recovered(vg_data) -> None:
    res = fit_structure(leaf(VARIANCE_GAMMA), vg_data, rng=np.random.default_rng(0))
    assert res.structure_key == "VG + Drift"
    (sigma, nu, theta), (mu,) = res.model.params
    # sampling error at n = 6000 weekly observations: a few percent on σ, ~10-15% on ν and θ
    assert sigma == pytest.approx(SIGMA, rel=0.08)
    assert nu == pytest.approx(NU, rel=0.25)
    assert theta == pytest.approx(THETA, rel=0.35)
    # the likelihood estimates the location; the mean agrees to sampling error
    assert abs(res.model.cumulants(1)[0] - vg_data.mean) < 3 * vg_data.standard_errors[0]
    assert res.g_pvalue > 0.01 and res.ks_distance < 0.02
    assert res.n_params == 4 and res.bic > res.aic


def test_brownian_motion_is_rejected_on_variance_gamma_data(vg_data) -> None:
    bm = fit_structure(leaf(BROWNIAN), vg_data)
    vg = fit_structure(leaf(VARIANCE_GAMMA), vg_data)
    assert bm.g_pvalue < 1e-6
    assert vg.bic < bm.bic - 50
    # the Gaussian fit matches the variance but has no excess kurtosis
    assert bm.cumulant_relative_errors[3] == pytest.approx(1.0)


def test_cumulant_method_matches_three_cumulants_exactly() -> None:
    target = Model(leaf(NIG_FAMILY) + leaf(DRIFT), [(9.0, -3.0, 0.5), (0.04,)])
    k = target.cumulants(4)
    stats_only = from_statistics(
        mean=k[0], variance=k[1], skewness=k[2] / k[1] ** 1.5, excess_kurtosis=k[3] / k[1] ** 2
    )
    res = fit_structure(leaf(NIG_FAMILY), stats_only, method="cumulants")
    assert res.max_relative_error < 1e-6
    # the cumulant objective profiles the drift: the mean matches exactly
    assert res.cumulant_relative_errors[0] < 1e-12
    np.testing.assert_allclose(res.model.params[0], (9.0, -3.0, 0.5), rtol=1e-4)
    assert res.loglik is None and res.aic is None and res.bic is None
    with pytest.raises(ValueError, match="sample"):
        fit_structure(leaf(NIG_FAMILY), stats_only, method="likelihood")
    with pytest.raises(ValueError):
        fit_structure(leaf(NIG_FAMILY), stats_only, method="moments")  # type: ignore[arg-type]


def test_initial_values_and_location() -> None:
    s = leaf(VARIANCE_GAMMA) + -leaf(GAMMA)
    assert with_location(s).key() == "VG + -Gamma + Drift"
    assert with_location(with_location(s)) == with_location(s)
    p = initial_params(with_location(s), std=0.2)
    m = Model(with_location(s), p)
    assert m.cumulants(2)[1] == pytest.approx(0.04, rel=1e-9)
    sub = (leaf(DRIFT) + leaf(BROWNIAN)) @ leaf(GAMMA)
    assert initial_params(sub, 0.3)[-1] == (2.0, 2.0)


def test_binned_data() -> None:
    x = np.random.default_rng(0).normal(size=1_000)
    b = BinnedData.from_sample(x, dt=1.0)
    assert b.n == 1_000 and len(b.counts) == 20
    assert np.all(np.diff(b.edges[1:-1]) >= 4 * b.bandwidth)
    assert np.isinf(b.edges[0]) and np.isinf(b.edges[-1])
    # logit-spaced levels: about tail_count = 10 observations in each outer bin
    assert b.raw_counts[0] == pytest.approx(10, abs=4)
    assert b.raw_counts[-1] == pytest.approx(10, abs=4)
    assert b.raw_counts.max() > 2 * b.raw_counts.mean()
    equal = BinnedData.from_sample(x, dt=1.0, tail_count=None)
    assert np.all(np.abs(equal.raw_counts - 50) <= 1)
    # smoothing moves mass between neighbouring bins but conserves it
    assert b.counts.sum() == pytest.approx(1_000, rel=1e-12)
    assert b.bandwidth == pytest.approx(0.02 * x.std(), rel=0.5)
    assert len(BinnedData.from_sample(x, 1.0, n_bins=12).counts) == 12
    # the model side is the matching smoothed law: N(0, 1 + h²) bin probabilities
    from scipy import stats

    p = b.probabilities(Model(leaf(BROWNIAN), [(1.0,)]))
    np.testing.assert_allclose(
        p, np.diff(stats.norm.cdf(b.edges, scale=np.sqrt(1 + b.bandwidth**2))), atol=1e-10
    )


def test_bins_merge_around_a_near_atom() -> None:
    # VG over a week with ν = 0.25: the Gamma clock is ~0 for a large share of weeks
    x = vg_increments(SIGMA, NU, THETA, DT, 3_000, seed=4, mu=MU)
    b = BinnedData.from_sample(x, DT)
    widths = np.diff(b.edges[1:-1])
    assert widths.min() >= 4 * b.bandwidth
    assert len(b.counts) < 60  # quantile edges inside the atom were merged


def test_goodness_of_fit_is_calibrated_under_the_null() -> None:
    # Gaussian data fitted by BM + Drift: the Wald test rejects at about its level
    pvals = []
    for seed in range(120):
        x = np.random.default_rng(1_000 + seed).normal(0.01, 0.1, size=800)
        pvals.append(fit_structure(leaf(BROWNIAN), summarize(x), n_starts=1).g_pvalue)
    rejections = np.mean(np.array(pvals) < 0.05)
    # binomial(120, 0.05): mean 6, sd 2.4 rejections
    assert 0.0 < rejections < 0.12


def test_warm_start(vg_data) -> None:
    cold = fit_structure(leaf(VARIANCE_GAMMA), vg_data, n_starts=1)
    warm = fit_structure(leaf(VARIANCE_GAMMA), vg_data, n_starts=1, initial=cold.model.params)
    assert warm.objective <= cold.objective + 1e-6
    np.testing.assert_allclose(warm.model.params[0], cold.model.params[0], rtol=1e-3)
    with pytest.raises(ValueError):
        fit_structure(leaf(VARIANCE_GAMMA), vg_data, initial=[(0.2, 0.3, 0.0)])


def test_tail_bins_recover_a_sum_of_pure_jump_laws() -> None:
    # daily VG(0.15, 0.2, -0.1) + Gamma(4, 25): true κ2, κ4 = 0.0309, 4.8e-4.
    # With equal quantile bins the fit's κ4 came out 3-4 times too large.
    dt, n = 1 / 252, 5_000
    x = vg_increments(0.15, 0.2, -0.1, dt, n, seed=1) + gamma_increments(4.0, 25.0, dt, n, seed=20)
    s = summarize(x, dt=dt)
    r = fit_structure(leaf(VARIANCE_GAMMA) + leaf(GAMMA), s, n_starts=3)
    (sigma, nu, theta), (a, b) = r.model.params[0], r.model.params[1]
    assert 0.12 < sigma < 0.2 and 0.12 < nu < 0.3 and -0.25 < theta < 0.0
    assert 2.0 < a < 9.0 and 15.0 < b < 50.0
    k = r.model_cumulants
    assert 0.025 < k[1] < 0.04
    assert 3e-4 < k[3] < 9.6e-4
    assert r.g_pvalue > 0.05


def test_moment_errors_and_z_scores() -> None:
    # the model is BM(0.2) exactly; targets are Laplace-like moments over dt = 1/4
    s = from_statistics(mean=0.0, variance=0.011, skewness=0.0, excess_kurtosis=3.0, dt=0.25, n=400)
    r = fit_structure(leaf(BROWNIAN), s, method="cumulants")
    e = r.moment_errors
    assert e is not None
    sigma = r.model.params[0][0]
    # variance error relative, skewness error absolute (|target| < 1), kurtosis relative
    assert e[0] == pytest.approx(abs(sigma**2 * 0.25 - 0.011) / 0.011, rel=1e-9)
    assert e[1] == pytest.approx(0.0, abs=1e-12)
    assert e[2] == pytest.approx(1.0, rel=1e-9)
    assert r.match_error == pytest.approx(max(e))
    z = r.cumulant_z_scores
    assert z is not None and z.shape == (4,)
    se = np.sqrt(np.diag(s.covariance))
    np.testing.assert_allclose(z, (r.model_cumulants - s.cumulants) / se)
    assert "moment error" in str(r)


def test_moment_errors_survive_a_zero_skewness_target() -> None:
    s = from_statistics(mean=0.0, variance=1.0, skewness=0.0, excess_kurtosis=0.0)
    r = fit_structure(leaf(NIG_FAMILY), s, method="cumulants")
    assert r.match_error is not None and r.match_error < 0.05


def test_kernel_matrix_and_pointwise_loglik(vg_data) -> None:
    b = BinnedData.from_sample(vg_data.sample, vg_data.dt)
    K = b.kernel_matrix()
    assert K.shape == (b.n, len(b.counts))
    np.testing.assert_allclose(K.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(K.sum(axis=0), b.counts, rtol=1e-12)
    res = fit_structure(leaf(VARIANCE_GAMMA), vg_data, binned=b, n_starts=1)
    ell = b.pointwise_loglik(res.model)
    assert ell.sum() == pytest.approx(res.loglik, rel=1e-12)
    bare = BinnedData(**{**b.__dict__, "sample": None})
    with pytest.raises(ValueError, match="sample"):
        bare.kernel_matrix()
