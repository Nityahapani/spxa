"""Posterior probabilities over structures and model averaging."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import logsumexp, softmax

from spxa.discovery import families as F
from spxa.discovery.bayes import GaussianPrior, default_prior, laplace_evidence
from spxa.discovery.density import model_distribution
from spxa.discovery.empirical import summarize
from spxa.discovery.expr import Model, leaf
from spxa.discovery.fit import BinnedData, fit_structure
from spxa.discovery.posterior import (
    MixtureDistribution,
    StructurePosterior,
    structure_posterior,
)
from spxa.discovery.pricing import cos_prices

DT = 1 / 252
BM = leaf(F.BROWNIAN)
VG = leaf(F.VARIANCE_GAMMA)
JUMPS = leaf(F.BROWNIAN) + leaf(F.NORMAL_JUMPS)


def _fits(model: Model, n: int, dt: float, seed: int, structures: list[Any]) -> Any:
    x = np.diff(model.to_process().simulate(n, 1, T=n * dt, rng=np.random.default_rng(seed))[0])
    summ = summarize(x, dt=dt)
    bd = BinnedData.from_sample(summ.sample, summ.dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = [
            fit_structure(s, summ, binned=bd, n_starts=3, rng=np.random.default_rng(1))
            for s in structures
        ]
    return summ, bd, fits


@pytest.fixture(scope="module")
def bm() -> Any:
    """2000 daily Brownian returns, with BM, VG and BM + NormalJumps fitted."""
    truth = Model(leaf(F.BROWNIAN) + leaf(F.DRIFT), [(0.2,), (0.05,)])
    summ, bd, fits = _fits(truth, 2000, DT, 1, [BM, VG, JUMPS])
    return summ, bd, fits, structure_posterior(fits, bd)


# ── Bayes' rule ───────────────────────────────────────────────────────────────


def test_probabilities_are_bayes_rule_on_laplace_evidences(bm: Any) -> None:
    _, bd, fits, P = bm
    assert isinstance(P, StructurePosterior) and P.method == "laplace"
    assert P.probabilities.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.all(np.diff(P.probabilities) <= 0)
    # each evidence is exactly laplace_evidence with the default prior
    std = float(np.std(bd.sample, ddof=1) / np.sqrt(bd.dt))  # per unit time, as discover's
    for f in fits:
        s = f.model.structure
        ev = laplace_evidence(f, bd, prior=default_prior(s, std))
        i = P.keys.index(s.key())
        assert P.log_evidences[i] == pytest.approx(ev.log_evidence, abs=1e-9)
    z = P.log_evidences
    np.testing.assert_allclose(P.probabilities, softmax(z - z.max()), rtol=1e-14, atol=1e-300)
    assert P.bayes_factor(0, 1) == pytest.approx(z[0] - z[1], abs=1e-12)
    assert np.all(P.standard_errors == 0) and np.all(P.probability_se == 0)


def test_lookup_by_key_or_structure(bm: Any) -> None:
    *_, P = bm
    for e, p in zip(P.evidences, P.probabilities):
        assert P.probability(e.structure) == p
        assert P.probability(e.structure.key()) == p
    assert P.probability("NIG + Drift") == 0.0
    assert P.best is P.evidences[0]


def test_simpler_structure_is_preferred_on_brownian_data(bm: Any) -> None:
    *_, P = bm
    assert P.keys[0] == "BM + Drift" and P.probabilities[0] > 0.5
    assert P.probability("VG + Drift") < 1e-3


def test_model_prior_shifts_the_log_odds_exactly(bm: Any) -> None:
    _, bd, fits, P = bm
    weights = {"BM + Drift": 3.0, "VG + Drift": 1.0, "BM + NormalJumps + Drift": 1.0}
    Q = structure_posterior(fits, bd, model_prior=weights)
    R = structure_posterior(fits, bd, model_prior=lambda s: weights[s.key()])
    for key in weights:
        assert Q.probability(key) == pytest.approx(R.probability(key), rel=1e-12)

    def log_odds(post: StructurePosterior, a: str, b: str) -> float:
        return float(np.log(post.probability(a) / post.probability(b)))

    a, b = "BM + Drift", "BM + NormalJumps + Drift"
    assert log_odds(Q, a, b) - log_odds(P, a, b) == pytest.approx(np.log(3.0), abs=1e-10)
    assert np.exp(Q.log_model_prior).sum() == pytest.approx(1.0)
    # a zero prior weight removes a structure; it stays listed with probability 0
    Z = structure_posterior(fits, bd, model_prior={"BM + Drift": 1.0, "VG + Drift": 1.0})
    assert Z.probability("BM + NormalJumps + Drift") == 0.0
    assert Z.probabilities.sum() == pytest.approx(1.0)
    for bad in ({"BM + Drift": -1.0}, {"nothing": 1.0}, {"BM + Drift": np.inf}):
        with pytest.raises(ValueError):
            structure_posterior(fits, bd, model_prior=bad)


def test_a_single_candidate_has_probability_one(bm: Any) -> None:
    _, bd, fits, _ = bm
    P = structure_posterior(fits[:1], bd)
    assert P.probabilities.tolist() == [1.0]


def test_top_keeps_the_best_candidates_by_bic(bm: Any) -> None:
    _, bd, fits, _ = bm
    P = structure_posterior(fits, bd, top=2)
    best_two = sorted(fits, key=lambda f: f.bic)[:2]
    assert set(P.keys) == {f.model.structure.key() for f in best_two}


def test_credible_set(bm: Any) -> None:
    *_, P = bm
    assert P.credible_set(1e-9) == P.evidences[:1]
    full = P.credible_set(1.0)
    assert len(full) <= len(P.evidences)
    cut = P.credible_set(0.95)
    mass = P.probabilities[: len(cut)].sum()
    assert mass >= 0.95 - 1e-12
    assert P.probabilities[: len(cut) - 1].sum() < 0.95
    for level in (0.0, 1.5):
        with pytest.raises(ValueError):
            P.credible_set(level)


def test_table_lists_probabilities_and_bic_weights(bm: Any) -> None:
    *_, P = bm
    text = str(P)
    assert "P(M|D)" in text and "BIC wt" in text and "se" not in text.split("\n")[0].split()
    assert text.count("\n") == len(P.evidences)
    w = P.bic_weights()
    assert w.sum() == pytest.approx(1.0) and w.argmax() == int(np.argmin(P.bic))


# ── importance sampling ───────────────────────────────────────────────────────


def test_delta_method_standard_errors_are_the_softmax_jacobian(bm: Any) -> None:
    *_, P = bm
    se = np.array([0.01, 0.05, 0.02])
    Q = dataclasses.replace(P, standard_errors=se)
    z, h = Q.log_evidences + Q.log_model_prior, 1e-6
    J = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        J[:, j] = (np.exp(z + e - logsumexp(z + e)) - np.exp(z - e - logsumexp(z - e))) / (2 * h)
    np.testing.assert_allclose(Q.probability_se, np.sqrt((J**2) @ se**2), rtol=1e-6)


def test_importance_refinement_and_its_monte_carlo_error(bm: Any) -> None:
    _, bd, fits, _ = bm
    runs = [
        structure_posterior(
            fits[::2], bd, evidence="importance", n_draws=1500, rng=np.random.default_rng(s)
        )
        for s in range(6)
    ]
    for Q in runs:
        assert Q.method == "importance" and len(Q.refined) == 2
        assert np.all(Q.standard_errors > 0)
        assert [i.log_evidence for i in Q.refined] == Q.log_evidences.tolist()
        z = Q.log_evidences + Q.log_model_prior
        np.testing.assert_allclose(Q.probabilities, softmax(z - z.max()), rtol=1e-14)
        assert "se" in str(Q).split("\n")[0].split()
    p_jumps = np.array([Q.probability("BM + NormalJumps + Drift") for Q in runs])
    se = np.array([Q.probability_se[Q.keys.index("BM + NormalJumps + Drift")] for Q in runs])
    # the reported error matches the spread across seeds
    assert 0.4 < np.std(p_jumps, ddof=1) / np.mean(se) < 2.5
    # Laplace understates the weakly identified jump structure; importance
    # sampling corrects it by more than its own error
    laplace = structure_posterior(fits[::2], bd).probability("BM + NormalJumps + Drift")
    assert np.mean(p_jumps) - laplace > 3 * np.mean(se)  # observed: 0.26 vs 0.19, 4 se


# ── clear cases ───────────────────────────────────────────────────────────────


def test_jump_diffusion_data_put_all_mass_on_jumps() -> None:
    truth = Model(
        leaf(F.BROWNIAN) + leaf(F.NORMAL_JUMPS) + leaf(F.DRIFT),
        [(0.15,), (20.0, -0.02, 0.04), (0.05,)],
    )
    _, bd, fits = _fits(truth, 3000, DT, 2, [BM, VG, JUMPS])
    P = structure_posterior(fits, bd)
    assert P.keys[0] == "BM + NormalJumps + Drift" and P.probabilities[0] > 1 - 1e-12
    (_, (rate, jump_mean, _), _) = P.best.model.params
    assert abs(jump_mean + 0.02) < 0.01 and 10 < rate < 30


def test_variance_gamma_data_put_all_mass_on_vg() -> None:
    truth = Model(leaf(F.VARIANCE_GAMMA) + leaf(F.DRIFT), [(0.2, 0.3, -0.1), (0.01,)])
    _, bd, fits = _fits(truth, 2000, 1 / 52, 3, [BM, VG, JUMPS])
    P = structure_posterior(fits, bd)
    assert P.keys[0] == "VG + Drift" and P.probabilities[0] > 1 - 1e-12


# ── mixtures ──────────────────────────────────────────────────────────────────


def _normal(mu: float, sigma: float, t: float = 1.0) -> Any:
    m = Model(leaf(F.BROWNIAN) + leaf(F.DRIFT), [(sigma,), (mu,)])
    return m, model_distribution(m, t)


def test_mixture_of_normals_is_exact() -> None:
    (m1, d1), (m2, d2) = _normal(-0.1, 0.2), _normal(0.3, 0.5)
    k = np.array([m1.cumulants(2), m2.cumulants(2)])
    mix = MixtureDistribution([0.3, 0.7], [d1, d2], k)
    x = np.linspace(-1.0, 1.5, 11)
    exact_cdf = 0.3 * stats.norm.cdf(x, -0.1, 0.2) + 0.7 * stats.norm.cdf(x, 0.3, 0.5)
    exact_pdf = 0.3 * stats.norm.pdf(x, -0.1, 0.2) + 0.7 * stats.norm.pdf(x, 0.3, 0.5)
    np.testing.assert_allclose(mix.cdf(x), exact_cdf, atol=1e-10)
    np.testing.assert_allclose(mix.pdf(x), exact_pdf, atol=1e-9)
    assert mix.mean == pytest.approx(0.3 * -0.1 + 0.7 * 0.3, abs=1e-14)
    var = 0.3 * (0.04 + 0.01) + 0.7 * (0.25 + 0.09) - mix.mean**2
    assert mix.variance == pytest.approx(var, rel=1e-13)
    q = np.array([1e-4, 0.05, 0.5, 0.95, 0.9999])
    np.testing.assert_allclose(mix.cdf(mix.quantile(q)), q, atol=1e-10)
    assert np.ndim(mix.quantile(0.5)) == 0
    # unnormalised weights are normalised
    same = MixtureDistribution([3.0, 7.0], [d1, d2], k)
    np.testing.assert_allclose(same.cdf(x), mix.cdf(x), atol=1e-15)


def test_mixture_mean_and_variance_match_its_density() -> None:
    (m1, d1), (m2, d2) = _normal(0.0, 0.1), _normal(0.2, 0.3)
    mix = MixtureDistribution([0.5, 0.5], [d1, d2], np.array([m1.cumulants(2), m2.cumulants(2)]))
    lo, hi = mix.support
    mass = integrate.quad(lambda x: float(mix.pdf(x)), lo, hi, limit=200)[0]
    m1_ = integrate.quad(lambda x: x * float(mix.pdf(x)), lo, hi, limit=200)[0]
    m2_ = integrate.quad(lambda x: x * x * float(mix.pdf(x)), lo, hi, limit=200)[0]
    assert mass == pytest.approx(1.0, abs=1e-9)
    assert m1_ == pytest.approx(mix.mean, abs=1e-9)
    assert m2_ - m1_**2 == pytest.approx(mix.variance, rel=1e-8)


def test_mixture_validation() -> None:
    m, d = _normal(0.0, 0.1)
    k = np.array([m.cumulants(2)])
    bad: list[tuple[list[float], Any]] = [
        ([], k),  # no weights
        ([1.0, 1.0], k),  # two weights, one component
        ([-1.0], k),
        ([0.0], k),
        ([1.0], k[:, :1]),  # cumulants of the wrong shape
    ]
    for w, kk in bad:
        with pytest.raises(ValueError):
            MixtureDistribution(w, [d], kk)
    mix = MixtureDistribution([1.0], [d], k)
    for q in (0.0, 1.0, [0.5, 1.2]):
        with pytest.raises(ValueError):
            mix.quantile(q)


# ── predictive and averaged prices ────────────────────────────────────────────


def test_predictive_moments_are_the_model_average(bm: Any) -> None:
    *_, P = bm
    for t in (None, 1 / 52, 1.0):
        horizon = P.dt if t is None else t
        d = P.predictive(t)
        k = np.array([e.model.cumulants(2, t=horizon) for e in P.evidences])
        keep = P.probabilities >= 1e-6
        w = P.probabilities[keep] / P.probabilities[keep].sum()
        mean = w @ k[keep, 0]
        assert d.mean == pytest.approx(mean, abs=1e-15)
        assert d.variance == pytest.approx(w @ (k[keep, 1] + k[keep, 0] ** 2) - mean**2, rel=1e-12)
        assert len(d.components) == int(keep.sum())
    # the mixture's averaged cumulants are those of cumulants()
    np.testing.assert_allclose(
        P.cumulants(2, t=1.0), P.probabilities @ [e.model.cumulants(2) for e in P.evidences]
    )
    with pytest.raises(ValueError):
        P.predictive(0.0)


def test_predictive_with_draws_adds_parameter_uncertainty(bm: Any) -> None:
    *_, P = bm
    d = P.predictive(1.0, n_draws=40, rng=np.random.default_rng(5))
    sets = P._parameter_sets(40, np.random.default_rng(5))
    keep = P.probabilities >= 1e-6
    w, k = [], []
    for p, models, kp in zip(P.probabilities, sets, keep):
        if kp:
            for m in models:
                w.append(p / len(models))
                k.append(m.cumulants(2, t=1.0))
    w_, k_ = np.array(w) / np.sum(w), np.array(k)
    assert d.mean == pytest.approx(w_ @ k_[:, 0], abs=1e-14)
    # law of total variance over structures and draws
    assert d.variance == pytest.approx(w_ @ k_[:, 1] + w_ @ k_[:, 0] ** 2 - d.mean**2, rel=1e-12)
    assert d.variance > P.predictive(1.0).variance
    with pytest.raises(ValueError):
        P.predictive(n_draws=-1)


def test_averaged_prices_without_draws_are_a_weighted_mixture(bm: Any) -> None:
    *_, P = bm
    strikes = np.array([80.0, 100.0, 120.0])
    out = P.option_prices(100.0, strikes, 0.5, 0.03, 0.01, True)
    each = np.array(
        [cos_prices(e.model, 100.0, strikes, 0.5, 0.03, 0.01, True) for e in P.evidences]
    )
    np.testing.assert_allclose(out.by_structure, each, rtol=1e-14)
    np.testing.assert_allclose(out.mean, P.probabilities @ each, rtol=1e-14)
    # without draws only the spread between structures remains
    between = P.probabilities @ (each - out.mean) ** 2
    np.testing.assert_allclose(out.sd, np.sqrt(between), rtol=1e-6, atol=1e-12)
    assert out.discarded.tolist() == [0] * len(P.evidences)


def test_averaged_prices_keep_put_call_parity(bm: Any) -> None:
    *_, P = bm
    K = np.array([90.0, 100.0, 110.0])
    kw = {"n_draws": 30}
    c = P.option_prices(100.0, K, 1.0, 0.03, 0.01, True, rng=np.random.default_rng(2), **kw)
    p = P.option_prices(100.0, K, 1.0, 0.03, 0.01, False, rng=np.random.default_rng(2), **kw)
    parity = 100.0 * np.exp(-0.01) - K * np.exp(-0.03)
    np.testing.assert_allclose(c.mean - p.mean, parity, atol=1e-8)
    assert np.all(c.sd > 0)
    # one jump draw here has jump sd 9.5 in log price: its correction is about
    # -5e18 and its prices overflow, so it is discarded and counted
    i = P.keys.index("BM + NormalJumps + Drift")
    assert c.discarded[i] == p.discarded[i] == 1
    assert c.discarded.sum() == 1


# ── reports and errors ────────────────────────────────────────────────────────


def test_report_shortcut_matches_the_function() -> None:
    from spxa.discovery import discover, grammar_from_names

    x = np.random.default_rng(4).normal(0.08 * DT, 0.2 * np.sqrt(DT), 1500)
    r = discover(x, dt=DT, grammar=grammar_from_names(["BM", "VG"], max_terms=1), n_starts=2)
    P = r.posterior()
    Q = structure_posterior(r)
    assert P.keys == Q.keys
    np.testing.assert_allclose(P.probabilities, Q.probabilities, rtol=1e-12)
    assert P.keys[0] == "BM + Drift"
    assert r.posterior(top=1).keys == [r.candidates[0].structure_key]


def test_candidates_without_a_laplace_approximation_are_skipped(
    bm: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import spxa.discovery.posterior as post

    _, bd, fits, _ = bm
    real = post.laplace_evidence

    def flaky(fit: Any, binned: Any, **kw: Any) -> Any:
        if fit.model.structure.key() == "VG + Drift":
            raise ValueError("not positive definite")
        return real(fit, binned, **kw)

    monkeypatch.setattr(post, "laplace_evidence", flaky)
    P = structure_posterior(fits, bd)
    assert "VG + Drift" not in P.keys and P.probabilities.sum() == pytest.approx(1.0)
    assert P.skipped == [("VG + Drift", "not positive definite")]
    assert "VG + Drift: skipped (not positive definite)" in P.table()

    def never(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("not positive definite")

    monkeypatch.setattr(post, "laplace_evidence", never)
    with pytest.raises(ValueError, match="no candidate"):
        structure_posterior(fits, bd)


def test_input_validation(bm: Any) -> None:
    summ, bd, fits, _ = bm
    with pytest.raises(ValueError, match="no candidates"):
        structure_posterior([], bd)
    with pytest.raises(ValueError, match="bins"):
        structure_posterior(fits)
    with pytest.raises(ValueError, match="evidence"):
        structure_posterior(fits, bd, evidence="bic")  # type: ignore[arg-type]
    moments = fit_structure(BM, summ, method="cumulants")
    with pytest.raises(ValueError, match="likelihood"):
        structure_posterior([moments], bd)
    with pytest.raises(ValueError, match="coordinates"):
        structure_posterior(fits, bd, prior=GaussianPrior([0.0], [1.0]))
    with pytest.raises(TypeError):
        structure_posterior(fits, bd, prior=lambda s: 1.0)
