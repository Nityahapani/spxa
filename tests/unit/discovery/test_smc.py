"""Sequential Monte Carlo evidence."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from scipy import integrate
from scipy.special import gammaln, logsumexp

from spxa.discovery import families as F
from spxa.discovery.bayes import _log_target, importance_evidence, laplace_evidence
from spxa.discovery.empirical import summarize
from spxa.discovery.expr import Model, leaf
from spxa.discovery.fit import BinnedData, fit_structure
from spxa.discovery.posterior import structure_posterior
from spxa.discovery.smc import SMCEvidence, smc_evidence, tempered_smc

from .test_evidence import _gaussian, _quad_log_evidence


def _pooled(runs: list[Any]) -> tuple[float, float]:
    """log mean Z and its delta-method standard error over independent runs."""
    z = np.array([r.log_evidence for r in runs])
    log_z = float(logsumexp(z) - np.log(z.size))
    return log_z, float(np.std(np.exp(z - log_z), ddof=1) / np.sqrt(z.size))


# ── the engine, on targets with known normalising constants ───────────────────


def test_gaussian_target_from_an_offset_reference() -> None:
    m = np.array([0.3, -0.2])
    S = np.array([[1.0, 0.6], [0.6, 2.0]])
    Si, C = np.linalg.inv(S), 5.0
    log_norm = 0.5 * np.log(np.linalg.det(2 * np.pi * S))

    def f(x: np.ndarray) -> float:
        return float(C - 0.5 * (x - m) @ Si @ (x - m) - log_norm)

    runs = [
        tempered_smc(f, m + 0.5, 1.44 * S, n_particles=400, rng=np.random.default_rng(s))
        for s in range(6)
    ]
    log_z, se = _pooled(runs)
    assert abs(log_z - C) < 4 * se + 1e-3
    assert se < 0.02
    for r in runs:
        assert r.temperatures[0] == 0.0 and r.temperatures[-1] == 1.0
        assert np.all(np.diff(r.temperatures) > 0)
        assert np.all((r.acceptance > 0.1) & (r.acceptance < 0.7))
        assert logsumexp(r.log_weights) == pytest.approx(0.0, abs=1e-12)
    # the weighted particles are a posterior sample
    x = np.concatenate([r.particles for r in runs])
    w = np.concatenate([np.exp(r.log_weights) for r in runs]) / len(runs)
    mean = w @ x
    np.testing.assert_allclose(mean, m, atol=0.15)
    cov = (x - mean).T @ ((x - mean) * w[:, None])
    np.testing.assert_allclose(cov, S, atol=0.35)


@pytest.mark.parametrize("a", [0.3, 2.0])
def test_skewed_log_gamma_target(a: float) -> None:
    # x = log G with G ~ Gamma(a): density exp(a x - e^x) / Gamma(a), very skewed for small a
    def f(x: np.ndarray) -> float:
        return float(a * x[0] - np.exp(x[0]))

    runs = [
        tempered_smc(f, [np.log(a)], [[1.44 / a]], n_particles=300, rng=np.random.default_rng(s))
        for s in range(10)
    ]
    log_z, se = _pooled(runs)
    assert abs(log_z - gammaln(a)) < 4 * se + 1e-3
    assert se < 0.02


def test_targets_that_vanish_on_half_the_reference() -> None:
    # a half-normal: half the reference draws have zero target density at once
    def f(x: np.ndarray) -> float:
        return float(-0.5 * x[0] ** 2 - 0.5 * np.log(2 * np.pi)) if x[0] > 0 else -np.inf

    runs = [
        tempered_smc(f, [0.0], [[1.0]], n_particles=400, rng=np.random.default_rng(s))
        for s in range(6)
    ]
    log_z, se = _pooled(runs)
    assert abs(log_z - np.log(0.5)) < 4 * se + 1e-3
    for r in runs:
        assert np.all(r.particles[np.exp(r.log_weights) > 0, 0] > 0)


def test_engine_validation() -> None:
    def f(x: np.ndarray) -> float:
        return float(-0.5 * x @ x)

    for kw in ({"n_particles": 1}, {"n_moves": 0}, {"cess": 1.0}, {"cess": 0.0}):
        with pytest.raises(ValueError):
            tempered_smc(f, [0.0], [[1.0]], **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="zero at every"):
        tempered_smc(lambda x: -np.inf, [0.0], [[1.0]], n_particles=10)
    # a target far narrower than the reference needs several temperatures
    with pytest.raises(RuntimeError, match="temperatures"):
        tempered_smc(lambda x: float(-0.5 * 1e6 * x @ x), [0.0], [[1.0]], max_steps=1)


# ── structures ────────────────────────────────────────────────────────────────


def test_smc_evidence_matches_quadrature() -> None:
    summ, bd = _gaussian(300, seed=11)
    fr = fit_structure(leaf(F.BROWNIAN), summ, binned=bd, location=False, n_starts=1)
    ev = laplace_evidence(fr, bd)
    assert ev.n_params == 1
    exact = _quad_log_evidence(ev, bd)
    sm = smc_evidence(ev, bd, n_particles=200, n_runs=4, rng=np.random.default_rng(1))
    assert isinstance(sm, SMCEvidence)
    assert abs(sm.log_evidence - exact) < 4 * sm.standard_error + 2e-3
    assert sm.standard_error < 0.02
    assert sm.log_evidence == pytest.approx(logsumexp(sm.run_log_evidences) - np.log(4), abs=1e-12)
    assert sm.laplace_error == pytest.approx(ev.log_evidence - sm.log_evidence)
    assert sm.n_evaluations == sum(r.n_evaluations for r in sm.runs)
    assert np.all(sm.n_temperatures >= 1)


def test_smc_posterior_draws_match_the_quadrature_posterior() -> None:
    summ, bd = _gaussian(300, seed=11)
    fr = fit_structure(leaf(F.BROWNIAN), summ, binned=bd, location=False, n_starts=1)
    ev = laplace_evidence(fr, bd)
    f = _log_target(ev, bd)
    c, s = f(ev.mode), float(np.sqrt(ev.covariance[0, 0]))
    lo, hi = ev.mode[0] - 15 * s, ev.mode[0] + 15 * s
    z = integrate.quad(lambda t: np.exp(f(np.array([t])) - c), lo, hi, epsrel=1e-12)[0]
    mean = integrate.quad(lambda t: t * np.exp(f(np.array([t])) - c), lo, hi, epsrel=1e-12)[0] / z
    sm = smc_evidence(ev, bd, n_particles=200, n_runs=4, rng=np.random.default_rng(2))
    x, w = sm.weighted_particles()
    assert w.sum() == pytest.approx(1.0) and x.shape == (800, 1)
    assert abs(w @ x[:, 0] - mean) < 4 * s / np.sqrt(1.0 / np.sum(w**2)) + 1e-4
    d = sm.draws(500, np.random.default_rng(3))
    assert d.shape == (500, 1)
    assert abs(d.mean() - mean) < 5 * s / np.sqrt(500)


def test_smc_and_importance_sampling_agree_on_a_skewed_posterior() -> None:
    # 250 weekly VG returns: Laplace is 0.2 nats off (see test_evidence)
    true = Model(leaf(F.VARIANCE_GAMMA) + leaf(F.DRIFT), [(0.2, 0.3, -0.1), (0.01,)])
    x = np.diff(true.to_process().simulate(250, 1, T=250 / 52, rng=np.random.default_rng(3))[0])
    summ = summarize(x, dt=1 / 52)
    bd = BinnedData.from_sample(summ.sample, summ.dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = fit_structure(leaf(F.VARIANCE_GAMMA), summ, binned=bd, n_starts=3)
    ev = laplace_evidence(fr, bd)
    im = importance_evidence(ev, bd, n_draws=4000, rng=np.random.default_rng(4))
    sm = smc_evidence(ev, bd, n_particles=250, n_runs=4, rng=np.random.default_rng(5))
    gap = abs(sm.log_evidence - im.log_evidence)
    assert gap < 4 * np.hypot(sm.standard_error, im.standard_error) + 5e-3
    assert sm.laplace_error < -5 * sm.standard_error  # Laplace is low here
    assert sm.standard_error < 0.06


def test_smc_evidence_validation() -> None:
    summ, bd = _gaussian(300, seed=11)
    ev = laplace_evidence(
        fit_structure(leaf(F.BROWNIAN), summ, binned=bd, location=False, n_starts=1), bd
    )
    with pytest.raises(ValueError, match="n_runs"):
        smc_evidence(ev, bd, n_runs=1)
    for kw in ({"df": 0.0}, {"inflate": -1.0}):
        with pytest.raises(ValueError):
            smc_evidence(ev, bd, **kw)  # type: ignore[arg-type]


def test_structure_posterior_with_smc_evidence() -> None:
    summ, bd = _gaussian(300, seed=11)
    fits = [
        fit_structure(s, summ, binned=bd, n_starts=2)
        for s in (leaf(F.BROWNIAN), leaf(F.VARIANCE_GAMMA))
    ]
    P = structure_posterior(fits, bd, evidence="smc", n_draws=400, rng=np.random.default_rng(6))
    assert P.method == "smc" and all(isinstance(r, SMCEvidence) for r in P.refined)
    assert P.log_evidences.tolist() == [r.log_evidence for r in P.refined]
    assert np.all(P.standard_errors > 0) and np.all(P.probability_se > 0)
    assert all(r.runs[0].particles.shape[0] == 100 for r in P.refined)
    assert "se" in P.table().split("\n")[0].split()
    with pytest.raises(ValueError, match="n_draws"):
        structure_posterior(fits, bd, evidence="smc", n_draws=4)
