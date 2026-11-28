"""Model evidence: priors, the Laplace approximation and importance sampling."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from scipy import integrate

from spxa.discovery import families as F
from spxa.discovery.bayes import (
    MODE_GAP_TOLERANCE,
    Evidence,
    GaussianPrior,
    _coerce_prior,
    _log_target,
    default_prior,
    importance_evidence,
    laplace_evidence,
)
from spxa.discovery.empirical import summarize
from spxa.discovery.expr import Model, leaf
from spxa.discovery.fit import BinnedData, fit_structure, initial_params, with_location

warnings.simplefilter("ignore")


def _gaussian(n: int, seed: int, mu: float = 0.0) -> tuple[Any, BinnedData]:
    rng = np.random.default_rng(seed)
    r = mu / 52 + 0.2 * np.sqrt(1 / 52) * rng.standard_normal(n)
    summ = summarize(r, dt=1 / 52)
    return summ, BinnedData.from_sample(summ.sample, summ.dt)


def _quad_log_evidence(ev: Evidence, bd: BinnedData) -> float:
    """log ∫ exp(ℓ + log π) by adaptive quadrature, for one or two coordinates."""
    f = _log_target(ev, bd)
    c = f(ev.mode)
    s = np.sqrt(np.diag(ev.covariance))
    m = ev.mode
    if ev.n_params == 1:
        val, _ = integrate.quad(
            lambda t: np.exp(f(np.array([t])) - c),
            m[0] - 15 * s[0],
            m[0] + 15 * s[0],
            epsabs=0,
            epsrel=1e-12,
            limit=200,
        )
    else:
        val, _ = integrate.dblquad(
            lambda b, a: np.exp(f(np.array([a, b])) - c),
            m[0] - 10 * s[0],
            m[0] + 10 * s[0],
            lambda a: m[1] - 10 * s[1],
            lambda a: m[1] + 10 * s[1],
            epsabs=0,
            epsrel=1e-8,
        )
    return float(c + np.log(val))


# ── priors ──────────────────────────────────────────────────────────────────


def test_gaussian_prior_density_gradient_and_hessian() -> None:
    p = GaussianPrior(np.array([0.3, -1.0]), np.array([2.0, 0.5]))
    assert p.n == 2
    tot, _ = integrate.dblquad(
        lambda b, a: np.exp(p.log_density([a, b])), -20, 20, lambda a: -10, lambda a: 10
    )
    assert tot == pytest.approx(1.0, abs=1e-9)
    x = np.array([1.1, 0.2])
    h = 1e-6
    fd = [(p.log_density(x + h * e) - p.log_density(x - h * e)) / (2 * h) for e in np.eye(2)]
    np.testing.assert_allclose(p.grad(x), fd, rtol=1e-8)
    np.testing.assert_array_equal(p.hess(), -np.diag([0.25, 4.0]))
    draws = p.sample(200_000, np.random.default_rng(0))
    np.testing.assert_allclose(draws.mean(0), p.mean, atol=0.02)
    np.testing.assert_allclose(draws.std(0), p.sd, rtol=0.01)


@pytest.mark.parametrize(
    "mean, sd",
    [([0.0, 1.0], [1.0]), ([0.0], [0.0]), ([np.nan], [1.0]), ([[0.0]], [[1.0]])],
)
def test_gaussian_prior_rejects_bad_input(mean: Any, sd: Any) -> None:
    with pytest.raises(ValueError):
        GaussianPrior(np.array(mean), np.array(sd))


def test_default_prior_centre_and_widths() -> None:
    s = with_location(leaf(F.VARIANCE_GAMMA) + leaf(F.NORMAL_JUMPS))
    std = 0.18
    p = default_prior(s, std, scale=1.5)
    np.testing.assert_allclose(p.mean, Model(s, initial_params(s, std)).to_free())
    # VG (log σ, log ν, θ), NormalJumps (log λ, m, log s), Drift (μ)
    np.testing.assert_allclose(p.sd, [1.5, 1.5, 1.5 * std, 1.5, 1.5 * std, 1.5, 1.5 * std])
    with pytest.raises(ValueError):
        default_prior(s, 0.0)
    with pytest.raises(ValueError):
        default_prior(s, std, scale=-1.0)


def test_coerce_prior() -> None:
    summ, bd = _gaussian(500, 0)
    s = with_location(leaf(F.BROWNIAN))
    d = _coerce_prior(None, s, bd)
    assert d.n == 2 and d.sd[1] == pytest.approx(2 * np.std(summ.sample, ddof=1) * np.sqrt(52))
    given = GaussianPrior(np.zeros(2), np.ones(2))
    assert _coerce_prior(given, s, bd) is given
    assert _coerce_prior(lambda st: given, s, bd) is given
    with pytest.raises(TypeError):
        _coerce_prior(lambda st: 3, s, bd)
    with pytest.raises(TypeError):
        _coerce_prior(3.0, s, bd)


# ── Laplace against quadrature ──────────────────────────────────────────────


def test_laplace_matches_quadrature_with_order_one_over_n_error() -> None:
    errors = {}
    occam = {}
    for n in (300, 2000):
        summ, bd = _gaussian(n, 5 + n)
        fr = fit_structure(leaf(F.BROWNIAN), summ, binned=bd, location=False, n_starts=1)
        ev = laplace_evidence(fr, bd)
        assert ev.converged and ev.n_params == 1 and ev.n_obs == n
        errors[n] = abs(ev.log_evidence - _quad_log_evidence(ev, bd))
        occam[n] = ev.log_occam_factor
    assert errors[300] < 5e-3
    assert errors[2000] < 1e-3
    # Laplace's error is O(1/n): a 6.7x larger sample shrinks it at least 3x
    assert errors[2000] < errors[300] / 3
    # one parameter: the Occam factor falls by about (1/2) log(2000/300)
    assert occam[2000] - occam[300] == pytest.approx(-0.5 * np.log(2000 / 300), abs=0.3)


def test_laplace_matches_two_dimensional_quadrature() -> None:
    summ, bd = _gaussian(1500, 9, mu=0.05)
    fr = fit_structure(leaf(F.BROWNIAN), summ, binned=bd, n_starts=1)
    ev = laplace_evidence(fr, bd)
    assert ev.n_params == 2
    assert abs(ev.log_evidence - _quad_log_evidence(ev, bd)) < 3e-3


def test_importance_sampling_matches_quadrature() -> None:
    for n, seed in ((300, 1), (1500, 2)):
        summ, bd = _gaussian(n, 40 + seed, mu=0.05)
        fr = fit_structure(leaf(F.BROWNIAN), summ, binned=bd, n_starts=1, location=seed == 2)
        ev = laplace_evidence(fr, bd)
        im = importance_evidence(ev, bd, n_draws=3000, rng=np.random.default_rng(seed))
        ref = _quad_log_evidence(ev, bd)
        assert abs(im.log_evidence - ref) < 4 * im.standard_error + 1e-4
        assert im.ess > 0.7 * im.n_draws
        assert im.laplace is ev


def test_importance_sampling_exposes_laplace_error_on_small_samples() -> None:
    true = Model(leaf(F.VARIANCE_GAMMA) + leaf(F.DRIFT), [(0.2, 0.3, -0.1), (0.01,)])
    gaps = {}
    for n in (250, 2000):
        x = np.diff(true.to_process().simulate(n, 1, T=n / 52, rng=np.random.default_rng(3))[0])
        summ = summarize(x, dt=1 / 52)
        bd = BinnedData.from_sample(summ.sample, summ.dt)
        fr = fit_structure(leaf(F.VARIANCE_GAMMA), summ, binned=bd, n_starts=3)
        ev = laplace_evidence(fr, bd)
        im = importance_evidence(ev, bd, n_draws=4000, rng=np.random.default_rng(4))
        assert im.ess > 1000
        gaps[n] = (abs(im.laplace_error), im.standard_error)
    # n = 250: the skewed posterior puts Laplace measurably off (observed 0.21 nats)
    assert gaps[250][0] > 5 * gaps[250][1]
    assert gaps[250][0] < 0.5
    # n = 2000: Laplace is within Monte Carlo error of the unbiased estimate
    assert gaps[2000][0] < 4 * gaps[2000][1] + 0.01


# ── the Evidence object ─────────────────────────────────────────────────────


def test_evidence_properties_and_draws() -> None:
    summ, bd = _gaussian(1000, 21, mu=0.05)
    fr = fit_structure(leaf(F.BROWNIAN), summ, binned=bd, n_starts=1)
    ev = laplace_evidence(fr, bd)
    np.testing.assert_allclose(ev.covariance @ ev.hessian, np.eye(2), atol=1e-10)
    assert ev.log_det_hessian == pytest.approx(np.linalg.slogdet(ev.hessian)[1], rel=1e-12)
    assert ev.bic_equivalent == pytest.approx(-2 * ev.log_evidence)
    assert ev.log_occam_factor == pytest.approx(ev.log_evidence - ev.log_likelihood)
    # ℓ at the mode, recomputed from the bins
    p = bd.probabilities(ev.model)
    assert ev.log_likelihood == pytest.approx(float(bd.counts @ np.log(p)), rel=1e-12)
    assert ev.log_prior == pytest.approx(ev.prior.log_density(ev.mode), rel=1e-12)
    # the mode sits next to the MLE: the prior is weak against 1000 points
    assert np.max(np.abs(ev.mode - fr.model.to_free()) / np.sqrt(np.diag(ev.covariance))) < 0.05
    d = ev.draws(40_000, np.random.default_rng(0))
    se = np.sqrt(np.diag(ev.covariance))
    assert np.max(np.abs(d.mean(0) - ev.mode) / se) < 0.03
    # on the correlation scale: the off-diagonal covariance is near zero here
    assert np.max(np.abs(np.cov(d.T) - ev.covariance) / np.outer(se, se)) < 0.03
    assert "log evidence" in str(ev)


def test_laplace_on_a_bare_structure_and_errors() -> None:
    summ, bd = _gaussian(800, 31)
    fr = fit_structure(leaf(F.BROWNIAN), summ, binned=bd, n_starts=1)
    s = fr.model.structure
    a = laplace_evidence(fr, bd)
    b = laplace_evidence(s, bd, x0=fr.model.to_free() + 0.1)
    assert b.log_evidence == pytest.approx(a.log_evidence, abs=1e-9)
    with pytest.raises(ValueError, match="x0"):
        laplace_evidence(s, bd)
    with pytest.raises(ValueError, match="coordinates"):
        laplace_evidence(fr, bd, prior=GaussianPrior(np.zeros(3), np.ones(3)))
    with pytest.raises(ValueError, match="not finite"):
        laplace_evidence(s, bd, x0=[np.nan, 0.0])
    with pytest.raises(ValueError):
        importance_evidence(a, bd, n_draws=1)
    with pytest.raises(ValueError):
        importance_evidence(a, bd, df=0.0)
    bare = BinnedData(**{**bd.__dict__, "sample": None})
    with pytest.raises(ValueError, match="without the sample"):
        laplace_evidence(fr, bare)


def test_a_tighter_prior_moves_the_mode_and_the_evidence() -> None:
    summ, bd = _gaussian(400, 8)
    fr = fit_structure(leaf(F.BROWNIAN), summ, binned=bd, location=False, n_starts=1)
    x_mle = fr.model.to_free()
    wide = laplace_evidence(fr, bd, prior=GaussianPrior(x_mle, np.array([2.0])))
    off = laplace_evidence(fr, bd, prior=GaussianPrior(x_mle - 0.2, np.array([0.02])))
    # conjugate-normal intuition: the mode is pulled most of the way to the prior
    assert off.mode[0] < x_mle[0] - 0.1
    assert off.log_evidence < wide.log_evidence
    # both stay exact up to Laplace's error
    for ev in (wide, off):
        assert abs(ev.log_evidence - _quad_log_evidence(ev, bd)) < 5e-3


def test_convergence_is_judged_by_the_newton_decrement() -> None:
    # BM fitted to jump-diffusion data: the binned likelihood's kinks (clip and
    # running maximum, about 1e-7 relative) stall the gradient above 1e-6 |v|,
    # yet the mode is accurate to well under a micro-nat
    rng = np.random.default_rng(2)
    truth = Model(
        leaf(F.BROWNIAN) + leaf(F.NORMAL_JUMPS) + leaf(F.DRIFT),
        [(0.15,), (20.0, -0.02, 0.04), (0.05,)],
    )
    x = np.diff(truth.to_process().simulate(3000, 1, T=3000 / 252, rng=rng)[0])
    summ = summarize(x, dt=1 / 252)
    bd = BinnedData.from_sample(summ.sample, summ.dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_structure(leaf(F.BROWNIAN), summ, binned=bd, rng=np.random.default_rng(1))
    # gtol = 0 can never be met by a nonzero gradient, so only the gap
    # criterion can declare convergence (where the search stalls varies with
    # the platform's floating point: |g| was 1.2e-2 on one machine, 6e-3 on another)
    ev = laplace_evidence(fit, bd, gtol=0.0)
    from spxa.graph.autodiff.discovery import likelihood_value_grad_hess

    v, g, _ = likelihood_value_grad_hess(ev.structure, bd, ev.mode)
    g = g - ev.prior.grad(ev.mode)
    assert np.max(np.abs(g)) > 0
    assert ev.mode_gap == pytest.approx(0.5 * g @ np.linalg.solve(ev.hessian, g), rel=1e-10)
    assert 0 <= ev.mode_gap < MODE_GAP_TOLERANCE and ev.converged
    # the default search reaches the same mode
    default = laplace_evidence(fit, bd)
    assert default.converged
    assert default.log_evidence == pytest.approx(ev.log_evidence, abs=1e-6)
    # and the gap is what it claims: the log posterior one Newton step on
    # rises by about that much, not more
    nxt = ev.mode - np.linalg.solve(ev.hessian, g)
    v2, _, _ = likelihood_value_grad_hess(ev.structure, bd, nxt)
    rise = (v - ev.prior.log_density(ev.mode)) - (v2 - ev.prior.log_density(nxt))
    assert rise < 1e-5
