"""DiscoveryReport rendering, weights and matching, on hand-built fits."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.discovery.empirical import from_statistics
from spxa.discovery.expr import Leaf, Model
from spxa.discovery.families import BROWNIAN, DRIFT, GAMMA, VARIANCE_GAMMA
from spxa.discovery.fit import FitResult
from spxa.discovery.report import DiscoveryReport

SUMMARY = from_statistics(
    mean=0.0, variance=0.01, skewness=-0.3, excess_kurtosis=2.0, dt=0.1, n=1000
)


def _fit(model: Model, loglik: float | None, ks: float | None = None) -> FitResult:
    return FitResult(
        model=model,
        method="likelihood" if loglik is not None else "cumulants",
        objective=0.0,
        n_obs=1000,
        loglik=loglik,
        g_statistic=10.0 if loglik is not None else None,
        g_pvalue=0.4 if loglik is not None else None,
        ks_distance=ks,
        target_cumulants=SUMMARY.cumulants.copy(),
        dt=SUMMARY.dt,
    )


VG_GAMMA = Model(
    Leaf(VARIANCE_GAMMA) + Leaf(GAMMA) + Leaf(DRIFT), [(0.2, 0.3, -0.1), (2.0, 25.0), (0.0,)]
)
VG = Model(Leaf(VARIANCE_GAMMA) + Leaf(DRIFT), [(0.2, 0.3, -0.1), (0.0,)])
BM = Model(Leaf(BROWNIAN) + Leaf(DRIFT), [(0.3,), (0.0,)])


def test_weights_follow_the_bic_differences() -> None:
    fits = [_fit(VG_GAMMA, -100.0, 0.004), _fit(VG, -101.0, 0.006), _fit(BM, -150.0, 0.05)]
    fits.sort(key=lambda f: f.bic)
    rep = DiscoveryReport(fits, SUMMARY, "bic", "likelihood", 0.02)
    bic = np.array([f.bic for f in fits])
    expected = np.exp(-0.5 * (bic - bic.min()))
    np.testing.assert_allclose(rep.weights(), expected / expected.sum())
    assert rep.score(fits[1]) == fits[1].bic


def test_headline_uses_the_cdf_gap_and_weighted_subordinators() -> None:
    # Gamma(a, b) is (1/b) * Gamma(a, 1): rendered as 0.04 * GammaProcess(a=2, b=1)
    fits = [_fit(VG_GAMMA, -100.0, 0.004), _fit(BM, -150.0, 0.05)]
    rep = DiscoveryReport(fits, SUMMARY, "bic", "likelihood", 0.02)
    head = rep.headline()
    assert head.startswith(
        "VarianceGamma(sigma=0.2, nu=0.3, theta=-0.1) + 0.04 * GammaProcess(a=2, b=1)"
    )
    assert "within 0.4% (largest gap between model and empirical CDFs" in head
    gap = fits[1].bic - fits[0].bic
    assert head.endswith(f"BIC lead over the runner-up {gap:.1f}")
    table = rep.table().splitlines()
    assert "CDF gap" in table[0] and "moment err" in table[0]
    assert table[1].split()[0] == "1" and float(table[1].split()[1]) == 0.0


def test_cumulant_reports_quote_the_moment_error() -> None:
    fits = [_fit(VG, None), _fit(BM, None)]
    fits.sort(key=lambda f: f.match_error)
    rep = DiscoveryReport(fits, SUMMARY, "bic", "cumulants", tolerance=10.0)
    assert rep.score(fits[0]) == fits[0].match_error
    assert "largest error in variance, skewness and excess kurtosis" in rep.headline()
    assert "moment err" in rep.table().splitlines()[0]
    assert rep.matching() == [f for f in fits if f.match_error <= 10.0]
    with pytest.raises(ValueError):
        rep.weights()


def test_warnings_are_printed() -> None:
    rep = DiscoveryReport([_fit(BM, -1.0, 0.1)], SUMMARY, "aic", "likelihood", 0.02, ["careful"])
    assert str(rep).endswith("warning: careful")
    assert "runner-up" not in rep.headline()


def test_option_reports() -> None:
    def opt(model: Model, sse: float) -> FitResult:
        n = 27
        return FitResult(
            model=model,
            method="options",
            objective=sse,
            n_obs=n,
            loglik=-0.5 * n * np.log(sse / n),
            extras={"iv_rmse": np.sqrt(sse / n), "iv_max_error": 0.004, "price_rel_rmse": 0.012},
        )

    fits = [opt(VG, 1e-5), opt(BM, 1e-3)]
    rep = DiscoveryReport(fits, None, "bic", "options", 0.0)
    head = rep.headline()
    assert head.startswith("VarianceGamma(sigma=0.2, nu=0.3, theta=-0.1) + Drift(mu=0)")
    assert f"within {100 * np.sqrt(1e-5 / 27):.3g} vol points RMS" in head
    assert "RMS relative price error 1.2%" in head
    lines = rep.table().splitlines()
    assert "IV RMSE" in lines[0] and len(lines) == 3
    assert rep.weights()[0] > 0.99


def test_degenerate_fits_are_marked_and_screened() -> None:
    from spxa.discovery.families import NIG_FAMILY
    from spxa.discovery.report import screen_degenerate

    edge = Model(Leaf(NIG_FAMILY) + Leaf(DRIFT), [(8.97e5, 8.97e5 - 1.0, 0.01), (0.0,)])
    fits = [_fit(edge, 110.0, 0.01), _fit(VG, 100.0, 0.02), _fit(BM, 50.0, 0.05)]
    assert [f.degenerate for f in fits] == [True, False, False]

    rep = DiscoveryReport(fits, SUMMARY, "bic", "likelihood", 0.02)
    lines = rep.table().splitlines()
    assert "! NIG(" in lines[1] and "! " not in lines[2]
    assert lines[-1].strip().startswith("! degenerate fit")
    assert [f.structure_key for f in rep.admissible()] == [
        fits[1].structure_key,
        fits[2].structure_key,
    ]
    assert (
        "! degenerate" not in DiscoveryReport(fits[1:], SUMMARY, "bic", "likelihood", 0.02).table()
    )

    def score(r: FitResult) -> float:
        assert r.bic is not None
        return float(r.bic)

    kept, out, notes = screen_degenerate(fits, False, score, "bic")
    assert kept == fits and out == [] and len(notes) == 1
    assert "is a limiting case (NIG.alpha at a boundary" in notes[0]
    assert f"BIC {score(fits[1]) - score(fits[0]):+.1f}" in notes[0]
    kept, out, notes = screen_degenerate(fits, True, score)
    assert kept == fits[1:] and out == fits[:1] and "1 degenerate fit excluded" in notes[0]
    kept, out, notes = screen_degenerate(fits[:1], True)
    assert kept == fits[:1] and out == [] and "every fitted structure" in notes[0]
    assert screen_degenerate(fits[1:], True) == (fits[1:], [], [])
