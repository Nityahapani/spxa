"""Structure search: beam and exhaustive discovery, ranking and the report."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.discovery.empirical import from_statistics, summarize
from spxa.discovery.expr import Leaf, Model
from spxa.discovery.families import BROWNIAN, GAMMA, VARIANCE_GAMMA
from spxa.discovery.fit import fit_structure, with_location
from spxa.discovery.grammar import grammar_from_names
from spxa.discovery.search import (
    DiscoveryReport,
    _strip_location,
    _term_params,
    _warm_start,
    discover,
)

from .conftest import vg_increments

DT = 1 / 52


@pytest.fixture(scope="module")
def vg_sample() -> np.ndarray:
    return vg_increments(0.2, 0.25, -0.3, DT, 2_000, seed=7, mu=0.1)


@pytest.fixture(scope="module")
def small_report(vg_sample: np.ndarray) -> DiscoveryReport:
    g = grammar_from_names(["BM", "VG", "NIG", "Gamma"], max_terms=2)
    return discover(vg_sample, dt=DT, grammar=g, n_starts=2)


def test_location_helpers() -> None:
    vg = Leaf(VARIANCE_GAMMA)
    assert _strip_location(with_location(vg)) == vg
    two = vg + Leaf(GAMMA)
    assert _strip_location(with_location(two)).key() == two.key()
    m = Model(with_location(two), [(0.2, 0.3, -0.1), (2.0, 4.0), (0.05,)])
    terms, drift = _term_params(m)
    assert terms == {"VG": [(0.2, 0.3, -0.1)], "Gamma": [(2.0, 4.0)]}
    assert drift == 0.05


def test_warm_start_keeps_the_parent_and_adds_a_typical_atom(vg_sample: np.ndarray) -> None:
    s = summarize(vg_sample, dt=DT)
    parent = fit_structure(Leaf(VARIANCE_GAMMA), s, n_starts=1)
    child = with_location(Leaf(VARIANCE_GAMMA) + Leaf(BROWNIAN))
    init = _warm_start(parent, child, 0.05)
    vg_params = parent.model.params[0]
    assert init[0] == vg_params
    assert init[1] == BROWNIAN.typical(0.05)
    assert init[2] == parent.model.params[-1]
    # the warm start is a valid parameter list for the child
    Model(child, init)


def test_likelihood_report(small_report: DiscoveryReport) -> None:
    r = small_report
    assert r.method == "likelihood" and r.criterion == "bic"
    bics = [c.bic for c in r.candidates]
    assert bics == sorted(bics)
    # pure-jump data: Brownian motion alone is decisively rejected
    bm = next(c for c in r.candidates if c.structure_key == "BM + Drift")
    assert bm.bic - r.best.bic > 100
    assert bm.g_pvalue < 1e-6
    # VG generated the data; it or a two-term superset of it wins
    assert "VG" in r.best.structure_key or "NIG" in r.best.structure_key
    w = r.weights()
    assert w.sum() == pytest.approx(1.0) and np.all(np.diff(w) <= 1e-12)
    assert r.best.g_pvalue > 0.01
    assert "matches your data within" in r.headline()
    assert "largest gap between model and empirical CDFs" in r.headline()
    lines = r.table(top=5).splitlines()
    assert len(lines) == 6 and "ΔBIC" in lines[0]
    assert str(r).startswith(r.headline())


def test_every_structure_is_fitted_once(small_report: DiscoveryReport) -> None:
    keys = [c.structure_key for c in small_report.candidates]
    assert len(keys) == len(set(keys))
    # four single atoms (BM, VG, NIG, ±Gamma = 5) all appear
    for k in ("BM + Drift", "VG + Drift", "NIG + Drift", "Gamma + Drift", "-Gamma + Drift"):
        assert k in keys


def test_beam_agrees_with_exhaustive_search(vg_sample: np.ndarray) -> None:
    g = grammar_from_names(["BM", "VG", "Gamma"], max_terms=2)
    beam = discover(vg_sample, dt=DT, grammar=g, n_starts=2, beam_width=5)
    full = discover(vg_sample, dt=DT, grammar=g, n_starts=2, search="exhaustive")
    assert len(full.candidates) == len(g)
    # with a beam as wide as the atom list both searches visit every structure
    assert {c.structure_key for c in beam.candidates} == {c.structure_key for c in full.candidates}
    assert beam.best.bic == pytest.approx(full.best.bic, abs=2.0)


def test_search_is_deterministic(vg_sample: np.ndarray) -> None:
    g = grammar_from_names(["BM", "VG"], max_terms=2)
    a = discover(vg_sample, dt=DT, grammar=g, n_starts=2, seed=3)
    b = discover(vg_sample, dt=DT, grammar=g, n_starts=2, seed=3)
    assert [c.structure_key for c in a.candidates] == [c.structure_key for c in b.candidates]
    assert [c.bic for c in a.candidates] == [c.bic for c in b.candidates]


def test_aic_ranking(vg_sample: np.ndarray) -> None:
    g = grammar_from_names(["BM", "VG"], max_terms=1)
    r = discover(vg_sample, dt=DT, grammar=g, criterion="aic", n_starts=1)
    assert [c.aic for c in r.candidates] == sorted(c.aic for c in r.candidates)
    assert "ΔAIC" in r.table()


def test_summary_statistics_rank_matching_structures_by_simplicity() -> None:
    # Gaussian moments: BM matches exactly, and so do richer families in a limit
    s = from_statistics(mean=0.001, variance=1e-4, skewness=0.0, excess_kurtosis=0.0, dt=1 / 252)
    g = grammar_from_names(["BM", "VG", "NIG"], max_terms=1)
    r = discover(s, grammar=g)
    assert r.method == "cumulants"
    assert r.best.structure_key == "BM + Drift"
    assert r.best.match_error < 1e-8
    assert r.best.model.params[0][0] == pytest.approx(np.sqrt(1e-4 * 252), rel=1e-6)
    assert "largest error in variance, skewness and excess kurtosis" in r.headline()
    with pytest.raises(ValueError, match="likelihood"):
        r.weights()


def test_summary_statistics_warn_about_non_identifiability() -> None:
    # skewed, fat-tailed daily moments: several 3-parameter families match all four
    s = from_statistics(
        mean=0.0003, variance=1.2e-4, skewness=-0.5, excess_kurtosis=4.0, dt=1 / 252
    )
    g = grammar_from_names(["BM", "VG", "NIG", "Meixner"], max_terms=1)
    r = discover(s, grammar=g, tolerance=0.02)
    matched = r.matching()
    assert {m.structure_key for m in matched} >= {"VG + Drift", "NIG + Drift"}
    assert all(m.match_error <= 0.02 for m in matched)
    assert r.best in matched
    assert any("equally valid explanations" in w for w in r.warnings)
    assert "BM + Drift" not in {m.structure_key for m in matched}


def test_dependence_warning() -> None:
    rng = np.random.default_rng(6)
    n = 2_000
    sig, r = np.empty(n), np.empty(n)
    sig[0], r[0] = 0.01, 0.0
    for t in range(1, n):  # GARCH(1,1)
        sig[t] = np.sqrt(1e-6 + 0.1 * r[t - 1] ** 2 + 0.88 * sig[t - 1] ** 2)
        r[t] = sig[t] * rng.normal()
    rep = discover(r, dt=1 / 252, grammar=grammar_from_names(["BM"], max_terms=1), n_starts=1)
    assert any("marginal law only" in w for w in rep.warnings)


def test_invalid_arguments(vg_sample: np.ndarray) -> None:
    g = grammar_from_names(["BM"], max_terms=1)
    with pytest.raises(ValueError, match="criterion"):
        discover(vg_sample, dt=DT, grammar=g, criterion="hqc")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="search"):
        discover(vg_sample, dt=DT, grammar=g, search="genetic")  # type: ignore[arg-type]
    # summaries without a sample cannot use the likelihood
    s = from_statistics(mean=0.0, variance=1.0)
    with pytest.raises(ValueError, match="sample"):
        discover(s, grammar=g, method="likelihood")
