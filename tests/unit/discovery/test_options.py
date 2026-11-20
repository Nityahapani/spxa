"""Risk-neutral structure discovery from option quotes."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.discovery import BROWNIAN, GAMMA, VARIANCE_GAMMA, Model, grammar_from_names, leaf
from spxa.discovery.options import (
    OptionQuotes,
    discover_from_options,
    fit_options,
    model_prices,
)
from spxa.market.black import black_scholes_price

K = np.tile(np.linspace(80.0, 120.0, 9), 3)
T = np.repeat([0.25, 0.5, 1.0], 9)
CALL = K >= 100


def _quotes(model: Model) -> OptionQuotes:
    shell = OptionQuotes.from_arrays(100.0, K, T, np.ones_like(K), CALL, rate=0.02, dividend=0.01)
    p = model_prices(model, shell, n_terms=4096)
    return OptionQuotes.from_arrays(100.0, K, T, p, CALL, rate=0.02, dividend=0.01)


def test_quotes() -> None:
    p = black_scholes_price(100.0, K, T, 0.02, 0.2, 0.01, np.where(CALL, "call", "put").tolist())
    q = OptionQuotes.from_arrays(100.0, K, T, np.asarray(p, dtype=float), CALL, 0.02, 0.01)
    assert len(q) == 27
    np.testing.assert_allclose(q.implied_vols(), 0.2, atol=1e-8)
    np.testing.assert_allclose(q.forwards, 100 * np.exp(0.01 * T))
    groups = q.maturity_groups()
    assert [g[0] for g in groups] == [0.25, 0.5, 1.0] and all(len(g[1]) == 9 for g in groups)
    with pytest.raises(ValueError):
        OptionQuotes.from_arrays(100.0, [-1.0], 1.0, [1.0])
    with pytest.raises(ValueError):
        OptionQuotes.from_arrays(100.0, [90.0, 100.0], 1.0, [1.0])


def test_black_scholes_quotes_give_brownian_motion() -> None:
    q = _quotes(Model(leaf(BROWNIAN), [(0.2,)]))
    r = fit_options(leaf(BROWNIAN), q)
    assert r.model.params[0][0] == pytest.approx(0.2, abs=1e-6)
    assert r.extras["iv_rmse"] < 1e-6 and r.method == "options" and r.n_obs == 27


def test_variance_gamma_parameters_from_a_surface() -> None:
    q = _quotes(Model(leaf(VARIANCE_GAMMA), [(0.15, 0.3, -0.2)]))
    r = fit_options(leaf(VARIANCE_GAMMA), q)
    np.testing.assert_allclose(r.model.params[0], (0.15, 0.3, -0.2), rtol=1e-3, atol=1e-4)
    assert r.extras["iv_rmse"] < 1e-5
    for t in (0.25, 0.5, 1.0):
        assert r.extras[f"iv_rmse@{t:g}"] < 1e-5


def test_discovery_ranks_the_generating_family_first() -> None:
    q = _quotes(Model(leaf(VARIANCE_GAMMA), [(0.15, 0.3, -0.2)]))
    rep = discover_from_options(q, grammar=grammar_from_names(["BM", "VG", "NIG"], max_terms=1))
    assert rep.method == "options" and rep.summary is None
    assert rep.best.structure_key == "VG"
    assert rep.candidates[-1].structure_key == "BM"  # no smile at all
    assert "matches your option prices within" in rep.headline()
    assert not rep.warnings


def test_two_term_search_and_maturity_warning() -> None:
    q = _quotes(Model(leaf(VARIANCE_GAMMA) + -leaf(GAMMA), [(0.15, 0.3, -0.1), (3.0, 30.0)]))
    rep = discover_from_options(q, grammar=grammar_from_names(["BM", "VG", "Gamma"], max_terms=2))
    assert rep.best.structure_key == "VG + -Gamma"
    assert rep.best.extras["iv_rmse"] < 5e-4
    # quotes no single Lévy law can produce: a steep short-dated smile, a flat
    # long-dated one
    a = _quotes(Model(leaf(VARIANCE_GAMMA), [(0.12, 0.8, -0.3)]))
    b = _quotes(Model(leaf(BROWNIAN), [(0.2,)]))
    mixed = OptionQuotes.from_arrays(
        100.0, K, T, np.where(T < 0.3, a.prices, b.prices), CALL, rate=0.02, dividend=0.01
    )
    bad = discover_from_options(mixed, grammar=grammar_from_names(["BM", "VG"], max_terms=1))
    assert any("depend on maturity" in w for w in bad.warnings)
    with pytest.raises(ValueError, match="criterion"):
        discover_from_options(q, criterion="hqc")  # type: ignore[arg-type]
