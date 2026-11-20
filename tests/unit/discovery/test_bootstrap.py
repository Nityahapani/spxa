"""Bootstrap selection frequencies and parameter intervals."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.discovery import discover, from_statistics, grammar_from_names
from spxa.discovery.bootstrap import _resample, bootstrap

from .conftest import vg_increments


def test_resampling_schemes() -> None:
    x = np.arange(100.0)
    rng = np.random.default_rng(0)
    iid = _resample(x, rng, 1)
    assert iid.size == 100 and set(iid) <= set(x)
    blocks = _resample(x, rng, 10)
    assert blocks.size == 100
    # moving blocks: consecutive runs of length 10
    runs = blocks.reshape(10, 10)
    assert np.all(np.diff(runs, axis=1) == 1)


def test_bootstrap_of_a_clear_winner() -> None:
    x = vg_increments(0.2, 0.25, -0.3, 1 / 52, 2_000, seed=7, mu=0.1)
    g = grammar_from_names(["BM", "VG", "NIG"], max_terms=1)
    r = discover(x, dt=1 / 52, grammar=g, n_starts=2)
    b = bootstrap(r, n_boot=12, top=3, seed=1)
    freq = b.selection_frequency
    assert sum(freq.values()) == pytest.approx(1.0)
    assert b.structures[0] == "VG + Drift" and freq["VG + Drift"] >= 0.9
    assert b.parameters[0].shape == (12, 4)
    ci = b.intervals(level=0.9)
    est = np.array([v for p in r.best.model.params for v in p])
    assert np.all(ci[:, 0] < est) and np.all(est < ci[:, 1])
    assert ci[0, 0] < 0.2 < ci[0, 1]  # σ
    assert "selection frequency over 12 resamples (BIC)" in str(b)
    assert b.intervals("BM + Drift").shape == (2, 2)
    same = bootstrap(r, n_boot=12, top=3, seed=1)
    np.testing.assert_array_equal(same.wins, b.wins)


def test_bootstrap_needs_a_sample() -> None:
    s = from_statistics(mean=0.0, variance=1e-4, dt=1 / 252)
    r = discover(s, grammar=grammar_from_names(["BM"], max_terms=1))
    with pytest.raises(ValueError, match="return sample"):
        bootstrap(r)
