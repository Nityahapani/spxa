"""Suites: selection, reproducible checks and the three report formats."""

from __future__ import annotations

import json

import pytest

from spxa.laws import expect, floats, law
from spxa.laws.suite import Suite, SuiteFailure, default_suite


@law(x=floats(0.0, 1.0), tags=("toy", "good"))
def square_is_small(x):
    """x^2 <= x on [0, 1]"""
    expect(x * x <= x, "x^2 > x")


@law(x=floats(0.0, 1.0), tags=("toy", "bad"))
def below_half(x):
    """x < 1/2 | pipes are escaped"""
    expect(x < 0.5, "too big")


@law(x=floats(0.0, 1.0), tags=("other",))
def always(x):
    """trivial"""
    return True


def _suite() -> Suite:
    return Suite("toys", [square_is_small, (below_half,)], always)


def test_check_counts_and_order() -> None:
    report = _suite().check(n_examples=30, seed=2)
    assert [r.law for r in report.results] == ["square_is_small", "below_half", "always"]
    assert report.counts == {"verified": 2, "falsified": 1, "error": 0, "unsatisfiable": 0}
    assert not report.ok
    assert [r.law for r in report.failures] == ["below_half"]
    assert report["always"].ok
    with pytest.raises(KeyError):
        report["missing"]


def test_reports_are_reproducible() -> None:
    a = _suite().check(n_examples=30, seed=5)["below_half"]
    b = below_half.check(n_examples=30, seed=5)
    assert a.counterexample == b.counterexample


def test_text_markdown_and_json() -> None:
    report = _suite().check(n_examples=20, seed=1)
    text = str(report)
    assert text.startswith("toys: 2 verified, 1 falsified (3 laws, 20 examples each, seed 1")
    assert "below_half: FALSIFIED" in text
    md = report.to_markdown()
    assert "| `square_is_small` | verified | 20 | 0 | x^2 <= x on [0, 1] |" in md
    assert "x < 1/2 \\| pipes are escaped" in md
    assert "<details><summary><code>below_half</code>" in md
    data = json.loads(json.dumps(report.to_dict()))
    assert data["ok"] is False and data["counts"]["falsified"] == 1
    assert data["results"][1]["counterexample"]["x"]


def test_raise_if_failed() -> None:
    with pytest.raises(SuiteFailure, match="below_half: FALSIFIED") as info:
        _suite().check(n_examples=20).raise_if_failed()
    assert info.value.report.counts["falsified"] == 1
    _suite().select(exclude_tags=["bad"]).check(n_examples=20).raise_if_failed()


def test_select() -> None:
    s = _suite()
    assert [x.name for x in s.select(tags=["toy"])] == ["square_is_small", "below_half"]
    assert [x.name for x in s.select(names=["always"])] == ["always"]
    assert [x.name for x in s.select(tags=["toy"], exclude_tags=["bad"])] == ["square_is_small"]
    assert s.tags == ["bad", "good", "other", "toy"]
    with pytest.raises(KeyError, match="nope"):
        s.select(names=["nope"])


def test_fail_fast_and_progress() -> None:
    seen: list[str] = []
    report = _suite().check(n_examples=20, fail_fast=True, progress=lambda r: seen.append(r.law))
    assert seen == ["square_is_small", "below_half"]
    assert len(report.results) == 2


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Suite("dup", always, [always])


def test_default_suite_holds_every_library_law() -> None:
    from spxa.laws import library

    suite = default_suite()
    expected = sum(len(getattr(library, name)) for name in library.__all__)
    assert len(suite) == expected == 46
    for tag in ("axiom", "algebra", "subordination", "triplet", "cumulants", "properties"):
        assert suite.select(tags=[tag])
    assert len(suite.select(tags=["simulation"])) == 7
