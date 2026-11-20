"""law_tests and assert_holds: laws as pytest cases."""

from __future__ import annotations

import pytest

from spxa.laws import LawFailure, expect, floats, law
from spxa.laws.library import AXIOMS
from spxa.laws.testing import assert_holds, law_tests, resolve_settings


@law(x=floats(0.0, 1.0))
def below_half(x):
    expect(x < 0.5, "too big")


# the library axioms, collected as five pytest cases
test_axioms = law_tests(AXIOMS, n_examples=10, seed=4)


def test_cases_are_named_after_the_laws() -> None:
    marks = test_axioms.pytestmark  # type: ignore[attr-defined]
    ids = marks[0].kwargs["ids"]
    assert ids == [x.name for x in AXIOMS]


def test_failure_is_an_assertion_with_the_report() -> None:
    with pytest.raises(AssertionError, match="below_half: FALSIFIED") as info:
        law_tests(below_half, n_examples=50)(below_half)
    assert isinstance(info.value, LawFailure)
    assert info.value.result.counterexample["x"] >= 0.5


def test_assert_holds_returns_the_result() -> None:
    result = assert_holds(AXIOMS[0], n_examples=5)
    assert result.ok and result.n_examples == 5


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_settings(100, 0) == (100, 0)
    monkeypatch.setenv("SPXA_LAW_EXAMPLES", "7")
    monkeypatch.setenv("SPXA_LAW_SEED", "11")
    assert resolve_settings(100, 0) == (7, 11)
    result = assert_holds(AXIOMS[0], n_examples=100)
    assert (result.n_examples, result.seed) == (7, 11)
    monkeypatch.setenv("SPXA_LAW_SEED", "x")
    with pytest.raises(ValueError, match="integers"):
        resolve_settings(1, 1)


def test_needs_a_law() -> None:
    with pytest.raises(ValueError):
        law_tests([])
