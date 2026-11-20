"""Law objects and the check runner."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.laws.expect import assume, expect, expect_close
from spxa.laws.law import Law, LawFailure, LawResult, evaluate, law
from spxa.laws.strategies import floats, integers


@law(x=floats(-10, 10), y=floats(-10, 10))
def addition_commutes(x: float, y: float) -> None:
    """x + y = y + x"""
    expect_close(x + y, y + x)


@law(x=floats(0, 10))
def square_root_inverts_square(x: float) -> None:
    expect_close(np.sqrt(x * x), x, rtol=1e-12)


@law(x=floats(0.0, 10.0))
def false_for_large(x: float) -> bool:
    return x < 7.5


def test_decorator_builds_a_law() -> None:
    assert isinstance(addition_commutes, Law)
    assert addition_commutes.name == "addition_commutes"
    assert addition_commutes.description == "x + y = y + x"
    assert addition_commutes(x=1.0, y=2.0) is None
    assert (
        repr(addition_commutes) == "Law('addition_commutes', x=floats(-10, 10), y=floats(-10, 10))"
    )
    with pytest.raises(ValueError):
        Law("empty", lambda: None, {})


def test_verified() -> None:
    r = addition_commutes.check(n_examples=50, seed=3)
    assert r.status == "verified" and r.ok and not r.failed
    assert r.n_examples == 50 and r.n_discarded == 0 and r.counterexample is None
    assert str(r) == "addition_commutes: VERIFIED (no counterexample in 50 examples)"
    r.raise_if_failed()
    assert square_root_inverts_square.check(200).ok


def test_falsified_by_a_false_return() -> None:
    r = false_for_large.check(n_examples=100, seed=0)
    assert r.status == "falsified" and r.failed
    assert r.counterexample is not None and r.counterexample["x"] >= 7.5
    # shrunk to the boundary of the failure region
    assert r.counterexample == {"x": 7.5} and r.original["x"] > 7.5 and r.shrink_steps >= 1
    unshrunk = false_for_large.check(n_examples=100, seed=0, shrink=False)
    assert unshrunk.counterexample == unshrunk.original == r.original
    assert r.message == "the law returned False"
    assert "FALSIFIED" in str(r) and "x = " in str(r)
    with pytest.raises(LawFailure) as e:
        r.raise_if_failed()
    assert e.value.result is r


def test_errors_are_counterexamples() -> None:
    @law(n=integers(-5, 5))
    def reciprocal(n: int) -> None:
        expect(1 / n != 0)

    r = reciprocal.check(n_examples=200, seed=1)
    assert r.status == "error" and r.exception == "ZeroDivisionError"
    assert r.counterexample == {"n": 0}
    assert "ZeroDivisionError: division by zero" in str(r)


def test_assume_and_unsatisfiable() -> None:
    @law(n=integers(0, 100))
    def even_halves(n: int) -> None:
        assume(n % 2 == 0)
        expect((n // 2) * 2 == n)

    r = even_halves.check(n_examples=40, seed=0)
    assert r.ok and r.n_examples == 40 and r.n_discarded > 0
    assert "discarded)" in str(r)

    @law(n=integers(0, 100))
    def never(n: int) -> None:
        assume(False)

    u = never.check(n_examples=10, max_discard_ratio=2.0)
    assert u.status == "unsatisfiable" and u.n_discarded == 21 and not u.failed
    assert "more than 20 examples discarded" in str(u)

    bad = Law("bad", lambda n: None, {"n": integers(0, 3).filter(lambda n: n > 10)})
    assert bad.check(5).status == "unsatisfiable"


def test_checks_are_reproducible() -> None:
    a = false_for_large.check(n_examples=100, seed=11)
    b = false_for_large.check(n_examples=100, seed=11)
    assert a.counterexample == b.counterexample and a.n_examples == b.n_examples
    c = false_for_large.check(n_examples=100, seed=12)
    assert c.original != a.original
    # both shrink close to the boundary x = 7.5
    for r in (a, c):
        assert 7.5 <= r.counterexample["x"] < 7.6


def test_evaluate_silences_floating_point_warnings() -> None:
    out = evaluate(lambda x: expect(np.isnan(np.log(x) * 0 + np.sqrt(x))), {"x": -1.0})
    assert out.kind == "pass"
    with pytest.raises(ValueError):
        addition_commutes.check(n_examples=0)


def test_holds_for_and_result_fields() -> None:
    out = false_for_large.holds_for(x=9.0)
    assert out.kind == "violation"
    r = LawResult("l", "verified", 3, 0, seed=0)
    assert r.ok and r.shrink_steps == 0


def test_shrinking_keeps_the_failure_kind() -> None:
    @law(x=floats(0.0, 100.0))
    def two_bugs(x: float) -> None:
        if x < 10.0:
            raise KeyError("small")  # a different failure below 10
        expect(x < 50.0, "too big")

    r = two_bugs.check(n_examples=200, seed=2)
    assert r.status == "falsified" and r.message == "too big"
    assert r.counterexample == {"x": 50.0}

    @law(x=floats(0.0, 100.0))
    def errors_only(x: float) -> None:
        if x > 20.0:
            raise ZeroDivisionError("big")
        if x > 5.0:
            raise ValueError("medium")

    e = errors_only.check(n_examples=200, seed=5)
    assert e.status == "error" and e.exception == "ZeroDivisionError"
    assert 20.0 < e.counterexample["x"] <= 25.0


def test_shrinking_budget() -> None:
    capped = false_for_large.check(n_examples=100, seed=0, max_shrinks=1)
    assert capped.shrink_steps <= 1


def test_composite_counterexamples_shrink_to_the_guilty_operand() -> None:
    from spxa.laws.processes import compositions, levy_processes

    @law(X=compositions(levy_processes(["BrownianMotion", "GammaProcess"]), max_depth=2))
    def no_gamma(X) -> None:  # type: ignore[no-untyped-def]
        expect("GammaProcess" not in repr(X), "contains a gamma process")

    r = no_gamma.check(n_examples=100, seed=0)
    assert r.status == "falsified"
    assert repr(r.counterexample["X"]) == "GammaProcess(a=1.0, b=1.0)"
    assert "GammaProcess" in repr(r.original["X"])


def test_result_to_dict_is_json_ready() -> None:
    import json

    from spxa.laws import expect, floats, law

    @law(x=floats(0.0, 1.0))
    def below_half(x):
        expect(x < 0.5, "too big", x=x)

    data = below_half.check(n_examples=50, seed=1).to_dict()
    json.dumps(data)
    assert data["status"] == "falsified"
    assert data["law"] == "below_half"
    assert float(data["counterexample"]["x"]) >= 0.5
    assert isinstance(data["details"]["x"], float)
