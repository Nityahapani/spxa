"""Expectations used inside laws."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.laws.expect import Discard, LawViolation, assume, expect, expect_close, implies


def test_expect_close_scalars() -> None:
    expect_close(1.0, 1.0 + 1e-12)
    expect_close(0.0, 1e-13)
    expect_close(1e10, 1e10 * (1 + 5e-10))
    expect_close(float("inf"), float("inf"))
    expect_close(float("nan"), float("nan"), equal_nan=True)
    expect_close(1 + 2j, 1 + 2j * (1 + 1e-12))
    with pytest.raises(LawViolation) as e:
        expect_close(1.0, 1.1, what="mean")
    v = e.value
    assert v.message.startswith("mean differ: 1.0 vs 1.1")
    assert v.details["abs_error"] == pytest.approx(0.1) and v.details["rel_error"] == pytest.approx(
        0.1 / 1.1
    )
    with pytest.raises(LawViolation):
        expect_close(float("nan"), float("nan"))
    with pytest.raises(LawViolation):
        expect_close(float("inf"), -float("inf"))
    with pytest.raises(LawViolation):
        expect_close(1.0, 1.0 + 1e-6, rtol=1e-9)
    expect_close(1.0, 1.0 + 1e-6, rtol=1e-5)
    # symmetric in a and b: the scale is max(|a|, |b|)
    for a, b in [(1.0, 1.0 + 3e-9), (1.0 + 3e-9, 1.0), (0.5, 1.0), (1.0, 0.5)]:
        with pytest.raises(LawViolation):
            expect_close(a, b, rtol=2e-9, atol=0.0)
    expect_close(1.0, 2.0, rtol=0.5, atol=0.0)
    expect_close(2.0, 1.0, rtol=0.5, atol=0.0)


def test_expect_close_arrays_report_the_worst_entry() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    expect_close(a, a * (1 + 1e-12))
    b = a.copy()
    b[1, 0] += 0.5
    b[0, 1] += 1e-3
    with pytest.raises(LawViolation) as e:
        expect_close(a, b, what="char_func")
    d = e.value.details
    assert d["index"] == (1, 0) and d["n_mismatched"] == 2
    assert "at index (1, 0) (2 of 4 mismatched)" in e.value.message
    expect_close(np.ones(3), 1.0)  # broadcasting
    with pytest.raises(LawViolation, match="shapes"):
        expect_close(np.ones(3), np.ones(2))
    c = a.copy()
    c[0, 0] = np.nan
    with pytest.raises(LawViolation) as e:
        expect_close(a, c)
    assert e.value.details["index"] == (0, 0)


def test_expect_implies_assume() -> None:
    expect(True)
    with pytest.raises(LawViolation, match="custom") as e:
        expect(False, "custom", value=3)
    assert e.value.details == {"value": 3}
    implies(False, False)
    implies(True, True)
    implies(False, lambda: 1 / 0)  # conclusion not evaluated
    with pytest.raises(LawViolation, match="premise holds"):
        implies(True, lambda: False)
    assume(True)
    with pytest.raises(Discard, match="too small"):
        assume(False, "too small")
