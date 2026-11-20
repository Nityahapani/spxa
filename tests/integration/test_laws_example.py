"""examples/09_property_based_laws.py runs end to end (quick mode)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "09_property_based_laws.py"


def test_laws_example_quick(capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("laws_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(quick=True)
    out = capsys.readouterr().out
    assert "mean_is_additive: VERIFIED (no counterexample in 20 examples)" in out
    assert "martingale_plus_constant: FALSIFIED" in out
    assert "X = Drift(mu=0.0)" in out
    assert "subordination_cf: VERIFIED" in out
    assert "terminal_value_matches_cf: VERIFIED" in out
    assert "example: 10 verified (10 laws, 10 examples each, seed 4" in out
