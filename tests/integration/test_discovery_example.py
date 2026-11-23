"""examples/08_process_discovery.py runs end to end (quick mode)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "08_process_discovery.py"


def test_discovery_example_quick(capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("discovery_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(quick=True)
    out = capsys.readouterr().out
    for title in ("1. Returns CSV", "2. Uncertainty", "3. Bootstrap", "4. Summary", "5. Option"):
        assert title in out
    assert "matches your data within" in out
    assert "selection frequency over 10 resamples" in out
    assert "summary statistics identify at most three shape parameters" in out
    assert "VarianceGamma(sigma=0.15, nu=0.3, theta=-0.2) matches your option prices" in out
