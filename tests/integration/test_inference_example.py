"""examples/10_inference_monitoring_carma.py runs end to end (quick mode)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "10_inference_monitoring_carma.py"


def test_inference_example_quick(capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("inference_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(quick=True)
    out = capsys.readouterr().out
    for title in ("1. Posterior", "2. Model-averaged", "3. Monitoring", "4. CARMA(2, 1)"):
        assert title in out
    assert "95% credible set: ['BM + Drift'" in out
    assert "in-control run length at least 10000 days" in out
    assert "alarm on day" in out and "no alarm" not in out
    assert "CARMA(2,1) Gaussian QML fit" in out
    assert "orders by BIC: (2,1)" in out
