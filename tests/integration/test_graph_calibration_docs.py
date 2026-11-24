"""The graph-calibration example in docs/computational_graph.md runs and recovers its parameters."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

from spxa.graph.autodiff.exactness import GradientExactness

DOC = (Path(__file__).resolve().parents[2] / "docs" / "computational_graph.md").read_text(
    encoding="utf-8"
)


def test_calibration_example_runs_and_is_exact() -> None:
    section = DOC.split("### B. Graph-Based Model Calibration", 1)[1].split("\n---", 1)[0]
    (code,) = re.findall(r"```python\n(.*?)```", section, flags=re.S)
    ns: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "computational_graph.md", "exec"), ns)
    result = ns["result"]
    assert result["success"]
    assert result["gradient_exactness"] == GradientExactness.EXACT
    assert result["optimal_parameters"]["slope"] == pytest.approx(3.5, rel=1e-6)
    assert result["optimal_parameters"]["intercept"] == pytest.approx(1.2, rel=1e-6)
