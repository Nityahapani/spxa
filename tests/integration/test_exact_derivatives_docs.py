"""Statements in docs/theory/exact_derivatives.md, checked against the code."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import spxa
from spxa.graph.autodiff import montecarlo
from spxa.laws.library import gradients
from spxa.laws.library.gradients import GRADIENTS

DOC = Path(__file__).resolve().parents[2] / "docs" / "theory" / "exact_derivatives.md"
TEXT = DOC.read_text(encoding="utf-8")


def test_first_example_runs_and_is_exact() -> None:
    block = re.search(r"```python\n(.*?)```", TEXT, re.S).group(1)
    ns: dict = {}
    exec(block, ns)  # noqa: S102
    vg, exkurt = ns["vg"], ns["exkurt"]
    g = spxa.grad(exkurt)(vg.values)
    for i in range(3):
        h = 1e-5
        e = np.eye(3)[i] * h
        assert g[i] == pytest.approx(
            (exkurt(vg.values + e) - exkurt(vg.values - e)) / (2 * h), rel=1e-8
        )


def test_counts_thresholds_and_defaults() -> None:
    assert "(seven laws)" in TEXT and len(GRADIENTS) == 7
    assert "$x \\le a + 1 + 4\\sqrt a$" in TEXT
    assert "cut = a + 1.0 + 4.0 * np.sqrt(a)" in inspect.getsource(
        montecarlo.gamma_shape_derivative
    )
    assert "steps of $10^{-4}$ relative" in TEXT
    assert "h = 1e-4 * max(abs(th[k]), 0.05)" in inspect.getsource(gradients)
    from spxa.discovery.fit import fit_structure
    from spxa.discovery.options import fit_options
    from spxa.graph.autodiff.fourier import cos_price
    from spxa.rates.sabr import calibrate_sabr

    assert inspect.signature(fit_structure).parameters["gradient"].default == "exact"
    assert inspect.signature(fit_options).parameters["jac"].default == "exact"
    assert inspect.signature(calibrate_sabr).parameters["jac"].default == "exact"
    assert inspect.signature(cos_price).parameters["track_interval"].default is True
