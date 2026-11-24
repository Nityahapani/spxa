"""Names in the API reference section on exact derivatives exist in the code."""

from __future__ import annotations

import re
from pathlib import Path

import spxa
from spxa.calibration.surface import optimizer
from spxa.discovery import fit, options
from spxa.graph.autodiff import (
    api,
    discovery,
    exactness,
    families,
    fourier,
    hessian,
    jet,
    montecarlo,
    process,
    taylor,
    tape,
)
from spxa.rates import sabr
from spxa.vol import calibration as vol_calibration

API = (Path(__file__).resolve().parents[2] / "docs" / "api.md").read_text(encoding="utf-8")


def test_section_names_exist() -> None:
    section = API.split("## Exact Derivatives (`spxa.graph.autodiff`)", 2)[-1].split("\n## ", 1)[0]
    names = set(re.findall(r"`(?:[A-Za-z_][\w.]*\.)?([A-Za-z_]\w*)\(", section))
    names |= set(re.findall(r"`([A-Z][A-Za-z]+)`", section))
    modules = (spxa, api, discovery, exactness, families, fourier, hessian, jet, montecarlo)
    modules += (process, taylor, tape, fit, options, sabr, optimizer, vol_calibration)
    modules += (tape.GradientTape, optimizer.LeastSquaresOptimizer, process.DiffProcess)
    modules += (exactness.GradientExactness, montecarlo.MCGradient)
    missing = sorted(n for n in names if not any(hasattr(m, n) for m in modules))
    assert not missing, missing
