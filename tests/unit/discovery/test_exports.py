"""The public discovery API."""

from __future__ import annotations

import spxa
import spxa.discovery as d


def test_lazy_subpackage_access() -> None:
    assert spxa.discovery is d
    assert "discovery" in spxa.__all__


def test_public_names() -> None:
    for name in (
        "discover",
        "DiscoveryReport",
        "Grammar",
        "fit_structure",
        "summarize",
        "from_statistics",
        "recognize",
        "Model",
        "CGFSeries",
        "VARIANCE_GAMMA",
    ):
        assert name in d.__all__ and hasattr(d, name)
    assert set(d.FAMILIES) >= {"Drift", "BM", "Gamma", "IG", "VG", "NIG", "Meixner", "CGMY"}
