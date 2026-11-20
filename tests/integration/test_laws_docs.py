"""Every numerical statement in docs/theory/laws.md, checked against the code."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import spxa.laws as L
from spxa.laws import library
from spxa.laws.library import _common
from spxa.laws.stats import LEVEL

DOC = Path(__file__).resolve().parents[2] / "docs" / "theory" / "laws.md"
TEXT = DOC.read_text(encoding="utf-8")


def test_library_table_counts() -> None:
    rows = dict(re.findall(r"^\| `([A-Z]+)` \| (\d+) \|", TEXT, flags=re.M))
    assert set(rows) == set(library.__all__)
    for name, count in rows.items():
        assert len(getattr(library, name)) == int(count), name
    assert sum(int(c) for c in rows.values()) == len(L.default_suite()) == 46
    assert "as 46 laws in seven groups" in TEXT


def test_tolerances_and_level() -> None:
    assert (_common.RTOL, _common.ATOL) == (1e-6, 1e-9)
    f = _common.frequencies()
    xs = [f.example(seed) for seed in range(200)]
    assert min(xs) >= -4.0 and max(xs) <= 4.0
    assert LEVEL == 1e-6
    assert "relative $10^{-6}$ and an absolute $10^{-9}$" in TEXT


def test_api_names_in_the_doc_exist() -> None:
    for name in (
        "Suite",
        "default_suite",
        "law_tests",
        "expect_close",
        "expect_not_rejected",
        "implies",
        "assume",
        "martingales",
        "compositions",
        "levy_processes",
        "alpha_stables",
        "PRIMITIVES",
    ):
        assert re.search(rf"\b{name}\b", TEXT) and hasattr(L, name), name


def test_first_snippet_runs_as_shown() -> None:
    code = re.search(r"```python\n(from spxa.laws import compositions.*?)```", TEXT, re.S)
    assert code is not None
    namespace: dict[str, object] = {}
    body = code.group(1).replace("print(", "result = (")
    exec(body, namespace)  # noqa: S102
    assert str(namespace["result"]) == (
        "mean_is_additive: VERIFIED (no counterexample in 200 examples)"
    )


def test_false_law_counterexample_matches_the_example() -> None:
    import importlib.util

    path = DOC.parents[2] / "examples" / "09_property_based_laws.py"
    spec = importlib.util.spec_from_file_location("ex09", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.martingale_plus_constant.check(n_examples=200, seed=1)
    shown = re.search(r"```text\n(martingale_plus_constant.*?)```", TEXT, re.S)
    assert shown is not None
    assert str(result).startswith(shown.group(1).strip())


def test_variance_identity_for_the_ecf() -> None:
    """Var Re(e^{iuX}) = (1 + Re phi(2u))/2 - (Re phi(u))^2, checked exactly on a discrete law."""
    x = np.array([-1.0, 0.3, 2.0])
    p = np.array([0.2, 0.5, 0.3])
    u = 0.7
    phi = lambda v: np.sum(p * np.exp(1j * v * x))  # noqa: E731
    var_re = np.sum(p * np.cos(u * x) ** 2) - np.sum(p * np.cos(u * x)) ** 2
    var_im = np.sum(p * np.sin(u * x) ** 2) - np.sum(p * np.sin(u * x)) ** 2
    assert np.isclose(var_re, (1 + phi(2 * u).real) / 2 - phi(u).real ** 2)
    assert np.isclose(var_im, (1 - phi(2 * u).real) / 2 - phi(u).imag ** 2)
