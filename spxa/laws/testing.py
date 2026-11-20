r"""
pytest integration.

Turn laws into ordinary pytest tests, one per law::

    # tests/test_my_laws.py
    from spxa.laws.library import ALGEBRA, SUBORDINATION
    from spxa.laws.testing import law_tests

    test_algebra = law_tests(ALGEBRA, n_examples=50)
    test_my_law = law_tests(my_law, n_examples=200, seed=3)

Each law becomes one parametrised case named after it. A failing case
raises :class:`~spxa.laws.law.LawFailure` (an ``AssertionError``), whose
message is the full report with the shrunk counterexample, so pytest
prints it as the failure.

For one-off assertions inside an existing test, use :func:`assert_holds`.

Two environment variables override the arguments without touching the
tests, for example in a nightly job:

``SPXA_LAW_EXAMPLES``
    Examples per law, in place of ``n_examples``.
``SPXA_LAW_SEED``
    The seed, in place of ``seed``. A failure message always states the
    seed it used, so a nightly counterexample can be replayed locally.

pytest is imported only when :func:`law_tests` is called, so spxa does not
depend on it.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from spxa.laws.law import Law, LawResult
from spxa.laws.suite import LawsLike, _flatten

__all__ = ["assert_holds", "law_tests", "resolve_settings"]


def resolve_settings(n_examples: int, seed: int) -> tuple[int, int]:
    """``(n_examples, seed)`` after the ``SPXA_LAW_EXAMPLES`` / ``SPXA_LAW_SEED`` overrides."""
    n = os.environ.get("SPXA_LAW_EXAMPLES")
    s = os.environ.get("SPXA_LAW_SEED")
    try:
        return (int(n) if n else n_examples, int(s) if s else seed)
    except ValueError as exc:
        raise ValueError(f"SPXA_LAW_EXAMPLES / SPXA_LAW_SEED must be integers: {exc}") from None


def assert_holds(law_: Law, n_examples: int = 100, seed: int = 0, **check_kw: Any) -> LawResult:
    """Check ``law_`` and raise :class:`~spxa.laws.law.LawFailure` unless it is verified."""
    n, s = resolve_settings(n_examples, seed)
    result = law_.check(n_examples=n, seed=s, **check_kw)
    result.raise_if_failed()
    return result


def law_tests(
    *laws: LawsLike, n_examples: int = 100, seed: int = 0, **check_kw: Any
) -> Callable[[Law], None]:
    """A pytest test function parametrised over ``laws``, one case per law."""
    import pytest

    flat = _flatten(laws)
    if not flat:
        raise ValueError("law_tests needs at least one law")

    @pytest.mark.parametrize("law_", flat, ids=[law_.name for law_ in flat])
    def test_law(law_: Law) -> None:
        assert_holds(law_, n_examples, seed, **check_kw)

    return test_law
