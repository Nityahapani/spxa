r"""
How a derivative was obtained.

Every gradient, Jacobian and Hessian that :mod:`spxa.graph.autodiff` returns
carries a :class:`GradientExactness` label:

``EXACT``
    The analytic derivative of the function that is actually computed,
    evaluated in floating point (reverse-mode adjoints, authored VJPs, or
    forward-mode jets). The only error is rounding.
``NUMERICAL``
    Some step on the path used finite differences, so the result carries a
    truncation error (and cancellation) on top of rounding.
``STOCHASTIC``
    Some step is a Monte Carlo estimator: the result is a random variable
    whose mean is the derivative; it is reported with a standard error.

Labels combine by taking the worst one along a path
(``EXACT < NUMERICAL < STOCHASTIC``), so a result is ``EXACT`` only if every
contributing step is exact. Pass ``strict=True`` to the differentiation entry
points to raise :class:`InexactGradientError` instead of returning anything
that is not exact.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "GradientExactness",
    "InexactGradientError",
    "GradientNote",
    "GradientResult",
    "worst",
]


class GradientExactness(enum.IntEnum):
    """How a derivative was obtained; larger is less exact."""

    EXACT = 0
    NUMERICAL = 1
    STOCHASTIC = 2

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


def worst(levels: Iterable[GradientExactness]) -> GradientExactness:
    """The least exact of ``levels`` (``EXACT`` for an empty iterable)."""
    out = GradientExactness.EXACT
    for lv in levels:
        if lv > out:
            out = GradientExactness(lv)
    return out


class InexactGradientError(RuntimeError):
    """Raised under ``strict=True`` when a derivative would not be exact."""


@dataclass(frozen=True)
class GradientNote:
    """Why one node of a graph did not contribute an exact derivative."""

    node: str
    exactness: GradientExactness
    method: str
    reason: str = ""

    def __str__(self) -> str:
        why = f": {self.reason}" if self.reason else ""
        return f"{self.node} [{self.exactness.name}, {self.method}]{why}"


class GradientResult(dict):  # type: ignore[type-arg]
    r"""
    Gradients keyed by source node, plus how each was obtained.

    A plain ``dict`` (``result[node]``, ``result.get(node, 0.0)``, iteration)
    with three extra attributes:

    ``exactness``
        The worst label over all sources.
    ``source_exactness``
        ``{source node: GradientExactness}``.
    ``notes``
        :class:`GradientNote` entries for every node whose derivative was not
        obtained exactly, or which used a fallback route.
    """

    def __init__(
        self,
        values: Optional[Dict[Any, Any]] = None,
        source_exactness: Optional[Dict[Any, GradientExactness]] = None,
        notes: Optional[List[GradientNote]] = None,
        standard_errors: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__(values or {})
        self.source_exactness: Dict[Any, GradientExactness] = dict(source_exactness or {})
        self.notes: List[GradientNote] = list(notes or [])
        self.standard_errors: Dict[Any, Any] = dict(standard_errors or {})

    @property
    def exactness(self) -> GradientExactness:
        return worst(self.source_exactness.values())

    @property
    def is_exact(self) -> bool:
        return self.exactness == GradientExactness.EXACT

    def by_name(self) -> Dict[str, Any]:
        """The gradients keyed by ``node.name``."""
        return {getattr(k, "name", k): v for k, v in self.items()}

    def __repr__(self) -> str:
        inner = ", ".join(f"{getattr(k, 'name', k)!s}: {v!r}" for k, v in self.items())
        return f"GradientResult({{{inner}}}, exactness={self.exactness.name})"


@dataclass
class DerivativeReport:
    """Exactness bookkeeping shared by Jacobian and Hessian results."""

    exactness: GradientExactness = GradientExactness.EXACT
    notes: List[GradientNote] = field(default_factory=list)

    def merge(self, other: Tuple[GradientExactness, List[GradientNote]]) -> None:
        lv, notes = other
        self.exactness = worst([self.exactness, lv])
        self.notes.extend(notes)


def check_strict(
    strict: bool, exactness: GradientExactness, notes: List[GradientNote], what: str
) -> None:
    """Raise :class:`InexactGradientError` if ``strict`` and ``exactness`` is not EXACT."""
    if strict and exactness != GradientExactness.EXACT:
        detail = "; ".join(str(n) for n in notes) or "no detail"
        raise InexactGradientError(f"{what} is {exactness.name}, not EXACT ({detail})")
