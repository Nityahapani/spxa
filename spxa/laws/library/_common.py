"""Shared strategies and tolerances for the law library."""

from __future__ import annotations

from spxa.laws.strategies import Strategy, floats

#: Relative and absolute tolerances for quantities that may come from
#: quadrature (composite characteristic functions, measure moments).
RTOL = 1e-6
ATOL = 1e-9


def frequencies() -> Strategy[float]:
    """A real frequency in [-4, 4], shrinking to 1."""
    return floats(-4.0, 4.0, simplest=1.0)


def times() -> Strategy[float]:
    """A time horizon in [0.1, 3], shrinking to 1."""
    return floats(0.1, 3.0, simplest=1.0)


def constants() -> Strategy[float]:
    """A non-zero scale factor, shrinking to 2 (1 would make scaling laws trivial)."""
    return floats(-3.0, 3.0, simplest=2.0).filter(lambda c: abs(c) > 0.05)
