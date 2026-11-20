"""
Ready-made laws of stochastic process algebra.

Each module holds laws about one part of the theory and lists them in a
module-level tuple:

- :mod:`~spxa.laws.library.axioms` (``AXIOMS``): what every Lévy
  characteristic function and cumulant sequence must satisfy.
- :mod:`~spxa.laws.library.algebra` (``ALGEBRA``): sums, scalings and
  negation.
"""

from __future__ import annotations

from spxa.laws.library.algebra import ALGEBRA
from spxa.laws.library.axioms import AXIOMS

__all__ = ["ALGEBRA", "AXIOMS"]
