"""
Ready-made laws of stochastic process algebra.

Each module holds laws about one part of the theory and lists them in a
module-level tuple:

- :mod:`~spxa.laws.library.axioms` (``AXIOMS``): what every Lévy
  characteristic function and cumulant sequence must satisfy.
- :mod:`~spxa.laws.library.algebra` (``ALGEBRA``): sums, scalings and
  negation.
- :mod:`~spxa.laws.library.subordination` (``SUBORDINATION``): random
  time changes.
"""

from __future__ import annotations

from spxa.laws.library.algebra import ALGEBRA
from spxa.laws.library.axioms import AXIOMS
from spxa.laws.library.subordination import SUBORDINATION

__all__ = ["ALGEBRA", "AXIOMS", "SUBORDINATION"]
