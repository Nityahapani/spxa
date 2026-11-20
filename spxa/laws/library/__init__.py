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
- :mod:`~spxa.laws.library.triplet` (``TRIPLET``): a primitive's closed
  forms against its Lévy–Khintchine triplet and measure.
"""

from __future__ import annotations

from spxa.laws.library.algebra import ALGEBRA
from spxa.laws.library.axioms import AXIOMS
from spxa.laws.library.subordination import SUBORDINATION
from spxa.laws.library.triplet import TRIPLET

__all__ = ["ALGEBRA", "AXIOMS", "SUBORDINATION", "TRIPLET"]
