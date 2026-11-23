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
- :mod:`~spxa.laws.library.cumulants` (``CUMULANTS``): cumulants
  recovered from the characteristic function; convexity of the CGF.
- :mod:`~spxa.laws.library.properties` (``PROPERTIES``): property flags
  (martingale, subordinator, moments, self-similarity) against the
  quantities they claim, and their closure under the operators.
- :mod:`~spxa.laws.library.simulation` (``SIMULATION``): simulated paths and
  ``sample_at`` draws against the characteristic function and cumulants,
  with statistical tests at a known false-alarm rate.
"""

from __future__ import annotations

from spxa.laws.library.algebra import ALGEBRA
from spxa.laws.library.axioms import AXIOMS
from spxa.laws.library.cumulants import CUMULANTS
from spxa.laws.library.properties import PROPERTIES
from spxa.laws.library.simulation import SIMULATION
from spxa.laws.library.subordination import SUBORDINATION
from spxa.laws.library.triplet import TRIPLET

__all__ = [
    "ALGEBRA",
    "AXIOMS",
    "CUMULANTS",
    "PROPERTIES",
    "SIMULATION",
    "SUBORDINATION",
    "TRIPLET",
]
