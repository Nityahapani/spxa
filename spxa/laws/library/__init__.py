"""
Ready-made laws of stochastic process algebra.

Each module holds laws about one part of the theory and lists them in a
module-level tuple:

- :mod:`~spxa.laws.library.axioms` (``AXIOMS``): what every Lévy
  characteristic function and cumulant sequence must satisfy.
"""

from __future__ import annotations

from spxa.laws.library.axioms import AXIOMS

__all__ = ["AXIOMS"]
