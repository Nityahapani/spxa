"""
Property-based testing for stochastic algebra.

Declare laws about processes, for example
:math:`E[X + Y] = E[X] + E[Y]` or the subordination identity, and spxa
searches randomised parameter grids for counterexamples, shrinking any it
finds to a minimal one.

Strategies
    :func:`floats`, :func:`integers`, :func:`sampled_from`, :func:`just`,
    :func:`one_of`, :func:`tuples`, :func:`fixed_dictionaries` and
    :func:`builds`, with ``map``, ``filter`` and ``flatmap``. Every generated
    example carries its own shrinks (:class:`Tree`).
"""

from spxa.laws.strategies import (
    Strategy,
    Tree,
    Unsatisfiable,
    builds,
    fixed_dictionaries,
    floats,
    integers,
    just,
    one_of,
    sampled_from,
    tuples,
)

__all__ = [
    "Strategy",
    "Tree",
    "Unsatisfiable",
    "builds",
    "fixed_dictionaries",
    "floats",
    "integers",
    "just",
    "one_of",
    "sampled_from",
    "tuples",
]
