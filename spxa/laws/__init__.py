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
Laws
    :func:`law` (decorator) and :class:`Law`. Inside a law, use
    :func:`expect_close`, :func:`expect`, :func:`implies` and
    :func:`assume`. :meth:`Law.check` returns a :class:`LawResult`.
Processes
    :data:`PRIMITIVES` and :data:`SUBORDINATORS` (one strategy per zoo
    family), :func:`levy_processes`, :func:`subordinators`,
    :func:`alpha_stables`, :func:`martingales`, :func:`compositions` (expression
    trees with ``+``, ``*``, ``-`` and ``@``) and :func:`scalars`.
Quantities and library
    :mod:`spxa.laws.quantities` evaluates characteristic functions, cumulants
    and exponential-moment strips for any composite.
    :mod:`spxa.laws.library` holds ready-made laws: ``AXIOMS``, ``ALGEBRA``,
    ``SUBORDINATION``, ``TRIPLET``, ``CUMULANTS`` and ``PROPERTIES``.
"""

from spxa.laws.expect import Discard, LawViolation, assume, expect, expect_close, implies
from spxa.laws.law import Law, LawFailure, LawResult, law
from spxa.laws.processes import (
    PRIMITIVES,
    SUBORDINATORS,
    alpha_stables,
    compositions,
    levy_processes,
    martingales,
    scalars,
    subordinators,
)
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

from spxa.laws import library, quantities

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
    # laws
    "Law",
    "LawFailure",
    "LawResult",
    "law",
    "LawViolation",
    "Discard",
    "assume",
    "expect",
    "expect_close",
    "implies",
    # processes
    "PRIMITIVES",
    "SUBORDINATORS",
    "alpha_stables",
    "compositions",
    "levy_processes",
    "martingales",
    "scalars",
    "subordinators",
    # modules
    "library",
    "quantities",
]
