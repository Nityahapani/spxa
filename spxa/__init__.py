"""
spxa — Stochastic Process Algebra

Stochastic processes as first-class algebraic objects.
Write Z = X + c*Y, get back a new process with its Lévy–Khintchine triplet
computed exactly, cumulants derived symbolically, and a property lattice
tracking what remains true about the result.
"""

__version__ = "0.0.1"

from spxa.core import (
    ExactnessLevel,
    ExactnessError,
    SpxaDegradationWarning,
    LevyMeasure,
    LevyTriplet,
    Process,
    ProcessProperties,
)
from spxa.zoo import (
    AlphaStable,
    BrownianMotion,
    CGMY,
    FractionalBrownianMotion,
    GammaProcess,
    HawkesProcess,
    NIG,
    OULevy,
    PoissonProcess,
    VarianceGamma,
)

__all__ = [
    "__version__",
    "ExactnessLevel",
    "ExactnessError",
    "SpxaDegradationWarning",
    "LevyMeasure",
    "LevyTriplet",
    "Process",
    "ProcessProperties",
    "AlphaStable",
    "BrownianMotion",
    "CGMY",
    "FractionalBrownianMotion",
    "GammaProcess",
    "HawkesProcess",
    "NIG",
    "OULevy",
    "PoissonProcess",
    "VarianceGamma",
]
