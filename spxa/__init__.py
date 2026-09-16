"""
spxa — Stochastic Process Algebra

Stochastic processes as first-class algebraic objects.
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
    InverseGaussianProcess,
    MeixnerProcess,
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
    "InverseGaussianProcess",
    "MeixnerProcess",
    "NIG",
    "OULevy",
    "PoissonProcess",
    "VarianceGamma",
]
