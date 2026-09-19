"""
spxa — Stochastic Process Algebra

Stochastic processes as first-class algebraic objects.
"""

__version__ = "1.0.0"

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
    CorrelatedLevy,
    FractionalBrownianMotion,
    GammaProcess,
    HawkesProcess,
    InverseGaussianProcess,
    MeixnerProcess,
    MultivariateBrownianMotion,
    MultivariateLevyTriplet,
    NegativeBinomialProcess,
    NIG,
    OULevy,
    PoissonProcess,
    TemperedStable,
    VarianceGamma,
    correlated_brownian_motion,
    independent_levy_vector,
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
    "CorrelatedLevy",
    "FractionalBrownianMotion",
    "GammaProcess",
    "HawkesProcess",
    "InverseGaussianProcess",
    "MeixnerProcess",
    "MultivariateBrownianMotion",
    "MultivariateLevyTriplet",
    "NegativeBinomialProcess",
    "NIG",
    "OULevy",
    "PoissonProcess",
    "TemperedStable",
    "VarianceGamma",
    "correlated_brownian_motion",
    "independent_levy_vector",
]
