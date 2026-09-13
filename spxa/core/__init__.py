"""Core algebraic engine for spxa."""

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError, SpxaDegradationWarning, SpxaError
from spxa.core.levy_measure import (
    CompoundLevyMeasure,
    DensityLevyMeasure,
    LevyMeasure,
    ScaledLevyMeasure,
)
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet

__all__ = [
    "ExactnessLevel",
    "ExactnessError",
    "SpxaDegradationWarning",
    "SpxaError",
    "CompoundLevyMeasure",
    "DensityLevyMeasure",
    "LevyMeasure",
    "ScaledLevyMeasure",
    "Process",
    "ProcessProperties",
    "LevyTriplet",
]
