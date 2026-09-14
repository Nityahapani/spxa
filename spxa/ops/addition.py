"""
Addition of independent Lévy processes.

The operator `+` on Process objects delegates here for the mathematical
justification. This module also exposes a standalone `add()` function
for composing more than two processes at once, and a `sum_processes()`
convenience for iterables.

Mathematical basis
------------------
For independent Lévy processes X, Y with triplets (b_X, σ²_X, ν_X) and
(b_Y, σ²_Y, ν_Y), the sum Z = X + Y is a Lévy process with triplet

    b_Z   = b_X + b_Y
    σ²_Z  = σ²_X + σ²_Y
    ν_Z   = ν_X + ν_Y   (measure addition)

The result is exact for independent Lévy processes and degrades to
MOMENT_PROPAGATION otherwise, with a warning stating the reason.

Reference: Sato (1999), Proposition 11.10.
"""

from __future__ import annotations

import warnings
from typing import Iterable

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import SpxaDegradationWarning
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def add(X: Process, Y: Process) -> Process:
    """
    Return the process Z = X + Y, assuming X and Y are independent.

    For independent Lévy processes this is exact: the triplet is computed
    by component-wise addition of drifts, diffusion matrices, and Lévy
    measures. For non-Lévy processes the result degrades to
    MOMENT_PROPAGATION with a descriptive warning.

    Parameters
    ----------
    X, Y :
        Independent stochastic processes.

    Returns
    -------
    Process
        A new process representing X + Y.

    References
    ----------
    Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*.
    Cambridge University Press. Proposition 11.10.
    """
    return X + Y


def sum_processes(processes: Iterable[Process]) -> Process:
    """
    Return the sum of an iterable of independent processes.

    Equivalent to functools.reduce(add, processes) but with a more
    informative error message for empty iterables.

    Parameters
    ----------
    processes :
        Iterable of Process objects. Must be non-empty.

    Returns
    -------
    Process
        Sum of all processes.

    Examples
    --------
    >>> from spxa.zoo.levy import BrownianMotion, GammaProcess
    >>> from spxa.ops.addition import sum_processes
    >>> procs = [BrownianMotion(sigma=1.0), BrownianMotion(sigma=0.5)]
    >>> Z = sum_processes(procs)
    >>> Z.triplet.sigma_sq  # 1.0 + 0.25 = 1.25
    1.25
    """
    proc_list = list(processes)
    if not proc_list:
        raise ValueError("sum_processes requires at least one process.")
    result = proc_list[0]
    for p in proc_list[1:]:
        result = result + p
    return result


def triplet_add(t1: LevyTriplet, t2: LevyTriplet) -> LevyTriplet:
    """
    Add two Lévy triplets component-wise.

    (b₁, σ²₁, ν₁) + (b₂, σ²₂, ν₂) = (b₁+b₂, σ²₁+σ²₂, ν₁+ν₂)

    This is the core arithmetic operation exposed for use in custom
    process implementations.

    Parameters
    ----------
    t1, t2 :
        LevyTriplet instances.

    Returns
    -------
    LevyTriplet

    References
    ----------
    Sato (1999), Proposition 11.10.
    """
    return t1 + t2


def exactness_for_addition(X: Process, Y: Process) -> tuple[ExactnessLevel, str]:
    """
    Determine the ExactnessLevel of X + Y and the reason if it degrades.

    Returns
    -------
    (level, reason) :
        level is the combined ExactnessLevel.
        reason is empty string if no degradation, otherwise a mathematical
        explanation of why exactness was lost.
    """
    combined = ExactnessLevel.combine(X.exactness, Y.exactness)
    if combined == ExactnessLevel.EXACT:
        return combined, ""
    reason = (
        f"Adding {type(X).__name__} (ExactnessLevel.{X.exactness.name}) "
        f"and {type(Y).__name__} (ExactnessLevel.{Y.exactness.name}): "
        "at least one operand is outside the Lévy class or has no exact triplet. "
        "Moments are propagated where available."
    )
    return combined, reason
