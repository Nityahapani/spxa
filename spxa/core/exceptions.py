"""Exceptions raised by spxa operations."""

from __future__ import annotations


class SpxaError(Exception):
    """Base class for all spxa exceptions."""


class ExactnessError(SpxaError):
    """
    Raised when a method requiring a higher ExactnessLevel is called on a process
    that does not meet that level.

    For example, calling `.triplet` on a composition that left the Lévy class.

    Parameters
    ----------
    method :
        The method that was called.
    required :
        The ExactnessLevel required by the method.
    actual :
        The ExactnessLevel of the process.
    reason :
        Mathematical explanation of why the level was degraded.
    """

    def __init__(self, method: str, required: str, actual: str, reason: str) -> None:
        self.method = method
        self.required = required
        self.actual = actual
        self.reason = reason
        super().__init__(
            f"'{method}' requires ExactnessLevel.{required}, "
            f"but this process has ExactnessLevel.{actual}.\n"
            f"Reason: {reason}"
        )


class SpxaDegradationWarning(UserWarning):
    """
    Emitted when an operation degrades the ExactnessLevel of a process.

    Includes the mathematical reason for the degradation so users are never
    surprised by a silent loss of analytical guarantees.
    """
