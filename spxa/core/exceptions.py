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


class TripletUnavailableError(ExactnessError):
    """
    Raised by ``.triplet`` on an exact process whose triplet is not implemented.

    A subordinated Lévy process :math:`X_{T_t}` is a Lévy process, so its
    characteristic function and cumulants are exact. Its Lévy measure,
    however, needs the law of :math:`X_s` for every :math:`s`
    (Sato 1999, Thm 30.1).
    """

    def __init__(self, reason: str) -> None:
        SpxaError.__init__(
            self,
            "'triplet' is not available for this process, although it is exact "
            "(char_func, cumulants and simulate are).\n"
            f"Reason: {reason}",
        )
        self.method = "triplet"
        self.required = "EXACT"
        self.actual = "EXACT"
        self.reason = reason


class SpxaDegradationWarning(UserWarning):
    """
    Emitted when an operation degrades the ExactnessLevel of a process.

    Includes the mathematical reason for the degradation so users are never
    surprised by a silent loss of analytical guarantees.
    """
