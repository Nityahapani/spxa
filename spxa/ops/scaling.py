"""
Scalar multiplication and linear transformation of Lévy processes.

Mathematical basis
------------------
If X has triplet (b, σ², ν) and c ∈ ℝ \ {0}, then Z = cX has triplet

    b_Z   = c·b + drift_correction(c, ν)
    σ²_Z  = c²·σ²
    ν_Z(B) = ν(B/c)   [image measure]

where drift_correction accounts for the shift of mass across the unit ball
boundary when the truncation function h(x) = 𝟙_{|x|≤1} is applied to the
scaled process.

For a linear map A : ℝᵈ → ℝᵏ applied to a d-dimensional Lévy process X:

    b_Z   = A·b + ∫(Ax·𝟙_{‖Ax‖≤1} − Ax·𝟙_{‖x‖≤1}) ν(dx)
    Σ_Z   = A·Σ·Aᵀ
    ν_Z   = ν ∘ A⁻¹   [image measure]

Reference: Sato (1999), proof of Proposition 11.10; Cont & Tankov (2004),
Proposition 4.1.
"""

from __future__ import annotations

import numpy as np

from spxa.core.process import Process
from spxa.core.triplet import LevyTriplet


def scale(X: Process, c: float) -> Process:
    """
    Return the process Z = c·X.

    The triplet is transformed exactly: the Lévy measure image ν(B/c) and
    the drift correction are computed analytically.

    Parameters
    ----------
    X :
        Input process.
    c :
        Non-zero scalar.

    Returns
    -------
    Process
        A new process representing c·X.

    References
    ----------
    Sato (1999), Proposition 11.10 (proof).
    """
    if c == 0:
        raise ValueError("Scaling by 0 is not supported — the result is not a stochastic process.")
    return c * X


def negate(X: Process) -> Process:
    """Return the process Z = -X."""
    return -X


def subtract(X: Process, Y: Process) -> Process:
    """Return the process Z = X - Y (assuming X, Y independent)."""
    return X - Y


def triplet_scale(triplet: LevyTriplet, c: float) -> LevyTriplet:
    """
    Return the triplet of c·X given the triplet of X.

    The Lévy measure transforms as the image measure ν_Z(B) = ν_X(B/c),
    and the drift is corrected for the change in the truncation region.

    Parameters
    ----------
    triplet :
        Lévy triplet of X.
    c :
        Non-zero scalar.

    Returns
    -------
    LevyTriplet

    References
    ----------
    Sato (1999), proof of Proposition 11.10.
    Cont & Tankov (2004), Proposition 4.1.
    """
    return triplet.scale(c)


def linear_transform_triplet(
    triplet: LevyTriplet,
    A: np.ndarray,
) -> dict[str, object]:
    """
    Compute the triplet components of A·X from the triplet of X.

    For a d→k linear map A, the image process Z = AX has:
      b_Z  = A·b + drift_correction
      Σ_Z  = A·Σ·Aᵀ
      ν_Z  = image measure of ν under A

    Returns a dict with keys 'b', 'sigma_sq' (if d=k=1), 'A_sigma_A' (matrix),
    and a description of ν_Z, since the image measure cannot always be
    represented as a named LevyMeasure subclass.

    This function is provided for reference. For scalar transforms, use
    `triplet_scale` or the `*` operator directly.

    Parameters
    ----------
    triplet :
        1-dimensional LevyTriplet of X.
    A :
        Scalar value (1×1 case only in current implementation).

    References
    ----------
    Sato (1999), Theorem 11.3.
    """
    if not np.isscalar(A):
        raise NotImplementedError(
            "linear_transform_triplet currently supports scalar A only. "
            "For matrices, use the multidimensional extension (planned for v0.2)."
        )
    c = float(A)
    return {"triplet": triplet.scale(c)}
