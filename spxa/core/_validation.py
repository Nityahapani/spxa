"""Shared domain checks for marginal times and simulation grids."""

from __future__ import annotations

import numpy as np


def validate_order(order: int) -> None:
    """Reject booleans/nonintegral orders instead of truncating or fabricating moments."""
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("Cumulant order must be at least 1 (a positive integer)")


def validate_time(t: float) -> None:
    """Require a finite, nonnegative marginal horizon."""
    if not np.isfinite(t) or t < 0:
        raise ValueError("Time t must be non-negative and finite")


def simulation_grid(
    n_steps: int, n_paths: int, dt: float | None = None, T: float | None = None
) -> float:
    """Validate dimensions/horizons, including consistency when both dt and T are given."""
    for name, value, minimum in (("n_steps", n_steps, 0), ("n_paths", n_paths, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if T is not None and (not np.isfinite(T) or T <= 0):
        raise ValueError("T must be strictly positive and finite")
    if dt is not None:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be strictly positive and finite")
        if T is not None and n_steps and not np.isclose(dt * n_steps, T, rtol=1e-12, atol=0):
            raise ValueError("dt * n_steps must equal T when both are supplied")
        return float(dt)
    return float((T if T is not None else 1.0) / n_steps) if n_steps else 1.0
