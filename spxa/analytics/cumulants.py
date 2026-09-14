"""
Cumulant analysis utilities.

Provides standardised cumulant-based diagnostics for comparing processes,
computing skewness and excess kurtosis, and checking moment existence.
"""

from __future__ import annotations

import numpy as np

from spxa.core.process import Process


def skewness(process: Process, t: float = 1.0) -> float:
    """
    Return the skewness of X_t: κ_3 / κ_2^{3/2}.

    Parameters
    ----------
    process :
        Process with at least MOMENT_PROPAGATION exactness.
    t :
        Time. Cumulants scale as κ_n(X_t) = t · κ_n(X_1).
    """
    kappas = process.cumulants(order=3)
    k2, k3 = kappas[2], kappas[3]
    if k2 <= 0:
        raise ValueError("Variance is zero or negative — skewness undefined.")
    return t * k3 / (t * k2) ** 1.5


def excess_kurtosis(process: Process, t: float = 1.0) -> float:
    """
    Return excess kurtosis of X_t: κ_4 / κ_2².

    Parameters
    ----------
    process :
        Process with at least MOMENT_PROPAGATION exactness.
    t :
        Time.
    """
    kappas = process.cumulants(order=4)
    k2, k4 = kappas[2], kappas[4]
    if k2 <= 0:
        raise ValueError("Variance is zero — kurtosis undefined.")
    return t * k4 / (t * k2) ** 2


def cumulant_table(process: Process, order: int = 4, t: float = 1.0) -> dict[str, float]:
    """
    Return a named cumulant summary for X_t.

    Parameters
    ----------
    process :
        Process.
    order :
        Highest cumulant order to include (min 4).
    t :
        Time.

    Returns
    -------
    dict with keys: mean, variance, std, skewness, excess_kurtosis,
    and kappa_n for n in 1..order.
    """
    order = max(order, 4)
    kappas = process.cumulants(order=order)
    result: dict[str, float] = {}
    for n, v in kappas.items():
        result[f"kappa_{n}"] = t * v
    k2t = t * kappas[2]
    result["mean"] = t * kappas[1]
    result["variance"] = k2t
    result["std"] = float(np.sqrt(max(k2t, 0.0)))
    if k2t > 0:
        result["skewness"] = t * kappas[3] / k2t**1.5
        result["excess_kurtosis"] = t * kappas[4] / k2t**2
    return result
