"""
Product of independent stochastic processes.

X·Y is not a Lévy process in general even when X and Y are independent Lévy
processes. spxa tracks moments of the product analytically using the
independence identity:

    E[X_t^m · Y_t^n] = E[X_t^m] · E[Y_t^n]

and computes the moments of the product process Z_t = X_t · Y_t via the
moment product formula. The result always has ExactnessLevel.MOMENT_PROPAGATION.

Higher-order moments require the raw moments of X and Y, which are derived
from their cumulants via the cumulant-to-moment conversion.

Reference: Stuart, A. & Ord, J.K. (1994). *Kendall's Advanced Theory of
Statistics*, Vol. 1. Chapter 3 (cumulant-moment relations).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError, SpxaDegradationWarning
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def cumulants_to_raw_moments(kappas: dict[int, float]) -> dict[int, float]:
    """
    Convert cumulants κ₁, …, κ_n to raw moments μ'₁, …, μ'_n.

    Uses the cumulant-moment relation (Stuart & Ord 1994, eq. 3.28):
      μ'_n = ∑_{partitions of {1,...,n}} ∏ κ_{|block|}

    Computed via the recurrence:
      μ'_n = ∑_{k=1}^{n} C(n-1, k-1) · κ_k · μ'_{n-k}    with μ'_0 = 1.

    Parameters
    ----------
    kappas :
        Dict mapping order → cumulant value. Keys must be 1..n contiguously.

    Returns
    -------
    dict
        Raw moments μ'_1, …, μ'_n.

    References
    ----------
    Stuart, A. & Ord, J.K. (1994). *Kendall's Advanced Theory of Statistics*,
    Vol. 1. Section 3.14.
    """
    from math import comb
    n_max = max(kappas.keys())
    moments: dict[int, float] = {0: 1.0}
    for n in range(1, n_max + 1):
        val = 0.0
        for k in range(1, n + 1):
            if k in kappas:
                val += comb(n - 1, k - 1) * kappas[k] * moments[n - k]
        moments[n] = val
    return {k: v for k, v in moments.items() if k >= 1}


def raw_moments_to_cumulants(moments: dict[int, float]) -> dict[int, float]:
    """
    Convert raw moments μ'₁, …, μ'_n back to cumulants κ₁, …, κ_n.

    Inverse of cumulants_to_raw_moments via Möbius inversion.

    Parameters
    ----------
    moments :
        Dict mapping order → raw moment value. Keys must be 1..n contiguously.

    Returns
    -------
    dict
        Cumulants κ_1, …, κ_n.

    References
    ----------
    Stuart & Ord (1994), Section 3.14.
    """
    from math import comb
    n_max = max(moments.keys())
    mu = {0: 1.0, **moments}
    kappas: dict[int, float] = {}
    for n in range(1, n_max + 1):
        val = mu[n]
        for k in range(1, n):
            if k in kappas:
                val -= comb(n - 1, k - 1) * kappas[k] * mu[n - k]
        kappas[n] = val
    return kappas


class ProductProcess(Process):
    """
    Product process Z_t = X_t · Y_t for independent X, Y.

    Not a Lévy process in general. Moments are computed analytically
    using the independence identity E[X^m Y^n] = E[X^m] E[Y^n].

    ExactnessLevel: MOMENT_PROPAGATION.

    Parameters
    ----------
    X, Y :
        Independent processes. Both must support cumulants().
    max_moment_order :
        Highest cumulant/moment order to track. Default 4.
    """

    def __init__(self, X: Process, Y: Process, max_moment_order: int = 4) -> None:
        self._X = X
        self._Y = Y
        self._max_order = max_moment_order

        node = CompositionNode(
            kind=NodeKind.MOMENT_APPROX,
            name=f"({X!r}) · ({Y!r})",
            children=[X._node, Y._node],
            math_note=(
                "Product process Z_t = X_t · Y_t; "
                "E[Z_t^n] = ∑_{k=0}^{n} C(n,k) E[X_t^k] E[Y_t^{n-k}] (independence); "
                "ExactnessLevel: MOMENT_PROPAGATION"
            ),
            reference="Stuart & Ord (1994), Kendall's Advanced Theory of Statistics, §3.14",
        )
        warnings.warn(
            f"ProductProcess({type(X).__name__}, {type(Y).__name__}): "
            "Product of Lévy processes is not Lévy. "
            "ExactnessLevel degraded to MOMENT_PROPAGATION. "
            "Only moments up to order {} are tracked.".format(max_moment_order),
            SpxaDegradationWarning,
            stacklevel=2,
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "ProductProcess has no Lévy–Khintchine triplet: "
                "the product of two Lévy processes is not a Lévy process in general. "
                "Use .cumulants() or .simulate() instead."
            ),
        )

    def _properties(self) -> ProcessProperties:
        px, py = self._X.properties, self._Y.properties
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=(px.has_finite_variance and py.has_finite_variance),
            has_finite_mean=(px.has_finite_mean and py.has_finite_mean),
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                f"Product of {type(self._X).__name__} and {type(self._Y).__name__}.",
                "Not a Lévy process — no independent increments.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        """
        Compute cumulants of Z_t = X_t · Y_t via moment products.

        E[Z^n] = E[(XY)^n] = E[X^n] · E[Y^n]  (independence, t=1)

        Then convert raw moments back to cumulants.

        Parameters
        ----------
        order :
            Maximum cumulant order.
        """
        o = min(order, self._max_order)

        kx = self._X.cumulants(order=o)
        ky = self._Y.cumulants(order=o)

        mx = cumulants_to_raw_moments(kx)
        my = cumulants_to_raw_moments(ky)

        mz = {n: mx[n] * my[n] for n in range(1, o + 1)}
        return raw_moments_to_cumulants(mz)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate Z_t = X_t · Y_t by simulating X and Y independently
        and taking the pointwise product.

        Parameters
        ----------
        n_steps, n_paths, T, rng :
            Standard simulation parameters.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1).
        """
        rng = rng or np.random.default_rng()
        paths_X = self._X.simulate(n_steps, n_paths, T, rng)
        paths_Y = self._Y.simulate(n_steps, n_paths, T, rng)
        return paths_X * paths_Y

    def __repr__(self) -> str:
        return f"ProductProcess({self._X!r}, {self._Y!r})"


def multiply(X: Process, Y: Process, max_moment_order: int = 4) -> ProductProcess:
    """
    Return the product process Z_t = X_t · Y_t.

    Parameters
    ----------
    X, Y :
        Independent processes.
    max_moment_order :
        Highest moment order to track analytically.

    Returns
    -------
    ProductProcess
    """
    return ProductProcess(X, Y, max_moment_order=max_moment_order)
