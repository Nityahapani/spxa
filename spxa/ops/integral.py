"""
Stochastic integrals of one process with respect to another.

The Itô stochastic integral ∫_0^t f(X_s) dY_s is only tractable in closed
form for restricted classes of integrands f and integrators Y. This module
implements the cases where analytical results are available and falls back to
Euler-Maruyama discretisation otherwise.

Exact cases implemented
-----------------------
1. Deterministic integrand f: ∫_0^t f(s) dX_s (stochastic convolution)
   For X a Lévy process with known characteristic exponent ψ, the
   characteristic function is:
     φ(u; t) = exp(∫_0^t ψ(u·f(s)) ds)

2. Linear SDE: dZ_t = a·Z_t dt + dX_t (Ornstein–Uhlenbeck type)
   Solution: Z_t = e^{at} Z_0 + ∫_0^t e^{a(t-s)} dX_s
   This is the basis of OULevy — see zoo/beyond/ou_levy.py.

3. Itô isometry: for X a square-integrable Lévy martingale and f deterministic,
     Var(∫_0^t f(s) dX_s) = κ_2(X_1) · ∫_0^t f(s)² ds

Approximate cases
-----------------
For random integrands, the Euler discretisation is used:
  ∫_0^t f(X_s) dY_s ≈ ∑_k f(X_{t_k}) · (Y_{t_{k+1}} - Y_{t_k})

References
----------
Protter, P.E. (2005). *Stochastic Integration and Differential Equations*,
2nd ed. Springer. Chapter II.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapter 8.

Applebaum, D. (2009). *Lévy Processes and Stochastic Calculus*, 2nd ed.
Cambridge University Press. Chapter 4.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError, SpxaDegradationWarning
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def ito_isometry_variance(
    integrator: Process,
    f: Callable[[float], float],
    t: float = 1.0,
    n_quad: int = 1000,
) -> float:
    """
    Compute Var(∫_0^t f(s) dX_s) via the Itô isometry.

    For a square-integrable Lévy martingale X and deterministic f:
      Var(∫_0^t f(s) dX_s) = κ_2(X_1) · ∫_0^t f(s)² ds

    Parameters
    ----------
    integrator :
        Square-integrable Lévy martingale X.
    f :
        Deterministic integrand function f : [0, t] → ℝ.
    t :
        Upper limit of integration.
    n_quad :
        Number of quadrature points for ∫ f(s)² ds.

    Returns
    -------
    float
        Variance of the stochastic integral.

    References
    ----------
    Applebaum (2009), Theorem 4.2.3 (Itô isometry for Lévy martingales).
    """
    kappas = integrator.cumulants(order=2)
    kappa2 = kappas[2]

    s_grid = np.linspace(0, t, n_quad)
    f_sq = np.array([f(s) ** 2 for s in s_grid])
    integral_f_sq = float(np.trapezoid(f_sq, s_grid))

    return kappa2 * integral_f_sq


def stochastic_convolution_char_func(
    integrator: Process,
    f: Callable[[float], float],
    u: float | np.ndarray,
    t: float = 1.0,
    n_quad: int = 500,
) -> np.ndarray:
    """
    Characteristic function of ∫_0^t f(s) dX_s for deterministic f.

    For X a Lévy process with characteristic exponent ψ_X:
      E[exp(iu ∫_0^t f(s) dX_s)] = exp(∫_0^t ψ_X(u·f(s)) ds)

    Computed by numerical quadrature on [0, t].

    Parameters
    ----------
    integrator :
        Lévy process X. Must have ExactnessLevel.EXACT.
    f :
        Deterministic integrand f : [0, t] → ℝ.
    u :
        Frequency argument(s).
    t :
        Upper limit.
    n_quad :
        Quadrature points.

    Returns
    -------
    np.ndarray
        Complex characteristic function values.

    References
    ----------
    Cont & Tankov (2004), Proposition 8.4.
    Protter (2005), Theorem II.30.
    """
    if integrator.exactness != ExactnessLevel.EXACT:
        raise ExactnessError(
            method="stochastic_convolution_char_func",
            required="EXACT",
            actual=integrator.exactness.name,
            reason="Characteristic function of stochastic convolution requires exact triplet.",
        )

    u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
    s_grid = np.linspace(0, t, n_quad)
    ds = s_grid[1] - s_grid[0]

    results = np.zeros(len(u_arr), dtype=complex)
    for j, uj in enumerate(u_arr):
        exponents = np.zeros(n_quad, dtype=complex)
        for i, s in enumerate(s_grid):
            fs = f(s)
            if abs(fs) < 1e-15:
                exponents[i] = 0.0
            else:
                cf_val = integrator.char_func(u=uj * fs, t=1.0)
                exponents[i] = np.log(complex(cf_val) + 1e-300)
        integral_psi = float(np.trapezoid(exponents.real, s_grid)) + \
                       1j * float(np.trapezoid(exponents.imag, s_grid))
        results[j] = np.exp(integral_psi)

    return results if results.shape != (1,) else results[0]


class StochasticIntegral(Process):
    """
    Process Z_t = ∫_0^t f(X_s) dY_s for random integrand f(X_s).

    For random integrands, no closed-form characteristic function is
    available in general. This class tracks:
      - Mean (zero if Y is a martingale and f is adapted)
      - Variance via the Itô isometry (when Y is square-integrable Lévy
        martingale and f(X) has finite second moment)
      - Simulation via Euler–Maruyama discretisation

    ExactnessLevel: MOMENT_PROPAGATION.

    Parameters
    ----------
    X :
        The process whose path drives the integrand.
    Y :
        The integrator process (the differential dY).
    f :
        The integrand function f : ℝ → ℝ applied to X.
    """

    def __init__(
        self,
        X: Process,
        Y: Process,
        f: Callable[[float], float],
    ) -> None:
        self._X = X
        self._Y = Y
        self._f = f

        node = CompositionNode(
            kind=NodeKind.MOMENT_APPROX,
            name=f"∫f({X!r})d({Y!r})",
            children=[X._node, Y._node],
            math_note=(
                "Z_t = ∫_0^t f(X_s) dY_s; simulated via Euler–Maruyama. "
                "Variance: Var(Z_t) ≈ κ_2(Y_1)·∫_0^t E[f(X_s)²] ds (Itô isometry)."
            ),
            reference="Protter (2005) Ch. II; Cont & Tankov (2004) Ch. 8; Applebaum (2009) Ch. 4",
        )
        warnings.warn(
            f"StochasticIntegral: random integrand f({type(X).__name__}) w.r.t. "
            f"{type(Y).__name__}. No exact characteristic function — "
            "using Euler–Maruyama simulation and moment propagation.",
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
                "StochasticIntegral with random integrand has no exact triplet. "
                "Use .simulate() for paths or .cumulants() for moment approximations."
            ),
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=self._Y.properties.is_martingale,
            has_finite_variance=(
                self._X.properties.has_finite_variance
                and self._Y.properties.has_finite_variance
            ),
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            notes=[
                f"Stochastic integral ∫f({type(self._X).__name__})d({type(self._Y).__name__}).",
                "Simulated via Euler–Maruyama. Moments approximate via Itô isometry.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        """
        Approximate cumulants via the Itô isometry (order ≤ 2 only).
        Higher orders are not tractable without strong assumptions.
        """
        result: dict[int, float] = {}
        if order >= 1:
            result[1] = 0.0 if self._Y.properties.is_martingale else float("nan")
        if order >= 2:
            kappa2_Y = self._Y.cumulants(order=2)[2]
            kappa2_X = self._X.cumulants(order=2)[2]
            result[2] = kappa2_Y * kappa2_X
        for n in range(3, order + 1):
            result[n] = float("nan")
        return result

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate Z_t = ∫_0^t f(X_s) dY_s via Euler–Maruyama.

        Z_{t_{k+1}} ≈ Z_{t_k} + f(X_{t_k}) · (Y_{t_{k+1}} - Y_{t_k})

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
        dY = np.diff(paths_Y, axis=1)

        paths_Z = np.zeros((n_paths, n_steps + 1))
        for k in range(n_steps):
            f_vals = np.vectorize(self._f)(paths_X[:, k])
            paths_Z[:, k + 1] = paths_Z[:, k] + f_vals * dY[:, k]

        return paths_Z

    def __repr__(self) -> str:
        return f"StochasticIntegral(X={self._X!r}, Y={self._Y!r})"
