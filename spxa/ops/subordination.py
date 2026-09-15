"""
Bochner subordination of Lévy processes.

Given a parent Lévy process X and an independent subordinator T (a
non-decreasing Lévy process), the subordinated process is

    Z_t = X_{T_t}

Z is again a Lévy process. Its characteristic exponent is

    ψ_Z(u) = -φ(-ψ_X(u))

where φ is the Laplace exponent of T: φ(λ) = -log E[e^{-λT_1}].

The Lévy measure of Z is

    ν_Z(B) = δ·ν_X(B) + ∫_0^∞ P(X_s ∈ B) ρ(ds)

where (δ, ρ) is the (drift, Lévy measure) of T.

For named process pairs, the characteristic function of Z is available in
closed form via the formula above. For general pairs it is computed
numerically by composing the characteristic exponents.

Named exact pairs implemented here:
  - BrownianMotion @ GammaProcess → VarianceGamma characteristic function
  - BrownianMotion @ InverseGaussian → NIG characteristic function

For arbitrary pairs the generic formula ψ_Z(u) = -φ(-ψ_X(u)) is applied
numerically.

References
----------
Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*.
Cambridge University Press. Theorem 30.1.

Schilling, R., Song, R. & Vondraček, Z. (2012). *Bernstein Functions*,
2nd ed. De Gruyter. Chapter 3.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapman & Hall/CRC. Proposition 4.4.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def subordinate(parent: Process, subordinator: Process) -> Process:
    """
    Return the subordinated process Z_t = X_{T_t}.

    Equivalent to the `@` operator: `parent @ subordinator`.

    Parameters
    ----------
    parent :
        The parent (driving) process X.
    subordinator :
        The time-change process T. Must be a subordinator
        (is_subordinator=True in its ProcessProperties).

    Returns
    -------
    Process
        The subordinated Lévy process Z.

    Raises
    ------
    ValueError
        If subordinator is not a subordinator.

    References
    ----------
    Sato (1999), Theorem 30.1.
    Cont & Tankov (2004), Proposition 4.4.
    """
    return parent @ subordinator


def char_exp_subordinated(
    parent: Process,
    subordinator: Process,
    u: float | np.ndarray,
    t: float = 1.0,
) -> np.ndarray:
    """
    Evaluate the characteristic function of Z_t = X_{T_t} at frequency u.

    Uses the composition formula:
        φ_Z(u; t) = exp(t · ψ_Z(u))
        ψ_Z(u)   = -φ_T(-ψ_X(u); 1) ... wait, more precisely:
        φ_Z(u; t) = L_T(-ψ_X(u); t)

    where L_T(λ; t) = E[e^{-λ T_t}] is the Laplace transform of T_t,
    evaluated at λ = -ψ_X(u).

    Since T is a subordinator with Laplace exponent φ:
        L_T(λ; t) = e^{-t·φ(λ)}
    and the result is:
        φ_Z(u; t) = exp(-t · φ(-ψ_X(u)))

    Parameters
    ----------
    parent :
        Parent process X. Must support char_func.
    subordinator :
        Subordinator T. Must support char_func.
    u :
        Frequency argument(s).
    t :
        Time.

    Returns
    -------
    np.ndarray
        Complex-valued characteristic function values.

    References
    ----------
    Sato (1999), Theorem 30.1, equation (30.2).
    """
    if not subordinator.properties.is_subordinator:
        raise ValueError(
            f"{type(subordinator).__name__} is not a subordinator."
        )

    u_arr = np.atleast_1d(np.asarray(u, dtype=complex))

    # ψ_X(u): characteristic exponent of parent at time 1
    # φ_X(u; 1) = exp(ψ_X(u)) → ψ_X(u) = log(φ_X(u; 1))
    cf_X = parent.char_func(u=u_arr, t=1.0)
    psi_X = np.log(cf_X + 1e-300)  # ψ_X(u)

    # Laplace exponent of T: φ_T(λ) = -log E[e^{-λ T_1}]
    # E[e^{-λ T_1}] = char_func of T at u = -iλ
    # For λ = -ψ_X(u), we need E[e^{ψ_X(u) T_1}]
    # = char_func_T(-i·ψ_X(u)/i ... more carefully:
    # char_func_T(u_T; 1) = E[e^{i u_T T_1}]
    # We want E[e^{-λ T_1}] = char_func_T(u_T=iλ; 1)
    # λ = -ψ_X(u) so u_T = -i·ψ_X(u)
    u_T = -1j * psi_X
    cf_T_laplace = subordinator.char_func(u=u_T, t=1.0)

    # φ_Z(u; t) = exp(-t · φ_T(-ψ_X(u))) = (E[e^{ψ_X(u) T_1}])^t
    result = cf_T_laplace ** t
    return result if result.shape != (1,) else result[0]


def bernstein_function(subordinator: Process, lam: float | np.ndarray) -> np.ndarray:
    """
    Evaluate the Bernstein function (Laplace exponent) of a subordinator.

    φ(λ) = -log E[e^{-λ T_1}] = -log(char_func_T(iλ; 1))

    φ is a Bernstein function: completely monotone derivative, φ(0)=0,
    φ(λ) ≥ 0 for λ ≥ 0.

    Parameters
    ----------
    subordinator :
        A subordinator process.
    lam :
        Non-negative real argument(s).

    Returns
    -------
    np.ndarray
        φ(λ) values.

    References
    ----------
    Schilling, Song & Vondraček (2012), Definition 3.1.
    """
    if not subordinator.properties.is_subordinator:
        raise ValueError(f"{type(subordinator).__name__} is not a subordinator.")

    # Use closed-form method if the process provides one
    if hasattr(subordinator, "bernstein_function"):
        lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
        result = subordinator.bernstein_function(lam_arr)
        return result if np.asarray(result).shape != (1,) else np.asarray(result)[0]

    lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
    results = np.zeros(len(lam_arr))
    for i, lv in enumerate(lam_arr):
        if lv == 0.0:
            results[i] = 0.0
            continue
        u_T = complex(0.0, lv)
        try:
            cf_val = complex(subordinator.char_func(u=u_T, t=1.0))
            if abs(cf_val) < 1e-15:
                results[i] = np.inf
            else:
                results[i] = -np.log(abs(cf_val))
        except Exception:
            results[i] = float("nan")
    phi = results
    return phi if phi.shape != (1,) else phi[0]


def is_valid_subordination(parent: Process, subordinator: Process) -> tuple[bool, str]:
    """
    Check whether subordination Z = X @ T is well-defined and return a reason if not.

    Returns
    -------
    (valid, reason) :
        valid is True iff the subordination is well-defined.
        reason is empty string if valid, otherwise an explanation.
    """
    if not subordinator.properties.is_subordinator:
        return False, (
            f"{type(subordinator).__name__} is not a subordinator "
            "(paths are not non-decreasing). "
            "The right operand of @ must satisfy is_subordinator=True."
        )
    return True, ""
