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


import numpy as np

from spxa.core.process import Process


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
    r"""
    Evaluate the characteristic function of :math:`Z_t = X_{T_t}` at frequency u.

    Conditioning on the clock gives

    .. math::
        \varphi_Z(u; t) = E\big[e^{T_t\psi_X(u)}\big] = e^{t\,\psi_T(-i\psi_X(u))},

    where :math:`\psi_X` and :math:`\psi_T` are the characteristic exponents
    (:math:`E[e^{iuX_t}] = e^{t\psi_X(u)}`). Since :math:`\operatorname{Re}\psi_X
    \le 0`, the clock's exponent is evaluated where its Laplace transform
    converges. Equivalently, with the Bernstein function :math:`\phi` of
    :math:`T`, :math:`\varphi_Z(u; t) = e^{-t\phi(-\psi_X(u))}`.

    :math:`\psi_X` is computed from the triplet, not as
    :math:`\log\varphi_X`. The principal logarithm is wrong whenever
    :math:`|\operatorname{Im}\psi_X(u)| > \pi`, e.g. for a drift
    :math:`\mu` once :math:`|\mu u| > \pi`. This is the same as
    ``(parent @ subordinator).char_func(u, t)``.

    Parameters
    ----------
    parent :
        Parent process X. Must be EXACT.
    subordinator :
        Subordinator T. Must be EXACT.
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
        raise ValueError(f"{type(subordinator).__name__} is not a subordinator.")
    return (parent @ subordinator).char_func(u, t)


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
