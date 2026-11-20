r"""
Laws of subordination: a Lévy process run on a random clock.

For a Lévy process :math:`X` and independent subordinators :math:`T, S`,
:math:`Z_t = X_{T_t}` is again a Lévy process (Sato 1999, Thm 30.1) and

=================================  ==============================================================
``subordination_cf``               :math:`\varphi^{Z}_t(u) = E[e^{T_t\psi_X(u)}] = \varphi^{T}_t(-i\psi_X(u))`
``subordination_mean``             :math:`E[Z_1] = E[X_1]\,E[T_1]`
``subordination_variance``         :math:`\mathrm{Var}\,Z_1 = E[T_1]\mathrm{Var}\,X_1 + \mathrm{Var}\,T_1\,E[X_1]^2`
``subordination_third_cumulant``   :math:`\kappa_3^Z = \kappa^T_1\kappa^X_3 + 3\kappa^T_2\kappa^X_1\kappa^X_2 + \kappa^T_3(\kappa^X_1)^3`
``deterministic_clock``            :math:`X_{ct}` has :math:`\varphi^X_{ct}`: ``X @ Drift(c)`` runs :math:`X` at speed :math:`c`
``clocks_add``                     :math:`\varphi^{X\circ(T+S)} = \varphi^{X\circ T}\varphi^{X\circ S}`
``subordination_associates``       :math:`(X\circ T)\circ S = X\circ(T\circ S)` in law
=================================  ==============================================================

``subordination_cf`` evaluates the clock's own characteristic function
at the complex frequency :math:`-i\psi_X(u)`, using its closed form for
zoo primitives. The implementation of ``@`` uses the clock's triplet,
so the two sides are computed independently.

``clocks_add`` holds because :math:`X_{T+S} = X_T + (X_{T+S} - X_T)`,
and the second term is an independent copy of :math:`X_S`. The
superficially similar :math:`(X+Y)\circ T = X\circ T + Y\circ T` is false:
the right side runs two independent clocks.
"""

from __future__ import annotations

import numpy as np

from spxa.core.process import Process
from spxa.laws.expect import expect_close
from spxa.laws.law import Law, law
from spxa.laws.library._common import ATOL, RTOL, frequencies, times
from spxa.laws.processes import compositions, subordinators
from spxa.laws.quantities import char_exponent, char_func, cumulants
from spxa.laws.strategies import floats
from spxa.zoo.levy import Drift

_X = compositions(max_depth=1, operations=("+", "*", "-"))
_T = subordinators()


@law(X=_X, T=_T, u=frequencies(), t=times(), tags=("subordination", "cf"))
def subordination_cf(X: Process, T: Process, u: float, t: float) -> None:
    """phi_{X@T}(u) = phi_T(-i psi_X(u))"""
    w = -1j * char_exponent(X, u)
    expected = np.atleast_1d(np.asarray(T.char_func(w, t), dtype=complex))
    expect_close(char_func(X @ T, u, t), expected, rtol=RTOL, atol=ATOL, what="phi_{X@T}")


@law(X=_X, T=_T, tags=("subordination", "moments"))
def subordination_mean(X: Process, T: Process) -> None:
    """E[X_T] = E[X_1] E[T_1]"""
    kx, kt = cumulants(X, 1), cumulants(T, 1)
    expect_close(cumulants(X @ T, 1)[0], kx[0] * kt[0], rtol=RTOL, atol=ATOL, what="mean")


@law(X=_X, T=_T, tags=("subordination", "moments"))
def subordination_variance(X: Process, T: Process) -> None:
    """Var X_T = E[T] Var X + Var T E[X]^2"""
    kx, kt = cumulants(X, 2), cumulants(T, 2)
    expect_close(
        cumulants(X @ T, 2)[1],
        kt[0] * kx[1] + kt[1] * kx[0] ** 2,
        rtol=RTOL,
        atol=ATOL,
        what="variance",
    )


@law(X=_X, T=_T, tags=("subordination", "moments"))
def subordination_third_cumulant(X: Process, T: Process) -> None:
    """kappa_3 = kT1 kX3 + 3 kT2 kX1 kX2 + kT3 kX1^3"""
    kx, kt = cumulants(X, 3), cumulants(T, 3)
    expected = kt[0] * kx[2] + 3 * kt[1] * kx[0] * kx[1] + kt[2] * kx[0] ** 3
    expect_close(cumulants(X @ T, 3)[2], expected, rtol=RTOL, atol=ATOL, what="kappa_3")


@law(
    X=_X,
    c=floats(0.1, 5.0, simplest=2.0),
    u=frequencies(),
    t=times(),
    tags=("subordination", "cf"),
)
def deterministic_clock(X: Process, c: float, u: float, t: float) -> None:
    """X @ Drift(c) has phi_X(u, ct)"""
    expect_close(
        char_func(X @ Drift(c), u, t),
        char_func(X, u, c * t),
        rtol=RTOL,
        atol=ATOL,
        what="phi_{X@Drift(c)}",
    )


@law(X=_X, T=_T, S=_T, u=frequencies(), tags=("subordination", "cf"))
def clocks_add(X: Process, T: Process, S: Process, u: float) -> None:
    """phi_{X@(T+S)} = phi_{X@T} phi_{X@S}"""
    expect_close(
        char_func(X @ (T + S), u),
        char_func(X @ T, u) * char_func(X @ S, u),
        rtol=RTOL,
        atol=ATOL,
        what="phi_{X@(T+S)}",
    )


@law(X=_X, T=_T, S=_T, u=frequencies(), tags=("subordination", "cf"))
def subordination_associates(X: Process, T: Process, S: Process, u: float) -> None:
    """(X@T)@S = X@(T@S) in law"""
    expect_close(
        char_func((X @ T) @ S, u),
        char_func(X @ (T @ S), u),
        rtol=RTOL,
        atol=ATOL,
        what="phi",
    )


SUBORDINATION: tuple[Law, ...] = (
    subordination_cf,
    subordination_mean,
    subordination_variance,
    subordination_third_cumulant,
    deterministic_clock,
    clocks_add,
    subordination_associates,
)
