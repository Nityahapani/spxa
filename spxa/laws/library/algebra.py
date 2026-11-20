r"""
Laws of the process algebra: sums, scalings and negation.

For independent Lévy processes :math:`X, Y` and a constant :math:`c \ne 0`:

=================================  ===============================================
``mean_is_additive``               :math:`E[X_t+Y_t] = E[X_t] + E[Y_t]`
``cumulants_are_additive``         :math:`\kappa_n(X+Y) = \kappa_n(X) + \kappa_n(Y)`, :math:`n \le 4`
``cf_is_multiplicative``           :math:`\varphi^{X+Y}_t = \varphi^X_t\varphi^Y_t`
``addition_commutes``              :math:`\varphi^{X+Y} = \varphi^{Y+X}`
``addition_associates``            :math:`\varphi^{(X+Y)+Z} = \varphi^{X+(Y+Z)}`
``cumulants_scale``                :math:`\kappa_n(cX) = c^n\kappa_n(X)`
``cf_scales``                      :math:`\varphi^{cX}_t(u) = \varphi^X_t(cu)`
``scaling_distributes``            :math:`\varphi^{c(X+Y)} = \varphi^{cX + cY}`
``negation_conjugates``            :math:`\varphi^{-X}_t(u) = \overline{\varphi^X_t(u)}`
``subtraction_adds_negation``      :math:`\kappa_n(X - Y) = \kappa_n(X) + (-1)^n\kappa_n(Y)`
=================================  ===============================================

The sum laws compare the closed-form characteristic functions of the
primitives with the triplet of the composite, which is built from the
Lévy measures (and quadrature where there is no closed form). They
therefore test the measures and drift conventions as well as the
operators.
"""

from __future__ import annotations

import numpy as np

from spxa.core.process import Process
from spxa.laws.expect import expect_close
from spxa.laws.law import Law, law
from spxa.laws.library._common import ATOL, RTOL, constants, frequencies, times
from spxa.laws.processes import compositions
from spxa.laws.quantities import char_func, cumulants, mean

_X = compositions(max_depth=1, operations=("+", "*", "-"))


@law(X=_X, Y=_X, t=times(), tags=("algebra", "addition", "moments"))
def mean_is_additive(X: Process, Y: Process, t: float) -> None:
    """E[X + Y] = E[X] + E[Y]"""
    expect_close(mean(X + Y, t), mean(X, t) + mean(Y, t), rtol=RTOL, atol=ATOL, what="mean")


@law(X=_X, Y=_X, tags=("algebra", "addition", "moments"))
def cumulants_are_additive(X: Process, Y: Process) -> None:
    """kappa_n(X + Y) = kappa_n(X) + kappa_n(Y), n <= 4"""
    expect_close(
        cumulants(X + Y), cumulants(X) + cumulants(Y), rtol=RTOL, atol=ATOL, what="cumulants"
    )


@law(X=_X, Y=_X, u=frequencies(), t=times(), tags=("algebra", "addition", "cf"))
def cf_is_multiplicative(X: Process, Y: Process, u: float, t: float) -> None:
    """phi_{X+Y} = phi_X phi_Y"""
    expect_close(
        char_func(X + Y, u, t),
        char_func(X, u, t) * char_func(Y, u, t),
        rtol=RTOL,
        atol=ATOL,
        what="phi_{X+Y} vs phi_X phi_Y",
    )


@law(X=_X, Y=_X, u=frequencies(), tags=("algebra", "addition", "cf"))
def addition_commutes(X: Process, Y: Process, u: float) -> None:
    """phi_{X+Y} = phi_{Y+X}"""
    expect_close(char_func(X + Y, u), char_func(Y + X, u), rtol=RTOL, atol=ATOL, what="phi")


@law(X=_X, Y=_X, Z=_X, u=frequencies(), tags=("algebra", "addition", "cf"))
def addition_associates(X: Process, Y: Process, Z: Process, u: float) -> None:
    """phi_{(X+Y)+Z} = phi_{X+(Y+Z)}"""
    expect_close(
        char_func((X + Y) + Z, u), char_func(X + (Y + Z), u), rtol=RTOL, atol=ATOL, what="phi"
    )


@law(X=_X, c=constants(), tags=("algebra", "scaling", "moments"))
def cumulants_scale(X: Process, c: float) -> None:
    """kappa_n(cX) = c^n kappa_n(X)"""
    powers = c ** (1 + np.arange(4))
    expect_close(cumulants(c * X), powers * cumulants(X), rtol=RTOL, atol=ATOL, what="cumulants")


@law(X=_X, c=constants(), u=frequencies(), t=times(), tags=("algebra", "scaling", "cf"))
def cf_scales(X: Process, c: float, u: float, t: float) -> None:
    """phi_{cX}(u) = phi_X(cu)"""
    expect_close(
        char_func(c * X, u, t), char_func(X, c * u, t), rtol=RTOL, atol=ATOL, what="phi_{cX}(u)"
    )


@law(X=_X, Y=_X, c=constants(), u=frequencies(), tags=("algebra", "scaling", "cf"))
def scaling_distributes(X: Process, Y: Process, c: float, u: float) -> None:
    """phi_{c(X+Y)} = phi_{cX+cY}"""
    expect_close(
        char_func(c * (X + Y), u), char_func(c * X + c * Y, u), rtol=RTOL, atol=ATOL, what="phi"
    )


@law(X=_X, u=frequencies(), t=times(), tags=("algebra", "scaling", "cf"))
def negation_conjugates(X: Process, u: float, t: float) -> None:
    """phi_{-X}(u) = conj phi_X(u)"""
    expect_close(
        char_func(-X, u, t), char_func(X, u, t).conj(), rtol=RTOL, atol=ATOL, what="phi_{-X}"
    )


@law(X=_X, Y=_X, tags=("algebra", "scaling", "moments"))
def subtraction_adds_negation(X: Process, Y: Process) -> None:
    """kappa_n(X - Y) = kappa_n(X) + (-1)^n kappa_n(Y)"""
    signs = np.array([-1.0, 1.0, -1.0, 1.0])
    expect_close(
        cumulants(X - Y),
        cumulants(X) + signs * cumulants(Y),
        rtol=RTOL,
        atol=ATOL,
        what="cumulants",
    )


ALGEBRA: tuple[Law, ...] = (
    mean_is_additive,
    cumulants_are_additive,
    cf_is_multiplicative,
    addition_commutes,
    addition_associates,
    cumulants_scale,
    cf_scales,
    scaling_distributes,
    negation_conjugates,
    subtraction_adds_negation,
)
