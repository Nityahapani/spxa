r"""
Consistency of a primitive's triplet with its closed forms.

Each zoo process is specified twice: by closed forms (``char_func``,
``cumulants``) and by its Lévy–Khintchine triplet :math:`(b, \sigma^2, \nu)`,
whose measure has its own closed forms for tails, moments and the jump
exponent. Composites are built from the triplets, so every disagreement
between the two descriptions surfaces as a wrong sum or scaling. These
laws compare them directly, with truncation :math:`h(x) = \mathbb 1_{|x|\le1}`:

=============================  ==========================================================
``levy_khintchine``            :math:`\varphi_t(u) = \exp\big(t[ibu - \tfrac12\sigma^2u^2 + J(u)]\big)`
``triplet_cumulants``          :math:`\kappa_1 = b + \int_{|x|>1}x\,\nu`, :math:`\kappa_2 = \sigma^2 + \int x^2\nu`,
                               :math:`\kappa_n = \int x^n\nu` (Sato 1999, Thm 25.3)
``measure_moments``            ``nu.moment(n)`` equals :math:`\int x^n k(x)\,dx` by quadrature, :math:`n = 2, 3, 4`
``measure_tail``               ``nu.tail(x)`` equals :math:`\int_x^\infty k(y)\,dy`
``first_moment_between``       ``nu.first_moment_between(a, b)`` equals :math:`\int_{a<|x|\le b} x\,k(x)\,dx`
``jump_exponent``              ``nu.jump_exponent(u)`` equals its quadrature (integrated up
                               to 0 with a series near the origin, unlike the default
                               jump exponent, which stops at :math:`10^{-8}`)
=============================  ==========================================================

The quadrature laws apply to measures with a density. Atomic measures
(Poisson, negative binomial) discard the example via
:func:`~spxa.laws.expect.assume`.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate  # type: ignore[import-untyped]

from spxa.core.levy_measure import LevyMeasure
from spxa.core.process import Process
from spxa.laws.expect import assume, expect_close
from spxa.laws.law import Law, law
from spxa.laws.library._common import ATOL, RTOL, frequencies, times
from spxa.laws.processes import levy_processes
from spxa.laws.quantities import char_func, cumulants
from spxa.laws.strategies import floats, integers

_P = levy_processes()


def _density(nu: LevyMeasure) -> object:
    """The density as a scalar function, or discard the example."""
    try:
        nu.density(np.array([0.5, -0.5]))
    except NotImplementedError:
        assume(False, "measure has no density")

    def k(x: float) -> float:
        return float(nu.density(np.array([x]))[0])

    return k


def _quad(f: object, a: float, b: float) -> float:
    pts = [p for p in (-1.0, 1.0) if a < p < b]
    total = 0.0
    edges = [a, *pts, b]
    for lo, hi in zip(edges[:-1], edges[1:]):
        total += integrate.quad(f, lo, hi, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
    return total


@law(X=_P, u=frequencies(), t=times(), tags=("triplet", "cf"))
def levy_khintchine(X: Process, u: float, t: float) -> None:
    """phi_t(u) = exp(t psi(u)) with psi from (b, sigma^2, nu)"""
    tr = X.triplet
    psi = 1j * tr.b * u - 0.5 * tr.sigma_sq * u**2 + tr.nu.jump_exponent(complex(u))
    expect_close(char_func(X, u, t), np.exp(t * psi), rtol=RTOL, atol=ATOL, what="phi vs LK")


@law(X=_P, tags=("triplet", "cumulants"))
def triplet_cumulants(X: Process) -> None:
    """kappa_1 = b + int_{|x|>1} x nu, kappa_2 = sigma^2 + m_2, kappa_n = m_n"""
    tr = X.triplet
    expected = [
        tr.b + tr.nu.first_moment_between(1.0, np.inf),
        tr.sigma_sq + tr.nu.moment(2),
        tr.nu.moment(3),
        tr.nu.moment(4),
    ]
    expect_close(cumulants(X), expected, rtol=RTOL, atol=ATOL, what="cumulants vs triplet")


@law(X=_P, n=integers(2, 4, simplest=2), tags=("triplet", "measure"))
def measure_moments(X: Process, n: int) -> None:
    """nu.moment(n) = int x^n k(x) dx"""
    nu = X.triplet.nu
    k = _density(nu)
    num = _quad(lambda x: x**n * k(x) if x != 0 else 0.0, -np.inf, np.inf)  # type: ignore[operator]
    expect_close(nu.moment(n), num, rtol=1e-6, atol=1e-10, what=f"moment {n}")


@law(X=_P, x=floats(0.05, 3.0, simplest=1.0), tags=("triplet", "measure"))
def measure_tail(X: Process, x: float) -> None:
    """nu.tail(x) = int_x^inf k(y) dy"""
    nu = X.triplet.nu
    k = _density(nu)
    expect_close(nu.tail(x), _quad(k, x, np.inf), rtol=1e-6, atol=1e-10, what=f"tail({x})")


@law(
    X=_P,
    a=floats(0.05, 1.0, simplest=0.5),
    width=floats(0.1, 5.0, simplest=1.0),
    tags=("triplet", "measure"),
)
def first_moment_between(X: Process, a: float, width: float) -> None:
    """nu.first_moment_between(a, b) = int_{a<|x|<=b} x k(x) dx"""
    nu = X.triplet.nu
    k = _density(nu)
    b = a + width
    num = _quad(lambda x: x * k(x), a, b) + _quad(lambda x: x * k(x), -b, -a)  # type: ignore[operator]
    expect_close(nu.first_moment_between(a, b), num, rtol=1e-6, atol=1e-10, what="int x nu")


def _compensated(u: float, x: float) -> complex:
    r""":math:`e^{iux} - 1 - iux\mathbb 1_{|x|\le1}` without cancellation near 0."""
    z = u * x
    if abs(x) <= 1 and abs(z) < 1e-3:
        return complex(-(z**2) / 2 + z**4 / 24, -(z**3) / 6)
    return complex(np.expm1(1j * z) - (1j * z if abs(x) <= 1 else 0.0))


@law(X=_P, u=frequencies(), tags=("triplet", "measure"))
def jump_exponent(X: Process, u: float) -> None:
    """nu.jump_exponent(u) = quadrature of (e^{iux} - 1 - iux 1{|x|<=1}) k(x)"""
    nu = X.triplet.nu
    k = _density(nu)
    parts = []
    for f in (lambda c: c.real, lambda c: c.imag):
        g = lambda x, f=f: f(_compensated(u, x)) * k(x)  # type: ignore[operator]  # noqa: E731
        parts.append(_quad(g, -np.inf, 0.0) + _quad(g, 0.0, np.inf))
    expect_close(
        nu.jump_exponent(complex(u)),
        complex(parts[0], parts[1]),
        rtol=1e-6,
        atol=1e-9,
        what="J(u) vs quadrature",
    )


TRIPLET: tuple[Law, ...] = (
    levy_khintchine,
    triplet_cumulants,
    measure_moments,
    measure_tail,
    first_moment_between,
    jump_exponent,
)
