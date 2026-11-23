r"""
Cumulants recovered from the characteristic function.

On its exponential-moment strip (:func:`~spxa.laws.quantities.exponential_moment_strip`)
the cumulant generating function :math:`K(z) = \psi(-iz)` is analytic. Its
Taylor coefficients at 0 are :math:`\kappa_n/n!`. Cauchy's formula on a
circle of radius :math:`r` inside the strip gives them as a discrete Fourier
transform:

.. math::
    \kappa_n = \frac{n!}{r^n}\,\frac1N\sum_{k=0}^{N-1} K(re^{i\theta_k})\,e^{-in\theta_k},
    \qquad \theta_k = 2\pi k/N.

The aliasing error is of order :math:`(r/R)^N`, where :math:`R` is the
distance to the nearest singularity. With :math:`r = R/2` and
:math:`N = 32` it is about :math:`2\times10^{-10}`.

=====================  =========================================================
``cumulants_from_cf``  the Cauchy coefficients of :math:`K` equal ``cumulants()``, :math:`n \le 4`
``cgf_is_convex``      :math:`K` is real and convex on the strip (Hölder)
=====================  =========================================================

``cumulants_from_cf`` sets two independent computations against each
other. One is the characteristic exponent, from closed forms, measures
and exponent composition. The other is the cumulants, from closed
forms, measure moments and series composition. Subordinated processes
are included.
"""

from __future__ import annotations

from math import factorial

import numpy as np

from spxa.core.process import Process
from spxa.laws.expect import assume, expect, expect_close
from spxa.laws.law import Law, law
from spxa.laws.processes import compositions
from spxa.laws.quantities import cumulants, exponential_moment_strip, laplace_exponent
from spxa.laws.strategies import floats

_X = compositions(max_depth=1, operations=("+", "*", "-", "@"))


def cauchy_cumulants(X: Process, order: int, radius: float, n_points: int = 128) -> np.ndarray:
    r"""
    :math:`\kappa_1, \dots, \kappa_{order}` from :math:`K` on the circle :math:`|z| = r`.

    ``radius`` must lie inside the exponential-moment strip, and well
    inside it for accuracy. The trapezoid rule aliases the coefficient
    :math:`c_{n+N}r^{N}` onto :math:`c_n`, an error of order
    :math:`\max_{|z|=R}|K(z)|\,(r/R)^N` for :math:`K` analytic on
    :math:`|z| < R`. A compound clock exponentiates the parent's exponent,
    so :math:`\max|K|` can be :math:`10^5` or more. With :math:`r = R/2`
    and 32 points this gave errors of :math:`10^{-6}`, e.g. for
    ``CGMY(1, 2.5, 2, 1.5) @ PoissonProcess(1, jump_size=2)``. With 128
    points it is negligible.
    """
    theta = 2 * np.pi * np.arange(n_points) / n_points
    z = radius * np.exp(1j * theta)
    k = laplace_exponent(X, z)
    coeffs = np.fft.fft(k) / n_points  # c_n r^n
    return np.array([(coeffs[n] / radius**n).real * factorial(n) for n in range(1, order + 1)])


def _radius(X: Process) -> float:
    lo, hi = exponential_moment_strip(X)
    width = min(-lo, hi)
    assume(width > 1e-3, "exponential moment strip too narrow")
    return float(min(0.5 * width, 1.0))


@law(X=_X, tags=("cumulants", "cf"))
def cumulants_from_cf(X: Process) -> None:
    """kappa_n = n! [z^n] K(z), from the characteristic exponent"""
    r = _radius(X)
    expected = cumulants(X, 4)
    got = cauchy_cumulants(X, 4, r)
    # the n-th coefficient carries the quadrature noise of K amplified by n!/r^n
    scale = np.max(np.abs(laplace_exponent(X, r * np.exp(1j * np.linspace(0, 2 * np.pi, 9)))))
    atol = 1e-8 * scale * np.array([factorial(n) / r**n for n in range(1, 5)])
    for n in range(4):
        expect_close(
            got[n], expected[n], rtol=1e-6, atol=float(atol[n]) + 1e-12, what=f"kappa_{n + 1}"
        )


@law(
    X=_X,
    s=floats(-0.9, 0.9, simplest=0.5),
    v=floats(-0.9, 0.9, simplest=-0.5),
    w=floats(0.0, 1.0, simplest=0.5),
    tags=("cumulants", "convexity"),
)
def cgf_is_convex(X: Process, s: float, v: float, w: float) -> None:
    """K(w s + (1-w) v) <= w K(s) + (1-w) K(v) on the strip"""
    lo, hi = exponential_moment_strip(X)
    assume(hi - lo > 1e-3, "exponential moment strip too narrow")
    span = min(-lo, hi, 5.0)
    zs, zv = s * span, v * span
    vals = laplace_exponent(X, np.array([zs, zv, w * zs + (1 - w) * zv]))
    expect(
        bool(np.all(np.abs(vals.imag) <= 1e-8 * (1 + np.abs(vals.real)))),
        "K is not real on the real axis",
        values=vals,
    )
    k_s, k_v, k_mid = vals.real
    chord = w * k_s + (1 - w) * k_v
    expect(
        k_mid <= chord + 1e-8 * (1 + abs(chord)),
        f"K above its chord: K(mid) = {k_mid!r} > {chord!r}",
        k_mid=k_mid,
        chord=chord,
    )


CUMULANTS: tuple[Law, ...] = (cumulants_from_cf, cgf_is_convex)
