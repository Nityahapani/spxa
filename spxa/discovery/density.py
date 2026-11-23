r"""
Densities and distribution functions from characteristic functions (COS method).

Fang & Oosterlee (2008) expand a density on :math:`[a, b]` in a cosine series
whose coefficients are read off the characteristic function:

.. math::
   f(x) \approx \sideset{}{'}\sum_{k=0}^{N-1} A_k \cos\!\Big(k\pi\frac{x-a}{b-a}\Big),
   \qquad A_k = \frac{2}{b-a}\operatorname{Re}\Big\{\varphi\big(\tfrac{k\pi}{b-a}\big)
   e^{-ik\pi a/(b-a)}\Big\},

where :math:`\sum'` halves the :math:`k=0` term. Integrating term by term gives
the distribution function

.. math::
   F(x) \approx \frac{x-a}{b-a} + \sum_{k=1}^{N-1} A_k \frac{b-a}{k\pi}
   \sin\!\Big(k\pi\frac{x-a}{b-a}\Big), \qquad a \le x \le b.

The range follows the cumulant rule of Fang & Oosterlee:
:math:`[c_1 - L\sqrt{c_2 + \sqrt{c_4}},\; c_1 + L\sqrt{c_2 + \sqrt{c_4}}]`, with
:math:`c_j` the cumulants of :math:`X_t`. The error decays geometrically in
:math:`N` for smooth densities. For laws with atoms, such as compound
Poisson over short horizons, the CDF converges in the mean and is used
through bin probabilities.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

CharFunc = Callable[[np.ndarray], np.ndarray]


def truncation_range(cumulants: np.ndarray, L: float = 12.0) -> tuple[float, float]:
    """COS interval from the first four cumulants of :math:`X_t`."""
    c1, c2 = float(cumulants[0]), float(cumulants[1])
    c4 = float(cumulants[3]) if len(cumulants) > 3 else 0.0
    half = L * np.sqrt(max(c2, 0.0) + np.sqrt(max(c4, 0.0)))
    if not half > 0:
        raise ValueError("degenerate law: the variance and fourth cumulant vanish")
    return c1 - half, c1 + half


def cos_coefficients(char_func: CharFunc, a: float, b: float, n_terms: int) -> np.ndarray:
    """The coefficients :math:`A_k`, with the :math:`k = 0` term already halved."""
    if not b > a:
        raise ValueError(f"need b > a, got [{a}, {b}]")
    k = np.arange(n_terms)
    u = k * np.pi / (b - a)
    phi = np.asarray(char_func(u), dtype=complex)
    coef = 2.0 / (b - a) * np.real(phi * np.exp(-1j * u * a))
    coef[0] *= 0.5
    return coef


def _pdf_from_coef(coef: np.ndarray, x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = np.arange(coef.size)
    inside = (x >= a) & (x <= b)
    out = np.zeros_like(x)
    out[inside] = np.cos(np.outer(np.pi * (x[inside] - a) / (b - a), k)) @ coef
    return out


def _cdf_from_coef(coef: np.ndarray, x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = np.arange(1, coef.size)
    clipped = np.clip(x, a, b)
    series = np.sin(np.outer(np.pi * (clipped - a) / (b - a), k)) @ (
        coef[1:] * (b - a) / (k * np.pi)
    )
    out = (clipped - a) / (b - a) + series
    return np.asarray(np.where(x <= a, 0.0, np.where(x >= b, 1.0, out)))


def cos_pdf(
    char_func: CharFunc, x: np.ndarray, a: float, b: float, n_terms: int = 1024
) -> np.ndarray:
    """Density on ``x`` (zero outside ``[a, b]``)."""
    return _pdf_from_coef(cos_coefficients(char_func, a, b, n_terms), x, a, b)


def cos_cdf(
    char_func: CharFunc, x: np.ndarray, a: float, b: float, n_terms: int = 1024
) -> np.ndarray:
    """Distribution function on ``x`` (0 below ``a``, 1 above ``b``)."""
    return _cdf_from_coef(cos_coefficients(char_func, a, b, n_terms), x, a, b)


class COSDistribution:
    r"""Law of :math:`X_t` for a process given by its characteristic function.

    Parameters
    ----------
    char_func :
        ``u -> E[e^{iuX_t}]`` at the horizon of interest (vectorised in ``u``).
    cumulants :
        :math:`\kappa_1..\kappa_4` of :math:`X_t`, used for the range.
    n_terms :
        Number of cosine terms.
    L :
        Range width in the cumulant rule.
    """

    def __init__(
        self,
        char_func: CharFunc,
        cumulants: np.ndarray,
        n_terms: int = 1024,
        L: float = 12.0,
        bounds: Optional[tuple[float, float]] = None,
    ) -> None:
        self.char_func = char_func
        self.a, self.b = (
            bounds if bounds is not None else truncation_range(np.asarray(cumulants), L)
        )
        self.n_terms = int(n_terms)
        self._coef = cos_coefficients(char_func, self.a, self.b, self.n_terms)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return _pdf_from_coef(self._coef, x, self.a, self.b)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return _cdf_from_coef(self._coef, x, self.a, self.b)

    def bin_probabilities(self, edges: np.ndarray, floor: float = 1e-300) -> np.ndarray:
        """Probabilities of the bins ``[edges[i], edges[i+1])``, floored and made monotone."""
        F = np.maximum.accumulate(np.clip(self.cdf(np.asarray(edges, dtype=float)), 0.0, 1.0))
        return np.maximum(np.diff(F), floor)


def model_distribution(model, t: float, n_terms: int = 1024, L: float = 12.0) -> COSDistribution:  # type: ignore[no-untyped-def]
    """:class:`COSDistribution` of :math:`X_t` for a discovery :class:`~spxa.discovery.expr.Model`."""
    cum = model.cumulants(4, t=t)
    return COSDistribution(lambda u: model.char_func(u, t=t), cum, n_terms=n_terms, L=L)
