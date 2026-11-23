r"""
Truncated power series for cumulant generating functions.

The cumulant generating function of :math:`X_1` is

.. math:: K(\theta) = \log E[e^{\theta X_1}] = \sum_{n \ge 1} \kappa_n \frac{\theta^n}{n!}.

For a Lévy process :math:`\log E[e^{\theta X_t}] = tK(\theta)`. The algebra of
processes acts on these series without any integration:

- independent sum: :math:`K_{X+Y} = K_X + K_Y`;
- scaling: :math:`K_{cX}(\theta) = K_X(c\theta)`, so :math:`\kappa_n \mapsto c^n\kappa_n`;
- subordination by an independent subordinator :math:`T`:
  :math:`E[e^{\theta X_{T_1}}] = E[e^{T_1 K_X(\theta)}]`, so
  :math:`K_{X \circ T} = K_T \circ K_X` (Sato 1999, Theorem 30.1).

The composition is formal and exact order by order because
:math:`K_X(0) = 0`. Cumulants of any finite expression therefore follow from
the cumulants of its primitives, as finite sums of products. This is Faà di
Bruno's formula organised as series arithmetic.
"""

from __future__ import annotations

from math import comb, factorial
from typing import Sequence

import numpy as np


def moments_to_cumulants(moments: Sequence[float]) -> np.ndarray:
    r"""Cumulants :math:`\kappa_1, \dots, \kappa_N` from raw moments :math:`m_1, \dots, m_N`.

    Uses :math:`\kappa_n = m_n - \sum_{k=1}^{n-1} \binom{n-1}{k-1}\kappa_k m_{n-k}`.
    """
    m = [1.0, *map(float, moments)]
    kappa = [0.0] * len(m)
    for n in range(1, len(m)):
        kappa[n] = m[n] - sum(comb(n - 1, k - 1) * kappa[k] * m[n - k] for k in range(1, n))
    return np.array(kappa[1:])


def cumulants_to_moments(cumulants: Sequence[float]) -> np.ndarray:
    r"""Raw moments from cumulants, :math:`m_n = \sum_{k=1}^{n} \binom{n-1}{k-1}\kappa_k m_{n-k}`."""
    kappa = [0.0, *map(float, cumulants)]
    m = [1.0] + [0.0] * (len(kappa) - 1)
    for n in range(1, len(kappa)):
        m[n] = sum(comb(n - 1, k - 1) * kappa[k] * m[n - k] for k in range(1, n + 1))
    return np.array(m[1:])


def _mul(a: np.ndarray, b: np.ndarray, order: int) -> np.ndarray:
    """Product of two ordinary power series truncated after θ^order."""
    return np.convolve(a, b)[: order + 1]


class CGFSeries:
    r"""Cumulant generating function truncated after :math:`\theta^N`.

    Parameters
    ----------
    cumulants :
        :math:`\kappa_1, \dots, \kappa_N` of :math:`X_1`.
    """

    __slots__ = ("_kappa",)

    def __init__(self, cumulants: Sequence[float]) -> None:
        k = np.asarray(cumulants, dtype=float).ravel()
        if k.size < 1:
            raise ValueError("need at least one cumulant")
        self._kappa = k
        self._kappa.setflags(write=False)

    @property
    def order(self) -> int:
        """The truncation order N."""
        return int(self._kappa.size)

    @property
    def cumulants(self) -> np.ndarray:
        """Read-only array :math:`(\\kappa_1, \\dots, \\kappa_N)`."""
        return self._kappa

    def cumulant(self, n: int) -> float:
        """:math:`\\kappa_n` for ``1 <= n <= order``."""
        if not 1 <= n <= self.order:
            raise ValueError(f"cumulant order must lie in [1, {self.order}], got {n}")
        return float(self._kappa[n - 1])

    def coefficients(self) -> np.ndarray:
        """Ordinary power-series coefficients :math:`a_0 = 0, a_n = \\kappa_n / n!`."""
        return np.concatenate(
            [[0.0], self._kappa / [factorial(n) for n in range(1, self.order + 1)]]
        )

    @classmethod
    def from_coefficients(cls, a: np.ndarray) -> CGFSeries:
        """Inverse of :meth:`coefficients`; ``a[0]`` must vanish."""
        if abs(a[0]) > 0:
            raise ValueError("a cumulant generating function has K(0) = 0")
        n = np.arange(1, a.size)
        return cls(a[1:] * np.array([factorial(int(k)) for k in n], dtype=float))

    @classmethod
    def from_moments(cls, moments: Sequence[float]) -> CGFSeries:
        """Series of a distribution with raw moments :math:`m_1, \\dots, m_N`."""
        return cls(moments_to_cumulants(moments))

    def moments(self) -> np.ndarray:
        """Raw moments :math:`m_1, \\dots, m_N` of :math:`X_1`."""
        return cumulants_to_moments(self._kappa)

    def _common(self, other: CGFSeries) -> int:
        return min(self.order, other.order)

    def __add__(self, other: CGFSeries) -> CGFSeries:
        """Independent sum: cumulants add."""
        n = self._common(other)
        return CGFSeries(self._kappa[:n] + other._kappa[:n])

    def scale(self, c: float) -> CGFSeries:
        """Series of :math:`cX`: :math:`\\kappa_n \\mapsto c^n \\kappa_n`."""
        n = np.arange(1, self.order + 1)
        return CGFSeries(self._kappa * float(c) ** n)

    def at_time(self, t: float) -> CGFSeries:
        """Series of :math:`X_t`: every cumulant is multiplied by ``t``."""
        return CGFSeries(self._kappa * float(t))

    def compose(self, inner: CGFSeries) -> CGFSeries:
        r"""Series of :math:`K_{\text{self}} \circ K_{\text{inner}}`.

        With ``self`` the series of a subordinator :math:`T` and ``inner`` that
        of :math:`X`, this is the series of the subordinated process
        :math:`X_{T_t}`.
        """
        n = self._common(inner)
        outer = self.coefficients()[: n + 1]
        a = inner.coefficients()[: n + 1]
        result = np.zeros(n + 1)
        power = np.zeros(n + 1)
        power[0] = 1.0
        for m in range(1, n + 1):
            power = _mul(power, a, n)
            result += outer[m] * power
        return CGFSeries.from_coefficients(result)

    def __call__(self, theta: complex | np.ndarray) -> np.ndarray:
        """Evaluate the truncated polynomial (valid only near 0)."""
        a = self.coefficients()
        return np.polyval(a[::-1], np.asarray(theta, dtype=complex))

    @property
    def mean(self) -> float:
        return self.cumulant(1)

    @property
    def variance(self) -> float:
        return self.cumulant(2)

    @property
    def skewness(self) -> float:
        """:math:`\\kappa_3 / \\kappa_2^{3/2}`."""
        return self.cumulant(3) / self.cumulant(2) ** 1.5

    @property
    def excess_kurtosis(self) -> float:
        """:math:`\\kappa_4 / \\kappa_2^2`."""
        return self.cumulant(4) / self.cumulant(2) ** 2

    def __repr__(self) -> str:
        body = ", ".join(f"{k:.6g}" for k in self._kappa)
        return f"CGFSeries([{body}])"
