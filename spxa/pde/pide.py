r"""
Partial integro-differential equations for exponential Lévy models: the jump operator.

Under :math:`S_t = S_0 e^{(r - q)t + X_t}` with :math:`X` a Lévy process with
Gaussian variance :math:`\sigma^2` and Lévy density :math:`\nu`, option
prices :math:`u(x, \tau)` in :math:`x = \log S` solve

.. math::
    u_\tau = \tfrac12\sigma^2 u_{xx} + \bigl(r - q - \tfrac12\sigma^2 - \omega\bigr)u_x
             - r u + \int \bigl(u(x + y) - u(x)\bigr)\,\nu(dy),
    \qquad \omega = \int (e^y - 1)\,\nu(dy).

Discretisation (uniform grid in :math:`x`, spacing :math:`h`):

* **Jumps** with :math:`|y| \ge h/2` are integrated against the hat functions
  of the grid, giving weights :math:`w_k = \int \nu(y)\varphi_k(y)\,dy` so
  that :math:`\int u(x_i + y)\nu(dy) \approx \sum_k w_k u_{i+k}` is exact for
  piecewise-linear :math:`u` (second order). Values outside the domain are
  taken from the payoff's asymptotics, so the convolution needs no
  truncation of the solution.
* **Small jumps** :math:`|y| < h/2` are replaced by a diffusion of variance
  :math:`\sigma_\varepsilon^2 = \int_{|y|<h/2} y^2\nu(dy)` (Asmussen–Rosiński),
  which matters for infinite-activity models (VG, CGMY, NIG).
* **Moment matching.** Linear interpolation over-weights :math:`y^2` on every
  cell by :math:`\epsilon_h = \sum_k w_k (kh)^2 - \int_{|y|\ge h/2} y^2\nu(dy)
  = O(h^{2-Y})`; for Blumenthal–Getoor index :math:`Y \ge 1` (NIG, CGMY) this
  would make the scheme first order. The excess is removed from the local
  diffusion, so the discrete operator reproduces the zeroth, first and second
  moments of :math:`\nu` and stays second order.
* With :math:`\tilde\lambda = \sum_k w_k` and
  :math:`\tilde\omega = \sum_k w_k(e^{kh} - 1)` the local part is
  :math:`a = \tfrac12(\sigma^2 + \sigma_\varepsilon^2 - \epsilon_h)`,
  :math:`b = r - q - a - \tilde\omega`, :math:`c = -(r + \tilde\lambda)`; the
  discrete jump operator then maps :math:`e^x` exactly as the continuous one
  (the forward is preserved). :math:`a` may be slightly negative for pure-jump
  models; the nearest-neighbour jump weights keep the stencil well posed.
References: Cont & Voltchkova (2005), *SIAM J. Numer. Anal.* 43(4);
d'Halluin, Forsyth & Vetzal (2005), *IMA J. Numer. Anal.* 25(1);
Asmussen & Rosiński (2001), *J. Appl. Probab.* 38(2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import quad
from scipy.signal import fftconvolve

from spxa.pde._utils import check_positive

__all__ = ["LevyJumpOperator"]

Density = Callable[[np.ndarray], np.ndarray]
FarField = Callable[[np.ndarray, float], np.ndarray]

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def _gauss(f: Density, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre nodes and weighted values of ``f`` on each ``[a_i, b_i]``."""
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    y = mid[:, None] + half[:, None] * _GL_X[None, :]
    fy = np.asarray(f(y.ravel()), dtype=float).reshape(y.shape)
    return y, fy * _GL_W[None, :] * half[:, None]


@dataclass
class LevyJumpOperator:
    r"""Discrete jump integral on a uniform grid of spacing ``h``.

    Parameters
    ----------
    density : callable
        Lévy density :math:`\nu(y)` (vectorised).
    h : float
        Grid spacing in :math:`x = \log S`.
    max_jump : float
        Jumps larger than this in absolute value are neglected; the
        neglected mass is reported in :attr:`tail_mass`.
    """

    density: Density
    h: float
    max_jump: float
    offsets: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    small_jump_variance: float = field(init=False)
    interpolation_variance: float = field(init=False)
    tail_mass: float = field(init=False)

    def __post_init__(self) -> None:
        h = check_positive("h", self.h)
        m = max(1, int(math.ceil(self.max_jump / h)))
        self.offsets = np.arange(-m, m + 1)
        w = np.zeros(2 * m + 1)
        # interval [j h, (j + 1) h] for j = -m .. m-1, cut at |y| < h/2
        j = np.arange(-m, m)
        a = j * h
        b = (j + 1) * h
        a = np.where(j == 0, 0.5 * h, a)
        b = np.where(j == -1, -0.5 * h, b)
        y, fw = _gauss(self.density, a, b)
        left_node = (j * h)[:, None]
        frac = (y - left_node) / h  # hat of node j+1 rises linearly on the interval
        w_lo = np.sum(fw * (1.0 - frac), axis=1)
        w_hi = np.sum(fw * frac, axis=1)
        np.add.at(w, j + m, w_lo)
        np.add.at(w, j + m + 1, w_hi)
        self.weights = w
        # linear interpolation over-weights y^2 on every cell; the excess is
        # removed from the local diffusion so the second moment is exact
        m2_cells = float(np.sum(fw * y * y))
        self.interpolation_variance = float(np.dot(w, (self.offsets * h) ** 2)) - m2_cells
        self.small_jump_variance = self._small_jump_variance(0.5 * h)
        self.tail_mass = self._tail_mass(m * h)

    def _integral(self, f: Callable[[float], float], a: float, b: float) -> float:
        val, _ = quad(f, a, b, limit=200, epsabs=1e-14, epsrel=1e-10)
        return float(val)

    def _point(self, y: float) -> float:
        return float(np.asarray(self.density(np.array([y])), dtype=float)[0])

    def _small_jump_variance(self, eps: float) -> float:
        def f(y: float) -> float:
            return y * y * self._point(y)

        return self._integral(f, -eps, 0.0) + self._integral(f, 0.0, eps)

    def _tail_mass(self, y_max: float) -> float:
        return self._integral(self._point, -np.inf, -y_max) + self._integral(
            self._point, y_max, np.inf
        )

    @property
    def local_variance(self) -> float:
        r"""Variance the local operator must carry for the jump part.

        :math:`\sigma_\varepsilon^2 - \bigl(\sum_k w_k (kh)^2 - \int_{|y| \ge h/2} y^2\nu(dy)\bigr)`:
        the small jumps minus the excess second moment created by the linear
        interpolation. With it the discrete operator reproduces the zeroth,
        first and second moments of :math:`\nu`, which keeps the scheme
        second-order accurate for infinite-variation measures.
        """
        return self.small_jump_variance - self.interpolation_variance

    @property
    def intensity(self) -> float:
        r""":math:`\tilde\lambda = \sum_k w_k` (jumps with :math:`|y| \ge h/2`)."""
        return float(np.sum(self.weights))

    @property
    def compensator(self) -> float:
        r""":math:`\tilde\omega = \sum_k w_k (e^{kh} - 1)`."""
        return float(np.dot(self.weights, np.expm1(self.offsets * self.h)))

    def apply(
        self,
        u: np.ndarray,
        x: np.ndarray,
        tau: float,
        far_left: FarField,
        far_right: FarField,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r""":math:`\sum_k w_k u(x_i + kh)` with far-field values outside the grid.

        ``weights`` overrides :attr:`weights` (same offsets), e.g. to exclude
        the near field that a solver treats implicitly.
        """
        m = int(self.offsets[-1])
        h = self.h
        xl = x[0] - h * np.arange(m, 0, -1)
        xr = x[-1] + h * np.arange(1, m + 1)
        ext = np.concatenate([far_left(xl, tau), u, far_right(xr, tau)])
        # sum_k w_k ext[i + m + k] is a correlation: convolve with reversed weights
        wts = self.weights if weights is None else weights
        return fftconvolve(ext, wts[::-1], mode="valid")
