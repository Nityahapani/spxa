r"""Solvers for the linear complementarity problems of American-style options.

With early exercise, each implicit time step becomes the linear
complementarity problem (LCP)

.. math::
    M u \ge b, \qquad u \ge g, \qquad (M u - b)^\top (u - g) = 0,

where :math:`M = I - \theta\Delta\tau L_h` and :math:`g` is the exercise
value. All solvers implement the
:class:`~spxa.pde.theta.LCPStepper` protocol and can be passed to
:class:`~spxa.pde.theta.ThetaSolver1D` (and the pricers built on it).

* :class:`PenaltyLCP`: the penalty iteration of Forsyth & Vetzal (2002).
  It works for any matrix structure, converges finitely in practice and is
  the default.
* :class:`PSORLCP`: projected successive over-relaxation (Cryer, 1971), the
  classical but slow iterative method.
* :class:`BrennanSchwartzLCP`: the direct method of Brennan & Schwartz
  (1977). It is exact in one sweep for tridiagonal systems whose exercise
  region is an interval at one end of the grid (puts, or calls with
  dividends).
* :class:`IkonenToivanenLCP`: the operator splitting of Ikonen & Toivanen
  (2004). One linear solve per step plus an explicit update of a Lagrange
  multiplier; it extends verbatim to ADI schemes in two dimensions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def complementarity_residual(
    M: sp.spmatrix, u: np.ndarray, rhs: np.ndarray, obstacle: np.ndarray, free: np.ndarray
) -> float:
    """Largest violation of the LCP conditions on the ``free`` rows (scaled).

    Returns ``max(max(-w), max(g - u), max |w * (u - g)|)`` with
    ``w = M u - rhs``, normalised by ``max(1, max|u|)``.
    """
    w = M @ u - rhs
    gap = u - obstacle
    scale = max(1.0, float(np.max(np.abs(u))))
    viol = max(
        float(np.max(-w[free], initial=0.0)),
        float(np.max(-gap[free], initial=0.0)),
        float(np.max(np.abs(w[free] * gap[free]), initial=0.0)) / scale,
    )
    return viol / scale


class PenaltyLCP:
    r"""Penalty iteration of Forsyth & Vetzal (2002).

    Iterates

    .. math::
        (M + P^k) u^{k+1} = b + P^k g, \qquad
        P^k_{ii} = \begin{cases} 1/\varepsilon & u^k_i < g_i \\ 0 & \text{else}\end{cases}

    until the active set stops changing or the relative update falls below
    ``tol``. The constraint is satisfied to :math:`O(\varepsilon)`.

    Parameters
    ----------
    penalty : float
        Penalty factor :math:`1/\varepsilon`.
    tol : float
        Relative convergence tolerance.
    max_iter : int
        Iteration cap per time step.
    """

    def __init__(self, penalty: float = 1e8, tol: float = 1e-10, max_iter: int = 100) -> None:
        if penalty <= 0 or tol <= 0 or max_iter < 1:
            raise ValueError("penalty, tol and max_iter must be positive")
        self.penalty = float(penalty)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.iterations: list[int] = []

    def reset(self, n: int) -> None:
        self.iterations = []

    def step(
        self,
        M: sp.csr_matrix,
        solve: Callable[[np.ndarray], np.ndarray],
        rhs: np.ndarray,
        obstacle: np.ndarray,
        guess: np.ndarray,
        dt: float,
        free: np.ndarray,
    ) -> np.ndarray:
        u = np.maximum(guess, np.where(free, obstacle, -np.inf))
        active = free & (u <= obstacle)
        n = u.size
        for k in range(1, self.max_iter + 1):
            if not np.any(active):
                u_new = solve(rhs)
            else:
                p = np.where(active, self.penalty, 0.0)
                A = (M + sp.diags(p, 0, shape=(n, n))).tocsc()
                u_new = splu(A).solve(rhs + p * obstacle)
            new_active = free & (u_new < obstacle)
            change = np.max(np.abs(u_new - u)) / max(1.0, float(np.max(np.abs(u_new))))
            u = u_new
            if np.array_equal(new_active, active) or change < self.tol:
                self.iterations.append(k)
                return u
            active = new_active
        self.iterations.append(self.max_iter)
        return u

    def describe(self) -> str:
        return f"penalty iteration (1/eps={self.penalty:.0e})"
