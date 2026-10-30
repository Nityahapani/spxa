r"""Finite-difference operators on non-uniform one-dimensional grids.

Stencils
--------
At an interior node :math:`x_i` with :math:`h_- = x_i - x_{i-1}` and
:math:`h_+ = x_{i+1} - x_i`, the three-point central approximations are

.. math::
    u'(x_i) &\approx -\frac{h_+}{h_-(h_- + h_+)} u_{i-1}
              + \frac{h_+ - h_-}{h_- h_+} u_i
              + \frac{h_-}{h_+(h_- + h_+)} u_{i+1}, \\
    u''(x_i) &\approx \frac{2}{h_-(h_- + h_+)} u_{i-1}
              - \frac{2}{h_- h_+} u_i
              + \frac{2}{h_+(h_- + h_+)} u_{i+1}.

The first-derivative stencil is exact for quadratics, so it is second-order
on any grid. The second-derivative stencil is exact for quadratics and has a
leading error of :math:`\tfrac13 (h_+ - h_-) u'''`, which is
:math:`O(h^2)` on smoothly stretched grids such as the sinh grids of
:mod:`spxa.pde.grid`.

Operator
--------
:class:`LinearPDE1D` stores the coefficients of the backward equation

.. math::
    u_\tau = a(x, \tau) u_{xx} + b(x, \tau) u_x + c(x, \tau) u + f(x, \tau),

and :class:`DiscreteOperator1D` assembles its affine discretisation
:math:`L_h u + s` together with the boundary closures of
:mod:`spxa.pde.boundary`.

Convection may be discretised by central differences (second order) or by
first-order upwinding. ``'auto'`` upwinds only at nodes where the central
scheme would produce a negative off-diagonal coefficient, which is where it
would lose the discrete maximum principle. Such nodes are those with a cell
Péclet number above one (:math:`|b| h > 2a`).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def central_first_weights(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central first-derivative weights ``(lower, diag, upper)`` at interior nodes.

    Returns arrays of length ``len(x) - 2``; entry ``k`` belongs to node ``k + 1``.
    """
    x = np.asarray(x, dtype=float)
    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]
    lower = -hp / (hm * (hm + hp))
    diag = (hp - hm) / (hm * hp)
    upper = hm / (hp * (hm + hp))
    return lower, diag, upper


def central_second_weights(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central second-derivative weights ``(lower, diag, upper)`` at interior nodes."""
    x = np.asarray(x, dtype=float)
    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]
    lower = 2.0 / (hm * (hm + hp))
    diag = -2.0 / (hm * hp)
    upper = 2.0 / (hp * (hm + hp))
    return lower, diag, upper


def one_sided_first_weights(x0: float, x1: float, x2: float) -> tuple[float, float, float]:
    """Second-order one-sided first-derivative weights at ``x0`` using ``x1``, ``x2``.

    Works for either side (``x1``, ``x2`` may lie to the left of ``x0``); the
    stencil is exact for quadratics.
    """
    d1 = x1 - x0
    d2 = x2 - x0
    w1 = d2 / (d1 * (d2 - d1))
    w2 = -d1 / (d2 * (d2 - d1))
    return -(w1 + w2), w1, w2


def derivative_matrices(x: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Sparse first- and second-derivative matrices with one-sided boundary rows.

    Interior rows use the central stencils. Boundary rows use second-order
    one-sided stencils for the first derivative; the second derivative is
    extrapolated from the adjacent interior row (first-order). These
    matrices are meant for evaluating derivatives of computed solutions
    (Greeks), not for imposing boundary conditions.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 nodes")
    l1, d1, u1 = central_first_weights(x)
    l2, d2, u2 = central_second_weights(x)
    idx = np.arange(1, n - 1)
    rows = np.concatenate([idx, idx, idx])
    cols = np.concatenate([idx - 1, idx, idx + 1])
    D1 = sp.lil_matrix(sp.csr_matrix((np.concatenate([l1, d1, u1]), (rows, cols)), shape=(n, n)))
    D2 = sp.lil_matrix(sp.csr_matrix((np.concatenate([l2, d2, u2]), (rows, cols)), shape=(n, n)))
    w = one_sided_first_weights(x[0], x[1], x[2])
    D1[0, 0:3] = np.array(w)
    w = one_sided_first_weights(x[-1], x[-2], x[-3])
    D1[n - 1, n - 1] = w[0]
    D1[n - 1, n - 2] = w[1]
    D1[n - 1, n - 3] = w[2]
    D2[0, 0:3] = np.array([l2[0], d2[0], u2[0]])
    D2[n - 1, n - 3 :] = np.array([l2[-1], d2[-1], u2[-1]])
    return D1.tocsr(), D2.tocsr()
