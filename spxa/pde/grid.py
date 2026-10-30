r"""Spatial and temporal grids for the finite-difference solvers.

Spatial grids
-------------
:class:`Grid1D` stores a strictly increasing set of nodes. Besides uniform
grids it provides the sinh-stretched grids of Tavella & Randall (2000) and
in 't Hout & Foulon (2010),

.. math::
    x(\xi) = c + \alpha \sinh\!\big(c_2 \xi + c_1 (1 - \xi)\big), \qquad
    \xi \in [0, 1],

with :math:`c_1 = \operatorname{asinh}((x_{\min} - c)/\alpha)` and
:math:`c_2 = \operatorname{asinh}((x_{\max} - c)/\alpha)`. Nodes cluster
around the centre :math:`c` (typically the strike or a barrier); a smaller
:math:`\alpha` concentrates them more strongly. Because :math:`x(\xi)` is a
smooth map of a uniform grid, central differences on it remain second-order
accurate.

Points of non-smoothness of the payoff (strikes, barriers) should lie exactly
on nodes or exactly at cell midpoints; :meth:`Grid1D.with_points` snaps the
nearest interior node onto each such point.

Temporal grids
--------------
:class:`TimeGrid` lives in time to maturity :math:`\tau = T - t`. Event dates
(dividends, monitoring dates, Bermudan exercise dates) are inserted exactly
as grid times so that jump conditions are applied at the right instant.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from spxa.pde._utils import check_positive, cubic_interpolate


class Grid1D:
    """A strictly increasing one-dimensional grid.

    Parameters
    ----------
    nodes : array_like
        Strictly increasing node locations (at least three).
    """

    def __init__(self, nodes: Sequence[float] | np.ndarray) -> None:
        arr = np.asarray(nodes, dtype=float).ravel()
        if arr.size < 3:
            raise ValueError("a Grid1D needs at least 3 nodes")
        if not np.all(np.isfinite(arr)):
            raise ValueError("grid nodes must be finite")
        if np.any(np.diff(arr) <= 0.0):
            raise ValueError("grid nodes must be strictly increasing")
        self._nodes = arr
        self._nodes.setflags(write=False)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def uniform(cls, lo: float, hi: float, n_nodes: int) -> "Grid1D":
        """Uniform grid with ``n_nodes`` nodes on ``[lo, hi]``."""
        if hi <= lo:
            raise ValueError("hi must exceed lo")
        if int(n_nodes) < 3:
            raise ValueError("n_nodes must be at least 3")
        return cls(np.linspace(lo, hi, int(n_nodes)))

    @classmethod
    def sinh(
        cls,
        lo: float,
        hi: float,
        n_nodes: int,
        center: float,
        alpha: Optional[float] = None,
    ) -> "Grid1D":
        r"""Sinh-stretched grid concentrated around ``center``.

        Parameters
        ----------
        lo, hi : float
            Domain end points.
        n_nodes : int
            Number of nodes.
        center : float
            Concentration point :math:`c` (need not lie inside ``[lo, hi]``).
        alpha : float, optional
            Stretching scale :math:`\alpha > 0`. Defaults to
            ``(hi - lo) / 10``. Smaller values give stronger clustering.
        """
        if hi <= lo:
            raise ValueError("hi must exceed lo")
        if int(n_nodes) < 3:
            raise ValueError("n_nodes must be at least 3")
        a = (hi - lo) / 10.0 if alpha is None else check_positive("alpha", alpha)
        c1 = np.arcsinh((lo - center) / a)
        c2 = np.arcsinh((hi - center) / a)
        xi = np.linspace(0.0, 1.0, int(n_nodes))
        nodes = center + a * np.sinh(c2 * xi + c1 * (1.0 - xi))
        nodes[0], nodes[-1] = lo, hi
        return cls(nodes)

    # ------------------------------------------------------------------ properties
    @property
    def nodes(self) -> np.ndarray:
        """Node locations (read-only array)."""
        return self._nodes

    @property
    def n(self) -> int:
        """Number of nodes."""
        return int(self._nodes.size)

    @property
    def lo(self) -> float:
        """Left end point."""
        return float(self._nodes[0])

    @property
    def hi(self) -> float:
        """Right end point."""
        return float(self._nodes[-1])

    @property
    def spacing(self) -> np.ndarray:
        """Interval lengths ``h_i = x_{i+1} - x_i`` (length ``n - 1``)."""
        return np.diff(self._nodes)

    @property
    def is_uniform(self) -> bool:
        """True when all intervals have the same length (to rounding)."""
        h = self.spacing
        return bool(np.allclose(h, h[0], rtol=1e-10, atol=0.0))

    @property
    def faces(self) -> np.ndarray:
        """Control-volume faces: domain ends and interval midpoints (length ``n + 1``)."""
        x = self._nodes
        return np.concatenate([[x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]])

    @property
    def cell_widths(self) -> np.ndarray:
        """Width of the control volume around each node (length ``n``)."""
        return np.diff(self.faces)

    # ------------------------------------------------------------------ operations
    def with_points(self, points: Iterable[float]) -> "Grid1D":
        """Return a grid in which the nearest interior node is moved onto each point.

        Points outside the open interval ``(lo, hi)`` are ignored. Raises if
        two points would claim the same node or if snapping would break
        monotonicity (refine the grid in that case).
        """
        x = self._nodes.copy()
        claimed: dict[int, float] = {}
        for p in sorted(float(q) for q in points):
            if not (x[0] < p < x[-1]):
                continue
            j = int(np.argmin(np.abs(self._nodes[1:-1] - p))) + 1
            if j in claimed and not np.isclose(claimed[j], p):
                raise ValueError(
                    f"points {claimed[j]} and {p} snap to the same node; refine the grid"
                )
            claimed[j] = p
            x[j] = p
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("snapping broke grid monotonicity; refine the grid")
        return Grid1D(x)

    def refine(self) -> "Grid1D":
        """Insert the midpoint of every interval (doubles the resolution)."""
        x = self._nodes
        out = np.empty(2 * x.size - 1)
        out[0::2] = x
        out[1::2] = 0.5 * (x[1:] + x[:-1])
        return Grid1D(out)

    def locate(self, x: float) -> int:
        """Index of the node nearest to ``x``."""
        return int(np.argmin(np.abs(self._nodes - float(x))))

    def contains(self, x: float) -> bool:
        """True when ``lo <= x <= hi``."""
        return bool(self.lo <= float(x) <= self.hi)

    def interpolate(self, values: np.ndarray, x: float | np.ndarray) -> np.ndarray:
        """Local cubic interpolation of nodal ``values`` at ``x``."""
        values = np.asarray(values, dtype=float)
        if values.shape != self._nodes.shape:
            raise ValueError("values must have one entry per node")
        return cubic_interpolate(self._nodes, values, np.asarray(x, dtype=float))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        kind = "uniform" if self.is_uniform else "non-uniform"
        return f"Grid1D({kind}, n={self.n}, [{self.lo:.6g}, {self.hi:.6g}])"
