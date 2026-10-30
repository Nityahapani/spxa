r"""The θ-scheme time stepper for one-dimensional backward equations.

Given the semi-discrete system :math:`u_\tau = L_h(\tau) u + s(\tau)`, one
step from :math:`\tau_n` to :math:`\tau_{n+1} = \tau_n + \Delta\tau` reads

.. math::
    \big(I - \theta \Delta\tau L_h^{n+1}\big) u^{n+1}
      = \big(I + (1 - \theta) \Delta\tau L_h^{n}\big) u^{n}
        + \Delta\tau \big(\theta s^{n+1} + (1 - \theta) s^{n}\big).

:math:`\theta = 1` is backward Euler (first order, L-stable),
:math:`\theta = \tfrac12` Crank–Nicolson (second order, A-stable but not
L-stable). Rows that belong to algebraic boundary conditions (Dirichlet,
zero curvature) are replaced by the constraint ``C u^{n+1} = g(tau_{n+1})``.

Event hooks
-----------
Jump conditions at event dates, such as discrete dividends, discrete barrier
monitoring or Bermudan exercise, are supplied as a mapping ``tau -> f(u, x,
tau)``. Each hook is applied right after the grid time ``tau`` has been
reached.

Free-boundary problems
----------------------
American-style constraints :math:`u \ge g` turn each step into a linear
complementarity problem. The stepper delegates those steps to an object that
follows the :class:`LCPStepper` protocol; the solvers live in
:mod:`spxa.pde.american`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D

EventHook = Callable[[np.ndarray, np.ndarray, float], np.ndarray]
"""Signature of an event hook: ``f(u, x, tau) -> u_new``."""

Obstacle = Union[np.ndarray, Callable[[np.ndarray, float], np.ndarray]]


class LCPStepper(Protocol):
    """Protocol for solvers of the per-step linear complementarity problem.

    Find ``u`` with ``M u >= rhs``, ``u >= g`` and
    ``(M u - rhs) . (u - g) = 0`` on the rows flagged ``free``; the other
    rows (boundary constraints) must satisfy ``M u = rhs`` exactly.
    """

    def reset(self, n: int) -> None:
        """Prepare for a new solve with ``n`` unknowns."""

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
        """Solve one complementarity step; ``solve`` applies ``M^{-1}``."""


@dataclass
class ThetaResult:
    """Raw output of :meth:`ThetaSolver1D.solve`.

    Attributes
    ----------
    grid : Grid1D
        Spatial grid.
    time_grid : TimeGrid
        Time grid that was used.
    values : np.ndarray
        Solution at ``tau = T``.
    previous_values : np.ndarray
        Solution at the second-to-last grid time (for the time derivative).
    history : np.ndarray or None
        All time levels, shape ``(n_steps + 1, n)``, if requested.
    info : dict
        Solver diagnostics (number of factorisations, LCP iterations, ...).
    """

    grid: Grid1D
    time_grid: TimeGrid
    values: np.ndarray
    previous_values: np.ndarray
    history: Optional[np.ndarray] = None
    info: dict[str, Any] = field(default_factory=dict)


def _normalise_events(
    events: Optional[Union[Mapping[float, Any], Iterable[tuple[float, Any]]]],
) -> list[tuple[float, list[EventHook]]]:
    if events is None:
        return []
    items = events.items() if isinstance(events, Mapping) else events
    merged: dict[float, list[EventHook]] = {}
    for tau, hook in items:
        hooks = list(hook) if isinstance(hook, (list, tuple)) else [hook]
        for h in hooks:
            if not callable(h):
                raise TypeError("event hooks must be callables f(u, x, tau)")
        merged.setdefault(float(tau), []).extend(hooks)
    return sorted(merged.items())


class ThetaSolver1D:
    r"""θ-scheme solver for a :class:`~spxa.pde.operators.DiscreteOperator1D`.

    Parameters
    ----------
    operator : DiscreteOperator1D
        Spatial discretisation with boundary closures.
    theta : float
        Implicitness :math:`\theta \in [0, 1]` (default Crank–Nicolson).
    """

    def __init__(self, operator: DiscreteOperator1D, theta: float = 0.5) -> None:
        if not isinstance(operator, DiscreteOperator1D):
            raise TypeError("operator must be a DiscreteOperator1D")
        theta = float(theta)
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        self.operator = operator
        self.theta = theta
        self._lu_cache: dict[tuple[float, float], Any] = {}
        self._asm_cache: Optional[tuple[sp.csr_matrix, np.ndarray]] = None
        self._n_factorisations = 0

    # ------------------------------------------------------------------ helpers
    def _assemble(self, tau: float) -> tuple[sp.csr_matrix, np.ndarray]:
        op = self.operator
        if op.time_homogeneous:
            if self._asm_cache is None:
                self._asm_cache = op.assemble(0.0)
            return self._asm_cache
        return op.assemble(tau)

    def _system(self, tau_new: float, dt: float, theta: float):
        op = self.operator
        n = op.n
        L_new, s_new = self._assemble(tau_new)
        C, g = op.constraints(tau_new)
        key = (round(dt, 15), theta)
        lu = self._lu_cache.get(key) if op.matrix_time_homogeneous else None
        rows = op.constraint_rows
        E = sp.csr_matrix((np.ones(rows.size), (rows, rows)), shape=(n, n))
        M = (sp.identity(n, format="csr") - theta * dt * L_new + (C - E)).tocsr()
        if lu is None:
            lu = splu(M.tocsc())
            self._n_factorisations += 1
            if op.matrix_time_homogeneous:
                self._lu_cache[key] = lu
        return M, lu, s_new, g

    def _step(
        self,
        u: np.ndarray,
        tau_old: float,
        tau_new: float,
        theta: float,
        lcp: Optional[LCPStepper],
        obstacle: Optional[Obstacle],
    ) -> np.ndarray:
        op = self.operator
        dt = tau_new - tau_old
        M, lu, s_new, g = self._system(tau_new, dt, theta)
        rhs = u.copy()
        if theta < 1.0:
            L_old, s_old = self._assemble(tau_old)
            rhs += (1.0 - theta) * dt * (L_old @ u + s_old)
        rhs += theta * dt * s_new
        rows = op.constraint_rows
        rhs[rows] = g[rows]
        if lcp is None or obstacle is None:
            return lu.solve(rhs)
        gvals = self._obstacle(obstacle, tau_new)
        free = np.ones(op.n, dtype=bool)
        free[rows] = False
        return lcp.step(M, lu.solve, rhs, gvals, u, dt, free)

    def _obstacle(self, obstacle: Obstacle, tau: float) -> np.ndarray:
        x = self.operator.grid.nodes
        vals = obstacle(x, tau) if callable(obstacle) else obstacle
        out = np.broadcast_to(np.asarray(vals, dtype=float), x.shape)
        return np.array(out)

    # ------------------------------------------------------------------ public API
    def solve(
        self,
        initial: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
        time_grid: TimeGrid,
        events: Optional[Union[Mapping[float, Any], Iterable[tuple[float, Any]]]] = None,
        lcp: Optional[LCPStepper] = None,
        obstacle: Optional[Obstacle] = None,
        store_history: bool = False,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    ) -> ThetaResult:
        r"""March the solution from :math:`\tau = 0` to :math:`\tau = T`.

        Parameters
        ----------
        initial : array or callable
            Terminal payoff :math:`u(x, 0)` on the grid nodes, or ``f(x)``.
        time_grid : TimeGrid
            Time levels; event times must be grid times.
        events : mapping or iterable of (tau, hook), optional
            Jump conditions ``hook(u, x, tau) -> u`` applied at ``tau``.
        lcp : LCPStepper, optional
            Complementarity solver used when ``obstacle`` is given.
        obstacle : array or callable, optional
            Lower obstacle :math:`g(x, \tau)` (e.g. the exercise value).
        store_history : bool
            Keep every time level in :attr:`ThetaResult.history`.
        callback : callable, optional
            ``callback(k, tau_k, u_k)`` invoked after every time level.
        """
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid")
        if (lcp is None) != (obstacle is None):
            raise ValueError("lcp and obstacle must be given together")
        op = self.operator
        x = op.grid.nodes
        u = np.asarray(initial(x) if callable(initial) else initial, dtype=float).copy()
        if u.shape != x.shape:
            raise ValueError("initial values must have one entry per grid node")
        event_list = _normalise_events(events)
        times = time_grid.times
        event_at: dict[int, list[EventHook]] = {}
        for tau_e, hooks in event_list:
            j = int(np.argmin(np.abs(times - tau_e)))
            if j not in time_grid.event_indices or not np.isclose(times[j], tau_e):
                raise ValueError(f"event tau={tau_e} is not an event time of the time grid")
            event_at.setdefault(j, []).extend(hooks)
        if lcp is not None:
            lcp.reset(op.n)
        history = [u.copy()] if store_history else None
        if callback is not None:
            callback(0, 0.0, u)
        prev = u.copy()
        for k in range(1, times.size):
            prev = u
            u = self._step(u, times[k - 1], times[k], self.theta, lcp, obstacle)
            for hook in event_at.get(k, []):
                u = np.asarray(hook(u, x, float(times[k])), dtype=float)
                if u.shape != x.shape:
                    raise ValueError("event hooks must preserve the grid shape")
            if history is not None:
                history.append(u.copy())
            if callback is not None:
                callback(k, float(times[k]), u)
        info: dict[str, Any] = {
            "theta": self.theta,
            "n_steps": time_grid.n_steps,
            "n_factorisations": self._n_factorisations,
            "n_events": sum(len(v) for v in event_at.values()),
        }
        return ThetaResult(
            grid=op.grid,
            time_grid=time_grid,
            values=u,
            previous_values=prev,
            history=None if history is None else np.array(history),
            info=info,
        )
