r"""Black–Scholes and local-volatility PDE pricers.

The value :math:`u(S, \tau)` of a claim on an asset with dynamics
:math:`dS = (r - q) S\,dt + \sigma(S, t) S\,dW` solves

.. math::
    u_\tau = \tfrac12 \sigma(S, T - \tau)^2 S^2 u_{SS} + (r - q) S u_S - r u,
    \qquad u(S, 0) = g(S).

Discretisation choices, all verified by the test-suite:

* **Grid.** Sinh-stretched in :math:`S` on :math:`[0, S_{\max}]`, clustered at
  the strike, with strikes and barriers aligned to nodes.
  :math:`S_{\max} = \max(S_0, K)\, e^{6 \sigma \sqrt{T}}` by default.
* **Boundaries.** At :math:`S = 0` the PDE degenerates to
  :math:`u_\tau = -r u` and is applied as is (Fichera: no condition may be
  imposed). At :math:`S_{\max}` the zero-gamma condition
  :math:`u_{SS} = 0` holds for every payoff that is asymptotically linear.
* **Payoff.** Cell averages of the payoff (Pooley–Vetzal–Forsyth smoothing).
* **Time stepping.** Crank–Nicolson with Rannacher start-up.

Together these give clean second-order convergence of prices and grid
Greeks.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from spxa.pde._utils import check_option_type, check_positive
from spxa.pde.boundary import BoundaryCondition, Linear, Natural
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D
from spxa.pde.payoffs import Payoff, digital, project_payoff, vanilla
from spxa.pde.solution import PDEPricingResult, PDESolution1D
from spxa.pde.theta import LCPStepper, ThetaSolver1D

Volatility = Union[float, Callable[[np.ndarray, float], np.ndarray]]


class BlackScholesPDE:
    r"""Finite-difference pricer for the Black–Scholes and local-volatility models.

    Parameters
    ----------
    r : float
        Risk-free rate.
    sigma : float or callable
        Constant volatility or local volatility ``sigma(S, t)`` with ``t``
        calendar time (vectorised in ``S``).
    q : float
        Continuous dividend yield.

    Examples
    --------
    >>> from spxa.pde import BlackScholesPDE
    >>> res = BlackScholesPDE(r=0.05, sigma=0.2).european(K=100, T=1.0, spot=100)
    >>> round(res.price, 3)
    10.451
    """

    def __init__(self, r: float, sigma: Volatility, q: float = 0.0) -> None:
        self.r = float(r)
        self.q = float(q)
        if not callable(sigma):
            check_positive("sigma", sigma)
            sigma = float(sigma)
        self.sigma = sigma

    # ------------------------------------------------------------------ model pieces
    @property
    def is_local_vol(self) -> bool:
        """True when the volatility is a function of ``(S, t)``."""
        return callable(self.sigma)

    def local_vol(self, s: np.ndarray, t: float) -> np.ndarray:
        """Volatility on the nodes ``s`` at calendar time ``t``."""
        s = np.asarray(s, dtype=float)
        if callable(self.sigma):
            return np.broadcast_to(np.asarray(self.sigma(s, t), dtype=float), s.shape)
        return np.full(s.shape, self.sigma)

    def _sigma_ref(self, spot: float, T: float) -> float:
        if not callable(self.sigma):
            return float(self.sigma)
        pts = spot * np.array([0.5, 0.8, 1.0, 1.25, 2.0])
        vals = [np.max(self.local_vol(pts, t)) for t in (0.0, 0.5 * T, T)]
        return float(max(max(vals), 1e-3))

    def pde(self, T: float) -> LinearPDE1D:
        """The backward PDE in ``S`` for maturity ``T``."""
        r, q = self.r, self.q
        if callable(self.sigma):

            def diffusion(s: np.ndarray, tau: float) -> np.ndarray:
                return 0.5 * self.local_vol(s, T - tau) ** 2 * s**2

            return LinearPDE1D(
                diffusion=diffusion,
                convection=lambda s, tau: (r - q) * s,
                reaction=-r,
                time_homogeneous=False,
            )
        sig = float(self.sigma)
        return LinearPDE1D(
            diffusion=lambda s, tau: 0.5 * sig**2 * s**2,
            convection=lambda s, tau: (r - q) * s,
            reaction=-r,
            time_homogeneous=True,
        )

    def default_s_max(self, spot: float, strike: float, T: float, n_std: float = 6.0) -> float:
        """Upper truncation ``max(S0, K) exp(n_std * sigma * sqrt(T))``."""
        sig = self._sigma_ref(spot, T)
        return float(max(spot, strike) * math.exp(n_std * sig * math.sqrt(T)))

    # ------------------------------------------------------------------ engine
    def solve(
        self,
        payoff: Payoff,
        T: float,
        *,
        center: float,
        n_space: int = 401,
        n_time: int = 200,
        s_min: float = 0.0,
        s_max: Optional[float] = None,
        left: Optional[BoundaryCondition] = None,
        right: Optional[BoundaryCondition] = None,
        align: Sequence[float] = (),
        events: Optional[dict[float, Any]] = None,
        event_times: Sequence[float] = (),
        lcp: Optional[LCPStepper] = None,
        obstacle: Any = None,
        store_history: bool = False,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        theta: float = 0.5,
        rannacher_steps: int = 2,
        smoothing: str = "average",
        grid: str = "sinh",
        concentration: float = 1.0,
        convection_scheme: str = "auto",
        label: str = "Black-Scholes PDE",
    ) -> PDESolution1D:
        r"""Solve the pricing PDE for an arbitrary payoff.

        Parameters
        ----------
        payoff : Payoff
            Terminal payoff (its kinks are aligned to grid nodes).
        T : float
            Maturity.
        center : float
            Grid concentration point (usually the strike).
        n_space, n_time : int
            Number of spatial nodes and time steps.
        s_min, s_max : float
            Computational domain (``s_max`` defaults to
            :meth:`default_s_max`).
        left, right : BoundaryCondition, optional
            Boundary closures (default: natural at ``S = 0``, zero gamma at
            ``s_max``; a positive ``s_min`` requires an explicit ``left``).
        align : sequence of float
            Extra points (barriers) to place exactly on nodes.
        events : dict, optional
            Additional event hooks ``{tau: hook}``; their times must be in
            ``event_times`` (calendar time).
        event_times : sequence of float
            Calendar times to insert into the time grid.
        lcp, obstacle :
            Free-boundary solver and obstacle for American-style problems.
        smoothing : {'average', 'none'}
            Payoff projection.
        grid : {'sinh', 'uniform'}
            Spatial grid type; ``concentration`` sets the sinh scale
            ``alpha = concentration * center``.
        """
        T = check_positive("T", T)
        s_max = self.default_s_max(center, center, T) if s_max is None else float(s_max)
        if not s_min < center < s_max:
            raise ValueError("center must lie strictly inside (s_min, s_max)")
        if s_min > 0.0 and left is None:
            raise ValueError("a positive s_min requires an explicit left boundary condition")
        if grid == "sinh":
            g = Grid1D.sinh(s_min, s_max, n_space, center=center, alpha=concentration * center)
        elif grid == "uniform":
            g = Grid1D.uniform(s_min, s_max, n_space)
        else:
            raise ValueError("grid must be 'sinh' or 'uniform'")
        g = g.with_points(list(payoff.kinks) + [float(a) for a in align])
        left = Natural() if left is None else left
        right = Linear() if right is None else right
        op = DiscreteOperator1D(self.pde(T), g, left, right, convection_scheme)
        u0 = project_payoff(payoff, g, method=smoothing)
        all_times = sorted(set(float(t) for t in event_times))
        tg = TimeGrid.from_calendar_events(T, n_time, all_times)
        hooks: dict[float, list[Any]] = {}
        for tau, hk in (events or {}).items():
            hooks.setdefault(float(tau), []).extend(hk if isinstance(hk, list) else [hk])
        # match hook times to the exact grid times
        ev: dict[float, list[Any]] = {}
        for tau, hk in hooks.items():
            j = int(np.argmin(np.abs(tg.times - tau)))
            ev.setdefault(float(tg.times[j]), []).extend(hk)
        solver = ThetaSolver1D(op, theta=theta, rannacher_steps=rannacher_steps)
        res = solver.solve(
            u0,
            tg,
            events=ev or None,
            lcp=lcp,
            obstacle=obstacle,
            store_history=store_history,
            callback=callback,
        )
        info = {
            "pde": "u_tau = 1/2 sigma^2 S^2 u_SS + (r - q) S u_S - r u",
            "boundary": f"left {left.describe()}, right {right.describe()}",
            "smoothing": f"{smoothing} payoff projection",
        }
        return PDESolution1D.from_theta_result(res, variable="S", label=label, info=info)

    # ------------------------------------------------------------------ results
    def _result(
        self,
        sol: PDESolution1D,
        spot: float,
        method: str,
        bump: Optional[Callable[["BlackScholesPDE"], float]] = None,
        exercise_boundary: Any = None,
    ) -> PDEPricingResult:
        g = sol.greeks(spot)
        vega = rho = None
        if bump is not None:
            h_s, h_r = 1e-3, 1e-4
            vega = (bump(self._bumped(dsigma=h_s)) - bump(self._bumped(dsigma=-h_s))) / (2 * h_s)
            rho = (bump(self._bumped(dr=h_r)) - bump(self._bumped(dr=-h_r))) / (2 * h_r)
        return PDEPricingResult(
            price=g["value"],
            delta=g["delta"],
            gamma=g["gamma"],
            theta=g.get("theta", float("nan")),
            spot=float(spot),
            solution=sol,
            vega=vega,
            rho=rho,
            exercise_boundary=exercise_boundary,
            method=method,
        )

    def _bumped(self, dsigma: float = 0.0, dr: float = 0.0) -> "BlackScholesPDE":
        if callable(self.sigma):
            base = self.sigma

            def sig(s: np.ndarray, t: float) -> np.ndarray:
                return np.asarray(base(s, t), dtype=float) + dsigma

            return BlackScholesPDE(self.r + dr, sig, self.q)
        return BlackScholesPDE(self.r + dr, float(self.sigma) + dsigma, self.q)

    # ------------------------------------------------------------------ products
    def european(
        self,
        K: float,
        T: float,
        spot: float,
        option: str = "call",
        bump_greeks: bool = False,
        **kwargs: Any,
    ) -> PDEPricingResult:
        """European call or put.

        Extra keyword arguments are forwarded to :meth:`solve` (resolution,
        grid, ...). ``bump_greeks=True`` adds vega and rho by
        central bump-and-revalue (four extra solves).
        """
        opt = check_option_type(option)
        spot = check_positive("spot", spot)
        kwargs.setdefault("s_max", self.default_s_max(spot, K, T))
        kwargs.setdefault("label", f"European {opt} K={K:g} T={T:g}")
        sol = self.solve(vanilla(K, opt), T, center=K, **kwargs)
        bump = None
        if bump_greeks:

            def bump(model: "BlackScholesPDE") -> float:
                return float(model.solve(vanilla(K, opt), T, center=K, **kwargs).value(spot))

        return self._result(sol, spot, f"Black-Scholes PDE European {opt}", bump)

    def digital(
        self,
        K: float,
        T: float,
        spot: float,
        option: str = "call",
        cash: float = 1.0,
        **kwargs: Any,
    ) -> PDEPricingResult:
        """Cash-or-nothing digital option."""
        opt = check_option_type(option)
        spot = check_positive("spot", spot)
        kwargs.setdefault("s_max", self.default_s_max(spot, K, T))
        kwargs.setdefault("label", f"Digital {opt} K={K:g} T={T:g}")
        sol = self.solve(digital(K, opt, cash), T, center=K, **kwargs)
        return self._result(sol, spot, f"Black-Scholes PDE digital {opt}")
