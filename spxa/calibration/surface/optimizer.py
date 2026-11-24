r"""Optimization engines and parameter transformation adapters for surface calibration.

Provides:
- LeastSquaresOptimizer: Damped Gauss-Newton / Levenberg-Marquardt / Trust Region Reflective (TRF)
- DifferentialEvolutionOptimizer: Stochastic global search for non-convex volatility landscapes
- HybridOptimizer: Global genetic exploration followed by machine-precision local polishing
- ParameterTransform: Logarithmic and logit mappings to enforce positivity and interval constraints
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize


@dataclass
class OptimizationResult:
    """Standardized result returned by calibration optimizers."""

    params: np.ndarray
    cost: float
    success: bool
    nit: int
    message: str
    residuals: Optional[np.ndarray] = None
    jacobian: Optional[np.ndarray] = None


class ParameterTransform:
    """Maps bounded parameters theta in [lb, ub] to unconstrained space in R."""

    def __init__(self, bounds: Sequence[Tuple[float, float]]) -> None:
        self.bounds = [tuple(b) for b in bounds]

    def to_unconstrained(self, params: np.ndarray) -> np.ndarray:
        """Map constrained params to unbounded real space."""
        p = np.asarray(params, dtype=float)
        uncon = np.empty_like(p)
        for i, (lb, ub) in enumerate(self.bounds):
            val = np.clip(p[i], lb + 1e-9, ub - 1e-9)
            # Logit transform: ln((x - lb) / (ub - x))
            normalized = (val - lb) / (ub - lb)
            uncon[i] = np.log(normalized / (1.0 - normalized))
        return uncon

    def to_constrained(self, uncon_params: np.ndarray) -> np.ndarray:
        """Map unbounded real vector back to constrained box [lb, ub]."""
        u = np.asarray(uncon_params, dtype=float)
        con = np.empty_like(u)
        for i, (lb, ub) in enumerate(self.bounds):
            # Sigmoid / logistic: 1 / (1 + exp(-u))
            sig = 1.0 / (1.0 + np.exp(-np.clip(u[i], -30.0, 30.0)))
            con[i] = lb + (ub - lb) * sig
        return con


class SurfaceOptimizer(abc.ABC):
    """Abstract base optimizer for calibration tasks."""

    @abc.abstractmethod
    def minimize(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """Run optimization problem given vector residual function."""
        raise NotImplementedError


class LeastSquaresOptimizer(SurfaceOptimizer):
    """Local bounded least-squares optimizer using Trust Region Reflective (TRF) algorithm."""

    def __init__(
        self,
        max_nfev: int = 500,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        method: str = "trf",
    ) -> None:
        self.max_nfev = max_nfev
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.method = method

    def minimize(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizationResult:
        """Bounded least squares; ``jac`` (residual Jacobian, e.g. from
        :mod:`spxa.graph.autodiff`) replaces SciPy's 2-point differences."""
        p0 = np.asarray(x0, dtype=float)
        if bounds is not None:
            lbs = [b[0] for b in bounds]
            ubs = [b[1] for b in bounds]
            scipy_bounds = (lbs, ubs)
            p0 = np.clip(p0, [lb + 1e-6 for lb in lbs], [ub - 1e-6 for ub in ubs])
        else:
            scipy_bounds = (-np.inf, np.inf)

        res = optimize.least_squares(
            residual_fn,
            p0,
            jac=jac if jac is not None else "2-point",
            bounds=scipy_bounds,
            method=self.method,
            max_nfev=self.max_nfev,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
        )

        return OptimizationResult(
            params=res.x,
            cost=float(res.cost),
            success=bool(res.success),
            nit=int(getattr(res, "nit", res.nfev)),
            message=str(res.message),
            residuals=res.fun,
            jacobian=res.jac if hasattr(res, "jac") else None,
        )


class DifferentialEvolutionOptimizer(SurfaceOptimizer):
    """Stochastic global optimizer for non-convex parameter spaces."""

    def __init__(
        self,
        maxiter: int = 150,
        popsize: int = 15,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        seed: Optional[int] = 42,
    ) -> None:
        self.maxiter = maxiter
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.seed = seed

    def minimize(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        if bounds is None:
            raise ValueError("DifferentialEvolutionOptimizer requires explicit parameter bounds.")

        def scalar_cost(p: np.ndarray) -> float:
            r = residual_fn(p)
            return float(np.sum(r**2))

        res = optimize.differential_evolution(
            scalar_cost,
            bounds=bounds,
            maxiter=self.maxiter,
            popsize=self.popsize,
            mutation=self.mutation,
            recombination=self.recombination,
            seed=self.seed,
            x0=x0,
        )

        final_residuals = residual_fn(res.x)
        return OptimizationResult(
            params=res.x,
            cost=float(res.fun),
            success=bool(res.success),
            nit=int(res.nit),
            message=str(res.message),
            residuals=final_residuals,
        )


class HybridOptimizer(SurfaceOptimizer):
    """2-Stage Global-to-Local Optimizer.

    Stage 1: Global exploration via Differential Evolution.
    Stage 2: Local polish via bounded Least Squares (TRF) to achieve machine-level convergence.
    """

    def __init__(
        self,
        global_maxiter: int = 50,
        local_max_nfev: int = 300,
        seed: Optional[int] = 42,
    ) -> None:
        self.global_opt = DifferentialEvolutionOptimizer(maxiter=global_maxiter, seed=seed)
        self.local_opt = LeastSquaresOptimizer(max_nfev=local_max_nfev)

    def minimize(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        # Stage 1: Global search
        global_res = self.global_opt.minimize(residual_fn, x0, bounds)

        # Stage 2: Local polish starting from best global candidate
        local_res = self.local_opt.minimize(residual_fn, global_res.params, bounds)

        # Pick the best result
        if local_res.cost <= global_res.cost:
            return local_res
        return global_res
