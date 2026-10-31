r"""
spxa.pde — finite-difference PDE and PIDE solvers for pricing and transition densities.

Every Markov process in spxa has a generator :math:`\mathcal{A}`; the
Feynman–Kac theorem turns expectations of the form

.. math::
    u(x, \tau) = \mathbb{E}\!\left[e^{-\int_0^\tau r(X_s)\,ds} g(X_\tau) \,\middle|\, X_0 = x\right]

into the backward equation :math:`u_\tau = \mathcal{A}u - r u`,
:math:`u(x, 0) = g(x)`, written here in time to maturity :math:`\tau`.
This subpackage discretises such equations with second-order accurate,
verified finite-difference schemes.
"""

from spxa.pde.american import (
    BrennanSchwartzLCP,
    IkonenToivanenLCP,
    PenaltyLCP,
    PSORLCP,
    complementarity_residual,
    make_lcp,
)
from spxa.pde.asian import AsianPDE, vecer_shares
from spxa.pde.black_scholes import BlackScholesPDE, DiscreteDividend
from spxa.pde.boundary import BoundaryCondition, Dirichlet, Linear, Natural, Neumann
from spxa.pde.convergence import (
    ConvergenceStudy,
    convergence_study,
    observed_order,
    observed_order_from_values,
    richardson_extrapolate,
)
from spxa.pde.fokker_planck import DensitySolution, FokkerPlanck1D
from spxa.pde.generator import DiffusionPDE, to_pde
from spxa.pde.pide import LevyJumpOperator, LevyPIDE
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.local_vol import (
    DupirePDE,
    DupireSurface,
    implied_volatility,
    local_vol_from_prices,
)
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D
from spxa.pde.payoffs import Payoff, asset_or_nothing, digital, project_payoff, vanilla
from spxa.pde.short_rate import ShortRatePDE
from spxa.pde.solution import ExerciseBoundary, PDEPricingResult, PDESolution1D
from spxa.pde.theta import LCPStepper, ThetaResult, ThetaSolver1D

__all__ = [
    "LevyJumpOperator",
    "LevyPIDE",
    "DiffusionPDE",
    "to_pde",
    "DensitySolution",
    "FokkerPlanck1D",
    "DupirePDE",
    "DupireSurface",
    "implied_volatility",
    "local_vol_from_prices",
    "ShortRatePDE",
    "AsianPDE",
    "vecer_shares",
    "BlackScholesPDE",
    "BrennanSchwartzLCP",
    "IkonenToivanenLCP",
    "PenaltyLCP",
    "PSORLCP",
    "complementarity_residual",
    "make_lcp",
    "BoundaryCondition",
    "ConvergenceStudy",
    "convergence_study",
    "observed_order",
    "observed_order_from_values",
    "richardson_extrapolate",
    "Dirichlet",
    "DiscreteDividend",
    "DiscreteOperator1D",
    "Linear",
    "LinearPDE1D",
    "Natural",
    "Neumann",
    "ExerciseBoundary",
    "PDEPricingResult",
    "PDESolution1D",
    "Payoff",
    "asset_or_nothing",
    "digital",
    "project_payoff",
    "vanilla",
    "Grid1D",
    "LCPStepper",
    "ThetaResult",
    "ThetaSolver1D",
    "TimeGrid",
]
