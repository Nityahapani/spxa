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

from spxa.pde.boundary import BoundaryCondition, Dirichlet, Linear, Natural, Neumann
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D
from spxa.pde.payoffs import Payoff, asset_or_nothing, digital, project_payoff, vanilla
from spxa.pde.solution import ExerciseBoundary, PDESolution1D
from spxa.pde.theta import LCPStepper, ThetaResult, ThetaSolver1D

__all__ = [
    "BoundaryCondition",
    "Dirichlet",
    "DiscreteOperator1D",
    "Linear",
    "LinearPDE1D",
    "Natural",
    "Neumann",
    "ExerciseBoundary",
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
