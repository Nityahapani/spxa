"""Process discovery: search the space of process compositions that explain data.

Inverse stochastic algebra. Given returns, or their summary statistics,
:func:`discover` searches algebraic combinations of Lévy primitives
(sums, scalings, subordination) and ranks those that explain the data:

>>> from spxa.discovery import discover                      # doctest: +SKIP
>>> report = discover(returns, dt=1 / 252)                    # doctest: +SKIP
>>> print(report.headline())                                  # doctest: +SKIP
VarianceGamma(sigma=0.129, nu=0.181, theta=-0.208) + 0.0631 * GammaProcess(a=4.36, b=1) + ...

Building blocks
---------------
Series
    :class:`CGFSeries`, truncated cumulant generating functions and their algebra.
Families
    :class:`Family` and the primitive families (:data:`VARIANCE_GAMMA`, ...).
Structures
    :class:`Leaf`, :class:`Sum`, :class:`Scale`, :class:`Subordinate` and the
    bound :class:`Model`; :func:`recognize` maps an spxa process to a model.
Data
    :func:`load_returns` reads returns or prices from a CSV file;
    :func:`summarize` and :func:`from_statistics` build an :class:`EmpiricalSummary`.
Fitting
    :func:`fit_structure` (smoothed binned likelihood or cumulants), :class:`BinnedData`.
Inference
    :func:`parameter_uncertainty` (sandwich or GMM standard errors) and
    :func:`vuong_test` for non-nested structures.
Search
    :class:`Grammar`, :func:`discover` and the :class:`DiscoveryReport`.
"""

from spxa.discovery.density import COSDistribution, model_distribution
from spxa.discovery.empirical import (
    EmpiricalSummary,
    acf,
    cumulant_covariance,
    from_statistics,
    k_statistics,
    ljung_box,
    rank_ljung_box,
    summarize,
)
from spxa.discovery.expr import Leaf, Model, Node, Scale, Subordinate, Sum, leaf
from spxa.discovery.families import (
    BROWNIAN,
    CGMY_FAMILY,
    DOUBLE_EXP_JUMPS,
    DRIFT,
    FAMILIES,
    GAMMA,
    INVERSE_GAUSSIAN,
    MEIXNER,
    NIG_FAMILY,
    NORMAL_JUMPS,
    POISSON,
    VARIANCE_GAMMA,
    Family,
    get_family,
)
from spxa.discovery.fit import BinnedData, FitResult, fit_structure, with_location
from spxa.discovery.inference import (
    ParameterUncertainty,
    VuongResult,
    parameter_uncertainty,
    vuong_test,
)
from spxa.discovery.io import ReturnSeries, infer_dt, load_returns
from spxa.discovery.grammar import Grammar, canonical, grammar_from_names, parse_atoms
from spxa.discovery.recognize import NotRecognizedError, recognize
from spxa.discovery.report import DiscoveryReport
from spxa.discovery.search import discover
from spxa.discovery.series import CGFSeries, cumulants_to_moments, moments_to_cumulants

__all__ = [
    # series
    "CGFSeries",
    "cumulants_to_moments",
    "moments_to_cumulants",
    # families
    "Family",
    "FAMILIES",
    "get_family",
    "DRIFT",
    "BROWNIAN",
    "GAMMA",
    "INVERSE_GAUSSIAN",
    "VARIANCE_GAMMA",
    "NIG_FAMILY",
    "MEIXNER",
    "POISSON",
    "NORMAL_JUMPS",
    "DOUBLE_EXP_JUMPS",
    "CGMY_FAMILY",
    # structures
    "Node",
    "Leaf",
    "leaf",
    "Sum",
    "Scale",
    "Subordinate",
    "Model",
    "recognize",
    "NotRecognizedError",
    # data
    "load_returns",
    "ReturnSeries",
    "infer_dt",
    "EmpiricalSummary",
    "summarize",
    "from_statistics",
    "k_statistics",
    "cumulant_covariance",
    "acf",
    "ljung_box",
    "rank_ljung_box",
    # densities
    "COSDistribution",
    "model_distribution",
    # fitting
    "BinnedData",
    "FitResult",
    "fit_structure",
    "with_location",
    # inference
    "ParameterUncertainty",
    "parameter_uncertainty",
    "VuongResult",
    "vuong_test",
    # search
    "Grammar",
    "canonical",
    "grammar_from_names",
    "parse_atoms",
    "discover",
    "DiscoveryReport",
]
