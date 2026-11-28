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
    :func:`vuong_test` for non-nested structures, :func:`bootstrap` for
    selection frequencies and parameter intervals.
Evidence
    :func:`laplace_evidence` (with the exact likelihood Hessian) and
    :func:`importance_evidence` compute the log evidence log p(D | M) under a
    :class:`GaussianPrior` on the free coordinates (:func:`default_prior`).
    :func:`smc_evidence` is a sequential Monte Carlo check for posteriors far
    from Gaussian. :func:`structure_posterior` turns them into posterior probabilities of the
    candidates (:class:`StructurePosterior`), with model-averaged predictive
    laws (:class:`MixtureDistribution`) and option prices (:class:`PosteriorPrices`).
Search
    :class:`Grammar`, :func:`discover` and the :class:`DiscoveryReport`.
Monitoring
    :class:`PageCUSUM` watches new increments for a change from a fitted law
    to a specified alternative, with a guaranteed in-control average run
    length (:class:`MonitorResult`, :class:`Alarm`).
Option prices
    :func:`cos_prices` prices European options with a fitted model as the
    risk-neutral driver (made a martingale by :func:`martingale_correction`).
    :class:`OptionQuotes`, :func:`fit_options` and :func:`discover_from_options`
    search for the structures that best reproduce quoted prices.
"""

from spxa.discovery.bootstrap import BootstrapResult, bootstrap
from spxa.discovery.bayes import (
    Evidence,
    GaussianPrior,
    ImportanceEvidence,
    default_prior,
    importance_evidence,
    laplace_evidence,
)
from spxa.discovery.density import COSDistribution, model_distribution
from spxa.discovery.monitor import Alarm, MonitorResult, PageCUSUM
from spxa.discovery.smc import SMCEvidence, smc_evidence
from spxa.discovery.posterior import (
    MixtureDistribution,
    PosteriorPrices,
    StructurePosterior,
    structure_posterior,
)
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
from spxa.discovery.options import OptionQuotes, discover_from_options, fit_options, model_prices
from spxa.discovery.pricing import cos_prices, martingale_correction
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
    "BootstrapResult",
    "bootstrap",
    # evidence
    "GaussianPrior",
    "default_prior",
    "Evidence",
    "laplace_evidence",
    "ImportanceEvidence",
    "importance_evidence",
    "SMCEvidence",
    "smc_evidence",
    "StructurePosterior",
    "structure_posterior",
    "MixtureDistribution",
    "PosteriorPrices",
    # monitoring
    "PageCUSUM",
    "MonitorResult",
    "Alarm",
    # search
    "Grammar",
    "canonical",
    "grammar_from_names",
    "parse_atoms",
    "discover",
    "DiscoveryReport",
    # option prices
    "cos_prices",
    "martingale_correction",
    "OptionQuotes",
    "model_prices",
    "fit_options",
    "discover_from_options",
]
