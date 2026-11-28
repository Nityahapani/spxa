r"""
Posterior probabilities over structures, and model averaging.

Given candidate structures :math:`M_1, \dots, M_m` fitted to the same bins, with
evidences :math:`p(D\mid M_i)` (:func:`~spxa.discovery.bayes.laplace_evidence`)
and prior probabilities :math:`\pi_i`,

.. math::
    P(M_i \mid D) = \frac{\pi_i\, p(D\mid M_i)}{\sum_j \pi_j\, p(D\mid M_j)}.

:func:`structure_posterior` returns these as a :class:`StructurePosterior`,
which also averages predictions over structures, and over each structure's
parameter uncertainty if asked:

- :meth:`~StructurePosterior.predictive` gives the posterior predictive law of
  :math:`X_t`. It is the mixture
  :math:`\sum_i P(M_i\mid D)\, p(x \mid M_i, D)`. Each component is the plug-in
  law at the mode, or an average over ``n_draws`` Laplace posterior draws.
- :meth:`~StructurePosterior.option_prices` gives model-averaged European
  prices with their posterior standard deviation. By the law of total
  variance, that is the within-structure parameter spread plus the spread
  between structures.

Unlike information-criterion weights, these probabilities use each
structure's actual posterior volume, through the Laplace determinant and the
prior. They remain conditional on the candidate set: a structure that was
never fitted has probability zero.

The Laplace evidence can be off by a fraction of a nat when a structure is
weakly identified, as with a rare-jump part on data without jumps. Pass
``evidence="importance"`` to replace each Laplace value with the unbiased
importance-sampling estimate (:func:`~spxa.discovery.bayes.importance_evidence`).
Its Monte Carlo error then propagates to the probabilities: with
:math:`P_i = \operatorname{softmax}_i(\log\pi + \log Z)` and independent
errors :math:`s_j` in :math:`\log \hat Z_j`, the delta method gives

.. math::
    \operatorname{se}(P_i)^2 = \sum_j \big(P_i(\delta_{ij} - P_j)\big)^2 s_j^2 .
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from scipy import optimize
from scipy.special import logsumexp

from spxa.discovery.bayes import (
    Evidence,
    GaussianPrior,
    ImportanceEvidence,
    _coerce_prior,
    importance_evidence,
    laplace_evidence,
)
from spxa.discovery.density import COSDistribution, model_distribution
from spxa.discovery.expr import Model, Node
from spxa.discovery.fit import BinnedData, FitResult
from spxa.discovery.pricing import cos_prices

__all__ = [
    "StructurePosterior",
    "structure_posterior",
    "MixtureDistribution",
    "PosteriorPrices",
]

ModelPrior = Union[None, Mapping[str, float], Callable[[Node], float]]


# ── mixtures ──────────────────────────────────────────────────────────────────


class MixtureDistribution:
    r"""A finite mixture :math:`\sum_k w_k F_k` of COS laws.

    Parameters
    ----------
    weights :
        Non-negative weights; normalised to sum to one.
    components :
        One :class:`~spxa.discovery.density.COSDistribution` per weight.
    cumulants :
        :math:`(\kappa_1, \kappa_2)` of each component, shape ``(k, 2)``, from
        which the mixture's mean and variance are computed exactly.
    """

    def __init__(
        self,
        weights: Sequence[float],
        components: Sequence[COSDistribution],
        cumulants: np.ndarray,
    ) -> None:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.size != len(components) or w.size == 0:
            raise ValueError("one weight per component is needed")
        if np.any(w < 0) or not w.sum() > 0:
            raise ValueError("weights must be non-negative and not all zero")
        k = np.asarray(cumulants, dtype=float)
        if k.shape != (w.size, 2):
            raise ValueError(f"cumulants must have shape ({w.size}, 2), got {k.shape}")
        self.weights = w / w.sum()
        self.components = list(components)
        self._k = k

    def pdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.asarray(sum(w * c.pdf(x) for w, c in zip(self.weights, self.components)))

    def cdf(self, x: Any) -> np.ndarray:
        """The mixture CDF, each component clipped to ``[0, 1]`` (COS ripple)."""
        x = np.asarray(x, dtype=float)
        return np.asarray(
            sum(w * np.clip(c.cdf(x), 0.0, 1.0) for w, c in zip(self.weights, self.components))
        )

    @property
    def support(self) -> tuple[float, float]:
        """The union of the components' COS truncation intervals."""
        return min(c.a for c in self.components), max(c.b for c in self.components)

    @property
    def mean(self) -> float:
        return float(self.weights @ self._k[:, 0])

    @property
    def variance(self) -> float:
        r"""Law of total variance: :math:`\sum w_k(\kappa_{2,k} + \kappa_{1,k}^2) - \bar\mu^2`."""
        m = self.mean
        return float(self.weights @ (self._k[:, 1] + self._k[:, 0] ** 2) - m * m)

    def quantile(self, q: Any) -> np.ndarray:
        """Quantiles by root-finding on the CDF over :attr:`support`."""
        qs = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((qs <= 0) | (qs >= 1)):
            raise ValueError("quantile levels must lie strictly between 0 and 1")
        lo, hi = self.support
        out = np.array(
            [
                optimize.brentq(
                    lambda x, p=p: float(np.ravel(self.cdf(x))[0]) - p, lo, hi, xtol=1e-14
                )
                for p in qs
            ]
        )
        return out if np.ndim(q) else out[0]


@dataclass(frozen=True)
class PosteriorPrices:
    """Model-averaged option prices.

    Attributes
    ----------
    mean :
        Posterior mean price per strike.
    sd :
        Posterior standard deviation per strike (within plus between structures).
    by_structure :
        Each structure's own posterior mean, shape ``(m, n_strikes)``.
    probabilities :
        The structure probabilities used, renormalised over the structures that
        could be priced.
    discarded :
        Per structure, the number of parameter draws without a risk-neutral price
        (:math:`E[e^{X_1}] = \\infty`), which were left out.
    """

    mean: np.ndarray
    sd: np.ndarray
    by_structure: np.ndarray
    probabilities: np.ndarray
    discarded: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


# ── the posterior ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StructurePosterior:
    """Posterior probabilities of candidate structures (see the module docstring).

    Attributes
    ----------
    evidences :
        One :class:`~spxa.discovery.bayes.Evidence` per structure, in
        decreasing posterior probability.
    probabilities :
        :math:`P(M_i\\mid D)`, aligned with ``evidences``.
    log_model_prior :
        :math:`\\log\\pi_i`, normalised over the structures kept.
    bic :
        Each structure's BIC from its maximum-likelihood fit, for comparison.
    skipped :
        ``(structure key, reason)`` for candidates without a Laplace
        approximation. They are left out of the normalisation.
    dt :
        Sampling interval of the data (the default prediction horizon).
    log_evidences :
        The :math:`\\log p(D\\mid M_i)` behind :attr:`probabilities`: the
        Laplace values, or the importance-sampling estimates.
    standard_errors :
        Monte Carlo standard errors of :attr:`log_evidences` (zero for Laplace).
    method :
        ``"laplace"`` or ``"importance"``.
    importance :
        The :class:`~spxa.discovery.bayes.ImportanceEvidence` per structure when
        ``method == "importance"``, else empty.
    """

    evidences: list[Evidence]
    probabilities: np.ndarray
    log_model_prior: np.ndarray
    bic: np.ndarray
    skipped: list[tuple[str, str]]
    dt: float
    log_evidences: np.ndarray = field(default_factory=lambda: np.zeros(0))
    standard_errors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    method: str = "laplace"
    importance: list[ImportanceEvidence] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.log_evidences.size == 0:
            object.__setattr__(
                self, "log_evidences", np.array([e.log_evidence for e in self.evidences])
            )
        if self.standard_errors.size == 0:
            object.__setattr__(self, "standard_errors", np.zeros(len(self.evidences)))

    @property
    def best(self) -> Evidence:
        return self.evidences[0]

    @property
    def keys(self) -> list[str]:
        return [e.structure.key() for e in self.evidences]

    def probability(self, structure: Union[str, Node]) -> float:
        """Posterior probability of one structure (by key or node), 0 if absent."""
        key = structure if isinstance(structure, str) else structure.key()
        for k, p in zip(self.keys, self.probabilities):
            if k == key:
                return float(p)
        return 0.0

    def bic_weights(self) -> np.ndarray:
        """BIC weights of the same structures, aligned with :attr:`evidences`."""
        d = self.bic - np.min(self.bic)
        w = np.exp(-0.5 * d)
        return np.asarray(w / w.sum())

    def bayes_factor(self, i: int = 0, j: int = 1) -> float:
        """:math:`\\log B_{ij} = \\log p(D\\mid M_i) - \\log p(D\\mid M_j)` (from :attr:`log_evidences`)."""
        return float(self.log_evidences[i] - self.log_evidences[j])

    @property
    def probability_se(self) -> np.ndarray:
        """Monte Carlo standard errors of :attr:`probabilities` (delta method; zero for Laplace)."""
        P = self.probabilities
        J = np.diag(P) - np.outer(P, P)  # dP_i / dlogZ_j
        return np.asarray(np.sqrt((J**2) @ (self.standard_errors**2)))

    def credible_set(self, level: float = 0.95) -> list[Evidence]:
        """The fewest most-probable structures whose probabilities sum to at least ``level``."""
        if not 0 < level <= 1:
            raise ValueError("level must lie in (0, 1]")
        c = np.cumsum(self.probabilities)
        n = int(np.searchsorted(c, level - 1e-12) + 1)
        return self.evidences[: min(n, len(self.evidences))]

    def table(self, top: int = 10, digits: int = 4) -> str:
        mc = self.method == "importance"
        head = f"{'#':>3}  {'P(M|D)':>8}  {'BIC wt':>8}  {'log Z':>11}"
        head += f"  {'se':>6}" if mc else ""
        rows = [head + f"  {'Occam':>8}  structure"]
        bw = self.bic_weights()
        for i, (e, p) in enumerate(zip(self.evidences[:top], self.probabilities[:top])):
            row = f"{i + 1:>3}  {p:8.4f}  {bw[i]:8.4f}  {self.log_evidences[i]:11.3f}"
            row += f"  {self.standard_errors[i]:6.3f}" if mc else ""
            rows.append(row + f"  {e.log_occam_factor:8.3f}  {e.model.render(digits)}")
        pad = 8 if mc else 0
        for key, why in self.skipped:
            rows.append(f"  -  {'':8}  {'':8}  {'':11}  {'':{pad}}{'':8}  {key}: skipped ({why})")
        return "\n".join(rows)

    def __str__(self) -> str:
        return self.table()

    # -- averaging ---------------------------------------------------------

    def _parameter_sets(
        self, n_draws: int, rng: Optional[np.random.Generator]
    ) -> list[list[Model]]:
        """Per structure: the mode (``n_draws = 0``) or valid Laplace draws."""
        if n_draws < 0:
            raise ValueError("n_draws must be non-negative")
        rng = rng if rng is not None else np.random.default_rng(0)
        out: list[list[Model]] = []
        for e in self.evidences:
            if n_draws == 0:
                out.append([e.model])
                continue
            models: list[Model] = []
            for x in e.draws(n_draws, rng):
                try:
                    models.append(Model.from_free(e.structure, x))
                except (ValueError, FloatingPointError, OverflowError):
                    continue
            out.append(models if models else [e.model])
        return out

    def predictive(
        self,
        t: Optional[float] = None,
        *,
        n_draws: int = 0,
        rng: Optional[np.random.Generator] = None,
        n_terms: int = 1024,
        min_probability: float = 1e-6,
    ) -> MixtureDistribution:
        r"""Posterior predictive law of :math:`X_t` (default: one sampling interval).

        With ``n_draws = 0`` each structure contributes its law at the
        posterior mode. Otherwise it contributes the average over ``n_draws``
        draws from its Laplace posterior. Structures below ``min_probability``
        are dropped and the rest renormalised.
        """
        horizon = self.dt if t is None else float(t)
        if not horizon > 0:
            raise ValueError("the horizon must be positive")
        keep = self.probabilities >= min_probability
        sets = self._parameter_sets(n_draws, rng)
        weights: list[float] = []
        comps: list[COSDistribution] = []
        cums: list[np.ndarray] = []
        for p, models, k in zip(self.probabilities, sets, keep):
            if not k:
                continue
            for m in models:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    comps.append(model_distribution(m, horizon, n_terms=n_terms))
                    cums.append(m.cumulants(2, t=horizon))
                weights.append(float(p) / len(models))
        return MixtureDistribution(weights, comps, np.array(cums))

    def cumulants(self, order: int = 4, t: float = 1.0) -> np.ndarray:
        r"""Posterior-mean cumulants :math:`\sum_i P(M_i\mid D)\,\kappa_n(\hat M_i)` at the modes.

        This averages the cumulants themselves, which is not the cumulant
        sequence of the mixture law. Use :meth:`predictive` for that law.
        """
        ks = np.array([e.model.cumulants(order, t=t) for e in self.evidences])
        return np.asarray(self.probabilities @ ks)

    def option_prices(
        self,
        spot: float,
        strikes: Any,
        maturity: float,
        rate: float = 0.0,
        dividend: float = 0.0,
        call: Any = True,
        *,
        n_draws: int = 0,
        rng: Optional[np.random.Generator] = None,
        n_terms: int = 1024,
    ) -> PosteriorPrices:
        r"""Model-averaged European prices (:func:`~spxa.discovery.pricing.cos_prices`).

        Each structure is made risk-neutral by its own martingale correction.
        With draws, a draw that has no finite :math:`E[e^{X_1}]` is discarded
        and counted. The mean and standard deviation follow the law of total
        expectation and variance over structures and draws.
        """
        sets = self._parameter_sets(n_draws, rng)
        means: list[np.ndarray] = []
        second: list[np.ndarray] = []
        probs: list[float] = []
        discarded: list[int] = []
        for p, models in zip(self.probabilities, sets):
            vals = []
            for m in models:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        vals.append(
                            cos_prices(
                                m, spot, strikes, maturity, rate, dividend, call, n_terms=n_terms
                            )
                        )
                except (ValueError, FloatingPointError, OverflowError):
                    continue
            discarded.append(len(models) - len(vals))
            if not vals:
                continue
            v = np.array(vals)
            means.append(v.mean(axis=0))
            second.append((v**2).mean(axis=0))
            probs.append(float(p))
        if not probs:
            raise ValueError("no candidate structure has a risk-neutral price")
        w = np.array(probs) / np.sum(probs)
        M = np.array(means)
        mean = w @ M
        var = np.maximum(w @ np.array(second) - mean**2, 0.0)
        return PosteriorPrices(
            mean=mean,
            sd=np.sqrt(var),
            by_structure=M,
            probabilities=w,
            discarded=np.array(discarded, dtype=int),
        )


def _log_model_prior(model_prior: ModelPrior, structures: Sequence[Node]) -> np.ndarray:
    if model_prior is None:
        raw = np.ones(len(structures))
    elif callable(model_prior):
        raw = np.array([float(model_prior(s)) for s in structures])
    else:
        raw = np.array([float(model_prior.get(s.key(), 0.0)) for s in structures])
    if np.any(raw < 0) or not np.all(np.isfinite(raw)):
        raise ValueError("model prior weights must be finite and non-negative")
    if not raw.sum() > 0:
        raise ValueError("the model prior gives every candidate weight zero")
    with np.errstate(divide="ignore"):
        return np.asarray(np.log(raw / raw.sum()))


def structure_posterior(
    candidates: Any,
    binned: Optional[BinnedData] = None,
    *,
    prior: Union[None, GaussianPrior, Callable[[Node], GaussianPrior]] = None,
    model_prior: ModelPrior = None,
    top: Optional[int] = None,
    evidence: Literal["laplace", "importance"] = "laplace",
    n_draws: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> StructurePosterior:
    """Posterior probabilities of fitted structures.

    Parameters
    ----------
    candidates :
        A likelihood-mode :class:`~spxa.discovery.report.DiscoveryReport`, or a
        sequence of likelihood :class:`~spxa.discovery.fit.FitResult` fitted to
        ``binned``.
    binned :
        The common bins (taken from the report when one is given).
    prior :
        Parameter prior: one :class:`~spxa.discovery.bayes.GaussianPrior`
        (only sensible for a single structure), a callable
        ``structure -> GaussianPrior``, or ``None`` for
        :func:`~spxa.discovery.bayes.default_prior`.
    model_prior :
        Prior over structures: ``None`` (uniform), a mapping from structure key
        (with drift, e.g. ``"BM + Drift"``) to a non-negative weight, or a
        callable ``structure -> weight``. Weights are normalised over the
        candidates.
    top :
        Use only the ``top`` best candidates by BIC.
    evidence :
        ``"laplace"`` (default) or ``"importance"``. The latter refines each
        Laplace evidence by importance sampling with ``n_draws`` draws from
        ``rng`` and reports Monte Carlo errors (:attr:`StructurePosterior.probability_se`).
    """
    from spxa.discovery.report import DiscoveryReport

    if evidence not in ("laplace", "importance"):
        raise ValueError(f"evidence must be 'laplace' or 'importance', got {evidence!r}")
    if isinstance(candidates, DiscoveryReport):
        if candidates.method != "likelihood" or candidates.binned is None:
            raise ValueError("posterior probabilities need a likelihood-mode report")
        fits = list(candidates.candidates)
        binned = candidates.binned if binned is None else binned
    else:
        fits = list(candidates)
    if binned is None:
        raise ValueError("pass the bins the candidates were fitted on")
    if not fits:
        raise ValueError("no candidates")
    for f in fits:
        if not isinstance(f, FitResult) or f.loglik is None:
            raise ValueError("every candidate must be a likelihood FitResult")
    fits.sort(key=lambda f: float(f.bic) if f.bic is not None else np.inf)
    if top is not None:
        fits = fits[: max(int(top), 1)]

    evidences: list[Evidence] = []
    bics: list[float] = []
    skipped: list[tuple[str, str]] = []
    for f in fits:
        s = f.model.structure
        try:
            ev = laplace_evidence(f, binned, prior=_coerce_prior(prior, s, binned))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            skipped.append((s.key(), str(exc)))
            continue
        evidences.append(ev)
        bics.append(float(f.bic) if f.bic is not None else np.nan)
    if not evidences:
        raise ValueError("no candidate has a Laplace approximation: " + "; ".join(skipped[0]))

    log_z = np.array([e.log_evidence for e in evidences])
    se = np.zeros(len(evidences))
    refined: list[ImportanceEvidence] = []
    if evidence == "importance":
        rng = rng if rng is not None else np.random.default_rng(0)
        for i, ev in enumerate(evidences):
            imp = importance_evidence(ev, binned, n_draws=n_draws, rng=rng)
            refined.append(imp)
            log_z[i], se[i] = imp.log_evidence, imp.standard_error
        if not np.all(np.isfinite(log_z)):
            bad = [e.structure.key() for e, z in zip(evidences, log_z) if not np.isfinite(z)]
            raise ValueError(f"importance sampling found no mass for {bad}")

    log_pi = _log_model_prior(model_prior, [e.structure for e in evidences])
    log_post = log_pi + log_z
    probs = np.exp(log_post - logsumexp(log_post))
    order = np.argsort(-probs, kind="stable")
    return StructurePosterior(
        evidences=[evidences[i] for i in order],
        probabilities=probs[order],
        log_model_prior=log_pi[order],
        bic=np.array(bics)[order],
        skipped=skipped,
        dt=float(binned.dt),
        log_evidences=log_z[order],
        standard_errors=se[order],
        method=evidence,
        importance=[refined[i] for i in order] if refined else [],
    )
