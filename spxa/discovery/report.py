r"""
The result of a discovery search: ranked structures and their fit quality.

A :class:`DiscoveryReport` holds the fitted candidates in rank order and
renders them for people. It has a one-sentence :meth:`~DiscoveryReport.headline`
("X + 0.3 * Y matches your data within 0.4%") and a ranking
:meth:`~DiscoveryReport.table`.

Fit-quality measures
--------------------
CDF gap
    :attr:`~spxa.discovery.fit.FitResult.ks_distance`: the largest
    difference between the fitted and the empirical distribution function
    over the bin edges. It is the "within x%" of the headline when a sample
    is available.
goodness-of-fit p
    The Wald test of :meth:`~spxa.discovery.fit.BinnedData.gof_statistic`.
moment error
    :attr:`~spxa.discovery.fit.FitResult.match_error`: the largest error in
    the variance, skewness and excess kurtosis. It is the "within x%" when
    only summary statistics are available. From a sample it is reported but
    not used for ranking: sample kurtosis of heavy-tailed returns is far
    noisier than the likelihood.
weights
    Akaike or Schwarz weights :math:`w_i \propto e^{-\Delta_i/2}`, the
    relative support of each structure within the candidate set (Burnham &
    Anderson 2002, §2.9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

import numpy as np

from spxa.discovery.empirical import EmpiricalSummary
from spxa.discovery.fit import BinnedData, FitResult

if TYPE_CHECKING:
    from spxa.discovery.expr import Model
    from spxa.discovery.monitor import PageCUSUM, ShiryaevRoberts
    from spxa.discovery.posterior import StructurePosterior
    from spxa.discovery.inference import ParameterUncertainty, VuongResult


@dataclass
class DiscoveryReport:
    """Ranked candidate structures and how well they explain the data."""

    candidates: list[FitResult]
    summary: Optional[EmpiricalSummary]
    criterion: str
    method: str
    tolerance: float
    warnings: list[str] = field(default_factory=list)
    excluded: list[FitResult] = field(default_factory=list)
    binned: Optional[BinnedData] = None

    @property
    def best(self) -> FitResult:
        return self.candidates[0]

    def score(self, r: FitResult) -> float:
        if self.method == "cumulants":
            return _err(r)
        val = r.bic if self.criterion == "bic" else r.aic
        assert val is not None
        return float(val)

    def weights(self) -> np.ndarray:
        """Information-criterion weights of the candidates (likelihood mode only)."""
        if self.method == "cumulants":
            raise ValueError(
                "weights need a likelihood; this report was built from summary statistics"
            )
        s = np.array([self.score(r) for r in self.candidates])
        w = np.exp(-0.5 * (s - s.min()))
        return w / w.sum()

    def matching(self) -> list[FitResult]:
        """Candidates whose moment error (:attr:`FitResult.match_error`) is within ``tolerance``."""
        return [r for r in self.candidates if _err(r) <= self.tolerance]

    def headline(self, digits: int = 3) -> str:
        """One sentence: the best explanation and how well it fits.

        From a sample, "within x%" is the largest gap between the model CDF
        and the empirical CDF over the bin edges. From summary statistics it
        is the moment error, i.e. the largest error in the variance, skewness
        and excess kurtosis (see :attr:`FitResult.moment_errors`).
        """
        b = self.best
        expr = b.model.render(digits=digits, weighted="subordinators")
        if self.method == "options":
            text = (
                f"{expr} matches your option prices within "
                f"{100 * b.extras['iv_rmse']:.3g} vol points RMS (largest implied-volatility "
                f"error {100 * b.extras['iv_max_error']:.3g}; RMS relative price error "
                f"{100 * b.extras['price_rel_rmse']:.3g}%)"
            )
        elif b.ks_distance is not None:
            text = (
                f"{expr} matches your data within {100 * b.ks_distance:.1f}% "
                f"(largest gap between model and empirical CDFs; goodness-of-fit "
                f"p = {b.g_pvalue:.2f})"
            )
        else:
            text = (
                f"{expr} matches your data within {100 * _err(b):.1f}% "
                "(largest error in variance, skewness and excess kurtosis)"
            )
        if self.method != "cumulants" and len(self.candidates) > 1:
            gap = self.score(self.candidates[1]) - self.score(b)
            text += f"; {self.criterion.upper()} lead over the runner-up {gap:.1f}"
        return text

    def _pick(self, which: int | FitResult) -> FitResult:
        return self.candidates[which] if isinstance(which, int) else which

    def uncertainty(self, which: int | FitResult = 0) -> ParameterUncertainty:
        """Standard errors of a candidate's parameters (default: the best).

        Uses :func:`~spxa.discovery.inference.parameter_uncertainty` on the
        bins the search shared, so the result matches the fit exactly.
        """
        from spxa.discovery.inference import parameter_uncertainty

        if self.summary is None:
            raise ValueError("standard errors need a return sample or summary statistics")
        return parameter_uncertainty(self._pick(which), self.summary, self.binned)

    def compare(
        self,
        first: int | FitResult = 0,
        second: int | FitResult = 1,
        correction: Literal["none", "aic", "bic"] = "bic",
        level: float = 0.05,
    ) -> VuongResult:
        """Vuong test of two candidates on the search's bins (likelihood reports only)."""
        from spxa.discovery.inference import vuong_test

        if self.method != "likelihood" or self.binned is None:
            raise ValueError("the Vuong test needs a likelihood report built from a return sample")
        return vuong_test(self._pick(first), self._pick(second), self.binned, correction, level)

    def posterior(self, **kwargs: Any) -> StructurePosterior:
        """Posterior probabilities of the candidates (:func:`~spxa.discovery.posterior.structure_posterior`).

        Keyword arguments (``prior``, ``model_prior``, ``top``) are passed on.
        Needs a likelihood-mode report.
        """
        from spxa.discovery.posterior import structure_posterior

        return structure_posterior(self, **kwargs)

    def monitor(
        self,
        alternative: Optional[Model] = None,
        *,
        which: int | FitResult = 0,
        kl: float = 0.02,
        arl0: float = 10_000.0,
    ) -> PageCUSUM | ShiryaevRoberts:
        """A change detector for new increments, with the fitted law of candidate ``which`` as null.

        With an ``alternative``, the result is a :class:`~spxa.discovery.monitor.PageCUSUM`.
        Without one, it is a :class:`~spxa.discovery.monitor.ShiryaevRoberts` mixture over
        :func:`~spxa.discovery.monitor.local_alternatives` at divergence ``kl``.
        Either way the in-control average run length is at least ``arl0``.
        Needs a likelihood-mode report built from a return sample, and new increments
        on the same time step.
        """
        from spxa.discovery.monitor import PageCUSUM, ShiryaevRoberts, local_alternatives

        if self.method != "likelihood" or self.binned is None:
            raise ValueError("monitoring needs a likelihood report built from a return sample")
        null = self._pick(which).model
        if alternative is not None:
            return PageCUSUM(null, alternative, self.binned, arl0=arl0)
        return ShiryaevRoberts(
            null, local_alternatives(null, self.binned, kl=kl), self.binned, arl0=arl0
        )

    def admissible(self) -> list[FitResult]:
        """Candidates that are not :attr:`~spxa.discovery.fit.FitResult.degenerate`, in rank order."""
        return [r for r in self.candidates if not r.degenerate]

    def table(self, top: int = 10, digits: int = 3) -> str:
        """Plain-text ranking of the first ``top`` candidates.

        Rows marked ``!`` are degenerate fits (see
        :meth:`~spxa.discovery.fit.FitResult.degeneracies`).
        """
        text = self._table(top, digits)
        if any(r.degenerate for r in self.candidates[:top]):
            text += (
                "\n  ! degenerate fit: a limiting case of its structure (FitResult.degeneracies())"
            )
        return text

    def _table(self, top: int, digits: int) -> str:
        rows = []
        if self.method == "options":
            w = self.weights()
            best = self.score(self.best)
            name = self.criterion.upper()
            header = f"{'#':>3}  {'Δ' + name:>8}  {'weight':>6}  {'IV RMSE':>8}  {'IV max':>7}  expression"
            for i, r in enumerate(self.candidates[:top], 1):
                rows.append(
                    f"{i:>3}  {self.score(r) - best:8.2f}  {w[i - 1]:6.3f}  "
                    f"{100 * r.extras['iv_rmse']:7.2f}%  {100 * r.extras['iv_max_error']:6.2f}%  "
                    f"{_flag(r)}{r.model.render(digits, weighted='subordinators')}"
                )
            return "\n".join([header] + rows)
        if self.method == "cumulants":
            header = f"{'#':>3}  {'params':>6}  {'moment err':>10}  expression"
            for i, r in enumerate(self.candidates[:top], 1):
                rows.append(
                    f"{i:>3}  {r.n_params:>6}  {_pct(_err(r)):>10}  "
                    f"{_flag(r)}{r.model.render(digits, weighted='subordinators')}"
                )
        else:
            w = self.weights()
            best = self.score(self.best)
            name = self.criterion.upper()
            header = (
                f"{'#':>3}  {'Δ' + name:>8}  {'weight':>6}  {'p':>6}  {'CDF gap':>8}  "
                f"{'moment err':>10}  expression"
            )
            for i, r in enumerate(self.candidates[:top], 1):
                rows.append(
                    f"{i:>3}  {self.score(r) - best:8.2f}  {w[i - 1]:6.3f}  {r.g_pvalue or 0:6.3f}  "
                    f"{100 * (r.ks_distance or 0):7.2f}%  {_pct(_err(r)):>10}  "
                    f"{_flag(r)}{r.model.render(digits, weighted='subordinators')}"
                )
        return "\n".join([header] + rows)

    def to_dict(self, top: Optional[int] = None) -> dict[str, Any]:
        """JSON-ready summary: headline, warnings and the ranked candidates.

        Each candidate entry has its rendered expression, structure key,
        named parameters (see :func:`~spxa.discovery.inference.parameter_names`),
        criterion values, fit-quality measures and degeneracy reasons.
        Values that do not apply are ``None``. Non-finite floats become
        ``None``, so the output is valid JSON.
        """
        from spxa.discovery.inference import parameter_names

        weights: Optional[np.ndarray] = None
        if self.method != "cumulants":
            weights = self.weights()

        def num(v: Any) -> Optional[float]:
            if v is None:
                return None
            f = float(v)
            return f if np.isfinite(f) else None

        rows = []
        for i, r in enumerate(self.candidates[: top or len(self.candidates)]):
            flat = [float(v) for p in r.model.params for v in p]
            rows.append(
                {
                    "rank": i + 1,
                    "expression": r.model.render(weighted="subordinators"),
                    "structure": r.structure_key,
                    "parameters": dict(zip(parameter_names(r.model), flat)),
                    "n_params": r.n_params,
                    "loglik": num(r.loglik),
                    "aic": num(r.aic),
                    "bic": num(r.bic),
                    "weight": None if weights is None else float(weights[i]),
                    "gof_pvalue": num(r.g_pvalue),
                    "cdf_gap": num(r.ks_distance),
                    "moment_error": num(r.match_error),
                    "extras": {k: num(v) for k, v in r.extras.items()},
                    "converged": bool(r.converged),
                    "degeneracies": r.degeneracies(),
                }
            )
        return {
            "headline": self.headline(),
            "method": self.method,
            "criterion": self.criterion,
            "n_obs": None if self.summary is None else self.summary.n,
            "dt": None if self.summary is None else self.summary.dt,
            "warnings": list(self.warnings),
            "candidates": rows,
            "excluded": [r.structure_key for r in self.excluded],
        }

    def __str__(self) -> str:
        lines = [self.headline(), "", self.table()]
        if self.warnings:
            lines += [""] + [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def _pct(v: float) -> str:
    """A fraction as a percentage, with absurd values collapsed to ``>9999%``."""
    if not np.isfinite(v) or v > 99.99:
        return ">9999%"
    return f"{100 * v:.2f}%"


def _flag(r: FitResult) -> str:
    return "! " if r.degenerate else ""


def screen_degenerate(
    ranked: list[FitResult],
    exclude: bool,
    score: Optional[Callable[[FitResult], float]] = None,
    criterion: str = "bic",
) -> tuple[list[FitResult], list[FitResult], list[str]]:
    """Split ranked fits into kept and excluded degenerate ones, with report notes.

    When ``exclude`` is false, nothing is removed. If the leading fit is
    degenerate, a note names the reasons and the best admissible
    alternative. When every fit is degenerate, all are kept.
    """
    bad = [r for r in ranked if r.degenerate]
    good = [r for r in ranked if not r.degenerate]
    notes: list[str] = []
    if exclude and bad and good:
        notes.append(
            f"{len(bad)} degenerate fit{'s' if len(bad) > 1 else ''} excluded from the ranking "
            "(see DiscoveryReport.excluded)"
        )
        return good, bad, notes
    if bad and not good:
        notes.append("every fitted structure is degenerate; the ranking shows them anyway")
    elif ranked and ranked[0].degenerate:
        alt = good[0]
        gap = (
            f" ({criterion.upper()} {score(alt) - score(ranked[0]):+.1f})"
            if score is not None
            else ""
        )
        notes.append(
            f"the leading structure {ranked[0].structure_key} is a limiting case "
            f"({'; '.join(ranked[0].degeneracies())}); the best non-degenerate structure is "
            f"{alt.structure_key}{gap}"
        )
    return ranked, [], notes


def _err(r: FitResult) -> float:
    e = r.match_error
    return float(e) if e is not None and np.isfinite(e) else float(np.inf)
