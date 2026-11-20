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

import numpy as np

from spxa.discovery.empirical import EmpiricalSummary
from spxa.discovery.fit import FitResult


@dataclass
class DiscoveryReport:
    """Ranked candidate structures and how well they explain the data."""

    candidates: list[FitResult]
    summary: EmpiricalSummary
    criterion: str
    method: str
    tolerance: float
    warnings: list[str] = field(default_factory=list)

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
        if b.ks_distance is not None:
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

    def table(self, top: int = 10, digits: int = 3) -> str:
        """Plain-text ranking of the first ``top`` candidates."""
        rows = []
        if self.method == "cumulants":
            header = f"{'#':>3}  {'params':>6}  {'moment err':>10}  expression"
            for i, r in enumerate(self.candidates[:top], 1):
                rows.append(
                    f"{i:>3}  {r.n_params:>6}  {100 * _err(r):9.2f}%  "
                    f"{r.model.render(digits, weighted='subordinators')}"
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
                    f"{100 * (r.ks_distance or 0):7.2f}%  {100 * _err(r):9.2f}%  "
                    f"{r.model.render(digits, weighted='subordinators')}"
                )
        return "\n".join([header] + rows)

    def __str__(self) -> str:
        lines = [self.headline(), "", self.table()]
        if self.warnings:
            lines += [""] + [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def _err(r: FitResult) -> float:
    e = r.match_error
    return float(e) if e is not None and np.isfinite(e) else float(np.inf)
