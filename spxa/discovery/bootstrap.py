r"""
Bootstrap stability of a discovery result.

An information criterion picks one structure, but a different sample from
the same process might have picked another. :func:`bootstrap` measures that.
It resamples the returns, refits the leading candidates of a
:class:`~spxa.discovery.report.DiscoveryReport` on each resample, and
records:

- how often each structure ranks first: its selection frequency, the
  bootstrap analogue of an Akaike weight (Buckland, Burnham & Augustin
  1997);
- percentile intervals for the parameters of each structure.

Only the ``top`` leading structures are refitted, each from its original
estimate, so a replicate costs ``top`` local fits rather than a full search.
Structures outside the top set are assumed never to win, which the report's
weights should support. ``block_length > 1`` uses the moving-block
bootstrap (Künsch 1989) to keep short-range dependence in the resamples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spxa.discovery.empirical import summarize
from spxa.discovery.fit import BinnedData, FitResult, fit_structure
from spxa.discovery.inference import parameter_names
from spxa.discovery.report import DiscoveryReport


@dataclass(frozen=True)
class BootstrapResult:
    """Selection frequencies and parameter replicates.

    Attributes
    ----------
    structures :
        Keys of the refitted structures, in the report's order.
    wins :
        Number of replicates in which each structure ranked first.
    parameters :
        For each structure, an ``(n_ok, n_params)`` array of flattened natural
        parameters from the replicates in which its fit succeeded.
    names :
        Parameter labels per structure.
    n_boot :
        Number of replicates.
    criterion :
        ``"bic"`` or ``"aic"``.
    """

    structures: list[str]
    wins: np.ndarray
    parameters: list[np.ndarray]
    names: list[list[str]]
    n_boot: int
    criterion: str

    @property
    def selection_frequency(self) -> dict[str, float]:
        return {k: float(w) / self.n_boot for k, w in zip(self.structures, self.wins)}

    def intervals(self, structure: Optional[str] = None, level: float = 0.9) -> np.ndarray:
        """Percentile intervals, ``(n_params, 2)``, for ``structure`` (default: the first)."""
        i = 0 if structure is None else self.structures.index(structure)
        p = self.parameters[i]
        if p.shape[0] == 0:
            return np.full((len(self.names[i]), 2), np.nan)
        a = 100 * (1 - level) / 2
        return np.percentile(p, [a, 100 - a], axis=0).T

    def table(self, level: float = 0.9, digits: int = 3) -> str:
        lines = [f"selection frequency over {self.n_boot} resamples ({self.criterion.upper()}):"]
        for k, f in self.selection_frequency.items():
            lines.append(f"  {100 * f:5.1f}%  {k}")
        ci = self.intervals(level=level)
        lines.append(f"{100 * level:.0f}% percentile intervals for {self.structures[0]}:")
        for n, (lo, hi) in zip(self.names[0], ci):
            lines.append(f"  {n:<20} [{lo:.{digits}g}, {hi:.{digits}g}]")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


def _resample(x: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    n = x.size
    if block <= 1:
        return x[rng.integers(0, n, size=n)]
    starts = rng.integers(0, n - block + 1, size=-(-n // block))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return x[idx]


def bootstrap(
    report: DiscoveryReport,
    n_boot: int = 50,
    top: int = 5,
    block_length: int = 1,
    seed: int = 0,
) -> BootstrapResult:
    """Selection frequencies and parameter intervals for the leading structures of ``report``."""
    if report.method != "likelihood" or report.summary.sample is None:
        raise ValueError("bootstrapping needs a report built from a return sample")
    if n_boot < 1 or top < 1:
        raise ValueError("n_boot and top must be positive")
    x = report.summary.sample
    dt = report.summary.dt
    leaders: list[FitResult] = report.candidates[:top]
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(leaders), dtype=int)
    params: list[list[np.ndarray]] = [[] for _ in leaders]
    for _ in range(n_boot):
        xb = _resample(x, rng, block_length)
        sb = summarize(xb, dt=dt)
        bins = BinnedData.from_sample(xb, dt)
        scores = np.full(len(leaders), np.inf)
        for i, c in enumerate(leaders):
            try:
                r = fit_structure(
                    c.model.structure,
                    sb,
                    location=False,
                    n_starts=1,
                    binned=bins,
                    initial=list(c.model.params),
                    rng=np.random.default_rng(rng.integers(2**32)),
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            val = r.bic if report.criterion == "bic" else r.aic
            if val is not None and np.isfinite(val):
                scores[i] = val
                params[i].append(np.array([v for p in r.model.params for v in p]))
        if np.isfinite(scores).any():
            wins[int(np.argmin(scores))] += 1
    return BootstrapResult(
        structures=[c.structure_key for c in leaders],
        wins=wins,
        parameters=[
            np.array(p) if p else np.empty((0, c.n_params)) for p, c in zip(params, leaders)
        ],
        names=[parameter_names(c.model) for c in leaders],
        n_boot=n_boot,
        criterion=report.criterion,
    )
