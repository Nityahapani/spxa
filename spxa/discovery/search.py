r"""
Structure search: which composition of primitives explains the data?

:func:`discover` searches a :class:`~spxa.discovery.grammar.Grammar`.

``search="beam"`` (default)
    Fit every atom, then repeatedly extend the ``beam_width`` best
    structures by one atom each, warm-starting every extension from its
    parent's fit, until ``max_terms`` atoms. This is forward selection with
    a beam, the usual strategy in symbolic regression.

``search="exhaustive"``
    Fit every structure in the grammar.

With a sample, candidates are ranked by BIC (or AIC) of the shared
smoothed binned likelihood. Information-criterion weights
:math:`w_i \propto e^{-\Delta_i/2}` give the relative support of each
structure (Burnham & Anderson 2002). With summary statistics only, the
four cumulants identify at most three shape parameters. Every structure
whose variance, skewness and excess kurtosis are within ``tolerance``
(:attr:`~spxa.discovery.fit.FitResult.match_error`) then counts as an
explanation, and they are ranked by simplicity (fewest parameters). The
report states the non-identifiability explicitly.
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Sequence, Union

import numpy as np

from spxa.discovery.empirical import EmpiricalSummary, summarize
from spxa.discovery.expr import Leaf, Model, Node, Sum
from spxa.discovery.families import DRIFT
from spxa.discovery.fit import BinnedData, FitResult, fit_structure, initial_params, with_location
from spxa.discovery.grammar import Grammar, canonical
from spxa.discovery.io import ReturnSeries, load_returns
from spxa.discovery.report import DiscoveryReport, _err, screen_degenerate

Criterion = Literal["bic", "aic"]


def _strip_location(structure: Node) -> Node:
    """Remove the trailing drift leaf that :func:`with_location` added."""
    if isinstance(structure, Sum):
        rest = [c for c in structure.children if not (isinstance(c, Leaf) and c.family is DRIFT)]
        if len(rest) == 1:
            return rest[0]
        return Sum(rest)
    return structure


def _term_params(model: Model) -> tuple[dict[str, list[tuple[float, ...]]], float]:
    """Parameters of each top-level term, keyed by the term's key, and the drift."""
    s = model.structure
    terms = s.children if isinstance(s, Sum) else (s,)
    params = list(model.params)
    out: dict[str, list[tuple[float, ...]]] = {}
    drift = 0.0
    for term in terms:
        n = len(term.leaves())
        chunk, params = params[:n], params[n:]
        if isinstance(term, Leaf) and term.family is DRIFT:
            drift = chunk[0][0]
        else:
            out[term.key()] = chunk
    return out, drift


def _warm_start(parent: FitResult, child: Node, std: float) -> list[tuple[float, ...]]:
    known, drift = _term_params(parent.model)
    terms = child.children if isinstance(child, Sum) else (child,)
    out: list[tuple[float, ...]] = []
    for term in terms:
        if isinstance(term, Leaf) and term.family is DRIFT:
            out.append((drift,))
        elif term.key() in known:
            out += known[term.key()]
        else:
            out += initial_params(term, std)
    return out


Data = Union[EmpiricalSummary, ReturnSeries, str, os.PathLike[str], Sequence[float], np.ndarray]


def _as_summary(data: Data, dt: Optional[float]) -> EmpiricalSummary:
    if isinstance(data, EmpiricalSummary):
        if dt is not None and not np.isclose(dt, data.dt):
            raise ValueError(f"dt={dt} conflicts with the summary's dt={data.dt}")
        return data
    if isinstance(data, (str, os.PathLike)):
        data = load_returns(data)
    if isinstance(data, ReturnSeries):
        step = dt if dt is not None else data.dt
        if step is None:
            raise ValueError(
                f"no dates in the data to infer the sampling interval from; pass dt= "
                f"(column {data.column!r})"
            )
        return summarize(data.returns, dt=step)
    return summarize(np.asarray(data, dtype=float), dt=1.0 if dt is None else dt)


def discover(
    data: Data,
    *,
    dt: Optional[float] = None,
    grammar: Optional[Grammar] = None,
    method: Optional[Literal["likelihood", "cumulants"]] = None,
    criterion: Criterion = "bic",
    search: Literal["beam", "exhaustive"] = "beam",
    beam_width: int = 4,
    max_terms: Optional[int] = None,
    n_starts: int = 3,
    extension_starts: int = 2,
    seed: int = 0,
    tolerance: float = 0.02,
    exclude_degenerate: bool = False,
) -> DiscoveryReport:
    """Search for the compositions of primitives that best explain ``data``.

    Parameters
    ----------
    data :
        One of:

        - increments (returns) sampled every ``dt``;
        - a path to a CSV file of returns or prices (read with
          :func:`~spxa.discovery.io.load_returns`);
        - a :class:`~spxa.discovery.io.ReturnSeries`;
        - an :class:`~spxa.discovery.empirical.EmpiricalSummary`. A summary
          built with :func:`~spxa.discovery.empirical.from_statistics` has no
          sample and is fitted on cumulants.
    dt :
        Sampling interval in model time units (years for annualised
        parameters). Defaults to the interval inferred from a file's dates,
        or the summary's own, or 1 for a bare array.
    grammar :
        Search space; :class:`~spxa.discovery.grammar.Grammar` by default.
    method :
        ``"likelihood"`` when a sample is available, otherwise ``"cumulants"``.
    criterion :
        ``"bic"`` or ``"aic"`` (likelihood mode).
    search, beam_width, max_terms :
        Search strategy (see the module docstring).
    n_starts, extension_starts, seed :
        Starting points per fit: ``n_starts`` for the single atoms (and every
        structure in exhaustive search), ``extension_starts`` for beam
        extensions, whose first start is the warm start from the parent. The
        search is deterministic for a given seed.
    tolerance :
        Moment error (:attr:`~spxa.discovery.fit.FitResult.match_error`)
        within which a structure "matches" the data.
    exclude_degenerate :
        Move degenerate fits (:meth:`~spxa.discovery.fit.FitResult.degeneracies`)
        out of the ranking into :attr:`DiscoveryReport.excluded`. By default
        they stay in the ranking: the table marks them with ``!``, and a
        warning appears when one of them leads.
    """
    summary = _as_summary(data, dt)
    grammar = grammar or Grammar()
    top = grammar.max_terms if max_terms is None else max_terms
    if method is None:
        method = "likelihood" if summary.sample is not None else "cumulants"
    if method == "likelihood" and summary.sample is None:
        raise ValueError(
            "the likelihood needs a sample of returns; this summary has none "
            "(use method='cumulants')"
        )
    if criterion not in ("bic", "aic"):
        raise ValueError(f"criterion must be 'bic' or 'aic', got {criterion!r}")
    binned = BinnedData.from_sample(summary.sample, summary.dt) if method == "likelihood" else None
    std = float(np.sqrt(summary.variance))
    report_warnings: list[str] = []
    if not summary.is_levy_consistent():
        report_warnings.append(
            "returns or squared returns are autocorrelated (Ljung–Box p < 0.01): no Lévy "
            "composition has dependent increments, so the ranking describes the marginal law only"
        )

    fitted: dict[str, FitResult] = {}
    rng = np.random.default_rng(seed)

    def fit(
        structure: Node,
        initial: Optional[list[tuple[float, ...]]] = None,
        starts: int = n_starts,
    ) -> Optional[FitResult]:
        key = with_location(structure).key()
        previous = fitted.get(key)
        if previous is not None and initial is None:
            return previous
        # A structure reached again from another parent is refitted from that
        # parent's warm start and the better optimum is kept: one warm start can
        # sit in a poor basin (e.g. many tiny jumps for a jump-diffusion), and
        # which parent arrives first must not decide the ranking.
        try:
            res = fit_structure(
                structure,
                summary,
                method=method,
                n_starts=starts,
                rng=np.random.default_rng(rng.integers(2**32)),
                binned=binned,
                initial=initial,
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            if previous is not None:
                return previous
            report_warnings.append(f"{structure.key()}: fit failed ({exc})")
            return None
        if previous is not None and not res.objective < previous.objective:
            return previous
        fitted[key] = res
        return res

    def rank_key(r: FitResult) -> tuple[float, ...]:
        if method == "cumulants":
            err = _err(r)
            return (
                0.0 if err <= tolerance else 1.0,
                float(r.n_params) if err <= tolerance else err,
                err,
            )
        val = r.bic if criterion == "bic" else r.aic
        return (float(val) if val is not None else np.inf,)

    if search == "exhaustive":
        for s in grammar.structures(top):
            fit(s)
    elif search == "beam":
        level = [r for r in (fit(a) for a in grammar.atoms()) if r is not None]
        for _ in range(2, top + 1):
            parents = sorted(level, key=rank_key)[:beam_width]
            level = []
            for parent in parents:
                for child in grammar.extensions(_strip_location(parent.model.structure)):
                    init = with_location(canonical(child))
                    res = fit(
                        canonical(child), _warm_start(parent, init, 0.5 * std), extension_starts
                    )
                    if res is not None:
                        level.append(res)
    else:
        raise ValueError(f"search must be 'beam' or 'exhaustive', got {search!r}")

    if not fitted:
        raise RuntimeError("no structure could be fitted")
    ranked = sorted(fitted.values(), key=rank_key)
    if method == "cumulants":
        matches = [r for r in ranked if _err(r) <= tolerance]
        if len(matches) > 1:
            report_warnings.append(
                f"{len(matches)} structures reproduce the four cumulants within {100 * tolerance:.0f}%: "
                "summary statistics identify at most three shape parameters, so these are equally "
                "valid explanations, ranked here by simplicity; a return sample would discriminate them"
            )
    ranked, excluded, notes = screen_degenerate(
        ranked,
        exclude_degenerate,
        None if method == "cumulants" else lambda r: float(rank_key(r)[0]),
        criterion,
    )
    return DiscoveryReport(
        candidates=ranked,
        summary=summary,
        criterion=criterion,
        method=method,
        tolerance=tolerance,
        warnings=report_warnings + notes,
        excluded=excluded,
        binned=binned,
    )
