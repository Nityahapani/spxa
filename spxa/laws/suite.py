r"""
Suites of laws and their reports.

A :class:`Suite` is a named collection of laws. It can be filtered by name
or tag and checked in one call. :meth:`Suite.check` returns a
:class:`SuiteReport` with one :class:`~spxa.laws.law.LawResult` per law, a
plain-text table (``str(report)``), Markdown (:meth:`SuiteReport.to_markdown`)
and a JSON-ready dict (:meth:`SuiteReport.to_dict`).

::

    from spxa.laws import Suite
    from spxa.laws.library import ALGEBRA, SUBORDINATION

    report = Suite("algebra", ALGEBRA + SUBORDINATION).check(n_examples=50)
    print(report)
    report.raise_if_failed()

Each law is checked with the same ``seed``, so a report is reproducible law
by law: rerunning one law with that seed reproduces its line.
:func:`default_suite` gathers every library law.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from spxa.laws.law import Law, LawResult

__all__ = ["Suite", "SuiteFailure", "SuiteReport", "default_suite"]

LawsLike = Union[Law, Iterable[Law]]


def _flatten(items: Iterable[LawsLike]) -> list[Law]:
    out: list[Law] = []
    for item in items:
        if isinstance(item, Law):
            out.append(item)
        else:
            out.extend(_flatten(item))
    return out


class SuiteFailure(AssertionError):
    """Raised by :meth:`SuiteReport.raise_if_failed`; carries the report."""

    def __init__(self, report: "SuiteReport") -> None:
        super().__init__(report.failure_text())
        self.report = report


@dataclass
class SuiteReport:
    """Results of checking a suite, in suite order."""

    name: str
    results: list[LawResult]
    n_examples: int
    seed: int
    elapsed: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """``True`` when every law was verified."""
        return all(r.ok for r in self.results)

    @property
    def counts(self) -> dict[str, int]:
        """Number of laws per status, always with all four keys."""
        out = {"verified": 0, "falsified": 0, "error": 0, "unsatisfiable": 0}
        for r in self.results:
            out[r.status] += 1
        return out

    @property
    def failures(self) -> list[LawResult]:
        """Results that are not verified (falsified, error or unsatisfiable)."""
        return [r for r in self.results if not r.ok]

    def __getitem__(self, name: str) -> LawResult:
        for r in self.results:
            if r.law == name:
                return r
        raise KeyError(name)

    def summary(self) -> str:
        """One line, e.g. ``algebra: 9 verified, 1 falsified (10 laws, 50 examples each, seed 0)``."""
        parts = [f"{v} {k}" for k, v in self.counts.items() if v]
        return (
            f"{self.name}: {', '.join(parts) or 'no laws'} ({len(self.results)} laws, "
            f"{self.n_examples} examples each, seed {self.seed}, {self.elapsed:.1f} s)"
        )

    def __str__(self) -> str:
        width = max([len(r.law) for r in self.results] + [3])
        lines = [self.summary(), ""]
        lines.append(f"  {'law':<{width}}  {'status':<13}  {'examples':>8}  {'discarded':>9}")
        for r in self.results:
            lines.append(
                f"  {r.law:<{width}}  {r.status:<13}  {r.n_examples:>8}  {r.n_discarded:>9}"
            )
        if self.failures:
            lines += ["", self.failure_text()]
        return "\n".join(lines)

    def failure_text(self) -> str:
        """The full message of every failure, blank-line separated."""
        return "\n\n".join(str(r) for r in self.failures) or f"{self.name}: all laws verified"

    def to_markdown(self) -> str:
        """A Markdown table of the results, followed by each counterexample."""
        lines = [
            f"**{self.summary()}**",
            "",
            "| law | status | examples | discarded | description |",
            "|---|---|---:|---:|---|",
        ]
        for r in self.results:
            desc = r.description.replace("|", "\\|")
            lines.append(f"| `{r.law}` | {r.status} | {r.n_examples} | {r.n_discarded} | {desc} |")
        for r in self.failures:
            lines += ["", f"<details><summary><code>{r.law}</code>: {r.status}</summary>", ""]
            lines += ["```text", str(r), "```", "", "</details>"]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready report."""
        return {
            "suite": self.name,
            "ok": self.ok,
            "counts": self.counts,
            "n_examples": self.n_examples,
            "seed": self.seed,
            "elapsed": self.elapsed,
            "meta": dict(self.meta),
            "results": [r.to_dict() for r in self.results],
        }

    def raise_if_failed(self) -> None:
        """Raise :class:`SuiteFailure` unless every law was verified."""
        if not self.ok:
            raise SuiteFailure(self)


class Suite:
    """A named, ordered collection of laws.

    Parameters
    ----------
    name :
        Shown in reports.
    laws :
        Laws, or iterables of laws such as the library tuples, in any
        nesting. Names must be unique.
    """

    def __init__(self, name: str, *laws: LawsLike) -> None:
        self.name = name
        self.laws = _flatten(laws)
        names = [law_.name for law_ in self.laws]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate law names in suite {name!r}: {duplicates}")

    def __len__(self) -> int:
        return len(self.laws)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.laws)

    def __repr__(self) -> str:
        return f"Suite({self.name!r}, {len(self.laws)} laws)"

    @property
    def tags(self) -> list[str]:
        """Every tag used by a law in the suite, sorted."""
        return sorted({t for law_ in self.laws for t in law_.tags})

    def select(
        self,
        *,
        names: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        exclude_tags: Sequence[str] = (),
    ) -> "Suite":
        """Laws whose name is in ``names`` and that carry any of ``tags``, minus ``exclude_tags``."""
        if names is not None:
            unknown = sorted(set(names) - {law_.name for law_ in self.laws})
            if unknown:
                raise KeyError(f"no such laws in {self.name!r}: {unknown}")
        chosen = [
            law_
            for law_ in self.laws
            if (names is None or law_.name in names)
            and (tags is None or set(tags) & set(law_.tags))
            and not set(exclude_tags) & set(law_.tags)
        ]
        return Suite(self.name, chosen)

    def check(
        self,
        n_examples: int = 100,
        seed: int = 0,
        *,
        progress: Optional[Callable[[LawResult], None]] = None,
        fail_fast: bool = False,
        **check_kw: Any,
    ) -> SuiteReport:
        """Check every law with the same ``n_examples`` and ``seed``.

        ``progress`` is called with each result as it arrives. With
        ``fail_fast`` the run stops at the first law that is not verified.
        Other keywords go to :meth:`~spxa.laws.law.Law.check`.
        """
        start = time.perf_counter()
        results: list[LawResult] = []
        for law_ in self.laws:
            result = law_.check(n_examples=n_examples, seed=seed, **check_kw)
            results.append(result)
            if progress is not None:
                progress(result)
            if fail_fast and not result.ok:
                break
        return SuiteReport(
            name=self.name,
            results=results,
            n_examples=n_examples,
            seed=seed,
            elapsed=time.perf_counter() - start,
        )


def default_suite() -> Suite:
    """Every law in :mod:`spxa.laws.library`, library by library."""
    from spxa.laws import library

    return Suite(
        "spxa",
        library.AXIOMS,
        library.ALGEBRA,
        library.SUBORDINATION,
        library.TRIPLET,
        library.CUMULANTS,
        library.PROPERTIES,
        library.SIMULATION,
    )
