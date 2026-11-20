r"""
Laws and the runner that checks them.

A :class:`Law` pairs a predicate with a strategy for each of its
arguments. :meth:`Law.check` draws examples, evaluates the predicate on
each one and returns a :class:`LawResult`:

``"verified"``
    No counterexample was found in ``n_examples`` examples. This is
    evidence, not proof. The result records how many examples were
    tried and how many were discarded by :func:`~spxa.laws.expect.assume`.
``"falsified"``
    The predicate raised :class:`~spxa.laws.expect.LawViolation` or
    returned ``False``. The counterexample is shrunk to a locally minimal
    one (see :meth:`Law._shrink`). The result keeps both the shrunk and the
    original counterexample, and the failure message of the shrunk one.
``"error"``
    Evaluation raised any other exception. That is also a
    counterexample: the law could not be evaluated where it should
    apply. The exception type and message are recorded.
``"unsatisfiable"``
    Too many examples were discarded, or a strategy could not produce
    values. Nothing was learned.

Every example is drawn from one ``numpy.random.Generator`` seeded by
``seed``, so a check is reproducible. Floating-point warnings raised
while evaluating are silenced: a law states its own tolerance.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional

import numpy as np

from spxa.laws.expect import Discard, LawViolation
from spxa.laws.strategies import Strategy, Tree, Unsatisfiable, fixed_dictionaries

Status = Literal["verified", "falsified", "error", "unsatisfiable"]


@dataclass
class Outcome:
    """The result of evaluating a law on one example."""

    kind: Literal["pass", "discard", "violation", "error"]
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None


@dataclass
class LawResult:
    """What a check found (see the module docstring for the statuses)."""

    law: str
    status: Status
    n_examples: int
    n_discarded: int
    seed: int
    counterexample: Optional[dict[str, Any]] = None
    original: Optional[dict[str, Any]] = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    shrink_steps: int = 0
    elapsed: float = 0.0
    description: str = ""

    @property
    def ok(self) -> bool:
        """``True`` when verified."""
        return self.status == "verified"

    @property
    def failed(self) -> bool:
        """``True`` when a counterexample was found (falsified or error)."""
        return self.status in ("falsified", "error")

    def raise_if_failed(self) -> None:
        """Raise :class:`LawFailure` unless the law was verified."""
        if not self.ok:
            raise LawFailure(self)

    def __str__(self) -> str:
        head = f"{self.law}: {self.status.upper()}"
        if self.status == "verified":
            tail = f" (no counterexample in {self.n_examples} examples"
            tail += f", {self.n_discarded} discarded)" if self.n_discarded else ")"
            return head + tail
        if self.status == "unsatisfiable":
            return (
                f"{head}: {self.message} "
                f"({self.n_examples} examples run, {self.n_discarded} discarded)"
            )
        lines = [head + f" after {self.n_examples} passing examples (seed {self.seed})"]
        if self.counterexample is not None:
            label = "counterexample"
            if self.shrink_steps:
                label += f" (shrunk in {self.shrink_steps} steps)"
            lines.append(f"  {label}:")
            lines += [f"    {k} = {v!r}" for k, v in self.counterexample.items()]
        what = f"{self.exception}: " if self.exception else ""
        lines.append(f"  {what}{self.message}")
        return "\n".join(lines)


class LawFailure(AssertionError):
    """Raised by :meth:`LawResult.raise_if_failed`; carries the result."""

    def __init__(self, result: LawResult) -> None:
        super().__init__(str(result))
        self.result = result


def evaluate(fn: Callable[..., Any], args: Mapping[str, Any]) -> Outcome:
    """Run ``fn(**args)`` and classify what happened."""
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            ret = fn(**args)
    except Discard as d:
        return Outcome("discard", str(d))
    except LawViolation as v:
        return Outcome("violation", v.message, v.details)
    except Exception as e:  # any other failure is a counterexample too
        return Outcome("error", str(e) or type(e).__name__, exception=type(e).__name__)
    if ret is False or (isinstance(ret, np.bool_) and not ret):
        return Outcome("violation", "the law returned False")
    return Outcome("pass")


def _same_failure(reference: Outcome, candidate: Outcome) -> bool:
    if reference.kind == "violation":
        return candidate.kind == "violation"
    return candidate.kind == "error" and candidate.exception == reference.exception


class Law:
    """A named predicate over arguments drawn from strategies.

    Parameters
    ----------
    name :
        Shown in reports.
    fn :
        Called with one keyword argument per strategy. It holds when it
        returns anything but ``False`` without raising.
    strategies :
        Argument name → :class:`~spxa.laws.strategies.Strategy`.
    description :
        One sentence, e.g. the formula; the docstring of ``fn`` by default.
    tags :
        Free labels used to select laws from a suite.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        strategies: Mapping[str, Strategy[Any]],
        description: str = "",
        tags: tuple[str, ...] = (),
    ) -> None:
        if not strategies:
            raise ValueError("a law needs at least one argument strategy")
        self.name = name
        self.fn = fn
        self.strategies = dict(strategies)
        self.description = description or (fn.__doc__ or "").strip().split("\n")[0]
        self.tags = tuple(tags)
        self._joint = fixed_dictionaries(self.strategies)

    def __call__(self, **kwargs: Any) -> Any:
        """Evaluate the predicate directly on given arguments."""
        return self.fn(**kwargs)

    def holds_for(self, **kwargs: Any) -> Outcome:
        """Evaluate on given arguments and classify the outcome."""
        return evaluate(self.fn, kwargs)

    def check(
        self,
        n_examples: int = 100,
        seed: int = 0,
        *,
        max_discard_ratio: float = 5.0,
        shrink: bool = True,
        max_shrinks: int = 500,
    ) -> LawResult:
        """Search for a counterexample (see the module docstring)."""
        if n_examples < 1:
            raise ValueError("n_examples must be at least 1")
        rng = np.random.default_rng(seed)
        start = time.perf_counter()
        passed = discarded = 0
        max_discards = int(max_discard_ratio * n_examples)

        def result(status: Status, **kw: Any) -> LawResult:
            return LawResult(
                law=self.name,
                status=status,
                n_examples=passed,
                n_discarded=discarded,
                seed=seed,
                elapsed=time.perf_counter() - start,
                description=self.description,
                **kw,
            )

        while passed < n_examples:
            try:
                tree = self._joint.generate(rng)
            except Unsatisfiable as exc:
                return result("unsatisfiable", message=str(exc))
            out = evaluate(self.fn, tree.value)
            if out.kind == "pass":
                passed += 1
            elif out.kind == "discard":
                discarded += 1
                if discarded > max_discards:
                    return result(
                        "unsatisfiable",
                        message=f"more than {max_discards} examples discarded by assume()",
                    )
            else:
                original = dict(tree.value)
                final, final_out, steps = (
                    self._shrink(tree, out, max_shrinks) if shrink else (tree, out, 0)
                )
                return result(
                    "falsified" if final_out.kind == "violation" else "error",
                    counterexample=dict(final.value),
                    original=original,
                    message=final_out.message,
                    details=final_out.details,
                    exception=final_out.exception,
                    shrink_steps=steps,
                )
        return result("verified")

    def _shrink(
        self, tree: Tree[dict[str, Any]], outcome: Outcome, max_shrinks: int
    ) -> tuple[Tree[dict[str, Any]], Outcome, int]:
        """Greedy descent to a locally minimal counterexample.

        At each step the children of the current counterexample are tried
        in order, simplest first. The first child that fails *in the same
        way* becomes the new counterexample: a violation must stay a
        violation, and an error must raise the same exception type. The
        descent stops when no child fails or after ``max_shrinks``
        evaluations. The result is locally minimal: none of its direct
        simplifications fails.
        """
        current, current_out = tree, outcome
        steps = evaluations = 0
        while evaluations < max_shrinks:
            for child in current.children():
                evaluations += 1
                out = evaluate(self.fn, child.value)
                if _same_failure(current_out, out):
                    current, current_out = child, out
                    steps += 1
                    break
                if evaluations >= max_shrinks:
                    break
            else:
                break
        return current, current_out, steps

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.strategies.items())
        return f"Law({self.name!r}, {args})"


def law(
    name: Optional[str] = None,
    *,
    description: str = "",
    tags: tuple[str, ...] = (),
    **strategies: Strategy[Any],
) -> Callable[[Callable[..., Any]], Law]:
    """Decorator: turn a predicate into a :class:`Law`.

    ::

        @law(X=levy_processes(), Y=levy_processes())
        def mean_is_additive(X, Y):
            expect_close(mean(X + Y), mean(X) + mean(Y))

        mean_is_additive.check(n_examples=200)
    """

    def wrap(fn: Callable[..., Any]) -> Law:
        return Law(name or fn.__name__, fn, strategies, description=description, tags=tags)

    return wrap
