r"""
Random example generation with integrated shrinking.

A :class:`Strategy` produces random examples together with the ways each
example can be simplified. A generated example is a *rose tree*
(:class:`Tree`). The root is the value, and the children are simpler
values, each with its own simpler values, and so on. Shrinking a
counterexample means walking down this tree, keeping a child whenever it
still falsifies the law. Because the tree is built alongside the value,
shrinking works through :meth:`Strategy.map`, :meth:`Strategy.filter`,
:meth:`Strategy.flatmap` and composite strategies without extra code. This
is the "integrated shrinking" of Hedgehog and of Hypothesis's internal
representation (MacIver & Donaldson 2020), here implemented directly.

Children are listed simplest first, so a greedy descent tries the most
aggressive simplification before the cautious ones.

Floats shrink towards a target: the value in range closest to 0 unless
``simplest`` is given. The candidates, in order, are the target, the
nearest integer, values rounded to 1, 2 and 3 significant digits, and
points that halve the distance to the target. Integers shrink towards
their target by halving. Tuples and dictionaries shrink one component at
a time.

Determinism: generation draws only from the ``numpy.random.Generator``
passed in. A strategy that needs randomness while shrinking, as
:meth:`Strategy.flatmap` does to rebuild an inner example, records a seed
at generation time, so the same counterexample shrinks the same way every
time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np

T = TypeVar("T")
U = TypeVar("U")

#: How many times :meth:`Strategy.filter` redraws before giving up.
FILTER_ATTEMPTS = 100


class Unsatisfiable(Exception):
    """A filtered strategy could not produce a value that passes its predicate."""


@dataclass(frozen=True)
class Tree(Generic[T]):
    """A value and a lazily computed list of simpler trees."""

    value: T
    _children: Callable[[], Iterable[Tree[T]]]

    def children(self) -> Iterator[Tree[T]]:
        return iter(self._children())

    def map(self, f: Callable[[T], U]) -> Tree[U]:
        return Tree(f(self.value), lambda: (c.map(f) for c in self.children()))

    def filter(self, pred: Callable[[T], bool], lookahead: int = 2) -> Tree[T]:
        """Keep only descendants whose value passes ``pred``; the root is assumed to pass.

        A child that fails is replaced by its passing descendants up to
        ``lookahead`` levels down. For example, under "odd integers" 7 shrinks
        to 3 and 5 through the even 6, instead of getting stuck.
        """

        def passing(t: Tree[T], depth: int) -> Iterator[Tree[T]]:
            for c in t.children():
                if _passes(pred, c.value):
                    yield c.filter(pred, lookahead)
                elif depth < lookahead:
                    yield from passing(c, depth + 1)

        return Tree(self.value, lambda: passing(self, 1))

    @staticmethod
    def leaf(value: T) -> Tree[T]:
        return Tree(value, lambda: ())


def _passes(pred: Callable[[Any], bool], value: Any) -> bool:
    try:
        return bool(pred(value))
    except Exception:
        return False


class Strategy(Generic[T]):
    """Base class: override :meth:`generate`."""

    def generate(self, rng: np.random.Generator) -> Tree[T]:
        raise NotImplementedError

    def example(self, seed: int = 0) -> T:
        """One value, for interactive use."""
        return self.generate(np.random.default_rng(seed)).value

    def map(self, f: Callable[[T], U], name: Optional[str] = None) -> Strategy[U]:
        return _Mapped(self, f, name)

    def filter(self, pred: Callable[[T], bool], name: Optional[str] = None) -> Strategy[T]:
        return _Filtered(self, pred, name)

    def flatmap(self, f: Callable[[T], Strategy[U]]) -> Strategy[U]:
        return _FlatMapped(self, f)

    def __or__(self, other: Strategy[Any]) -> Strategy[Any]:
        return one_of(self, other)

    def __repr__(self) -> str:
        return type(self).__name__


class _Mapped(Strategy[U]):
    def __init__(self, base: Strategy[Any], f: Callable[[Any], U], name: Optional[str]) -> None:
        self.base, self.f, self.name = base, f, name

    def generate(self, rng: np.random.Generator) -> Tree[U]:
        return self.base.generate(rng).map(self.f)

    def __repr__(self) -> str:
        return self.name or f"{self.base!r}.map({getattr(self.f, '__name__', 'f')})"


class _Filtered(Strategy[T]):
    def __init__(self, base: Strategy[T], pred: Callable[[T], bool], name: Optional[str]) -> None:
        self.base, self.pred, self.name = base, pred, name

    def generate(self, rng: np.random.Generator) -> Tree[T]:
        for _ in range(FILTER_ATTEMPTS):
            tree = self.base.generate(rng)
            if _passes(self.pred, tree.value):
                return tree.filter(self.pred)
        raise Unsatisfiable(f"{self!r}: no value passed the filter in {FILTER_ATTEMPTS} attempts")

    def __repr__(self) -> str:
        return self.name or f"{self.base!r}.filter(...)"


class _FlatMapped(Strategy[U]):
    def __init__(self, base: Strategy[Any], f: Callable[[Any], Strategy[U]]) -> None:
        self.base, self.f = base, f

    def generate(self, rng: np.random.Generator) -> Tree[U]:
        outer = self.base.generate(rng)
        seed = int(rng.integers(2**63))
        return self._bind(outer, seed)

    def _bind(self, outer: Tree[Any], seed: int) -> Tree[U]:
        inner = self.f(outer.value).generate(np.random.default_rng(seed))

        def children() -> Iterator[Tree[U]]:
            for o in outer.children():  # simpler outer value, same inner randomness
                try:
                    yield self._bind(o, seed)
                except Unsatisfiable:
                    continue
            yield from inner.children()

        return Tree(inner.value, children)


# ── numbers ───────────────────────────────────────────────────────────────


def _float_candidates(x: float, target: float, lo: float, hi: float) -> list[float]:
    out: list[float] = []

    def add(v: float) -> None:
        if lo <= v <= hi and v != x and abs(v - target) < abs(x - target) and v not in out:
            out.append(v)

    add(target)
    if math.isfinite(x):
        add(float(round(x)))
        if x != 0:
            for digits in (1, 2, 3):
                add(float(f"{x:.{digits}g}"))
        d = x - target
        for k in (1, 2, 3, 4):
            add(target + d * (1 - 0.5**k))
    return out


def _float_tree(x: float, target: float, lo: float, hi: float, depth: int = 0) -> Tree[float]:
    def children() -> Iterator[Tree[float]]:
        if depth >= 64:
            return
        for v in _float_candidates(x, target, lo, hi):
            yield _float_tree(v, target, lo, hi, depth + 1)

    return Tree(x, children)


class _Floats(Strategy[float]):
    def __init__(self, lo: float, hi: float, log: bool, simplest: Optional[float]) -> None:
        if not lo < hi:
            raise ValueError(f"need lo < hi, got [{lo}, {hi}]")
        if log and lo <= 0:
            raise ValueError("log-uniform floats need lo > 0")
        self.lo, self.hi, self.log = float(lo), float(hi), log
        if simplest is None:
            simplest = 0.0 if lo <= 0.0 <= hi else (lo if lo > 0 else hi)
        if not lo <= simplest <= hi:
            raise ValueError(f"simplest={simplest} lies outside [{lo}, {hi}]")
        self.simplest = float(simplest)

    def generate(self, rng: np.random.Generator) -> Tree[float]:
        if self.log:
            x = float(np.exp(rng.uniform(np.log(self.lo), np.log(self.hi))))
        else:
            x = float(rng.uniform(self.lo, self.hi))
        return _float_tree(x, self.simplest, self.lo, self.hi)

    def __repr__(self) -> str:
        kind = "log_floats" if self.log else "floats"
        return f"{kind}({self.lo:g}, {self.hi:g})"


def floats(
    lo: float, hi: float, *, log: bool = False, simplest: Optional[float] = None
) -> Strategy[float]:
    """Uniform (or log-uniform) floats on ``[lo, hi]``, shrinking towards ``simplest``."""
    return _Floats(lo, hi, log, simplest)


def _int_tree(n: int, target: int) -> Tree[int]:
    def children() -> Iterator[Tree[int]]:
        seen = set()
        d = n - target
        for v in (target, *(target + d - d // 2**k for k in range(1, 64) if d // 2**k)):
            if v != n and abs(v - target) < abs(d) and v not in seen:
                seen.add(v)
                yield _int_tree(v, target)

    return Tree(n, children)


class _Integers(Strategy[int]):
    def __init__(self, lo: int, hi: int, simplest: Optional[int]) -> None:
        if lo > hi:
            raise ValueError(f"need lo <= hi, got [{lo}, {hi}]")
        self.lo, self.hi = int(lo), int(hi)
        if simplest is None:
            simplest = 0 if lo <= 0 <= hi else (lo if lo > 0 else hi)
        self.simplest = int(simplest)

    def generate(self, rng: np.random.Generator) -> Tree[int]:
        return _int_tree(int(rng.integers(self.lo, self.hi + 1)), self.simplest)

    def __repr__(self) -> str:
        return f"integers({self.lo}, {self.hi})"


def integers(lo: int, hi: int, *, simplest: Optional[int] = None) -> Strategy[int]:
    """Uniform integers on ``[lo, hi]`` (inclusive), shrinking towards ``simplest``."""
    return _Integers(lo, hi, simplest)


# ── combinators ───────────────────────────────────────────────────────────


class _Just(Strategy[T]):
    def __init__(self, value: T) -> None:
        self.value = value

    def generate(self, rng: np.random.Generator) -> Tree[T]:
        return Tree.leaf(self.value)

    def __repr__(self) -> str:
        return f"just({self.value!r})"


def just(value: T) -> Strategy[T]:
    """Always ``value``."""
    return _Just(value)


class _SampledFrom(Strategy[T]):
    def __init__(self, values: Sequence[T]) -> None:
        if not values:
            raise ValueError("sampled_from needs at least one value")
        self.values = list(values)

    def generate(self, rng: np.random.Generator) -> Tree[T]:
        return _int_tree(int(rng.integers(len(self.values))), 0).map(self.values.__getitem__)

    def __repr__(self) -> str:
        return f"sampled_from({self.values!r})"


def sampled_from(values: Sequence[T]) -> Strategy[T]:
    """One of ``values``, shrinking towards the first."""
    return _SampledFrom(values)


class _OneOf(Strategy[Any]):
    def __init__(self, options: Sequence[Strategy[Any]]) -> None:
        if not options:
            raise ValueError("one_of needs at least one strategy")
        self.options = list(options)

    def generate(self, rng: np.random.Generator) -> Tree[Any]:
        i = int(rng.integers(len(self.options)))
        tree = self.options[i].generate(rng)
        seeds = [int(s) for s in rng.integers(2**63, size=i)]

        def children() -> Iterator[Tree[Any]]:
            for j in range(i):  # an example of an earlier, simpler alternative
                try:
                    yield self.options[j].generate(np.random.default_rng(seeds[j]))
                except Unsatisfiable:
                    continue
            yield from tree.children()

        return Tree(tree.value, children)

    def __repr__(self) -> str:
        return "one_of(" + ", ".join(map(repr, self.options)) + ")"


def one_of(*options: Strategy[Any]) -> Strategy[Any]:
    """A value from one of ``options``, chosen uniformly; earlier options count as simpler."""
    flat: list[Strategy[Any]] = []
    for o in options:
        flat += o.options if isinstance(o, _OneOf) else [o]
    return _OneOf(flat)


def _product(trees: Sequence[Tree[Any]]) -> Tree[tuple[Any, ...]]:
    def children() -> Iterator[Tree[tuple[Any, ...]]]:
        for k, t in enumerate(trees):
            for c in t.children():
                yield _product([*trees[:k], c, *trees[k + 1 :]])

    return Tree(tuple(t.value for t in trees), children)


class _Tuples(Strategy[tuple[Any, ...]]):
    def __init__(self, parts: Sequence[Strategy[Any]]) -> None:
        self.parts = list(parts)

    def generate(self, rng: np.random.Generator) -> Tree[tuple[Any, ...]]:
        return _product([p.generate(rng) for p in self.parts])

    def __repr__(self) -> str:
        return "tuples(" + ", ".join(map(repr, self.parts)) + ")"


def tuples(*parts: Strategy[Any]) -> Strategy[tuple[Any, ...]]:
    """A tuple with one value from each strategy; shrinks one component at a time."""
    return _Tuples(parts)


class _FixedDicts(Strategy[dict[str, Any]]):
    def __init__(self, parts: Mapping[str, Strategy[Any]]) -> None:
        self.keys = list(parts)
        self.inner = _Tuples([parts[k] for k in self.keys])

    def generate(self, rng: np.random.Generator) -> Tree[dict[str, Any]]:
        return self.inner.generate(rng).map(lambda vals: dict(zip(self.keys, vals)))

    def __repr__(self) -> str:
        return "fixed_dictionaries({" + ", ".join(f"{k!r}: ..." for k in self.keys) + "})"


def fixed_dictionaries(parts: Mapping[str, Strategy[Any]]) -> Strategy[dict[str, Any]]:
    """A dict with a value for each key; shrinks one entry at a time."""
    return _FixedDicts(parts)


def builds(target: Callable[..., T], /, **kwargs: Strategy[Any]) -> Strategy[T]:
    """``target(**values)`` with each keyword drawn from its strategy.

    The tree is shrunk on the keyword values and rebuilt with ``target``.
    A keyword combination that ``target`` rejects with ``ValueError`` is
    skipped during shrinking.
    """
    name = getattr(target, "__name__", repr(target))
    inner = fixed_dictionaries(kwargs)

    class _Builds(Strategy[T]):
        def generate(self, rng: np.random.Generator) -> Tree[T]:
            for _ in range(FILTER_ATTEMPTS):
                tree = inner.generate(rng)
                try:
                    return _build_tree(tree)
                except ValueError:
                    continue
            raise Unsatisfiable(f"builds({name}): every draw was rejected")

        def __repr__(self) -> str:
            return f"builds({name})"

    def _build_tree(tree: Tree[dict[str, Any]]) -> Tree[T]:
        value = target(**tree.value)

        def children() -> Iterator[Tree[T]]:
            for c in tree.children():
                try:
                    yield _build_tree(c)
                except ValueError:
                    continue

        return Tree(value, children)

    return _Builds()


def shrink_values(tree: Tree[T], limit: int = 50) -> list[T]:
    """The first ``limit`` direct shrinks of ``tree``, for inspection and tests."""
    out = []
    for c in tree.children():
        out.append(c.value)
        if len(out) >= limit:
            break
    return out
