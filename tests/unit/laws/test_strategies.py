"""Strategies: generation, determinism and integrated shrinking."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

from spxa.laws.strategies import (
    Strategy,
    Tree,
    Unsatisfiable,
    builds,
    fixed_dictionaries,
    floats,
    integers,
    just,
    one_of,
    sampled_from,
    shrink_values,
    tuples,
)


def descend(tree: Tree[Any], fails: Callable[[Any], bool]) -> Any:
    """Greedy shrink: follow the first failing child until none fails."""
    while True:
        for c in tree.children():
            if fails(c.value):
                tree = c
                break
        else:
            return tree.value


def draws(s: Strategy[Any], n: int = 200, seed: int = 0) -> list[Any]:
    rng = np.random.default_rng(seed)
    return [s.generate(rng).value for _ in range(n)]


def test_floats_stay_in_range_and_are_deterministic() -> None:
    xs = draws(floats(-2.0, 3.0))
    assert min(xs) >= -2.0 and max(xs) <= 3.0 and len(set(xs)) == 200
    assert xs == draws(floats(-2.0, 3.0))
    logs = np.array(draws(floats(1e-3, 1e3, log=True), n=2000))
    assert logs.min() >= 1e-3 and logs.max() <= 1e3
    # log-uniform: a third of the mass per factor 100
    assert np.mean(logs < 0.1) == pytest.approx(1 / 3, abs=0.05)
    with pytest.raises(ValueError):
        floats(1.0, 1.0)
    with pytest.raises(ValueError):
        floats(0.0, 1.0, log=True)
    with pytest.raises(ValueError):
        floats(0.0, 1.0, simplest=2.0)


def test_float_shrinking_finds_the_boundary_of_a_failure() -> None:
    tree = floats(0.0, 10.0).generate(np.random.default_rng(1))
    assert tree.value > 1.5
    # fails for every x > 1.5: the shrunk value is just above 1.5 and short
    x = descend(tree, lambda v: v > 1.5)
    assert 1.5 < x <= 2.0
    # target first, then integers and short decimals
    kids = shrink_values(floats(-5.0, 5.0).generate(np.random.default_rng(3)))
    assert kids[0] == 0.0 and all(abs(k) < 5.0 for k in kids)
    assert descend(floats(1.0, 4.0).generate(np.random.default_rng(0)), lambda v: True) == 1.0


def test_float_children_are_strictly_simpler() -> None:
    def key(v: float) -> tuple[int, float]:
        return len(repr(v)), abs(v - 1.0)

    rng = np.random.default_rng(5)
    for _ in range(50):
        t = floats(-3.0, 7.0, simplest=1.0).generate(rng)
        for c in t.children():
            assert key(c.value) < key(t.value) and -3.0 <= c.value <= 7.0


def test_floats_shrink_to_short_decimals() -> None:
    # any non-zero value fails: the simplest failing value is a one-digit decimal
    for seed in range(10):
        t = floats(-1.0, 1.0).generate(np.random.default_rng(seed))
        assert descend(t, lambda x: x != 0.0) in (0.1, -0.1)
        # a threshold: the boundary itself, or the next short decimal
        u = floats(0.0, 10.0).generate(np.random.default_rng(seed))
        if u.value >= 7.5:
            assert descend(u, lambda x: x >= 7.5) in (7.5, 8.0)


def test_integers() -> None:
    ns = draws(integers(-3, 9))
    assert set(ns) == set(range(-3, 10))
    t = integers(0, 1000).generate(np.random.default_rng(2))
    assert descend(t, lambda n: n >= 37) == 37
    assert descend(integers(5, 50).generate(np.random.default_rng(0)), lambda n: True) == 5


def test_sampled_from_just_and_one_of() -> None:
    assert set(draws(sampled_from("abc"))) == {"a", "b", "c"}
    t = sampled_from(["x", "y", "z"]).generate(np.random.default_rng(4))
    assert descend(t, lambda v: True) == "x"
    assert draws(just(7), n=3) == [7, 7, 7]
    s = one_of(just("simple"), floats(0, 1)) | just("third")
    vals = draws(s, n=300)
    assert {v for v in vals if isinstance(v, str)} == {"simple", "third"}
    # a float from the second option shrinks to the first option
    rng = np.random.default_rng(0)
    tree = next(t for t in (s.generate(rng) for _ in range(50)) if isinstance(t.value, float))
    assert descend(tree, lambda v: True) == "simple"


def test_tuples_and_dictionaries_shrink_componentwise() -> None:
    t = tuples(floats(0, 10), integers(0, 100)).generate(np.random.default_rng(8))
    x, n = descend(t, lambda v: v[0] + v[1] > 20)
    assert x + n > 20 and (x <= 1.0 or n <= 20)
    d = fixed_dictionaries({"a": floats(0, 1), "b": integers(0, 5)}).generate(
        np.random.default_rng(1)
    )
    assert set(d.value) == {"a", "b"}
    assert descend(d, lambda v: True) == {"a": 0.0, "b": 0}


def test_map_filter_flatmap() -> None:
    sq = floats(0, 3).map(lambda x: x * x)
    assert all(0 <= v <= 9 for v in draws(sq))
    assert descend(sq.generate(np.random.default_rng(0)), lambda v: v > 2.0) == pytest.approx(
        2.25, abs=0.3
    )
    pos = integers(-10, 10).filter(lambda n: n % 2 == 1)
    assert all(n % 2 == 1 for n in draws(pos))
    # shrinking respects the filter
    t = pos.generate(np.random.default_rng(3))
    assert descend(t, lambda n: True) in (1, -1)
    with pytest.raises(Unsatisfiable):
        integers(0, 10).filter(lambda n: n > 100).example()
    # flatmap: a list length, then that many floats
    vec = integers(1, 6).flatmap(lambda k: tuples(*[floats(0, 1)] * k))
    t = vec.generate(np.random.default_rng(7))
    assert 1 <= len(t.value) <= 6
    shrunk = descend(t, lambda v: len(v) >= 2)
    assert len(shrunk) == 2


def test_builds_rebuilds_and_skips_rejected_shrinks() -> None:
    class Interval:
        def __init__(self, lo: float, hi: float) -> None:
            if not lo < hi:
                raise ValueError("empty")
            self.lo, self.hi = lo, hi

    s = builds(Interval, lo=floats(0, 5), hi=floats(0, 10))
    ivs = draws(s, n=100)
    assert all(iv.lo < iv.hi for iv in ivs)
    t = s.generate(np.random.default_rng(2))
    small = descend(t, lambda iv: iv.hi - iv.lo > 1.0)
    assert small.hi - small.lo > 1.0 and small.lo == 0.0
    assert repr(s) == "builds(Interval)"
    with pytest.raises(Unsatisfiable):
        builds(Interval, lo=just(1.0), hi=just(0.0)).example()


def test_reprs() -> None:
    assert repr(floats(0, 1)) == "floats(0, 1)"
    assert repr(floats(1, 10, log=True)) == "log_floats(1, 10)"
    assert repr(integers(0, 3)) == "integers(0, 3)"
    assert "one_of(just(1), floats(0, 1))" == repr(one_of(just(1), floats(0, 1)))
    assert repr(floats(0, 1).map(abs, name="abs_floats")) == "abs_floats"
