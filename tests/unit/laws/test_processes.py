"""Process strategies: primitives, subordinators and compositions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spxa.core.process import Process
from spxa.laws.processes import (
    PRIMITIVES,
    SUBORDINATORS,
    alpha_stables,
    compositions,
    levy_processes,
    scalars,
    subordinators,
)
from spxa.laws.strategies import Tree, floats, just


def _draw(strategy, n: int = 40, seed: int = 0) -> list[Process]:
    rng = np.random.default_rng(seed)
    return [strategy.generate(rng).value for _ in range(n)]


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_every_primitive_has_exact_finite_cumulants(name: str) -> None:
    for p in _draw(PRIMITIVES[name](), n=15, seed=1):
        assert isinstance(p, Process) and type(p).__name__ == name
        k = p.cumulants(2)
        assert np.isfinite(k[1]) and np.isfinite(k[2]) and k[2] >= 0
        cf = p.char_func(np.array([0.0, 1.0]), 1.0)
        assert cf[0] == pytest.approx(1.0) and abs(cf[1]) <= 1 + 1e-12


def test_subordinators_are_subordinators() -> None:
    assert set(SUBORDINATORS) <= set(PRIMITIVES)
    for p in _draw(subordinators(), n=60):
        assert p.properties.is_subordinator
    with pytest.raises(ValueError, match="not subordinator"):
        subordinators(["BrownianMotion"])


def test_levy_processes_selection() -> None:
    kinds = {type(p).__name__ for p in _draw(levy_processes(), n=300)}
    assert kinds == set(PRIMITIVES)
    only = {type(p).__name__ for p in _draw(levy_processes(["VarianceGamma", "NIG"]), n=40)}
    assert only == {"VarianceGamma", "NIG"}
    heavy = {type(p).__name__ for p in _draw(levy_processes(heavy_tails=True), n=300)}
    assert "AlphaStable" in heavy
    assert all(1.1 <= p.alpha <= 1.9 for p in _draw(alpha_stables(), n=20))
    with pytest.raises(ValueError, match="unknown families"):
        levy_processes(["Wiener"])


def test_parameters_shrink_to_round_values() -> None:
    tree = PRIMITIVES["GammaProcess"]().generate(np.random.default_rng(3))
    while True:  # shrink as far as possible
        kids = list(tree.children())
        if not kids:
            break
        tree = kids[0]
    assert repr(tree.value) == "GammaProcess(a=1.0, b=1.0)"


def test_compositions_cover_every_operation() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reprs = [repr(p) for p in _draw(compositions(max_depth=2), n=150, seed=4)]
    assert any(" + " in r for r in reprs)
    assert any("@" in r for r in reprs)
    assert any(r.startswith("-") or "*" in r or "Scaled" in r for r in reprs)
    leaves_only = compositions(max_depth=0)
    assert all("+" not in repr(p) for p in _draw(leaves_only, n=20))
    with pytest.raises(ValueError, match="unknown operations"):
        compositions(operations=("^",))


def test_composites_shrink_to_their_operands_first() -> None:
    s = compositions(
        levy_processes(["BrownianMotion", "GammaProcess"]),
        max_depth=1,
        operations=("+",),
        leaf_probability=0.0,
    )
    tree: Tree[Process] = s.generate(np.random.default_rng(0))
    assert " + " in repr(tree.value)
    kids = [c.value for c in tree.children()]
    assert type(kids[0]).__name__ in {"BrownianMotion", "GammaProcess"}
    assert type(kids[1]).__name__ in {"BrownianMotion", "GammaProcess"}
    assert " + " in repr(kids[2])  # then operands shrink in place


def test_scaled_compositions_use_nonzero_constants() -> None:
    assert all(abs(c) > 1e-3 for c in _draw(scalars(), n=200))
    base = PRIMITIVES["Drift"]().example()
    s = compositions(
        just(base),
        operations=("*",),
        max_depth=1,
        scale_factors=floats(2.0, 3.0),
        leaf_probability=0.0,
    )
    ratio = s.example().cumulants(1)[1] / base.cumulants(1)[1]
    assert 2.0 <= ratio <= 3.0
