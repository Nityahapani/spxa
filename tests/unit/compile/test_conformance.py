"""
Conformance of spxa.compile on random composition trees.

Seeded random expressions over every family with a lowering rule (sums,
scalar multiples, subordination by every subordinator family, nesting to
depth three) are checked end to end: the compiled exponent against the zoo,
the compiled gradient against :class:`~spxa.graph.autodiff.process.DiffProcess`,
the Taylor-mode cumulants against the zoo's exact cumulants, the fused
sampler against the characteristic function, and the Numba and JAX
backends against NumPy.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, List

import numpy as np
import pytest

from spxa.compile import BackendUnsupported, CompiledProcess
from spxa.graph.autodiff.process import DiffProcess, UnsupportedProcessError
from spxa.zoo import levy as L
from spxa.zoo.levy import compound_poisson as cp

U = np.linspace(-12.0, 12.0, 49)

#: summands with exact vectorised samplers (also usable under a clock)
SAMPLED: List[Callable[[np.random.Generator], Any]] = [
    lambda r: L.BrownianMotion(r.uniform(-0.1, 0.1), r.uniform(0.1, 0.4)),
    lambda r: L.Drift(r.uniform(-0.2, 0.2)),
    lambda r: L.VarianceGamma(r.uniform(0.1, 0.3), r.uniform(0.1, 0.5), r.uniform(-0.2, 0.2)),
    lambda r: L.NIG(
        r.uniform(8, 12), r.uniform(-3, 3), r.uniform(0.3, 0.8), r.uniform(-0.05, 0.05)
    ),
    lambda r: L.MertonJumpDiffusion(
        r.uniform(-0.1, 0.1), r.uniform(0.1, 0.3), r.uniform(0.5, 2), r.uniform(-0.2, 0.1), 0.15
    ),
    lambda r: L.KouJumpDiffusion(
        r.uniform(-0.1, 0.1), r.uniform(0.1, 0.3), r.uniform(0.5, 2), 0.4, 9.0, r.uniform(4, 7)
    ),
    lambda r: cp.CompoundPoisson(r.uniform(0.5, 2), cp.NormalJump(r.uniform(-0.2, 0.2), 0.1)),
    lambda r: cp.CompoundPoisson(
        r.uniform(0.5, 2), cp.DoubleExponentialJump(0.3, r.uniform(5, 9), 4.0)
    ),
    lambda r: cp.CompoundPoisson(r.uniform(0.5, 2), cp.UniformJump(-0.3, r.uniform(0.1, 0.5))),
    lambda r: cp.CompoundPoisson(1.0, cp.DiscreteJump([-0.5, 0.25, 1.0], [0.3, 0.5, 0.2])),
    lambda r: L.CGMY(r.uniform(0.5, 1.5), r.uniform(4, 6), r.uniform(6, 9), r.uniform(0.2, 0.8)),
]
#: summands drawn through their own sampler (top level only)
EXTRA: List[Callable[[np.random.Generator], Any]] = [
    lambda r: L.MeixnerProcess(r.uniform(0.2, 0.4), r.uniform(-1, 1), r.uniform(0.5, 1.5), 0.0),
    lambda r: L.NormalTemperedStableProcess(0.5, r.uniform(4, 8), 0.5, r.uniform(-0.2, 0.2), 0.2),
    lambda r: L.GeneralizedHyperbolicProcess(r.uniform(-1, 1), 10.0, r.uniform(-2, 2), 0.5, 0.0),
]
CLOCKS: List[Callable[[np.random.Generator], Any]] = [
    lambda r: L.GammaProcess(r.uniform(3, 6), r.uniform(3, 6)),
    lambda r: L.InverseGaussianProcess(1.0, r.uniform(1, 4)),
    lambda r: L.TemperedStable(r.uniform(0.3, 0.7), 1.0, r.uniform(2, 4)),
    lambda r: L.NegativeBinomialProcess(r.uniform(1, 3), r.uniform(0.2, 0.4)),
    lambda r: L.PoissonProcess(r.uniform(1, 3)),
    lambda r: cp.CompoundPoisson(r.uniform(1, 3), cp.ExponentialJump(r.uniform(2, 5))),
]


def _tree(r: np.random.Generator, depth: int, top: bool) -> Any:
    pool = SAMPLED + (EXTRA if top else [])
    if depth == 0 or r.random() < 0.3:
        return pool[r.integers(len(pool))](r)
    kind = r.integers(3)
    if kind == 0:
        return _tree(r, depth - 1, top) + _tree(r, depth - 1, top)
    if kind == 1:
        return float(r.choice([-1.5, 0.5, 2.0])) * _tree(r, depth - 1, top)
    clock = CLOCKS[r.integers(len(CLOCKS))](r)
    if r.random() < 0.25:
        clock = clock @ CLOCKS[r.integers(len(CLOCKS))](r)
    return _tree(r, depth - 1, False) @ clock


def _trees(n: int) -> List[Any]:
    r = np.random.default_rng(20261023)
    return [_tree(r, 3, True) for _ in range(n)]


TREES = _trees(24)


def _scaled_err(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))


def _zoo_psi(proc: Any) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(proc._char_exponent(U), dtype=complex)


def _check_gradient_by_differences(c: CompiledProcess, g: np.ndarray) -> None:
    """Fourth-order central differences of the zoo exponent of the rebuilt process."""
    th0 = c.values
    for j in range(c.n_params):
        h = 1e-3 * max(1.0, abs(th0[j]))
        vals = []
        for step in (2, 1, -1, -2):
            th = th0.copy()
            th[j] += step * h
            vals.append(_zoo_psi(c.to_process(th)))
        fd = (-vals[0] + 8 * vals[1] - 8 * vals[2] + vals[3]) / (12 * h)
        assert _scaled_err(g[:, j], fd) < 1e-6, c.names[j]


@pytest.mark.parametrize("k", range(len(TREES)))
def test_exponent_gradient_and_cumulants(k: int) -> None:
    proc = TREES[k]
    c = CompiledProcess(proc)
    ref = _zoo_psi(proc)
    assert _scaled_err(c.char_exponent(U), ref) < 1e-9, c.story()
    _, g = c.char_exponent_and_grad(U)
    try:
        dp = DiffProcess(proc)
    except UnsupportedProcessError:  # compound Poisson leaves: differences of the zoo
        _check_gradient_by_differences(c, g)
    else:
        assert dp.names == c.names
        _, g_ref = dp.char_exponent_and_grad(U)
        assert _scaled_err(g, np.asarray(g_ref)) < 1e-9
    kap = c.cumulants(4)
    zoo = proc.cumulants(4)
    np.testing.assert_allclose(kap, [zoo[n] for n in range(1, 5)], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("k", range(0, len(TREES), 2))
def test_sampler_law(k: int) -> None:
    c = CompiledProcess(TREES[k])
    n = 20_000
    x = c.sample_at(np.full(n, 0.5), np.random.default_rng(k))
    emp = np.exp(1j * np.outer(U[::6], x)).mean(axis=1)
    err = np.abs(emp - c.char_func(U[::6], 0.5)).max() * np.sqrt(n)
    assert err < 4.5, c.story()


@pytest.mark.parametrize("backend", ["numba", "jax"])
def test_backends_agree(backend: str) -> None:
    pytest.importorskip(backend)
    for proc in TREES[:6]:
        ref = CompiledProcess(proc)
        try:
            c = CompiledProcess(proc, backend)
            v, g = c.char_exponent_and_grad(U)
        except BackendUnsupported as exc:  # only GH's Bessel functions are numpy-only
            assert "kve" in str(exc)
            continue
        rv, rg = ref.char_exponent_and_grad(U)
        assert _scaled_err(v, rv) < 1e-12 and _scaled_err(g, rg) < 1e-12


def test_trees_cover_the_families() -> None:
    stories = " ".join(CompiledProcess(t).story() for t in TREES)
    for fam in ["@", "BrownianMotion", "CGMY", "NIG", "CompoundPoisson", "GammaProcess"]:
        assert fam in stories
