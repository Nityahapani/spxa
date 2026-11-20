r"""
Strategies for processes: zoo primitives and their compositions.

Primitives
    One strategy per Lévy family in the zoo. Each draws parameters from
    ranges wide enough to cover qualitatively different behaviour (light
    and heavy jumps, near-symmetric and skewed, low and high activity),
    and narrow enough that the closed forms stay numerically meaningful.
    Parameters shrink towards round values (1 for scales and rates, 0
    for drifts and skews), so shrunk counterexamples read like textbook
    examples. :data:`PRIMITIVES` maps family names to strategies.

Compositions
    :func:`compositions` builds expression trees with ``+``, ``c * X``,
    unary ``-`` and subordination ``X @ T``, up to a given depth. A
    composite counterexample shrinks first to its operands, then each
    operand shrinks in place. A law that fails for ``X + Y`` because of
    ``X`` alone is therefore reported with ``X``.

Martingales
    :func:`martingales` centres draws, :math:`X_t - E[X_1]t`, so laws
    with a martingale hypothesis are not vacuous.

Heavy tails
    :func:`levy_processes` excludes families without a finite mean
    (:class:`~spxa.zoo.levy.stable.AlphaStable`) unless
    ``heavy_tails=True``, because most moment laws presuppose it.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np

from spxa.core.process import Process
from spxa.laws.strategies import (
    Strategy,
    Tree,
    Unsatisfiable,
    builds,
    floats,
    one_of,
)
from spxa.zoo.levy import (
    CGMY,
    NIG,
    AlphaStable,
    BrownianMotion,
    CompoundPoisson,
    Drift,
    GammaProcess,
    InverseGaussianProcess,
    KouJumpDiffusion,
    MeixnerProcess,
    MertonJumpDiffusion,
    NegativeBinomialProcess,
    PoissonProcess,
    TemperedStable,
    VarianceGamma,
)
from spxa.zoo.levy.compound_poisson import NormalJump


def _quiet(f: Callable[..., Process]) -> Callable[..., Process]:
    def g(**kw: Any) -> Process:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f(**kw)

    g.__name__ = getattr(f, "__name__", "process")
    return g


def _nig(alpha: float, skew: float, delta: float, mu: float) -> Process:
    return NIG(alpha=alpha, beta=skew * alpha, delta=delta, mu=mu)


def _compound_normal(rate: float, mean: float, std: float) -> Process:
    return CompoundPoisson(rate, NormalJump(mean, std))


_nig.__name__ = "NIG"
_compound_normal.__name__ = "CompoundPoisson"


def drifts() -> Strategy[Process]:
    return builds(_quiet(Drift), mu=floats(-1.0, 1.0))


def brownian_motions() -> Strategy[Process]:
    return builds(
        _quiet(BrownianMotion), mu=floats(-0.5, 0.5), sigma=floats(0.05, 1.0, simplest=1.0)
    )


def gamma_processes() -> Strategy[Process]:
    return builds(
        _quiet(GammaProcess), a=floats(0.2, 5.0, simplest=1.0), b=floats(0.5, 10.0, simplest=1.0)
    )


def inverse_gaussian_processes() -> Strategy[Process]:
    return builds(
        _quiet(InverseGaussianProcess),
        mu=floats(0.2, 3.0, simplest=1.0),
        lam=floats(0.2, 5.0, simplest=1.0),
    )


def variance_gammas() -> Strategy[Process]:
    return builds(
        _quiet(VarianceGamma),
        sigma=floats(0.05, 0.5),
        nu=floats(0.05, 1.0, simplest=1.0),
        theta=floats(-0.5, 0.5),
    )


def nigs() -> Strategy[Process]:
    return builds(
        _quiet(_nig),
        alpha=floats(1.0, 20.0, simplest=1.0),
        skew=floats(-0.9, 0.9),
        delta=floats(0.1, 2.0, simplest=1.0),
        mu=floats(-0.2, 0.2),
    )


def meixners() -> Strategy[Process]:
    return builds(
        _quiet(MeixnerProcess),
        alpha=floats(0.1, 2.0, simplest=1.0),
        beta=floats(-2.5, 2.5),
        delta=floats(0.1, 2.0, simplest=1.0),
        m=floats(-0.2, 0.2),
    )


def cgmys() -> Strategy[Process]:
    return builds(
        _quiet(CGMY),
        C=floats(0.1, 2.0, simplest=1.0),
        G=floats(1.0, 10.0, simplest=1.0),
        M=floats(1.5, 10.0, simplest=2.0),
        Y=floats(0.1, 1.9, simplest=0.5),
    )


def poisson_processes() -> Strategy[Process]:
    return builds(
        _quiet(PoissonProcess),
        rate=floats(0.2, 5.0, simplest=1.0),
        jump_size=floats(0.1, 2.0, simplest=1.0),
    )


def compound_poissons() -> Strategy[Process]:
    return builds(
        _quiet(_compound_normal),
        rate=floats(0.2, 5.0, simplest=1.0),
        mean=floats(-0.3, 0.3),
        std=floats(0.02, 0.5, simplest=0.1),
    )


def merton_jump_diffusions() -> Strategy[Process]:
    return builds(
        _quiet(MertonJumpDiffusion),
        drift=floats(-0.3, 0.3),
        sigma=floats(0.05, 0.5, simplest=0.2),
        rate=floats(0.2, 5.0, simplest=1.0),
        jump_mean=floats(-0.3, 0.3),
        jump_std=floats(0.02, 0.5, simplest=0.1),
    )


def kou_jump_diffusions() -> Strategy[Process]:
    return builds(
        _quiet(KouJumpDiffusion),
        drift=floats(-0.3, 0.3),
        sigma=floats(0.05, 0.5, simplest=0.2),
        rate=floats(0.2, 5.0, simplest=1.0),
        p=floats(0.1, 0.9, simplest=0.5),
        eta1=floats(2.0, 30.0, simplest=10.0),
        eta2=floats(2.0, 30.0, simplest=10.0),
    )


def tempered_stables() -> Strategy[Process]:
    return builds(
        _quiet(TemperedStable),
        alpha=floats(0.1, 0.9, simplest=0.5),
        C=floats(0.1, 2.0, simplest=1.0),
        lam=floats(0.5, 10.0, simplest=1.0),
    )


def negative_binomials() -> Strategy[Process]:
    return builds(
        _quiet(NegativeBinomialProcess),
        r=floats(0.2, 5.0, simplest=1.0),
        p=floats(0.1, 0.9, simplest=0.5),
    )


def alpha_stables() -> Strategy[Process]:
    """Strictly heavy-tailed: :math:`1 < \\alpha < 2`, so the mean exists but the variance does not."""
    return builds(
        _quiet(AlphaStable),
        alpha=floats(1.1, 1.9, simplest=1.5),
        beta=floats(-1.0, 1.0),
        sigma=floats(0.1, 1.0, simplest=1.0),
        mu=floats(-0.5, 0.5),
    )


#: Family name → strategy, for the light-tailed Lévy primitives.
PRIMITIVES: dict[str, Callable[[], Strategy[Process]]] = {
    "Drift": drifts,
    "BrownianMotion": brownian_motions,
    "GammaProcess": gamma_processes,
    "InverseGaussianProcess": inverse_gaussian_processes,
    "VarianceGamma": variance_gammas,
    "NIG": nigs,
    "MeixnerProcess": meixners,
    "CGMY": cgmys,
    "PoissonProcess": poisson_processes,
    "CompoundPoisson": compound_poissons,
    "MertonJumpDiffusion": merton_jump_diffusions,
    "KouJumpDiffusion": kou_jump_diffusions,
    "TemperedStable": tempered_stables,
    "NegativeBinomialProcess": negative_binomials,
}

#: Primitives that are subordinators for every parameter value drawn.
SUBORDINATORS: tuple[str, ...] = (
    "GammaProcess",
    "InverseGaussianProcess",
    "PoissonProcess",
    "TemperedStable",
    "NegativeBinomialProcess",
)


def levy_processes(
    families: Optional[Sequence[str]] = None, *, heavy_tails: bool = False
) -> Strategy[Process]:
    """One primitive from ``families`` (default: all of :data:`PRIMITIVES`)."""
    names = list(PRIMITIVES) if families is None else list(families)
    unknown = [n for n in names if n not in PRIMITIVES and n != "AlphaStable"]
    if unknown:
        raise ValueError(f"unknown families: {unknown}; known: {sorted(PRIMITIVES)}")
    options = [PRIMITIVES[n]() for n in names if n in PRIMITIVES]
    if heavy_tails or "AlphaStable" in names:
        options.append(alpha_stables())
    return one_of(*options)


def subordinators(families: Optional[Sequence[str]] = None) -> Strategy[Process]:
    """A subordinator: Gamma, inverse Gaussian, Poisson, tempered stable or negative binomial."""
    names = list(SUBORDINATORS) if families is None else list(families)
    bad = [n for n in names if n not in SUBORDINATORS]
    if bad:
        raise ValueError(f"not subordinator families: {bad}")
    return one_of(*[PRIMITIVES[n]() for n in names])


# ── compositions ──────────────────────────────────────────────────────────

OPERATIONS = ("+", "*", "-", "@")


def _combine(op: str, args: Sequence[Any]) -> Process:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if op == "+":
            return args[0] + args[1]
        if op == "*":
            return args[0] * args[1]
        if op == "-":
            return -args[0]
        return args[0] @ args[1]


def _node(op: str, trees: Sequence[Tree[Any]], operands: Sequence[int]) -> Tree[Process]:
    """A composite tree: shrinks to its process operands, then shrinks operands in place."""
    value = _combine(op, [t.value for t in trees])

    def children() -> Iterator[Tree[Process]]:
        for k in operands:
            yield trees[k]
        for k, t in enumerate(trees):
            for c in t.children():
                try:
                    yield _node(op, [*trees[:k], c, *trees[k + 1 :]], operands)
                except Exception:
                    continue

    return Tree(value, children)


class _Compositions(Strategy[Process]):
    def __init__(
        self,
        leaves: Strategy[Process],
        clocks: Strategy[Process],
        scalars: Strategy[float],
        max_depth: int,
        operations: Sequence[str],
        leaf_probability: float,
    ) -> None:
        bad = [o for o in operations if o not in OPERATIONS]
        if bad:
            raise ValueError(f"unknown operations {bad}; choose from {OPERATIONS}")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        self.leaves, self.clocks, self.scalars = leaves, clocks, scalars
        self.max_depth, self.operations = max_depth, tuple(operations)
        self.leaf_probability = leaf_probability

    def _gen(self, rng: np.random.Generator, depth: int) -> Tree[Process]:
        if depth == 0 or not self.operations or rng.uniform() < self.leaf_probability:
            return self.leaves.generate(rng)
        op = self.operations[int(rng.integers(len(self.operations)))]
        if op == "+":
            return _node(op, [self._gen(rng, depth - 1), self._gen(rng, depth - 1)], (0, 1))
        if op == "*":
            return _node(op, [self._gen(rng, depth - 1), self.scalars.generate(rng)], (0,))
        if op == "-":
            return _node(op, [self._gen(rng, depth - 1)], (0,))
        return _node(op, [self._gen(rng, depth - 1), self.clocks.generate(rng)], (0, 1))

    def generate(self, rng: np.random.Generator) -> Tree[Process]:
        for _ in range(100):
            try:
                return self._gen(rng, self.max_depth)
            except (ValueError, TypeError):
                continue
        raise Unsatisfiable("compositions: every draw was rejected")

    def __repr__(self) -> str:
        return f"compositions(max_depth={self.max_depth}, operations={''.join(self.operations)})"


def scalars(lo: float = -2.0, hi: float = 2.0) -> Strategy[float]:
    """Non-zero scale factors, shrinking towards 1."""
    return floats(lo, hi, simplest=1.0).filter(lambda c: abs(c) > 1e-3, name="scalars")


def compositions(
    leaves: Optional[Strategy[Process]] = None,
    *,
    max_depth: int = 2,
    operations: Sequence[str] = OPERATIONS,
    clocks: Optional[Strategy[Process]] = None,
    scale_factors: Optional[Strategy[float]] = None,
    leaf_probability: float = 0.3,
) -> Strategy[Process]:
    """Expression trees over ``leaves`` with ``+``, ``*``, ``-`` and ``@``.

    ``clocks`` (default :func:`subordinators`) supplies the right operand of
    ``@``, and ``scale_factors`` (default :func:`scalars`) supplies the
    constants of ``*``. At each level below ``max_depth``, a leaf is drawn
    with probability ``leaf_probability``.
    """
    return _Compositions(
        leaves if leaves is not None else levy_processes(),
        clocks if clocks is not None else subordinators(),
        scale_factors if scale_factors is not None else scalars(),
        max_depth,
        operations,
        leaf_probability,
    )


def _centre(X: Process) -> Process:
    """``X`` if it is flagged a martingale, else ``X + Drift(-E[X_1])``."""
    if X.properties.is_martingale:
        return X
    return X + Drift(-float(X.cumulants(1)[1]))


def martingales(base: Optional[Strategy[Process]] = None) -> Strategy[Process]:
    r"""Martingales: processes from ``base`` with their mean drift removed.

    A Lévy process is a martingale exactly when :math:`E|X_1| < \infty` and
    :math:`E[X_1] = 0`. Each draw from ``base`` (default: depth-1
    :func:`compositions`) with a finite mean is centred to
    :math:`X_t - E[X_1]\,t`. Generic strategies almost never hit
    :math:`E[X_1] = 0` exactly, so laws with a martingale hypothesis
    need this strategy, or they hold vacuously.
    """
    if base is None:
        base = compositions(max_depth=1)
    finite = base.filter(lambda X: bool(X.properties.has_finite_mean), name="finite mean")
    return finite.map(_centre, name="martingales")
