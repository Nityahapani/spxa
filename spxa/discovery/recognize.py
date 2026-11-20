"""
Translate spxa processes into discovery models.

:func:`recognize` walks an expression built from zoo primitives with ``+``,
scalar ``*``, unary ``-`` and ``@``. It returns the equivalent
:class:`~spxa.discovery.expr.Model`, which has exact cumulants of every
order and a closed-form characteristic function. The core gives neither for
subordinated processes, because ``_ComposedProcess`` has no triplet for ``@``.
"""

from __future__ import annotations

from spxa.core.process import Process, _ComposedProcess, _ScaledProcess
from spxa.discovery.expr import Leaf, Model, Node, Scale, Subordinate, Sum
from spxa.discovery.families import (
    BROWNIAN,
    CGMY_FAMILY,
    DOUBLE_EXP_JUMPS,
    DRIFT,
    GAMMA,
    INVERSE_GAUSSIAN,
    MEIXNER,
    NIG_FAMILY,
    NORMAL_JUMPS,
    POISSON,
    VARIANCE_GAMMA,
)

_Parts = list[tuple[Node, tuple[float, ...]]]


class NotRecognizedError(TypeError):
    """The process (or a part of it) has no counterpart among the discovery families."""


def _with_drift(parts: _Parts, mu: float) -> _Parts:
    return parts + [(Leaf(DRIFT), (float(mu),))] if mu != 0 else parts


def _primitive(proc: Process) -> _Parts:
    from spxa.zoo.levy import (
        CGMY,
        NIG,
        BrownianMotion,
        CompoundPoisson,
        DoubleExponentialJump,
        Drift,
        GammaProcess,
        InverseGaussianProcess,
        KouJumpDiffusion,
        MeixnerProcess,
        MertonJumpDiffusion,
        NormalJump,
        PoissonProcess,
        VarianceGamma,
    )

    if isinstance(proc, Drift):
        return [(Leaf(DRIFT), (proc.mu,))]
    if isinstance(proc, BrownianMotion):
        return _with_drift([(Leaf(BROWNIAN), (proc.sigma,))], proc.mu)
    if isinstance(proc, GammaProcess):
        return [(Leaf(GAMMA), (proc.a, proc.b))]
    if isinstance(proc, InverseGaussianProcess):
        return [(Leaf(INVERSE_GAUSSIAN), (proc.mu, proc.lam))]
    if isinstance(proc, VarianceGamma):
        return [(Leaf(VARIANCE_GAMMA), (proc.sigma, proc.nu, proc.theta))]
    if isinstance(proc, NIG):
        return _with_drift([(Leaf(NIG_FAMILY), (proc.alpha, proc.beta, proc.delta))], proc.mu)
    if isinstance(proc, MeixnerProcess):
        return _with_drift([(Leaf(MEIXNER), (proc.alpha, proc.beta, proc.delta))], proc.m)
    if isinstance(proc, PoissonProcess):
        c = float(proc.jump_size)
        if c == 0:
            raise NotRecognizedError("a Poisson process with zero jump size is the zero process")
        node: Node = Leaf(POISSON) if c > 0 else Scale(-1.0, Leaf(POISSON))
        return [(node, (proc.rate, abs(c)))]
    if isinstance(proc, MertonJumpDiffusion):
        parts: _Parts = [
            (Leaf(BROWNIAN), (proc.sigma,)),
            (Leaf(NORMAL_JUMPS), (proc.rate, proc.jump_mean, proc.jump_std)),
        ]
        return _with_drift(parts, proc.drift)
    if isinstance(proc, KouJumpDiffusion):
        if not 0 < proc.p < 1:
            raise NotRecognizedError(
                "Kou jumps with p in {0, 1} are one-sided; no family covers them"
            )
        parts = [
            (Leaf(BROWNIAN), (proc.sigma,)),
            (Leaf(DOUBLE_EXP_JUMPS), (proc.rate, proc.p, proc.eta1, proc.eta2)),
        ]
        return _with_drift(parts, proc.drift)
    if isinstance(proc, CompoundPoisson):
        jd = proc.jump_dist
        if isinstance(jd, NormalJump):
            return [(Leaf(NORMAL_JUMPS), (proc.rate, jd.mean, jd.std))]
        if isinstance(jd, DoubleExponentialJump) and 0 < jd.p < 1:
            return [(Leaf(DOUBLE_EXP_JUMPS), (proc.rate, jd.p, jd.eta1, jd.eta2))]
        raise NotRecognizedError(
            f"compound Poisson with {type(jd).__name__} jumps has no discovery family"
        )
    if isinstance(proc, CGMY):
        if not 0 < proc.Y < 1:
            raise NotRecognizedError(f"the CGMY family covers Y in (0, 1), got Y={proc.Y}")
        return [(Leaf(CGMY_FAMILY), (proc.C, proc.G, proc.M, proc.Y))]
    raise NotRecognizedError(
        f"{type(proc).__name__} has no counterpart among the discovery families"
    )


def _walk(proc: Process) -> tuple[Node, list[tuple[float, ...]]]:
    if isinstance(proc, _ScaledProcess):
        node, params = _walk(proc._base)
        return Scale(float(proc._scale), node), params
    if isinstance(proc, _ComposedProcess):
        left, lp = _walk(proc._left)
        right, rp = _walk(proc._right)
        if proc._operation == "+":
            return left + right, lp + rp
        if proc._operation == "@":
            return Subordinate(left, right), lp + rp
        raise NotRecognizedError(f"operation {proc._operation!r} is not a Lévy composition")
    parts = _primitive(proc)
    if len(parts) == 1:
        return parts[0][0], [parts[0][1]]
    return Sum([n for n, _ in parts]), [p for _, p in parts]


def recognize(process: Process) -> Model:
    """Return the discovery :class:`Model` equal in law to ``process``.

    Raises
    ------
    NotRecognizedError
        If some primitive in the expression has no discovery family (for
        example a stable process, a Heston model or compound Poisson with
        uniform jumps).
    """
    node, params = _walk(process)
    return Model(node, params)
