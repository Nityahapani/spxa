r"""
Lowering: a process expression → one IR exponent over a flat parameter vector.

The algebra of the composition becomes substitution in the IR:

.. math::
    \psi_{X+Y}(u) = \psi_X(u) + \psi_Y(u), \qquad
    \psi_{cX}(u) = \psi_X(cu), \qquad
    \psi_{X\circ T}(u) = \psi_T\big(-i\,\psi_X(u)\big),

so a subordinated process is the clock's exponent built on the parent's
exponent as its argument, and the whole tree becomes a single DAG with
shared subexpressions. Leaves come from :mod:`spxa.compile.rules`. A leaf
without a rule (any other exact Lévy process) becomes an opaque ``call``
node evaluated by its own ``_char_exponent``; it is fixed (its parameters
are not inputs of the kernel) and has no parameter derivative.

Parameters are numbered in the order :class:`~spxa.graph.autodiff.process.DiffProcess`
uses (depth first, left before right, the base of ``c * X`` before ``c``),
and named ``"<Family>.<param>"`` with ``"[k]"`` for repeated families and
``"scale"`` (``"scale[k]"``) for scale factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from spxa.compile import ir as I
from spxa.compile.ir import CompileError, Expr
from spxa.compile.rules import LeafSpec, spec_for

__all__ = [
    "Lowered",
    "LNode",
    "LLeaf",
    "LScale",
    "LSum",
    "LSub",
    "LOpaque",
    "lower",
    "is_analytic",
]


# ── structural tree (kept for rebuilding processes and for sampling) ────────


class LNode:
    def psi(self, u: Expr, P: Sequence[Expr]) -> Expr:  # pragma: no cover - abstract
        raise NotImplementedError

    def build(self, theta: Sequence[float]) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def story(self, theta: Sequence[float]) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class LLeaf(LNode):
    spec: LeafSpec
    slots: Tuple[int, ...]
    label: str

    def psi(self, u: Expr, P: Sequence[Expr]) -> Expr:
        return self.spec.psi(u, [P[i] for i in self.slots])

    def build(self, theta: Sequence[float]) -> Any:
        return self.spec.build([float(theta[i]) for i in self.slots])

    def story(self, theta: Sequence[float]) -> str:
        body = ", ".join(f"{n}={float(theta[i]):.6g}" for n, i in zip(self.spec.names, self.slots))
        return f"{self.spec.family}({body})"


@dataclass
class LScale(LNode):
    base: LNode
    slot: int

    def psi(self, u: Expr, P: Sequence[Expr]) -> Expr:
        return self.base.psi(P[self.slot] * u, P)

    def build(self, theta: Sequence[float]) -> Any:
        return float(theta[self.slot]) * self.base.build(theta)

    def story(self, theta: Sequence[float]) -> str:
        return f"{float(theta[self.slot]):.6g} * {self.base.story(theta)}"


@dataclass
class LSum(LNode):
    a: LNode
    b: LNode

    def psi(self, u: Expr, P: Sequence[Expr]) -> Expr:
        return self.a.psi(u, P) + self.b.psi(u, P)

    def build(self, theta: Sequence[float]) -> Any:
        return self.a.build(theta) + self.b.build(theta)

    def story(self, theta: Sequence[float]) -> str:
        return f"({self.a.story(theta)} + {self.b.story(theta)})"


@dataclass
class LSub(LNode):
    x: LNode
    clock: LNode

    def psi(self, u: Expr, P: Sequence[Expr]) -> Expr:
        return self.clock.psi(-1j * self.x.psi(u, P), P)

    def build(self, theta: Sequence[float]) -> Any:
        return self.x.build(theta) @ self.clock.build(theta)

    def story(self, theta: Sequence[float]) -> str:
        return f"({self.x.story(theta)} @ {self.clock.story(theta)})"


@dataclass
class LOpaque(LNode):
    process: Any

    def psi(self, u: Expr, P: Sequence[Expr]) -> Expr:
        return I.call(self.process, u)

    def build(self, theta: Sequence[float]) -> Any:
        return self.process

    def story(self, theta: Sequence[float]) -> str:
        return f"{self.process!r}"


@dataclass
class Lowered:
    """The result of :func:`lower`."""

    psi: Expr
    names: Tuple[str, ...]
    values: np.ndarray
    tree: LNode
    params: Tuple[Expr, ...]
    opaque: Tuple[Any, ...]

    @cached_property
    def analytic(self) -> bool:
        """Whether :math:`\\psi` is analytic in ``u`` (cumulants, complex ``u``)."""
        return is_analytic(self.psi)

    def build(self, theta: Optional[Sequence[float]] = None) -> Any:
        """The process at ``theta`` (default: the lowered values)."""
        th = self.values if theta is None else np.asarray(theta, dtype=float)
        return self.tree.build(th)

    def story(self, theta: Optional[Sequence[float]] = None) -> str:
        th = self.values if theta is None else np.asarray(theta, dtype=float)
        return self.tree.story(th)


def _parse(
    proc: Any,
    values: List[float],
    names: List[str],
    counts: Dict[str, int],
    opaque: List[Any],
    allow_opaque: bool,
) -> LNode:
    from spxa.core.exactness import ExactnessLevel
    from spxa.core.process import _ComposedProcess, _ScaledProcess

    if isinstance(proc, _ScaledProcess):
        base = _parse(proc._base, values, names, counts, opaque, allow_opaque)
        slot = len(values)
        values.append(float(proc._scale))
        k = counts.get("scale", 0)
        counts["scale"] = k + 1
        names.append("scale" if k == 0 else f"scale[{k}]")
        return LScale(base, slot)
    if isinstance(proc, _ComposedProcess):
        left = _parse(proc._left, values, names, counts, opaque, allow_opaque)
        right = _parse(proc._right, values, names, counts, opaque, allow_opaque)
        if proc._operation == "+":
            return LSum(left, right)
        if proc._operation == "@":
            return LSub(left, right)
        raise CompileError(f"composition {proc._operation!r} cannot be compiled")
    spec = spec_for(proc)
    if spec is None:
        if not allow_opaque:
            raise CompileError(f"{type(proc).__name__} has no lowering rule")
        if getattr(proc, "exactness", None) != ExactnessLevel.EXACT:
            raise CompileError(
                f"{type(proc).__name__} is not an exact Lévy process "
                f"(ExactnessLevel.{getattr(proc, 'exactness', None)}); only the Lévy algebra "
                "(exact processes under +, scaling and @) compiles"
            )
        opaque.append(proc)
        return LOpaque(proc)
    k = counts.get(spec.family, 0)
    counts[spec.family] = k + 1
    label = spec.family if k == 0 else f"{spec.family}[{k}]"
    slots = []
    for pname, v in zip(spec.names, spec.values):
        slots.append(len(values))
        values.append(float(v))
        names.append(f"{label}.{pname}")
    return LLeaf(spec, tuple(slots), label)


def lower(process: Any, allow_opaque: bool = True) -> Lowered:
    """
    Lower ``process`` (a zoo Lévy process or a ``+``/``*``/``@`` composition).

    Parameters
    ----------
    process :
        The process expression.
    allow_opaque :
        Accept exact Lévy leaves without a rule as opaque calls (default);
        if False, raise :class:`~spxa.compile.ir.CompileError` instead.
    """
    from spxa.core.process import Process

    if not isinstance(process, Process):
        raise CompileError(f"expected a spxa Process, got {type(process).__name__}")
    values: List[float] = []
    names: List[str] = []
    opaque: List[Any] = []
    tree = _parse(process, values, names, {}, opaque, allow_opaque)
    params = tuple(I.param(i, n) for i, n in enumerate(names))
    psi = tree.psi(I.U, params)
    if psi.kind != I.COMPLEX:
        psi = psi + I.const(0j)
    return Lowered(psi, tuple(names), np.array(values, dtype=float), tree, params, tuple(opaque))


def is_analytic(e: Union[Expr, Sequence[Expr]]) -> bool:
    r"""
    Whether the expression is analytic in :math:`u`.

    It is not when ``real``, ``imag``, ``abs`` or ``sign`` of a
    :math:`u`-dependent value is used as a value (rather than only inside a
    comparison, as in the overflow-free :math:`\log\cosh`).
    """
    roots = [e] if isinstance(e, Expr) else list(e)
    nodes = I.postorder(roots)
    users: Dict[int, List[Expr]] = {}
    for n in nodes:
        for a in n.args:
            users.setdefault(id(a), []).append(n)
    root_ids = {id(r) for r in roots}
    for n in nodes:
        if n.op in ("real", "imag", "abs", "sign") and n.depends_u:
            if id(n) in root_ids:
                return False
            for user in users.get(id(n), []):
                if user.op not in ("ge", "gt"):
                    return False
    return True
