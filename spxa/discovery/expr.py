r"""
Composition expressions over primitive families.

A *structure* is a tree whose leaves are :class:`~spxa.discovery.families.Family`
objects. Its inner nodes are:

- :class:`Sum`: an independent sum, :math:`K = \sum_i K_i`;
- :class:`Scale`: multiplication by a fixed constant :math:`c`,
  :math:`K(z) = K_X(cz)`;
- :class:`Subordinate`: an independent random time change,
  :math:`K_{X \circ T} = K_T \circ K_X` (Sato 1999, Thm 30.1).

A :class:`Model` binds a structure to parameter values. It then gives
cumulants of any order exactly, through :class:`~spxa.discovery.series.CGFSeries`,
and the characteristic function in closed form, by composing the leaves'
cumulant exponents. It also builds the equivalent spxa :class:`~spxa.core.process.Process`
with the ordinary operators ``+``, ``*`` and ``@``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence, Union

import numpy as np
from scipy.optimize import brentq

from spxa.core.process import Process
from spxa.discovery.families import Family
from spxa.discovery.series import CGFSeries

Weighting = Union[bool, Literal["subordinators"]]

ParamList = tuple[tuple[float, ...], ...]


class Node(ABC):
    """A composition structure; parameters are supplied per leaf, in :meth:`leaves` order."""

    @abstractmethod
    def leaves(self) -> list[Leaf]: ...

    @property
    @abstractmethod
    def is_subordinator(self) -> bool:
        """Whether every parameter value gives non-decreasing paths."""

    @abstractmethod
    def key(self) -> str:
        """Canonical structural signature, e.g. ``"VG + -Gamma"``."""

    @abstractmethod
    def _series(self, params: list[tuple[float, ...]], order: int) -> CGFSeries: ...

    @abstractmethod
    def _log_mgf(self, z: np.ndarray, params: list[tuple[float, ...]]) -> np.ndarray: ...

    @abstractmethod
    def _strip(self, params: list[tuple[float, ...]]) -> tuple[float, float]: ...

    @abstractmethod
    def _process(self, params: list[tuple[float, ...]]) -> Process: ...

    @abstractmethod
    def _render(self, params: list[tuple[float, ...]], digits: int, weighted: Weighting) -> str: ...

    @property
    def n_params(self) -> int:
        return sum(leaf.family.n_params for leaf in self.leaves())

    @property
    def size(self) -> int:
        """Number of leaves, the structural complexity."""
        return len(self.leaves())

    def __add__(self, other: Node) -> Sum:
        left = self.children if isinstance(self, Sum) else (self,)
        right = other.children if isinstance(other, Sum) else (other,)
        return Sum((*left, *right))

    def __neg__(self) -> Node:
        return Scale(-1.0, self)

    def __rmul__(self, c: float) -> Scale:
        return Scale(float(c), self)

    def __matmul__(self, clock: Node) -> Subordinate:
        return Subordinate(self, clock)

    def __repr__(self) -> str:
        return f"<structure {self.key()}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())


class Leaf(Node):
    """A single primitive family."""

    def __init__(self, family: Family) -> None:
        self.family = family

    def leaves(self) -> list[Leaf]:
        return [self]

    @property
    def is_subordinator(self) -> bool:
        return self.family.is_subordinator

    def key(self) -> str:
        return self.family.name

    def _series(self, params: list[tuple[float, ...]], order: int) -> CGFSeries:
        return CGFSeries(self.family.cumulants(params.pop(0), order))

    def _log_mgf(self, z: np.ndarray, params: list[tuple[float, ...]]) -> np.ndarray:
        return self.family.log_mgf(z, params.pop(0))

    def _strip(self, params: list[tuple[float, ...]]) -> tuple[float, float]:
        return self.family.strip(params.pop(0))

    def _process(self, params: list[tuple[float, ...]]) -> Process:
        return self.family.to_process(params.pop(0))

    def _render(self, params: list[tuple[float, ...]], digits: int, weighted: Weighting) -> str:
        return self._render_scaled(params, digits, weighted, 1.0)

    def _weights(self, weighted: Weighting) -> bool:
        if self.family.name == "Drift" or not weighted:
            return False
        return weighted is True or self.family.is_subordinator

    def _render_scaled(
        self, params: list[tuple[float, ...]], digits: int, weighted: Weighting, factor: float
    ) -> str:
        """Render ``factor * leaf``, folding the factor into the weight when weighting."""
        p = params.pop(0)
        if self._weights(weighted):
            c, unit = self.family.unit_scale(p)
            c *= factor
            if not np.isclose(c, 1.0, rtol=10.0 ** (-digits)):
                if np.isclose(c, -1.0, rtol=10.0 ** (-digits)):
                    return f"-{self.family.format(unit, digits)}"
                return f"{c:.{digits}g} * {self.family.format(unit, digits)}"
            return self.family.format(unit, digits)
        text = self.family.format(p, digits)
        if factor == 1.0:
            return text
        return f"-{text}" if factor == -1.0 else f"{factor:.{digits}g} * {text}"


class Scale(Node):
    """Multiplication by a fixed (structural) constant."""

    def __init__(self, c: float, child: Node) -> None:
        if c == 0 or not np.isfinite(c):
            raise ValueError(f"scale must be finite and non-zero, got {c}")
        self.c = float(c)
        self.child = child

    def leaves(self) -> list[Leaf]:
        return self.child.leaves()

    @property
    def is_subordinator(self) -> bool:
        return self.c > 0 and self.child.is_subordinator

    def key(self) -> str:
        inner = self.child.key()
        if not isinstance(self.child, Leaf):
            inner = f"({inner})"
        return f"-{inner}" if self.c == -1.0 else f"{self.c:g}*{inner}"

    def _series(self, params: list[tuple[float, ...]], order: int) -> CGFSeries:
        return self.child._series(params, order).scale(self.c)

    def _log_mgf(self, z: np.ndarray, params: list[tuple[float, ...]]) -> np.ndarray:
        return self.child._log_mgf(self.c * z, params)

    def _strip(self, params: list[tuple[float, ...]]) -> tuple[float, float]:
        lo, hi = self.child._strip(params)
        return (lo / self.c, hi / self.c) if self.c > 0 else (hi / self.c, lo / self.c)

    def _process(self, params: list[tuple[float, ...]]) -> Process:
        proc = self.child._process(params)
        return -proc if self.c == -1.0 else self.c * proc

    def _render(self, params: list[tuple[float, ...]], digits: int, weighted: Weighting) -> str:
        if isinstance(self.child, Leaf):
            return self.child._render_scaled(params, digits, weighted, self.c)
        inner = f"({self.child._render(params, digits, weighted)})"
        return f"-{inner}" if self.c == -1.0 else f"{self.c:.{digits}g} * {inner}"


class Sum(Node):
    """Independent sum of two or more structures."""

    def __init__(self, children: Sequence[Node]) -> None:
        if len(children) < 2:
            raise ValueError("a Sum needs at least two terms")
        self.children = tuple(children)

    def leaves(self) -> list[Leaf]:
        return [leaf for child in self.children for leaf in child.leaves()]

    @property
    def is_subordinator(self) -> bool:
        return all(child.is_subordinator for child in self.children)

    def key(self) -> str:
        return " + ".join(f"({c.key()})" if isinstance(c, Sum) else c.key() for c in self.children)

    def _series(self, params: list[tuple[float, ...]], order: int) -> CGFSeries:
        total = self.children[0]._series(params, order)
        for child in self.children[1:]:
            total = total + child._series(params, order)
        return total

    def _log_mgf(self, z: np.ndarray, params: list[tuple[float, ...]]) -> np.ndarray:
        return sum((child._log_mgf(z, params) for child in self.children), np.zeros_like(z))

    def _strip(self, params: list[tuple[float, ...]]) -> tuple[float, float]:
        strips = [child._strip(params) for child in self.children]
        return max(s[0] for s in strips), min(s[1] for s in strips)

    def _process(self, params: list[tuple[float, ...]]) -> Process:
        total = self.children[0]._process(params)
        for child in self.children[1:]:
            total = total + child._process(params)
        return total

    def _render(self, params: list[tuple[float, ...]], digits: int, weighted: Weighting) -> str:
        parts = [child._render(params, digits, weighted) for child in self.children]
        out = parts[0]
        for part in parts[1:]:
            out += f" - {part[1:]}" if part.startswith("-") else f" + {part}"
        return out


class Subordinate(Node):
    """Random time change :math:`X_{T_t}` by an independent subordinator ``clock``."""

    def __init__(self, base: Node, clock: Node) -> None:
        if not clock.is_subordinator:
            raise ValueError(f"the clock {clock.key()!r} is not a subordinator")
        self.base = base
        self.clock = clock

    def leaves(self) -> list[Leaf]:
        return self.base.leaves() + self.clock.leaves()

    @property
    def is_subordinator(self) -> bool:
        return self.base.is_subordinator

    def key(self) -> str:
        return f"({self.base.key()}) @ ({self.clock.key()})"

    def _series(self, params: list[tuple[float, ...]], order: int) -> CGFSeries:
        inner = self.base._series(params, order)
        return self.clock._series(params, order).compose(inner)

    def _log_mgf(self, z: np.ndarray, params: list[tuple[float, ...]]) -> np.ndarray:
        inner = self.base._log_mgf(z, params)
        return self.clock._log_mgf(inner, params)

    def _strip(self, params: list[tuple[float, ...]]) -> tuple[float, float]:
        n_base = len(self.base.leaves())
        base_params = params[:n_base]
        lo, hi = self.base._strip(list(base_params))
        _, clock_hi = self.clock._strip(params[n_base:])
        del params[: n_base + len(self.clock.leaves())]
        if not np.isfinite(clock_hi):
            return lo, hi

        def excess(z: float) -> float:
            return (
                float(self.base._log_mgf(np.array([z + 0j]), list(base_params)).real[0]) - clock_hi
            )

        # K_X is convex with K_X(0) = 0 < clock_hi, so {K_X < clock_hi} is an interval around 0
        def edge(bound: float, direction: float) -> float:
            far = bound if np.isfinite(bound) else direction
            if np.isfinite(bound):
                far = bound - direction * 1e-12 * max(1.0, abs(bound))
            else:
                while excess(far) < 0 and abs(far) < 1e8:
                    far *= 2.0
                if excess(far) < 0:
                    return direction * np.inf
            if excess(far) < 0:
                return bound
            return float(brentq(excess, 0.0, far, xtol=1e-14, rtol=1e-12))

        return edge(lo, -1.0), edge(hi, 1.0)

    def _process(self, params: list[tuple[float, ...]]) -> Process:
        base = self.base._process(params)
        clock = self.clock._process(params)
        return base @ clock

    def _render(self, params: list[tuple[float, ...]], digits: int, weighted: Weighting) -> str:
        base = self.base._render(params, digits, weighted)
        clock = self.clock._render(params, digits, weighted)
        return f"({base}) @ ({clock})"


def leaf(family: Family) -> Leaf:
    """Wrap a family as a structure."""
    return Leaf(family)


class Model:
    """A structure with parameter values: a fully specified composite process.

    Parameters
    ----------
    structure :
        The composition tree.
    params :
        One natural-parameter tuple per leaf, in :meth:`Node.leaves` order.
    """

    def __init__(self, structure: Node, params: Sequence[Sequence[float]]) -> None:
        leaves = structure.leaves()
        if len(params) != len(leaves):
            raise ValueError(
                f"{structure.key()} has {len(leaves)} leaves, got {len(params)} parameter sets"
            )
        self.structure = structure
        self.params: ParamList = tuple(lf.family.validate(p) for lf, p in zip(leaves, params))

    # ── parameter vectors ──────────────────────────────────────────────────
    @property
    def n_params(self) -> int:
        return self.structure.n_params

    def to_free(self) -> np.ndarray:
        """Concatenated unconstrained coordinates of all leaves."""
        return np.concatenate(
            [lf.family.to_free(p) for lf, p in zip(self.structure.leaves(), self.params)]
        )

    @classmethod
    def from_free(cls, structure: Node, x: Sequence[float]) -> Model:
        x = np.asarray(x, dtype=float)
        if x.size != structure.n_params:
            raise ValueError(
                f"{structure.key()} has {structure.n_params} free parameters, got {x.size}"
            )
        params, i = [], 0
        for lf in structure.leaves():
            params.append(lf.family.from_free(x[i : i + lf.family.n_params]))
            i += lf.family.n_params
        return cls(structure, params)

    # ── analytics ──────────────────────────────────────────────────────────
    def cgf_series(self, order: int = 4) -> CGFSeries:
        """Exact cumulant series of :math:`X_1` to ``order``."""
        return self.structure._series(list(self.params), order)

    def cumulants(self, order: int = 4, t: float = 1.0) -> np.ndarray:
        """:math:`\\kappa_1..\\kappa_{order}` of :math:`X_t`."""
        return self.cgf_series(order).at_time(t).cumulants.copy()

    def log_mgf(self, z: complex | np.ndarray) -> np.ndarray:
        """:math:`K(z) = \\log E[e^{zX_1}]`, composed from the leaves."""
        z_arr = np.asarray(z, dtype=complex)
        return self.structure._log_mgf(z_arr, list(self.params))

    def char_exponent(self, u: float | np.ndarray) -> np.ndarray:
        return self.log_mgf(1j * np.asarray(u, dtype=float))

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """:math:`E[e^{iuX_t}] = e^{t\\psi(u)}`."""
        return np.exp(t * self.char_exponent(u))

    def strip(self) -> tuple[float, float]:
        """Real :math:`z` interval on which :math:`E[e^{zX_1}] < \\infty`."""
        return self.structure._strip(list(self.params))

    @property
    def is_subordinator(self) -> bool:
        return self.structure.is_subordinator

    # ── output ─────────────────────────────────────────────────────────────
    def to_process(self) -> Process:
        """The equivalent spxa process, built with ``+``, ``*`` and ``@``."""
        return self.structure._process(list(self.params))

    def render(self, digits: int = 4, weighted: Weighting = False) -> str:
        """Human-readable expression.

        With ``weighted=True`` every non-drift primitive ``P`` is written as
        ``c * P_unit``, a multiple of the family's unit-scale member (for example
        ``0.3 * GammaProcess(a=2, b=1)``). A structural scale on a leaf is folded
        into that weight. ``weighted="subordinators"`` weights only the
        subordinator families (Gamma, inverse Gaussian) and leaves the
        return families in their natural parameters, giving output such as
        ``VarianceGamma(sigma=0.2, nu=0.3, theta=-0.1) + 0.04 * GammaProcess(a=4, b=1)``.
        """
        return self.structure._render(list(self.params), digits, weighted)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return f"Model({self.render()})"
