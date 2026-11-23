r"""
Differentiable processes: exact derivatives through the process algebra.

:class:`DiffProcess` mirrors a :class:`~spxa.core.process.Process` built from
zoo Lévy families with ``+``, scalar ``*`` and subordination ``@``, and
evaluates its characteristic exponent as a function of the flat parameter
vector :math:`\theta`, using the closed-form rules of
:mod:`spxa.graph.autodiff.families` at the leaves and the algebra at the
inner nodes:

.. math::
    \psi_{X+Y}(u) = \psi_X(u) + \psi_Y(u), \qquad
    \psi_{cX}(u) = \psi_X(cu), \qquad
    \psi_{X\circ T}(u) = \psi_T\big(-i\,\psi_X(u)\big).

Because every step is closed form, :math:`\partial\psi/\partial\theta` and
:math:`\partial^2\psi/\partial\theta^2` are exact (forward-mode jets), and the
cumulants :math:`\kappa_n(\theta) = n!\,[z^n]\,\psi(-iz)` and their gradients
are exact too (Taylor series with jet coefficients). Nothing is perturbed and
the original process is never mutated; :meth:`DiffProcess.to_process` builds
a new process at any :math:`\theta`.

Parameters are named ``"<Family>.<param>"`` (with ``"[k]"`` appended when a
family occurs more than once), plus ``"scale"`` entries for differentiable
scale factors. Choose the free subset with ``free=``; the others are held at
their values.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from spxa.core.process import Process
from spxa.graph.autodiff.exactness import GradientExactness
from spxa.graph.autodiff.families import FamilyRule, rule_for
from spxa.graph.autodiff.jet import Jet, value_of, variable_vector
from spxa.graph.autodiff.taylor import TaylorError, taylor_variable

__all__ = ["Param", "DiffProcess", "UnsupportedProcessError"]


class UnsupportedProcessError(TypeError):
    """The process (or part of it) has no differentiable closed form."""


@dataclass(frozen=True)
class Param:
    """A named scalar parameter value (used to mark or override parameters by name)."""

    value: float
    name: Optional[str] = None

    def __float__(self) -> float:
        return float(self.value)


# ── expression tree ───────────────────────────────────────────────────────────


class _Node:
    def psi(self, u: Any, th: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def build(self, th: Sequence[float]) -> Process:  # pragma: no cover - abstract
        raise NotImplementedError

    def story(self, th: Sequence[float]) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    analytic = True


class _Leaf(_Node):
    def __init__(self, rule: FamilyRule, slots: List[int], label: str) -> None:
        self.rule = rule
        self.slots = slots
        self.label = label
        self.analytic = rule.analytic

    def _p(self, th: Any) -> Tuple[Any, ...]:
        return tuple(th[i] for i in self.slots)

    def psi(self, u: Any, th: Any) -> Any:
        return self.rule.psi(u, self._p(th))

    def build(self, th: Sequence[float]) -> Process:
        return self.rule.build(tuple(float(th[i]) for i in self.slots))

    def story(self, th: Sequence[float]) -> str:
        body = ", ".join(f"{n}={float(th[i]):.6g}" for n, i in zip(self.rule.params, self.slots))
        return f"{self.rule.cls_name}({body})"


class _Sum(_Node):
    def __init__(self, a: _Node, b: _Node) -> None:
        self.a, self.b = a, b
        self.analytic = a.analytic and b.analytic

    def psi(self, u: Any, th: Any) -> Any:
        return self.a.psi(u, th) + self.b.psi(u, th)

    def build(self, th: Sequence[float]) -> Process:
        return self.a.build(th) + self.b.build(th)

    def story(self, th: Sequence[float]) -> str:
        return f"({self.a.story(th)} + {self.b.story(th)})"


class _Scale(_Node):
    def __init__(self, base: _Node, slot: int) -> None:
        self.base, self.slot = base, slot
        self.analytic = base.analytic

    def psi(self, u: Any, th: Any) -> Any:
        return self.base.psi(th[self.slot] * u, th)

    def build(self, th: Sequence[float]) -> Process:
        return float(th[self.slot]) * self.base.build(th)

    def story(self, th: Sequence[float]) -> str:
        return f"{float(th[self.slot]):.6g} * {self.base.story(th)}"


class _Subordinate(_Node):
    def __init__(self, x: _Node, clock: _Node) -> None:
        self.x, self.clock = x, clock
        self.analytic = x.analytic and clock.analytic

    def psi(self, u: Any, th: Any) -> Any:
        return self.clock.psi(-1j * self.x.psi(u, th), th)

    def build(self, th: Sequence[float]) -> Process:
        return self.x.build(th) @ self.clock.build(th)

    def story(self, th: Sequence[float]) -> str:
        return f"{self.x.story(th)} @ {self.clock.story(th)}"


def _parse(proc: Process, values: List[float], names: List[str], counts: Dict[str, int]) -> _Node:
    from spxa.core.process import _ComposedProcess, _ScaledProcess

    if isinstance(proc, _ScaledProcess):
        base = _parse(proc._base, values, names, counts)
        slot = len(values)
        values.append(float(proc._scale))
        k = counts.get("scale", 0)
        counts["scale"] = k + 1
        names.append("scale" if k == 0 else f"scale[{k}]")
        return _Scale(base, slot)
    if isinstance(proc, _ComposedProcess):
        left = _parse(proc._left, values, names, counts)
        right = _parse(proc._right, values, names, counts)
        if proc._operation == "+":
            return _Sum(left, right)
        if proc._operation == "@":
            return _Subordinate(left, right)
        raise UnsupportedProcessError(f"composition {proc._operation!r} is not supported")
    rule = rule_for(proc)
    if rule is None:
        raise UnsupportedProcessError(
            f"{type(proc).__name__} has no differentiable closed form; supported: "
            "see spxa.graph.autodiff.families.RULES"
        )
    k = counts.get(rule.cls_name, 0)
    counts[rule.cls_name] = k + 1
    label = rule.cls_name if k == 0 else f"{rule.cls_name}[{k}]"
    slots = []
    for pname, v in zip(rule.params, rule.getter(proc)):
        slots.append(len(values))
        values.append(float(v))
        names.append(f"{label}.{pname}")
    return _Leaf(rule, slots, label)


# ── the public class ──────────────────────────────────────────────────────────


class DiffProcess:
    r"""
    A process whose characteristic exponent is differentiable in its parameters.

    Parameters
    ----------
    process :
        A zoo Lévy process or a composition of them (``+``, ``c *``, ``@``).
    free :
        Names (or unique suffixes such as ``"sigma"``) of the parameters to
        differentiate; default all.

    Examples
    --------
    >>> from spxa.zoo.levy.vg import VarianceGamma
    >>> from spxa.zoo.levy.drift import Drift
    >>> dp = DiffProcess(VarianceGamma(0.2, 0.3, -0.1) + Drift(0.01))
    >>> psi, dpsi = dp.char_exponent_and_grad([0.5, 1.0])
    >>> dpsi.shape
    (2, 4)
    """

    def __init__(self, process: Process, free: Optional[Sequence[str]] = None) -> None:
        values: List[float] = []
        names: List[str] = []
        self._tree = _parse(process, values, names, {})
        self.process = process
        self._all_values = np.array(values, dtype=float)
        self._all_names = tuple(names)
        if free is None:
            self._free = list(range(len(names)))
        else:
            self._free = [self._resolve(f) for f in free]
            if len(set(self._free)) != len(self._free):
                raise ValueError("a parameter is listed twice in free=")

    def _resolve(self, key: Union[str, Param]) -> int:
        key = key.name if isinstance(key, Param) else key
        if key in self._all_names:
            return self._all_names.index(key)
        hits = [i for i, n in enumerate(self._all_names) if n.endswith("." + str(key))]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise KeyError(f"no parameter {key!r}; have {self._all_names}")
        raise KeyError(f"{key!r} is ambiguous among {[self._all_names[i] for i in hits]}")

    # -- parameters ----------------------------------------------------------
    @property
    def names(self) -> Tuple[str, ...]:
        """Names of the free parameters, in the order of :attr:`values`."""
        return tuple(self._all_names[i] for i in self._free)

    @property
    def values(self) -> np.ndarray:
        """Current values of the free parameters."""
        return self._all_values[self._free].copy()

    @property
    def all_names(self) -> Tuple[str, ...]:
        return self._all_names

    @property
    def analytic(self) -> bool:
        """Whether :math:`\\psi` is analytic in ``u`` (needed for cumulants and Fourier strips)."""
        return self._tree.analytic

    @property
    def exactness(self) -> GradientExactness:
        """Derivatives of :math:`\\psi` and of the cumulants are exact."""
        return GradientExactness.EXACT

    def _full(self, theta: Any) -> List[Any]:
        """Full parameter list with the free entries replaced by ``theta`` (array or Jet)."""
        full: List[Any] = list(self._all_values)
        if theta is None:
            return full
        if isinstance(theta, Jet):
            if theta.shape != (len(self._free),):
                raise ValueError(f"theta must have shape ({len(self._free)},)")
            for k, i in enumerate(self._free):
                full[i] = theta[k]
            return full
        th = np.asarray(theta, dtype=float).ravel()
        if th.size != len(self._free):
            raise ValueError(f"theta must have {len(self._free)} entries, got {th.size}")
        for k, i in enumerate(self._free):
            full[i] = float(th[k])
        return full

    def with_values(self, theta: Sequence[float]) -> DiffProcess:
        """A copy at new free-parameter values (the wrapped process is rebuilt)."""
        return DiffProcess(self.to_process(theta), free=self.names)

    def to_process(self, theta: Optional[Sequence[float]] = None) -> Process:
        """The spxa process at ``theta`` (default: the current values)."""
        full = self._full(None if theta is None else np.asarray(theta, float))
        return self._tree.build(full)

    # -- characteristic exponent ------------------------------------------------
    def char_exponent(self, u: Any, theta: Any = None) -> Any:
        r""":math:`\psi(u;\theta)`; ``theta`` may be a :class:`Jet` (shape ``(n,)``)."""
        uu = u if (isinstance(u, Jet) or hasattr(u, "coeffs")) else np.asarray(u, dtype=complex)
        return self._tree.psi(uu, self._full(theta))

    def log_char(self, u: Any, t: Any = 1.0, theta: Any = None) -> Any:
        r""":math:`\log E[e^{iuX_t}] = t\,\psi(u)` (continuous; no principal-branch jumps)."""
        return t * self.char_exponent(u, theta)

    def char_func(self, u: Any, t: Any = 1.0, theta: Any = None) -> Any:
        return np.exp(self.log_char(u, t, theta))

    def log_mgf(self, z: Any, theta: Any = None) -> Any:
        r""":math:`K(z) = \psi(-iz)`, inside the exponential-moment strip."""
        return self.char_exponent(-1j * np.asarray(z, dtype=complex), theta)

    def char_exponent_and_grad(self, u: Any, theta: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        :math:`\psi(u)` and the exact :math:`\partial\psi/\partial\theta`.

        Returns arrays of shapes ``u.shape`` and ``u.shape + (n,)``.
        """
        th = variable_vector(self.values if theta is None else theta)
        out = self.char_exponent(u, th)
        if not isinstance(out, Jet):
            n = len(self._free)
            return np.asarray(out), np.zeros(np.shape(out) + (n,), dtype=complex)
        return np.asarray(out.value), np.asarray(out.grad)

    def char_exponent_hessian(
        self, u: Any, theta: Any = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r""":math:`\psi`, :math:`\partial_\theta\psi` and :math:`\partial^2_\theta\psi` (exact)."""
        th = variable_vector(self.values if theta is None else theta, order=2)
        out = self.char_exponent(u, th)
        n = len(self._free)
        if not isinstance(out, Jet):
            z = np.zeros(np.shape(out) + (n,), dtype=complex)
            return np.asarray(out), z, np.zeros(np.shape(out) + (n, n), dtype=complex)
        return np.asarray(out.value), np.asarray(out.grad), np.asarray(out.hess)

    # -- cumulants ---------------------------------------------------------------
    def cumulants(self, order: int, theta: Any = None) -> Any:
        r"""
        :math:`\kappa_1, \dots, \kappa_{order}` of :math:`X_1`, exactly.

        From the Taylor coefficients of :math:`K(z) = \psi(-iz)` at :math:`z = 0`:
        :math:`\kappa_n = n!\,[z^n]K`. With a :class:`Jet` ``theta`` the result
        is a Jet (exact parameter derivatives).
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        if not self.analytic:
            raise UnsupportedProcessError(
                "cumulants need an exponent analytic at 0 (finite moments)"
            )
        z = taylor_variable(0.0 + 0.0j, order)
        try:
            K = self.char_exponent(-1j * z, theta)
        except TaylorError as exc:  # pragma: no cover - defensive
            raise UnsupportedProcessError(str(exc)) from exc
        ks = [np.real(factorial(n) * K.coeffs[n]) for n in range(1, order + 1)]
        if any(isinstance(k, Jet) for k in ks):
            from spxa.graph.autodiff.jet import stack

            return stack(ks)
        return np.array([float(value_of(k)) for k in ks])

    def cumulants_and_jacobian(
        self, order: int, theta: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cumulants and their exact Jacobian ``d kappa_n / d theta_k`` (shape ``(order, n)``)."""
        th = variable_vector(self.values if theta is None else theta)
        out = self.cumulants(order, th)
        if not isinstance(out, Jet):
            return np.asarray(out), np.zeros((order, len(self._free)))
        return np.asarray(out.value, dtype=float), np.asarray(out.grad, dtype=float)

    # -- reporting -------------------------------------------------------------------
    def __story__(self) -> str:
        """How derivatives are obtained, route by route."""
        lines = [
            f"DiffProcess {self._tree.story(list(self._all_values))}",
            "  psi: closed form per family; + adds exponents, c*X rescales u, X@T composes "
            "psi_T(-i psi_X(u))",
            "  d psi / d theta, d2 psi / d theta2: forward-mode jets through the closed forms "
            "(EXACT)",
            "  cumulants and their gradients: Taylor coefficients of psi(-iz) with jet "
            "coefficients (EXACT)",
            f"  free parameters: {', '.join(self.names)}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DiffProcess({self._tree.story(list(self._all_values))}, free={list(self.names)})"


def check_against_zoo(dp: DiffProcess, u: Any, rtol: float = 1e-12) -> float:
    """Max relative difference between ``dp`` and the wrapped process's own exponent."""
    u = np.asarray(u, dtype=float)
    ours = np.asarray(dp.char_exponent(u))
    proc = dp.process
    fn = getattr(proc, "char_exponent", None) or proc._char_exponent
    theirs = np.asarray(fn(np.asarray(u, dtype=complex)), dtype=complex)
    return float(np.max(np.abs(ours - theirs) / np.maximum(1.0, np.abs(theirs))))
