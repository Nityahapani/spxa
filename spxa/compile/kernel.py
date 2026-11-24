r"""
Compiled kernels for the characteristic exponent, its gradient and Hessian.

:class:`ExponentKernel` owns the generated programs for one lowered
exponent :math:`\psi(u;\theta)`:

* ``psi(u, theta)`` — :math:`\psi`;
* ``psi_and_grad(u, theta)`` — :math:`\psi` and :math:`\nabla_\theta\psi`
  from one fused program that shares every subexpression;
* ``psi_grad_hess(u, theta)`` — adds :math:`\nabla^2_\theta\psi`
  (upper triangle generated, mirrored);

each for one parameter vector (``theta`` of shape ``(n,)``) or a batch
(shape ``(B, n)``), on the NumPy, Numba or JAX backend. Only the *free*
parameters are inputs; the fixed ones are folded into the code as
constants.

Branches on parameters (``where`` nodes whose condition depends on
:math:`\theta` only, such as the CGMY series bands) are specialised: before
a call the kernel evaluates those conditions; when every parameter set of
the call agrees, it runs a program generated with the conditions replaced
by constants (only the live branch is computed), compiled once per pattern
and cached. A batch that straddles a branch runs the general program.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from spxa.compile import ir as I
from spxa.compile.codegen import Program, generate
from spxa.compile.diff import derivative
from spxa.compile.ir import CompileError, Expr
from spxa.compile.lower import Lowered

__all__ = ["ExponentKernel", "resolve_free"]


def resolve_free(
    names: Sequence[str], free: Optional[Sequence[Union[str, int]]]
) -> Tuple[int, ...]:
    """Indices of the free parameters (all when ``free`` is None)."""
    if free is None:
        return tuple(range(len(names)))
    out: List[int] = []
    for f in free:
        if isinstance(f, (int, np.integer)):
            k = int(f)
            if not 0 <= k < len(names):
                raise CompileError(f"parameter index {k} out of range (0..{len(names) - 1})")
        else:
            if f not in names:
                raise CompileError(f"unknown parameter {f!r}; known: {list(names)}")
            k = list(names).index(f)
        if k in out:
            raise CompileError(f"parameter {names[k]!r} listed twice in free")
        out.append(k)
    return tuple(out)


class _Family:
    """Programs for one set of outputs, specialised on parameter-only branch guards."""

    def __init__(
        self,
        outputs: Sequence[Expr],
        n_params: int,
        backend: str,
        name: str,
        parallel: bool,
    ) -> None:
        self.outputs = list(outputs)
        self.n_params = n_params
        self.backend = backend
        self.name = name
        self.parallel = parallel
        guards: List[Expr] = []
        seen: set = set()
        for n in I.postorder(self.outputs):
            if n.op == "where" and not n.args[0].depends_u and id(n.args[0]) not in seen:
                seen.add(id(n.args[0]))
                guards.append(n.args[0])
        self.guards = guards
        self.guard_prog: Optional[Program] = (
            generate(guards, n_params, "numpy", f"{name}_guards") if guards else None
        )
        self.cache: Dict[Optional[Tuple[bool, ...]], Program] = {}

    def program(self, params: Sequence[Any]) -> Program:
        key: Optional[Tuple[bool, ...]] = None
        if self.guard_prog is not None:
            vals = self.guard_prog(np.zeros(()), *params)
            pattern = []
            for v in vals:
                arr = np.asarray(v, dtype=bool)
                if arr.all():
                    pattern.append(True)
                elif not arr.any():
                    pattern.append(False)
                else:
                    pattern = []
                    break
            key = tuple(pattern) if pattern else None
        prog = self.cache.get(key)
        if prog is None:
            outs = self.outputs
            if key is not None:
                outs = I.substitute(outs, {g: bool(b) for g, b in zip(self.guards, key)})
            if not self.guards:
                pname = self.name
            elif key is None:
                pname = f"{self.name}_general"
            else:
                pname = f"{self.name}_" + "".join("1" if b else "0" for b in key)
            prog = generate(outs, self.n_params, self.backend, pname, self.parallel)
            self.cache[key] = prog
        return prog

    def __call__(self, u: Any, params: Sequence[Any]) -> Tuple[Any, ...]:
        return self.program(params)(u, *params)


class ExponentKernel:
    r"""
    Compiled :math:`\psi`, gradient and Hessian of a lowered exponent.

    Parameters
    ----------
    lowered :
        Result of :func:`spxa.compile.lower.lower`.
    backend :
        ``"numpy"`` (default), ``"numba"`` or ``"jax"``.
    free :
        Names or indices of the parameters that are inputs (default: all).
        The others are folded in at their lowered values.
    fixed :
        Optional overrides ``{name: value}`` for fixed parameters.
    parallel :
        Numba only: parallel loop over (parameter set, frequency).
    """

    def __init__(
        self,
        lowered: Lowered,
        backend: str = "numpy",
        free: Optional[Sequence[Union[str, int]]] = None,
        fixed: Optional[Dict[str, float]] = None,
        parallel: bool = False,
    ) -> None:
        self.lowered = lowered
        self.backend = backend
        self.parallel = parallel
        self.all_names = lowered.names
        self.free = resolve_free(lowered.names, free)
        self.names = tuple(lowered.names[k] for k in self.free)
        values = np.array(lowered.values, dtype=float)
        for nm, v in (fixed or {}).items():
            if nm not in lowered.names:
                raise CompileError(f"unknown parameter {nm!r} in fixed")
            k = lowered.names.index(nm)
            if k in self.free:
                raise CompileError(f"parameter {nm!r} is free; it cannot also be fixed")
            values[k] = float(v)
        self.all_values = values
        self.theta0 = values[list(self.free)].copy()
        # renumber: free parameters become p0..p{n-1}; fixed ones become constants
        mapping: Dict[Expr, Any] = {}
        self._new_params: List[Expr] = []
        for j, k in enumerate(self.free):
            q = I.param(j, lowered.names[k])
            mapping[lowered.params[k]] = q
            self._new_params.append(q)
        for k in range(len(lowered.names)):
            if k not in self.free:
                mapping[lowered.params[k]] = float(values[k])
        self.psi_expr: Expr = I.substitute(lowered.psi, mapping)
        if self.psi_expr.kind != I.COMPLEX:
            self.psi_expr = self.psi_expr + I.const(0j)
        self._grad_exprs: Optional[List[Expr]] = None
        self._hess_exprs: Optional[List[Expr]] = None
        self._families: Dict[str, _Family] = {}

    # ── expressions ─────────────────────────────────────────────────────────
    @property
    def n_free(self) -> int:
        return len(self.free)

    @property
    def grad_exprs(self) -> List[Expr]:
        if self._grad_exprs is None:
            self._grad_exprs = [derivative(self.psi_expr, q) for q in self._new_params]  # type: ignore[misc]
        return self._grad_exprs

    @property
    def hess_exprs(self) -> List[Expr]:
        """Upper triangle of the Hessian, row-major."""
        if self._hess_exprs is None:
            out: List[Expr] = []
            for i, g in enumerate(self.grad_exprs):
                for j in range(i, self.n_free):
                    out.append(derivative(g, self._new_params[j]))  # type: ignore[arg-type]
            self._hess_exprs = out
        return self._hess_exprs

    def _family(self, what: str) -> _Family:
        fam = self._families.get(what)
        if fam is None:
            if what == "psi":
                outs = [self.psi_expr]
            elif what == "grad":
                outs = [self.psi_expr] + self.grad_exprs
            else:
                outs = [self.psi_expr] + self.grad_exprs + self.hess_exprs
            fam = _Family(outs, self.n_free, self.backend, f"psi_{what}", self.parallel)
            self._families[what] = fam
        return fam

    def program(self, what: str = "psi", theta: Optional[Any] = None) -> Program:
        """The generated :class:`~spxa.compile.codegen.Program` (for inspection)."""
        th, _ = self._theta(theta)
        _, params = self._prepare(np.zeros(1), th)
        return self._family(what).program(params)

    # ── argument handling ───────────────────────────────────────────────────
    def _theta(self, theta: Optional[Any]) -> Tuple[np.ndarray, bool]:
        th = self.theta0 if theta is None else np.asarray(theta, dtype=float)
        if th.ndim == 1 and th.shape[0] == self.n_free:
            return th, False
        if th.ndim == 2 and th.shape[1] == self.n_free:
            return th, True
        raise ValueError(
            f"theta must have shape ({self.n_free},) or (batch, {self.n_free}); got {th.shape}"
        )

    def _prepare(self, u: Any, th: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
        uu = np.asarray(u, dtype=complex)
        if th.ndim == 1:
            return uu, [th[k] for k in range(self.n_free)]
        if self.backend == "numba":
            return uu, [np.ascontiguousarray(th[:, k]) for k in range(self.n_free)]
        tail = (1,) * uu.ndim
        return uu, [th[:, k].reshape((th.shape[0],) + tail) for k in range(self.n_free)]

    def _run(
        self, what: str, u: Any, theta: Optional[Any]
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[int, ...]]:
        th, batched = self._theta(theta)
        uu, params = self._prepare(u, th)
        shape = ((th.shape[0],) if batched else ()) + uu.shape
        outs = self._family(what)(uu, params)
        full = tuple(np.broadcast_to(np.asarray(o), shape) for o in outs)
        return full, shape

    # ── public evaluation ───────────────────────────────────────────────────
    def psi(self, u: Any, theta: Optional[Any] = None) -> np.ndarray:
        r""":math:`\psi(u;\theta)`, shape ``u.shape`` or ``(B,) + u.shape``."""
        outs, shape = self._run("psi", u, theta)
        return np.asarray(outs[0], dtype=complex).copy()

    def psi_and_grad(self, u: Any, theta: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        r""":math:`\psi` and :math:`\partial\psi/\partial\theta` (last axis: free parameters)."""
        outs, shape = self._run("grad", u, theta)
        psi = np.asarray(outs[0], dtype=complex).copy()
        grad = np.empty(shape + (self.n_free,), dtype=complex)
        for k in range(self.n_free):
            grad[..., k] = outs[1 + k]
        return psi, grad

    def psi_grad_hess(
        self, u: Any, theta: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r""":math:`\psi`, gradient and Hessian (last two axes: free parameters)."""
        outs, shape = self._run("hess", u, theta)
        n = self.n_free
        psi = np.asarray(outs[0], dtype=complex).copy()
        grad = np.empty(shape + (n,), dtype=complex)
        for k in range(n):
            grad[..., k] = outs[1 + k]
        hess = np.empty(shape + (n, n), dtype=complex)
        pos = 1 + n
        for i in range(n):
            for j in range(i, n):
                hess[..., i, j] = outs[pos]
                hess[..., j, i] = outs[pos]
                pos += 1
        return psi, grad, hess

    def char_func(self, u: Any, t: float = 1.0, theta: Optional[Any] = None) -> np.ndarray:
        r""":math:`E[e^{iuX_t}] = e^{t\psi(u)}`."""
        return np.exp(float(t) * self.psi(u, theta))

    def full_theta(self, theta: Optional[Any] = None) -> np.ndarray:
        """The full parameter vector(s) (fixed values filled in)."""
        th, batched = self._theta(theta)
        if not batched:
            out = self.all_values.copy()
            out[list(self.free)] = th
            return out
        out2 = np.tile(self.all_values, (th.shape[0], 1))
        out2[:, list(self.free)] = th
        return out2

    def source(self, what: str = "psi", theta: Optional[Any] = None) -> str:
        """Generated source of the program that would run for ``theta``."""
        return self.program(what, theta).source

    def __repr__(self) -> str:
        return (
            f"ExponentKernel(backend={self.backend!r}, free={list(self.names)}, "
            f"nodes={I.count_nodes(self.psi_expr)})"
        )
