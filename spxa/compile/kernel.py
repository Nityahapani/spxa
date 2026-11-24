r"""
Compiled kernels for the characteristic exponent, its gradient and Hessian.

:class:`ExponentKernel` owns the generated programs for one lowered
exponent :math:`\psi(u;\theta)`:

* ``psi(u, theta)`` — :math:`\psi`;
* ``psi_and_grad(u, theta)`` — :math:`\psi` and :math:`\nabla_\theta\psi`
  from one fused program that shares every subexpression;
* ``psi_grad_hess(u, theta)`` — adds :math:`\nabla^2_\theta\psi`
  (upper triangle generated, mirrored);
* ``psi_du_and_grad(u, theta)`` — :math:`\psi`, :math:`\partial\psi/\partial u`
  and :math:`\nabla_\theta\psi`, the inputs of the COS price gradient,
  optionally with a separate ``u`` grid per parameter set (``rowwise``);

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


#: Ops of ``u`` for which only Taylor mode gives exact cumulants (see
#: :meth:`ExponentKernel._symbolic_cumulants`).
_TAYLOR_ONLY = frozenset({"sinc", "sinc_d1", "kve_dnu", "call", "gamma", "digamma", "polygamma"})
#: Highest cumulant order taken from generated derivatives.
_SYMBOLIC_ORDER = 4


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
        self._cumulant_nodes: Optional[List[Expr]] = None
        self._u_derivs: Optional[List[Expr]] = None
        self._symbolic_ok: Optional[bool] = None
        self._hess_exprs: Optional[List[Expr]] = None
        self._du_expr: Optional[Expr] = None
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

    @property
    def du_expr(self) -> Expr:
        r""":math:`\partial\psi/\partial u`."""
        if self._du_expr is None:
            d = derivative(self.psi_expr, I.U)
            assert isinstance(d, Expr)
            self._du_expr = d if d.kind == I.COMPLEX else d + I.const(0j)
        return self._du_expr

    def _family(self, what: str) -> _Family:
        fam = self._families.get(what)
        if fam is None:
            if what == "psi":
                outs = [self.psi_expr]
            elif what == "grad":
                outs = [self.psi_expr] + self.grad_exprs
            elif what == "du_grad":
                outs = [self.psi_expr, self.du_expr] + self.grad_exprs
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

    def _run_rowwise(
        self, what: str, u: Any, theta: Optional[Any]
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[int, ...]]:
        """Like ``_run``, with row ``b`` of ``u`` evaluated at parameter set ``b`` only."""
        th, batched = self._theta(theta)
        uu = np.asarray(u, dtype=complex)
        if not batched or uu.ndim < 1 or uu.shape[0] != th.shape[0]:
            raise ValueError("rowwise needs theta (B, n_free) and u of shape (B, ...)")
        if self.backend == "numba":  # the generated loop shares u across the batch
            rows = [self._run(what, uu[b], th[b])[0] for b in range(th.shape[0])]
            return tuple(np.stack([r[k] for r in rows]) for k in range(len(rows[0]))), uu.shape
        tail = (1,) * (uu.ndim - 1)
        params = [th[:, k].reshape((th.shape[0],) + tail) for k in range(self.n_free)]
        res = self._family(what)(uu, params)
        return tuple(np.broadcast_to(np.asarray(o), uu.shape) for o in res), uu.shape

    def psi_rowwise(self, u: Any, theta: Any) -> np.ndarray:
        r""":math:`\psi(u_b;\theta_b)` for ``theta`` of shape ``(B, n_free)`` and ``u`` of shape ``(B, ...)``."""
        outs, _ = self._run_rowwise("psi", u, theta)
        return np.asarray(outs[0], dtype=complex).copy()

    def psi_du_and_grad(
        self, u: Any, theta: Optional[Any] = None, rowwise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        :math:`\psi`, :math:`\partial\psi/\partial u` and :math:`\partial\psi/\partial\theta`.

        With ``rowwise=True``, ``theta`` is a ``(B, n_free)`` batch and ``u``
        has shape ``(B, ...)``: row ``b`` of ``u`` is evaluated at parameter
        set ``b`` only (each set has its own frequency grid).
        """
        run = self._run_rowwise if rowwise else self._run
        outs, shape = run("du_grad", u, theta)
        psi = np.asarray(outs[0], dtype=complex).copy()
        dpsi = np.asarray(outs[1], dtype=complex).copy()
        grad = np.empty(shape + (self.n_free,), dtype=complex)
        for k in range(self.n_free):
            grad[..., k] = outs[2 + k]
        return psi, dpsi, grad

    # ── cumulants (Taylor mode on the same expressions) ────────────────────
    def _taylor_params(self, theta: Optional[Any]) -> Tuple[List[Any], bool]:
        th, batched = self._theta(theta)
        if batched:
            return [th[:, k] for k in range(self.n_free)], True
        return [th[k] for k in range(self.n_free)], False

    def _require_analytic(self) -> None:
        if not self.lowered.analytic:
            raise CompileError(
                "the exponent is not analytic at u = 0 (an alpha-stable component has no "
                "moments), so it has no cumulants"
            )

    def _symbolic_cumulants_ok(self) -> bool:
        """Whether generated u-derivatives at 0 reproduce Taylor mode exactly."""
        if self._symbolic_ok is None:
            self._symbolic_ok = not any(
                n.depends_u and n.op in _TAYLOR_ONLY for n in I.postorder([self.psi_expr])
            )
        return self._symbolic_ok

    def _symbolic_cumulants(
        self, order: int, params: Sequence[Any], lead: Tuple[int, ...], jac: bool
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        r"""
        Cumulants (and their Jacobian) from generated :math:`\partial_u^n\psi(0)`.

        :math:`\kappa_n = \mathrm{Re}\,(-i)^n\psi^{(n)}(0)`. Symbolic and Taylor
        differentiation agree node by node except where Taylor mode uses
        dedicated series for removable singularities (``sinc``, ``sinc_d1``, the
        Bessel order derivative, opaque calls) or refuses (Gamma-type functions
        of ``u``); those exponents, orders above four and any non-finite result
        return ``None`` and fall back to Taylor mode. A ``where`` picks its branch
        from the condition at the expansion point in both.
        """
        if order > _SYMBOLIC_ORDER or not self._symbolic_cumulants_ok():
            return None
        what = "cum_jac" if jac else "cum"
        fam = self._families.get(what)
        if fam is None:
            if self._u_derivs is None:
                ds: List[Expr] = [self.psi_expr]
                for _ in range(_SYMBOLIC_ORDER):
                    d = derivative(ds[-1], I.U)
                    assert isinstance(d, Expr)
                    ds.append(d)
                self._u_derivs = ds[1:]
            outs = list(self._u_derivs)
            if jac:
                for d in self._u_derivs:
                    outs += [derivative(d, q) for q in self._new_params]  # type: ignore[misc]
            # always NumPy: one point per parameter set, so compile time would dominate
            fam = _Family(outs, self.n_free, "numpy", f"psi_{what}", False)
            self._families[what] = fam
        with np.errstate(all="ignore"):
            res = fam(np.zeros(()), params)
        rot = [(-1j) ** n for n in range(1, _SYMBOLIC_ORDER + 1)]
        kappa = np.empty(lead + (order,))
        for n in range(order):
            kappa[..., n] = np.real(rot[n] * np.asarray(res[n]))
        jmat: Optional[np.ndarray] = None
        if jac:
            m = self.n_free
            jmat = np.empty(lead + (order, m))
            for n in range(order):
                for k in range(m):
                    jmat[..., n, k] = np.real(rot[n] * np.asarray(res[_SYMBOLIC_ORDER + n * m + k]))
        if not np.all(np.isfinite(kappa)) or (jmat is not None and not np.all(np.isfinite(jmat))):
            return None
        return kappa, jmat

    def cumulants(self, order: int = 4, theta: Optional[Any] = None) -> np.ndarray:
        r"""
        :math:`\kappa_1..\kappa_{order}` of :math:`X_1`, exactly.

        Shape ``(order,)``, or ``(B, order)`` for a batch of parameters. Up to
        order four they come from generated derivatives of :math:`\psi` at 0;
        otherwise from Taylor mode.
        """
        from spxa.compile.evaluate import cumulants_from_ir

        self._require_analytic()
        params, batched = self._taylor_params(theta)
        lead = (len(params[0]),) if batched else ()
        fast = self._symbolic_cumulants(order, params, lead, jac=False)
        if fast is not None:
            return fast[0]
        out = cumulants_from_ir(self.psi_expr, order, params)
        return np.broadcast_to(out, lead + (order,)).copy()

    def cumulants_and_jacobian(
        self, order: int = 4, theta: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Cumulants and :math:`\partial\kappa_n/\partial\theta_k`, shape ``(..., order, n_free)``.

        Taylor coefficients in :math:`u` commute with parameter derivatives, so
        the Jacobian is the cumulant series of :math:`\partial\psi/\partial\theta_k`.
        """
        from spxa.compile.evaluate import cumulants_from_ir_many

        self._require_analytic()
        params, batched = self._taylor_params(theta)
        lead = (len(params[0]),) if batched else ()
        fast = self._symbolic_cumulants(order, params, lead, jac=True)
        if fast is not None and fast[1] is not None:
            return fast[0], fast[1]
        # one Taylor pass: the gradient expressions share most nodes with psi
        roots = [self.psi_expr] + self.grad_exprs
        if self._cumulant_nodes is None:
            self._cumulant_nodes = I.postorder(roots)
        series = cumulants_from_ir_many(roots, order, params, self._cumulant_nodes)
        kappa = np.broadcast_to(series[0], lead + (order,))
        jac = np.empty(lead + (order, self.n_free))
        for k, ser in enumerate(series[1:]):
            jac[..., k] = np.broadcast_to(ser, lead + (order,))
        return kappa.copy(), jac

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
