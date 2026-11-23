r"""
Reverse- and forward-mode differentiation of computational graphs.

Every node contributes the derivative of the function it actually computes,
by the most exact route available:

1. its authored vector-Jacobian product (``OperatorNode.vjp_fn``);
2. otherwise a *local jet*: the node's own ``forward`` is evaluated on
   :class:`~spxa.graph.autodiff.jet.Jet` inputs, which yields its exact
   Jacobian (the jet value is checked against the plain forward value, so a
   node that silently ignores jets is detected and not trusted);
3. otherwise central finite differences, one input element at a time, with
   step :math:`h_j = \epsilon^{1/3}\max(1, |x_j|)`.

Routes 1–2 are ``EXACT``; route 3 is ``NUMERICAL``. Nodes may declare
``gradient_exactness`` (for instance ``STOCHASTIC`` for Monte Carlo
estimators) and ``jet_capable = False``. Each result reports the worst label
on any path from the source to the target; ``strict=True`` raises
:class:`~spxa.graph.autodiff.exactness.InexactGradientError` instead.

Conventions
-----------
* Adjoints have the shape of the value they belong to: broadcasting in the
  forward pass is undone by summation in the reverse pass, so a scalar
  parameter always receives a scalar gradient.
* Complex values use the JAX cotangent convention. For a real objective
  :math:`L` and a complex intermediate :math:`z = x + iy`, the cotangent is
  :math:`\bar z = \partial L/\partial x - i\,\partial L/\partial y`, so that
  :math:`dL = \operatorname{Re}(\bar z\,dz)`. For holomorphic :math:`w = f(z)`
  this gives :math:`\bar z = \bar w\,f'(z)`, the familiar formula, and a real
  input receives :math:`\operatorname{Re}(\ldots)`.
* For an array or complex target, ``gradient`` differentiates
  ``sum(target)``; for a complex target and real sources the result is
  :math:`d\,\mathrm{sum}(w)/d\theta = d\operatorname{Re}/d\theta + i\,d\operatorname{Im}/d\theta`.
  Use :meth:`GradientTape.jacobian` for the full Jacobian.
* ``feed_dict`` entries are leaves: derivatives do not flow through a fed
  node (unless it is itself a source).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from spxa.graph.autodiff.exactness import (
    GradientExactness,
    GradientNote,
    GradientResult,
    check_strict,
    worst,
)
from spxa.graph.autodiff.jet import Jet, JetError, _coerce
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node, OutputNode

__all__ = [
    "GradientTape",
    "value_and_grad",
    "jvp",
    "FD_EPS",
]

EXACT = GradientExactness.EXACT
NUMERICAL = GradientExactness.NUMERICAL

#: Relative step for central differences, :math:`\epsilon_{\rm mach}^{1/3}`.
FD_EPS = float(np.finfo(float).eps ** (1.0 / 3.0))

#: Upper bound on ``output size x directions`` held in one local-jet evaluation.
_JET_CHUNK_ELEMS = 4_000_000

# Exceptions that mean "this node cannot run on jets" (as opposed to a bug).
_JET_FAILURES = (JetError, TypeError, ValueError, AttributeError, NotImplementedError)


# ── small helpers ─────────────────────────────────────────────────────────────


def _is_numeric(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (int, float, complex, np.number)):
        return True
    return isinstance(v, np.ndarray) and v.dtype.kind in "fciu"


def _is_complex(v: Any) -> bool:
    return bool(np.iscomplexobj(v))


def _shape(v: Any) -> Tuple[int, ...]:
    return tuple(np.shape(v))


def _unbroadcast(g: Any, shape: Tuple[int, ...], complex_ok: bool) -> np.ndarray:
    """Sum ``g`` down to ``shape`` (undoing NumPy broadcasting); drop Im for real targets."""
    g = np.asarray(g)
    if g.shape != shape:
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        for ax, (gs, s) in enumerate(zip(g.shape, shape)):
            if s == 1 and gs != 1:
                g = g.sum(axis=ax, keepdims=True)
        if g.shape != shape:
            g = np.array(np.broadcast_to(g, shape))
    if not complex_ok and np.iscomplexobj(g):
        g = g.real
    return g


def _finalize(g: Any, like: Any) -> Any:
    """Return ``g`` with the type of ``like``: Python scalar for scalars, else an array."""
    g = np.asarray(g)
    if np.ndim(like) == 0 and g.ndim == 0:
        return complex(g) if np.iscomplexobj(g) else float(g)
    return g


def _scalar_like(x: np.ndarray, like: Any) -> Any:
    """Pass perturbed scalars back in the caller's scalar type (forwards may expect float)."""
    if np.ndim(like) == 0:
        return complex(x) if np.iscomplexobj(x) else float(x)
    return x


def _values_match(jet_val: Any, val: Any) -> bool:
    a, b = np.asarray(jet_val), np.asarray(val)
    if a.shape != b.shape:
        return False
    if not (_is_numeric(b) or b.dtype.kind == "c"):
        return False
    fin = np.isfinite(b)
    scale = float(np.max(np.abs(b[fin]))) if np.any(fin) else 1.0
    return bool(np.allclose(a, b, rtol=1e-10, atol=1e-13 * max(scale, 1e-300), equal_nan=True))


def _relevant(
    order: Sequence[Node], target: Node, sources: Sequence[Node], fed: Set[int]
) -> Set[int]:
    """Nodes on some path source -> target (fed non-source nodes are leaves)."""
    src = {s.node_id for s in sources}
    down: Set[int] = set()
    for node in order:
        nid = node.node_id
        if nid in src or (nid not in fed and any(i.node_id in down for i in node.inputs)):
            down.add(nid)
    up: Set[int] = {target.node_id}
    for node in reversed(order):
        if node.node_id in up and node.node_id not in (fed - src):
            up.update(i.node_id for i in node.inputs)
    return down & up


# ── local derivative routes for one node ─────────────────────────────────────


def _direction_layout(
    inp_vals: Sequence[Any], need: Sequence[int]
) -> Tuple[List[Tuple[int, int, int, bool]], int]:
    """[(input index, offset, size, complex)] and the total number of real directions."""
    layout = []
    off = 0
    for i in need:
        v = inp_vals[i]
        size = int(np.size(v))
        cplx = _is_complex(v)
        layout.append((i, off, size, cplx))
        off += size * (2 if cplx else 1)
    return layout, off


def _seeded_inputs(
    inp_vals: Sequence[Any],
    layout: List[Tuple[int, int, int, bool]],
    start: int,
    stop: int,
    order: int = 1,
) -> List[Any]:
    """Inputs with jets seeded on real directions ``start:stop`` (others are constants)."""
    n = stop - start
    out = list(inp_vals)
    for i, off, size, cplx in layout:
        v = np.asarray(inp_vals[i])
        dt = complex if cplx else float
        g = np.zeros((size, n), dtype=dt)
        for part, unit in ((0, 1.0), (1, 1j)) if cplx else ((0, 1.0),):
            lo = off + part * size
            a, b = max(lo, start), min(lo + size, stop)
            if a < b:
                e = np.arange(a, b)
                g[e - lo, e - start] = unit
        h = np.zeros((size, n, n), dtype=dt) if order == 2 else None
        out[i] = Jet(
            v.astype(dt),
            g.reshape(v.shape + (n,)),
            None if h is None else h.reshape(v.shape + (n, n)),
        )
    return out


def _local_jet_vjp(
    node: Node, inp_vals: Sequence[Any], out_val: Any, ct: np.ndarray, need: Sequence[int]
) -> Dict[int, np.ndarray]:
    """Exact VJP by running ``node.forward`` on jets. Raises on failure."""
    layout, total = _direction_layout(inp_vals, need)
    out_size = max(1, int(np.size(out_val)))
    chunk = max(1, min(total, _JET_CHUNK_ELEMS // out_size))
    contrib = np.zeros(total)
    ct = np.asarray(ct)
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        res = _coerce(node.forward(*_seeded_inputs(inp_vals, layout, start, stop)))
        if isinstance(res, Jet):
            if not _values_match(res.value, out_val):
                raise JetError("jet evaluation does not reproduce the node's value")
            G = res.grad
            contrib[start:stop] = np.real(
                np.tensordot(ct, G, axes=(list(range(ct.ndim)), list(range(ct.ndim))))
            ).reshape(-1)
        else:
            # Jets went in but none came out. A node that truly ignores these inputs
            # would do this, but so would one that swallowed the jets (e.g. caught a
            # JetError and fell back to cached floats). Do not guess: refuse.
            raise JetError("node returned a value without derivatives for jet inputs")
    out: Dict[int, np.ndarray] = {}
    for i, off, size, cplx in layout:
        shp = _shape(inp_vals[i])
        dx = contrib[off : off + size].reshape(shp)
        if cplx:
            dy = contrib[off + size : off + 2 * size].reshape(shp)
            out[i] = dx - 1j * dy
        else:
            out[i] = dx
    return out


def _fd_vjp(
    node: Node,
    inp_vals: Sequence[Any],
    ct: np.ndarray,
    need: Sequence[int],
    eps: float,
) -> Dict[int, np.ndarray]:
    """Central differences, one real direction at a time (``NUMERICAL``)."""
    out: Dict[int, np.ndarray] = {}
    for i in need:
        v0 = inp_vals[i]
        cplx = _is_complex(v0)
        x = np.array(v0, dtype=complex if cplx else float)
        flat = x.reshape(-1)
        res = np.zeros(flat.shape, dtype=complex if cplx else float)
        for e in range(flat.size):
            for unit in (1.0, 1j) if cplx else (1.0,):
                h = eps * max(1.0, abs(flat[e]))
                xp, xm = flat.copy(), flat.copy()
                xp[e] += h * unit
                xm[e] -= h * unit
                args_p, args_m = list(inp_vals), list(inp_vals)
                args_p[i] = _scalar_like(xp.reshape(x.shape), v0)
                args_m[i] = _scalar_like(xm.reshape(x.shape), v0)
                d = (np.asarray(node.forward(*args_p)) - np.asarray(node.forward(*args_m))) / (
                    2.0 * h
                )
                val = float(np.real(np.sum(ct * d)))
                if unit == 1.0:
                    res[e] = res[e] + val
                else:
                    res[e] = res[e] - 1j * val
        out[i] = res.reshape(x.shape)
    return out


# ── the tape ──────────────────────────────────────────────────────────────────


class GradientTape:
    r"""
    Reverse-mode differentiation of a :class:`ComputationalGraph`.

    Parameters
    ----------
    graph :
        The graph to differentiate.
    strict :
        Default for :meth:`gradient`: raise instead of returning a derivative
        that is not ``EXACT``.
    fd_eps :
        Relative step of the finite-difference fallback.
    """

    def __init__(
        self, graph: ComputationalGraph, strict: bool = False, fd_eps: float = FD_EPS
    ) -> None:
        self.graph = graph
        self.strict = strict
        self.fd_eps = float(fd_eps)
        self._no_jet: Set[int] = set()  # node ids whose forward cannot run on jets

    # -- forward ------------------------------------------------------------
    def _forward(self, feed_dict: Optional[Dict[Node, Any]]) -> Tuple[List[Node], Dict[int, Any]]:
        order = self.graph.topological_sort()
        values: Dict[int, Any] = {}
        for n, val in (feed_dict or {}).items():
            values[n.node_id] = val
        for node in order:
            if node.node_id in values:
                continue
            values[node.node_id] = node.forward(*[values[i.node_id] for i in node.inputs])
        return order, values

    def _resolve(self, target: Union[str, Node]) -> Node:
        return self.graph.outputs[target] if isinstance(target, str) else target

    # -- one node -------------------------------------------------------------
    def _node_vjp(
        self,
        node: Node,
        inp_vals: List[Any],
        out_val: Any,
        ct: np.ndarray,
        need: List[int],
        notes: List[GradientNote],
    ) -> Tuple[Dict[int, np.ndarray], GradientExactness]:
        declared = GradientExactness(getattr(node, "gradient_exactness", EXACT))
        if isinstance(node, OutputNode):
            return {0: ct}, EXACT
        vjp_fn = getattr(node, "vjp_fn", None)
        if vjp_fn is not None:
            vjps = vjp_fn(ct, *inp_vals)
            vjps = tuple(vjps) if isinstance(vjps, (tuple, list)) else (vjps,)
            if len(vjps) != len(inp_vals):
                raise ValueError(
                    f"{node.name}: vjp_fn returned {len(vjps)} cotangents for "
                    f"{len(inp_vals)} inputs"
                )
            return {i: vjps[i] for i in need}, declared
        if getattr(node, "jet_capable", True) and node.node_id not in self._no_jet:
            try:
                return _local_jet_vjp(node, inp_vals, out_val, ct, need), declared
            except _JET_FAILURES as exc:
                self._no_jet.add(node.node_id)
                reason = f"forward does not support jets ({type(exc).__name__}: {exc})"
        else:
            reason = "node is marked jet-incapable"
        if not all(_is_numeric(inp_vals[i]) for i in need):
            raise TypeError(f"{node.name}: cannot differentiate through non-numeric inputs")
        notes.append(GradientNote(node.name, NUMERICAL, "central differences", reason))
        return _fd_vjp(node, inp_vals, ct, need, self.fd_eps), worst([declared, NUMERICAL])

    # -- reverse pass -------------------------------------------------------
    def _reverse(
        self,
        order: List[Node],
        values: Dict[int, Any],
        target: Node,
        sources: List[Node],
        seed: np.ndarray,
        fed: Set[int],
        notes: List[GradientNote],
    ) -> Tuple[Dict[int, Any], Dict[int, GradientExactness]]:
        active = _relevant(order, target, sources, fed)
        adj: Dict[int, Any] = {}
        lvl: Dict[int, GradientExactness] = {}
        if target.node_id in active:
            adj[target.node_id] = seed
            lvl[target.node_id] = EXACT
        for node in reversed(order):
            nid = node.node_id
            if nid not in adj or nid not in active:
                continue
            if nid in fed:  # fed nodes (including fed sources) are leaves
                continue
            need = [k for k, i in enumerate(node.inputs) if i.node_id in active]
            if not need:
                continue
            ct = adj[nid]
            if not np.any(ct != 0):
                continue
            inp_vals = [values[i.node_id] for i in node.inputs]
            local, level = self._node_vjp(node, inp_vals, values[nid], ct, need, notes)
            out_level = worst([lvl.get(nid, EXACT), level])
            for k in need:
                inp = node.inputs[k]
                iv = values[inp.node_id]
                g = _unbroadcast(local[k], _shape(iv), _is_complex(iv))
                if inp.node_id in adj:
                    prev = adj[inp.node_id]
                    if _is_complex(g) and not _is_complex(prev):
                        prev = np.asarray(prev, dtype=complex)
                    adj[inp.node_id] = prev + g
                else:
                    adj[inp.node_id] = g
                lvl[inp.node_id] = worst([lvl.get(inp.node_id, EXACT), out_level])
        return adj, lvl

    # -- public API -----------------------------------------------------------
    def gradient(
        self,
        target_output: Union[str, Node],
        sources: Optional[List[Node]] = None,
        feed_dict: Optional[Dict[Node, Any]] = None,
        grad_output: Any = None,
        *,
        strict: Optional[bool] = None,
    ) -> GradientResult:
        r"""
        Gradients of ``target_output`` with respect to ``sources``.

        Parameters
        ----------
        target_output :
            Output name or node. Array targets are summed (or contracted with
            ``grad_output``).
        sources :
            Nodes to differentiate against (default: every ``ParameterNode``).
        feed_dict :
            Node values that override the graph; fed nodes are leaves.
        grad_output :
            Cotangent of the target (JAX convention). Default: ones, i.e. the
            gradient of ``sum(target)``.
        strict :
            Raise :class:`InexactGradientError` unless every gradient is exact.

        Returns
        -------
        GradientResult
            ``{source: gradient}`` with the shape of each source's value, plus
            ``exactness``, ``source_exactness`` and ``notes``.
        """
        strict = self.strict if strict is None else strict
        target = self._resolve(target_output)
        if sources is None:
            sources = list(self.graph.get_parameters().values())
        order, values = self._forward(feed_dict)
        fed = {n.node_id for n in (feed_dict or {})}
        tval = values[target.node_id]
        if not _is_numeric(tval):
            raise TypeError(f"target {target.name!r} is not numeric: {type(tval).__name__}")
        notes: List[GradientNote] = []
        tshape = _shape(tval)
        if grad_output is not None:
            seeds = [np.broadcast_to(np.asarray(grad_output), tshape)]
        elif _is_complex(tval):
            seeds = [np.ones(tshape), np.full(tshape, -1j)]
        else:
            seeds = [np.ones(tshape)]
        passes = [self._reverse(order, values, target, sources, s, fed, notes) for s in seeds]
        result: Dict[Node, Any] = {}
        levels: Dict[Node, GradientExactness] = {}
        for s in sources:
            sval = values.get(s.node_id, 0.0)
            parts = [p[0].get(s.node_id) for p in passes]
            if all(pt is None for pt in parts):
                result[s] = _finalize(np.zeros(_shape(sval)), sval)
                levels[s] = EXACT
                continue
            parts = [np.zeros(_shape(sval)) if pt is None else pt for pt in parts]
            if len(parts) == 2 and not _is_complex(sval):
                g = np.real(parts[0]) + 1j * np.real(parts[1])
            else:
                g = parts[0]
            result[s] = _finalize(g, sval)
            levels[s] = worst(p[1].get(s.node_id, EXACT) for p in passes)
        out = GradientResult(result, levels, _dedupe(notes))
        check_strict(strict, out.exactness, out.notes, f"gradient of {target.name!r}")
        return out

    def jacobian(
        self,
        targets: List[Union[str, Node]],
        sources: List[Node],
        feed_dict: Optional[Dict[Node, Any]] = None,
        *,
        strict: Optional[bool] = None,
        return_exactness: bool = False,
    ) -> Any:
        r"""
        Full Jacobian :math:`J_{ij} = \partial\,\mathrm{target}_i/\partial\,\mathrm{source}_j`.

        Rows are the flattened elements of the targets in order, columns the
        flattened elements of the sources, so scalar targets and sources give
        a ``(len(targets), len(sources))`` matrix. The Jacobian is computed in
        forward mode with jets when every node on the way supports them, and
        otherwise row by row in reverse mode.
        """
        strict = self.strict if strict is None else strict
        try:
            J, level, notes = _forward_jacobian(self, targets, sources, feed_dict)
        except _ForwardUnavailable:
            J, level, notes = self._reverse_jacobian(targets, sources, feed_dict)
        check_strict(strict, level, notes, "jacobian")
        if return_exactness:
            return J, level, notes
        return J

    def _reverse_jacobian(
        self,
        targets: List[Union[str, Node]],
        sources: List[Node],
        feed_dict: Optional[Dict[Node, Any]],
    ) -> Tuple[np.ndarray, GradientExactness, List[GradientNote]]:
        order, values = self._forward(feed_dict)
        fed = {n.node_id for n in (feed_dict or {})}
        notes: List[GradientNote] = []
        level = EXACT
        rows: List[np.ndarray] = []
        cplx = False
        for t in targets:
            tn = self._resolve(t)
            tval = np.asarray(values[tn.node_id])
            for m in range(max(1, tval.size)):
                row_parts = []
                for unit in (1.0, -1j) if _is_complex(tval) else (1.0,):
                    seed = np.zeros(tval.shape, dtype=complex if unit != 1.0 else float)
                    seed.reshape(-1)[m] = unit
                    adj, lvl = self._reverse(order, values, tn, sources, seed, fed, notes)
                    level = worst([level, *[lvl.get(s.node_id, EXACT) for s in sources]])
                    row_parts.append(
                        np.concatenate(
                            [
                                np.ravel(
                                    np.real(adj[s.node_id])
                                    if s.node_id in adj
                                    else np.zeros(np.size(values.get(s.node_id, 0.0)))
                                )
                                for s in sources
                            ]
                        )
                    )
                if len(row_parts) == 2:
                    cplx = True
                    rows.append(row_parts[0] + 1j * row_parts[1])
                else:
                    rows.append(row_parts[0])
        J = np.array(rows, dtype=complex if cplx else float)
        return J, level, _dedupe(notes)


def _dedupe(notes: List[GradientNote]) -> List[GradientNote]:
    seen = set()
    out = []
    for n in notes:
        key = (n.node, n.method)
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


# ── forward mode through the graph ────────────────────────────────────────────


class _ForwardUnavailable(Exception):
    """Some node on the way cannot run on jets (forward mode needs a fallback)."""


def _seed_sources(
    sources: Sequence[Node], values: Dict[int, Any], order: int
) -> Tuple[Dict[int, Jet], List[Tuple[Node, int, int]]]:
    """One real direction per element of each (real) source."""
    layout = []
    off = 0
    for s in sources:
        v = values.get(s.node_id, 0.0)
        if _is_complex(v):
            raise _ForwardUnavailable("complex sources are differentiated in reverse mode")
        layout.append((s, off, int(np.size(v))))
        off += int(np.size(v))
    n = off
    jets: Dict[int, Jet] = {}
    for s, o, size in layout:
        v = np.asarray(values.get(s.node_id, 0.0), dtype=float)
        g = np.zeros((size, n))
        g[np.arange(size), o + np.arange(size)] = 1.0
        h = np.zeros((size, n, n)) if order == 2 else None
        jets[s.node_id] = Jet(
            v, g.reshape(v.shape + (n,)), None if h is None else h.reshape(v.shape + (n, n))
        )
    return jets, layout


def _jet_pass(
    tape: GradientTape,
    order_nodes: List[Node],
    values: Dict[int, Any],
    seeds: Dict[int, Jet],
    fed: Set[int],
) -> Dict[int, Any]:
    """Propagate jets from ``seeds``; raises _ForwardUnavailable on a jet-incapable node."""
    jv: Dict[int, Any] = dict(seeds)
    for node in order_nodes:
        nid = node.node_id
        if nid in jv or nid in fed:
            continue
        ins = [jv.get(i.node_id, values[i.node_id]) for i in node.inputs]
        if not any(isinstance(x, Jet) for x in ins):
            continue
        if not getattr(node, "jet_capable", True) or nid in tape._no_jet:
            raise _ForwardUnavailable(node.name)
        if getattr(node, "gradient_exactness", EXACT) != EXACT:
            raise _ForwardUnavailable(node.name)
        try:
            res = _coerce(node.forward(*ins))
        except _JET_FAILURES as exc:
            tape._no_jet.add(nid)
            raise _ForwardUnavailable(node.name) from exc
        if not isinstance(res, Jet) or not _values_match(res.value, values[nid]):
            # no derivatives came out (possibly swallowed), or a different value
            tape._no_jet.add(nid)
            raise _ForwardUnavailable(node.name)
        jv[nid] = res
    return jv


def _forward_jacobian(
    tape: GradientTape,
    targets: List[Union[str, Node]],
    sources: List[Node],
    feed_dict: Optional[Dict[Node, Any]],
) -> Tuple[np.ndarray, GradientExactness, List[GradientNote]]:
    order, values = tape._forward(feed_dict)
    fed = {n.node_id for n in (feed_dict or {})} - {s.node_id for s in sources}
    seeds, layout = _seed_sources(sources, values, 1)
    n = sum(size for _, _, size in layout)
    jv = _jet_pass(tape, order, values, seeds, fed)
    rows = []
    for t in targets:
        tn = tape._resolve(t)
        res = jv.get(tn.node_id)
        tval = np.asarray(values[tn.node_id])
        if isinstance(res, Jet):
            rows.append(np.asarray(res.grad).reshape(max(1, tval.size), n))
        else:
            rows.append(np.zeros((max(1, tval.size), n)))
    J = np.concatenate(rows, axis=0)
    if not np.iscomplexobj(J):
        J = J.astype(float)
    return J, EXACT, []


def forward_hessian(
    tape: GradientTape,
    target: Union[str, Node],
    sources: List[Node],
    feed_dict: Optional[Dict[Node, Any]] = None,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Value, gradient and Hessian of ``sum(target)`` by second-order jets (exact).

    Raises ``_ForwardUnavailable`` if a node on the way cannot run on jets.
    """
    order, values = tape._forward(feed_dict)
    fed = {n.node_id for n in (feed_dict or {})} - {s.node_id for s in sources}
    seeds, layout = _seed_sources(sources, values, 2)
    n = sum(size for _, _, size in layout)
    jv = _jet_pass(tape, order, values, seeds, fed)
    tn = tape._resolve(target)
    res = jv.get(tn.node_id)
    if not isinstance(res, Jet):
        return values[tn.node_id], np.zeros(n), np.zeros((n, n))
    axes = tuple(range(res.value.ndim))
    return (
        values[tn.node_id],
        np.sum(res.grad, axis=axes),
        np.sum(res.hess, axis=axes),
    )


def jvp(
    graph: ComputationalGraph,
    target_output: Union[str, Node],
    tangents: Dict[Node, Any],
    feed_dict: Optional[Dict[Node, Any]] = None,
    *,
    strict: bool = False,
) -> Tuple[Any, Any, GradientExactness]:
    r"""
    Forward-mode directional derivative :math:`\sum_s \partial y/\partial s\cdot t_s`.

    Returns ``(y, dy, exactness)``. Exact when every node on the way supports
    jets; otherwise falls back to the reverse-mode Jacobian (exact where the
    nodes have VJPs) contracted with the tangents.
    """
    tape = GradientTape(graph)
    sources = list(tangents)
    order, values = tape._forward(feed_dict)
    tn = tape._resolve(target_output)
    fed = {n.node_id for n in (feed_dict or {})} - {s.node_id for s in sources}
    try:
        seeds = {}
        for s, t in tangents.items():
            v = np.asarray(values.get(s.node_id, 0.0))
            tt = np.broadcast_to(np.asarray(t), v.shape)
            seeds[s.node_id] = Jet(v, tt[..., None])
        jv = _jet_pass(tape, order, values, seeds, fed)
        res = jv.get(tn.node_id)
        y = values[tn.node_id]
        dy = res.grad[..., 0] if isinstance(res, Jet) else np.zeros_like(np.asarray(y), float)
        return y, _finalize(dy, y), EXACT
    except _ForwardUnavailable:
        J, level, notes = tape._reverse_jacobian([tn], sources, feed_dict)
        check_strict(strict, level, notes, "jvp")
        tvec = np.concatenate(
            [
                np.ravel(np.broadcast_to(np.asarray(t), np.shape(values.get(s.node_id, 0.0))))
                for s, t in tangents.items()
            ]
        )
        y = values[tn.node_id]
        dy = (J @ tvec).reshape(np.shape(y))
        return y, _finalize(dy, y), level


def value_and_grad(
    graph: ComputationalGraph,
    target_output: str,
    param_names: Optional[List[str]] = None,
    *,
    strict: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    r"""
    Value of ``target_output`` and its gradient keyed by parameter name.

    Scalar targets and parameters give floats. Names not in the graph are
    ignored. With ``strict=True`` an inexact gradient raises.
    """
    tape = GradientTape(graph)
    params = graph.get_parameters()
    if param_names is None:
        param_list = list(params.values())
    else:
        param_list = [params[name] for name in param_names if name in params]
    val = graph.evaluate(return_outputs_only=True)[target_output]
    grads = tape.gradient(target_output, sources=param_list, strict=strict)
    val_out = _finalize(np.asarray(val), val) if _is_numeric(val) else val
    return val_out, {node.name: grads[node] for node in param_list}
