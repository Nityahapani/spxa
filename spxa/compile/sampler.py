r"""
Fused exact samplers for lowered process expressions.

A :class:`Sampler` turns the structural tree of a :class:`~spxa.compile.lower.Lowered`
expression into a sampling plan once, then draws :math:`X_t` for arrays of
times (and paths on any grid) with a fixed number of vectorised NumPy calls,
for one parameter vector or a ``(B, n)`` batch of them in the same call.

Every node of the plan draws :math:`X_t` *exactly* at arbitrary,
entry-wise times :math:`t \ge 0`:

* ``X + Y`` adds independent draws, ``c X`` scales them.
* ``X @ T`` draws the clock :math:`T_t` and then :math:`X` at those random
  times; :math:`X_{T_t}` given :math:`T_t = s` is :math:`X_s` (Sato 1999, Thm. 30.1).
* Gaussian parts are merged: the Brownian, drift, Merton/Kou diffusion and NIG
  drift terms of one sum become a single :math:`N(mt, vt)` draw with
  :math:`m = \sum c_i\mu_i`, :math:`v = \sum c_i^2\sigma_i^2`.
* Jump sums are aggregated: given :math:`N \sim \text{Poisson}(\lambda t)`,
  :math:`N` normal jumps are :math:`N(Nm, Ns^2)`, :math:`N` exponential jumps
  are :math:`\text{Gamma}(N)/\eta`, double-exponential jumps split by a
  :math:`\text{Binomial}(N, p)` count into two Gamma sums, discrete jumps are
  a multinomial count vector, and uniform jumps are :math:`Na + (b-a)S_N`
  with :math:`S_N` a segmented sum of :math:`N` standard uniforms.
* Inverse Gaussian draws use
  :func:`~spxa.zoo.levy.inverse_gaussian.sample_inverse_gaussian`, exact at
  the tiny clock increments that occur under subordination.

Leaves without a sampling rule (tempered stable, NTS, Meixner, GH, CGMY,
:math:`\alpha`-stable and opaque leaves) call the process's own
:meth:`~spxa.core.process.Process.sample_at`. The plan is exact when each of
those has a direct (``_vector_sampler``) sampler; :attr:`Sampler.exact` reports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from spxa.compile.lower import LLeaf, LNode, Lowered, LOpaque, LScale, LSub, LSum
from spxa.zoo.levy.inverse_gaussian import sample_inverse_gaussian

__all__ = ["Sampler"]

Param = Union[float, np.ndarray]


# ── leaf rules ──────────────────────────────────────────────────────────────
# Each kind maps to (gauss, jump). gauss(p) -> (drift rate, variance rate) or
# None; jump(t, p, rng, extra) -> draws of the non-Gaussian part, shaped like t.


def _bc(p: Param, shape: Tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(np.asarray(p, dtype=float), shape)


def _masked(t: np.ndarray, *ps: Param) -> Tuple[Any, np.ndarray, List[Any]]:
    """Positions with t > 0 (``...`` when all are), and the parameters there."""
    if t.size and t.min() > 0:
        return Ellipsis, t, list(ps)
    pos = t > 0
    return pos, t[pos], [_bc(p, t.shape)[pos] for p in ps]


def _gamma_jump(t: np.ndarray, p: Sequence[Param], rng: np.random.Generator, _: Any) -> np.ndarray:
    a, b = p
    out = np.zeros(t.shape)
    pos, tp, (ap, bp) = _masked(t, a, b)
    out[pos] = rng.gamma(ap * tp, 1.0 / bp)
    return out


def _ig_jump(t: np.ndarray, p: Sequence[Param], rng: np.random.Generator, _: Any) -> np.ndarray:
    mu, lam = p
    return sample_inverse_gaussian(mu * t, lam * t**2, rng)


def _negbin_jump(t: np.ndarray, p: Sequence[Param], rng: np.random.Generator, _: Any) -> np.ndarray:
    r, q = p
    out = np.zeros(t.shape)
    pos, tp, (rp, qp) = _masked(t, r, q)
    # numpy counts failures before n successes with success probability 1 - q
    out[pos] = rng.negative_binomial(rp * tp, 1.0 - qp)
    return out


def _poisson_jump(
    t: np.ndarray, p: Sequence[Param], rng: np.random.Generator, _: Any
) -> np.ndarray:
    rate, c = p
    return c * rng.poisson(_bc(rate * t, t.shape))


def _vg_jump(t: np.ndarray, p: Sequence[Param], rng: np.random.Generator, _: Any) -> np.ndarray:
    sigma, nu, theta = p
    g = np.zeros(t.shape)
    pos, tp, (nup,) = _masked(t, nu)
    g[pos] = rng.gamma(tp / nup, nup)
    return theta * g + sigma * np.sqrt(g) * rng.standard_normal(t.shape)


def _nig_jump(t: np.ndarray, p: Sequence[Param], rng: np.random.Generator, _: Any) -> np.ndarray:
    alpha, beta, delta, _mu = p
    gam = np.sqrt(np.asarray(alpha) ** 2 - np.asarray(beta) ** 2)
    s = sample_inverse_gaussian(delta * t / gam, (delta * t) ** 2, rng)
    return beta * s + np.sqrt(s) * rng.standard_normal(t.shape)


def _segment_sums(counts: np.ndarray, draw: Callable[[int], np.ndarray]) -> np.ndarray:
    """``out[k] = sum of counts[k] fresh draws`` for integer ``counts``."""
    flat = counts.reshape(-1).astype(np.int64)
    total = int(flat.sum())
    out = np.zeros(flat.shape)
    if total:
        vals = draw(total)
        live = flat > 0
        starts = np.concatenate([[0], np.cumsum(flat)[:-1]])[live]
        out[live] = np.add.reduceat(vals, starts)
    return out.reshape(counts.shape)


def _jump_sums(
    kind: str, n: np.ndarray, p: Sequence[Param], rng: np.random.Generator, extra: Dict[str, Any]
) -> np.ndarray:
    """Exact sums of ``n`` i.i.d. jumps of law ``kind`` with parameters ``p``."""
    live = n > 0
    out = np.zeros(n.shape)
    if not live.any():
        return out
    # jumps are rare on fine grids: work on the entries with at least one
    out[live] = _jump_sums_dense(kind, n[live], [_bc(q, n.shape)[live] for q in p], rng, extra)
    return out


def _jump_sums_dense(
    kind: str, n: np.ndarray, p: Sequence[Param], rng: np.random.Generator, extra: Dict[str, Any]
) -> np.ndarray:
    nf = n.astype(float)
    if kind == "normal":
        m, s = p
        return m * nf + s * np.sqrt(nf) * rng.standard_normal(n.shape)
    if kind == "exponential":
        (eta,) = p
        return rng.standard_gamma(nf) / eta
    if kind == "double_exponential":
        q, e1, e2 = p
        k_up = rng.binomial(n, _bc(q, n.shape))
        return (
            rng.standard_gamma(k_up.astype(float)) / e1
            - rng.standard_gamma((n - k_up).astype(float)) / e2
        )
    if kind == "uniform":
        lo, hi = p
        return lo * nf + (hi - lo) * _segment_sums(n, rng.random)
    if kind == "discrete":
        values = np.asarray(extra["values"], dtype=float)
        counts = rng.multinomial(n, np.asarray(extra["probs"], dtype=float))
        return np.asarray(counts @ values, dtype=float)
    raise ValueError(f"unknown jump law {kind!r}")  # pragma: no cover - rules are closed


def _merton_jump(t: np.ndarray, p: Sequence[Param], rng: np.random.Generator, _: Any) -> np.ndarray:
    _d, _s, rate, jm, js = p
    n = rng.poisson(_bc(rate * t, t.shape))
    return _jump_sums("normal", n, (jm, js), rng, {})


def _kou_jump(t: np.ndarray, p: Sequence[Param], rng: np.random.Generator, _: Any) -> np.ndarray:
    _d, _s, rate, q, e1, e2 = p
    n = rng.poisson(_bc(rate * t, t.shape))
    return _jump_sums("double_exponential", n, (q, e1, e2), rng, {})


def _cp_jump(
    t: np.ndarray, p: Sequence[Param], rng: np.random.Generator, extra: Dict[str, Any]
) -> np.ndarray:
    n = rng.poisson(_bc(p[0] * t, t.shape))
    return _jump_sums(extra["jump"], n, p[1:], rng, extra)


GaussFn = Callable[[Sequence[Param]], Tuple[Param, Param]]
JumpFn = Callable[[np.ndarray, Sequence[Param], np.random.Generator, Any], np.ndarray]

#: kind -> (Gaussian part, non-Gaussian part)
LEAF_SAMPLERS: Dict[str, Tuple[Optional[GaussFn], Optional[JumpFn]]] = {
    "gaussian": (lambda p: (p[0], p[1] ** 2), None),
    "drift": (lambda p: (p[0], 0.0), None),
    "gamma": (None, _gamma_jump),
    "inverse_gaussian": (None, _ig_jump),
    "negative_binomial": (None, _negbin_jump),
    "poisson": (None, _poisson_jump),
    "variance_gamma": (None, _vg_jump),
    "nig": (lambda p: (p[3], 0.0), _nig_jump),
    "merton": (lambda p: (p[0], p[1] ** 2), _merton_jump),
    "kou": (lambda p: (p[0], p[1] ** 2), _kou_jump),
    "compound_poisson": (None, _cp_jump),
}


# ── plan ────────────────────────────────────────────────────────────────────


@dataclass
class _Group:
    """A flattened sum: ``sum_k prod(theta[scales_k]) * item_k``."""

    gauss: List[Tuple[Tuple[int, ...], LLeaf]] = field(default_factory=list)
    jumps: List[Tuple[Tuple[int, ...], LLeaf]] = field(default_factory=list)
    fallback: List[Tuple[Tuple[int, ...], LNode]] = field(default_factory=list)
    subs: List[Tuple[Tuple[int, ...], "_Group", "_Group"]] = field(default_factory=list)


def _plan(node: LNode, scales: Tuple[int, ...] = (), group: Optional[_Group] = None) -> _Group:
    g = _Group() if group is None else group
    if isinstance(node, LSum):
        _plan(node.a, scales, g)
        _plan(node.b, scales, g)
    elif isinstance(node, LScale):
        _plan(node.base, scales + (node.slot,), g)
    elif isinstance(node, LSub):
        g.subs.append((scales, _plan(node.x), _plan(node.clock)))
    elif isinstance(node, LLeaf) and node.spec.sampler in LEAF_SAMPLERS:
        gauss, jump = LEAF_SAMPLERS[node.spec.sampler]
        if gauss is not None:
            g.gauss.append((scales, node))
        if jump is not None:
            g.jumps.append((scales, node))
    elif isinstance(node, (LLeaf, LOpaque)):
        g.fallback.append((scales, node))
    else:  # pragma: no cover - lower() produces only the node types above
        raise TypeError(f"cannot sample {type(node).__name__}")
    return g


@dataclass
class _Ctx:
    theta: np.ndarray  # (n,) or (B, n)
    params: List[Param]  # theta[i] as a float, or (B, 1, ..., 1)
    batch: Optional[int]
    rng: np.random.Generator


def _scale(ctx: _Ctx, slots: Tuple[int, ...]) -> Param:
    c: Param = 1.0
    for s in slots:
        c = c * ctx.params[s]
    return c


def _leaf_params(ctx: _Ctx, leaf: LLeaf) -> List[Param]:
    return [ctx.params[i] for i in leaf.slots]


def _draw(g: _Group, t: np.ndarray, ctx: _Ctx) -> np.ndarray:
    out = np.zeros(t.shape)
    if g.gauss:
        m: Param = 0.0
        v: Param = 0.0
        for slots, leaf in g.gauss:
            gauss = LEAF_SAMPLERS[str(leaf.spec.sampler)][0]
            assert gauss is not None
            c = _scale(ctx, slots)
            gm, gv = gauss(_leaf_params(ctx, leaf))
            m = m + c * gm
            v = v + c * c * gv
        out += m * t
        if not (np.ndim(v) == 0 and float(v) == 0.0):  # drift-only sums need no normal
            out += np.sqrt(v * t) * ctx.rng.standard_normal(t.shape)
    for slots, leaf in g.jumps:
        jump = LEAF_SAMPLERS[str(leaf.spec.sampler)][1]
        assert jump is not None
        out += _scale(ctx, slots) * jump(t, _leaf_params(ctx, leaf), ctx.rng, leaf.spec.extra)
    for slots, node in g.fallback:
        out += _scale(ctx, slots) * _fallback(node, t, ctx)
    for slots, x, clock in g.subs:
        tau = _draw(clock, t, ctx)
        if np.any(tau < 0):
            raise ValueError("the clock of a subordination drew a negative time")
        out += _scale(ctx, slots) * _draw(x, tau, ctx)
    return out


def _fallback(node: LNode, t: np.ndarray, ctx: _Ctx) -> np.ndarray:
    if ctx.batch is None:
        return np.asarray(node.build(ctx.theta).sample_at(t, ctx.rng), dtype=float)
    out = np.empty(t.shape)
    for b in range(ctx.batch):
        out[b] = node.build(ctx.theta[b]).sample_at(t[b], ctx.rng)
    return out


def _fallback_exact(node: LNode, theta: np.ndarray) -> bool:
    return bool(getattr(node.build(theta), "_vector_sampler", False))


def _all_fallbacks(g: _Group) -> List[LNode]:
    out = [n for _, n in g.fallback]
    for _, x, clock in g.subs:
        out += _all_fallbacks(x) + _all_fallbacks(clock)
    return out


# ── public ──────────────────────────────────────────────────────────────────

#: Entries per vectorised call when sampling a batch of parameter vectors.
_CHUNK = 1 << 16


class Sampler:
    """
    Exact draws of a lowered expression at arbitrary times, for one or many
    parameter vectors.

    Parameters
    ----------
    lowered :
        The output of :func:`~spxa.compile.lower.lower`.

    All methods take ``theta``: a full parameter vector (``lowered.names``
    order) or a ``(B, n)`` batch, in which case the output gains a leading
    batch axis. Default: ``lowered.values``.
    """

    def __init__(self, lowered: Lowered) -> None:
        self.lowered = lowered
        self.plan = _plan(lowered.tree)
        self._fallbacks = _all_fallbacks(self.plan)

    @property
    def fallback_leaves(self) -> int:
        """Number of leaves drawn through their process's own ``sample_at``."""
        return len(self._fallbacks)

    def exact(self, theta: Optional[Any] = None) -> bool:
        """Whether every leaf at ``theta`` (one vector) has an exact sampler."""
        th = self._theta(theta)
        if th.ndim != 1:
            return all(self.exact(row) for row in th)
        return all(_fallback_exact(n, th) for n in self._fallbacks)

    def _theta(self, theta: Optional[Any]) -> np.ndarray:
        th = self.lowered.values if theta is None else np.asarray(theta, dtype=float)
        n = len(self.lowered.names)
        if th.ndim not in (1, 2) or th.shape[-1] != n:
            raise ValueError(f"theta must have shape ({n},) or (B, {n}), got {th.shape}")
        return th

    def _ctx(self, theta: np.ndarray, ndim: int, rng: np.random.Generator) -> _Ctx:
        if theta.ndim == 1:
            return _Ctx(theta, [float(v) for v in theta], None, rng)
        shape = (theta.shape[0],) + (1,) * ndim
        return _Ctx(
            theta, [theta[:, i].reshape(shape) for i in range(theta.shape[1])], len(theta), rng
        )

    def sample_at(
        self,
        times: Any,
        rng: Optional[np.random.Generator] = None,
        theta: Optional[Any] = None,
    ) -> np.ndarray:
        r"""
        Independent draws of :math:`X_{t_k}`, one per entry of ``times``.

        Returns an array shaped like ``times`` (with a leading batch axis for
        a batch of ``theta``).
        """
        rng = rng or np.random.default_rng()
        th = self._theta(theta)
        t = np.asarray(times, dtype=float)
        if np.any(t < 0) or not np.all(np.isfinite(t)):
            raise ValueError("times must be finite and non-negative")
        if th.ndim == 1:
            return _draw(self.plan, t, self._ctx(th, t.ndim, rng))
        # A batch runs in chunks of rows: whole rows with scalar parameters when
        # they are large (NumPy's scalar-parameter fast paths, no broadcast
        # temporaries), several rows per vectorised call when they are small.
        out = np.empty((th.shape[0],) + t.shape)
        rows = max(1, _CHUNK // max(t.size, 1))
        for lo in range(0, th.shape[0], rows):
            block = th[lo : lo + rows]
            if len(block) == 1:
                out[lo] = _draw(self.plan, t, self._ctx(block[0], t.ndim, rng))
            else:
                tb = np.broadcast_to(t, (len(block),) + t.shape).copy()
                out[lo : lo + len(block)] = _draw(self.plan, tb, self._ctx(block, t.ndim, rng))
        return out

    def paths(
        self,
        times: Any,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
        theta: Optional[Any] = None,
    ) -> np.ndarray:
        r"""
        Paths observed at the non-decreasing ``times`` (any grid, :math:`t_0 \ge 0`).

        Returns ``(n_paths, len(times))`` (or ``(B, n_paths, len(times))``),
        column ``k`` being :math:`X_{t_k}` with :math:`X_0 = 0`.
        """
        grid = np.asarray(times, dtype=float)
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError("times must be a non-empty 1-D grid")
        dt = np.diff(grid, prepend=0.0)
        if np.any(dt < 0):
            raise ValueError("times must be non-negative and non-decreasing")
        incr = self.sample_at(np.broadcast_to(dt, (int(n_paths), dt.size)), rng, theta)
        return np.cumsum(incr, axis=-1)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,  # noqa: N803
        rng: Optional[np.random.Generator] = None,
        theta: Optional[Any] = None,
    ) -> np.ndarray:
        """``(n_paths, n_steps + 1)`` paths on an even grid over ``[0, T]`` (batch-leading)."""
        if n_steps < 1 or n_paths < 1 or not T > 0:
            raise ValueError("need n_steps >= 1, n_paths >= 1 and T > 0")
        incr = self.sample_at(np.full((int(n_paths), int(n_steps)), T / n_steps), rng, theta)
        zero = np.zeros(incr.shape[:-1] + (1,))
        return np.concatenate([zero, np.cumsum(incr, axis=-1)], axis=-1)

    def __repr__(self) -> str:
        return f"Sampler({self.lowered.story()}, fallback_leaves={self.fallback_leaves})"
