r"""
Code generation: IR → Python source for NumPy, Numba or JAX.

A :class:`Program` is one generated function ``fn(u, p0, p1, ...)`` that
returns a tuple with one array per output expression. The source is plain,
readable Python (see :attr:`Program.source`) and is registered with
:mod:`linecache`, so tracebacks point into it.

The generator

* emits every shared subexpression once (the IR is hash-consed, so sharing
  is structural);
* hoists the parameter-only part of the DAG ahead of the :math:`u`-dependent
  part (loop-invariant code motion), so per-frequency work is minimal;
* inlines single-use subexpressions into their user, so the source reads
  like the formula it came from;
* keeps NumPy semantics for real nodes (parameters are passed as NumPy
  floats, never Python floats, so ``(-1.0) ** 0.5`` is NaN and ``x / 0`` is
  ``inf``, as in the zoo).

Backends:

``"numpy"``
    Straight-line array code; always available.
``"numba"``
    The hoisted part runs in NumPy; the :math:`u`-dependent part becomes one
    ``@njit`` loop over (parameter set, frequency) with no temporaries.
    Needs ``numba``; special functions of :math:`u` (``kve``) and opaque
    calls are not supported there.
``"jax"``
    ``jax.numpy`` code under ``jax.jit``, evaluated with 64-bit types.
    Needs ``jax``; ``kve`` and opaque calls are not supported.
"""

from __future__ import annotations

import itertools
import linecache
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spxa.compile import ir as I
from spxa.compile.ir import CompileError, Expr

__all__ = ["Program", "generate", "BackendUnavailable", "BackendUnsupported", "BACKENDS"]

BACKENDS = ("numpy", "numba", "jax")

_COUNTER = itertools.count()
#: Inline at most this many nested operations before naming a temporary.
_MAX_INLINE_DEPTH = 12


class BackendUnavailable(ImportError):
    """The backend's library is not installed."""


class BackendUnsupported(CompileError):
    """The expression uses an operation the backend cannot generate."""


@dataclass
class Program:
    """A generated function and its source."""

    backend: str
    name: str
    source: str
    fn: Callable[..., Tuple[Any, ...]]
    n_params: int
    n_outputs: int
    hoisted: int = 0
    namespace: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __call__(self, u: Any, *params: Any) -> Tuple[Any, ...]:
        if len(params) != self.n_params:
            raise TypeError(f"{self.name} takes {self.n_params} parameters, got {len(params)}")
        if self.backend == "jax":
            return self.fn(u, *params)
        # NumPy semantics: real parameters are NumPy floats, u is complex
        uu = np.asarray(u, dtype=complex)
        ps = tuple(np.asarray(p, dtype=float)[()] for p in params)
        return self.fn(uu, *ps)


# ── shared emission machinery ───────────────────────────────────────────────


def _fmt_float(v: float) -> Optional[str]:
    if math.isfinite(v):
        r = repr(float(v))
        return r if not r.startswith("-") else f"({r})"
    return None


class _Emitter:
    def __init__(self, outputs: Sequence[Expr], n_params: int) -> None:
        self.outputs = list(outputs)
        self.n_params = n_params
        self.nodes = I.postorder(self.outputs)
        self.uses: Dict[int, int] = {}
        for n in self.nodes:
            for a in n.args:
                self.uses[id(a)] = self.uses.get(id(a), 0) + 1
        for o in self.outputs:
            self.uses[id(o)] = self.uses.get(id(o), 0) + 2  # outputs are always named
        self.consts: Dict[str, Any] = {}
        self.const_names: Dict[Tuple[Any, ...], str] = {}
        self.procs: Dict[int, str] = {}
        self.namespace: Dict[str, Any] = {}

    def const_ref(self, node: Expr) -> str:
        v = node.payload
        if isinstance(v, bool):
            return repr(v)
        if isinstance(v, float):
            s = _fmt_float(v)
            if s is not None:
                return s
        key = I._payload_key("const", v)
        if key not in self.const_names:
            name = f"_c{len(self.const_names)}"
            self.const_names[key] = name
            self.namespace[name] = v
        return self.const_names[key]

    def proc_ref(self, node: Expr) -> str:
        if id(node.payload) not in self.procs:
            name = f"_proc{len(self.procs)}"
            self.procs[id(node.payload)] = name
            self.namespace[name] = node.payload
        return self.procs[id(node.payload)]


def _register(name: str, src: str) -> str:
    filename = f"<spxa.compile:{name}:{next(_COUNTER)}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    return filename


def _exec(src: str, filename: str, namespace: Dict[str, Any], fname: str) -> Callable[..., Any]:
    code = compile(src, filename, "exec")
    exec(code, namespace)  # noqa: S102 - generated from the IR, never from user strings
    return namespace[fname]  # type: ignore[no-any-return]


# ── array backends (NumPy, JAX) ─────────────────────────────────────────────

_ARRAY_FUNCS = {
    "numpy": {
        "exp": "np.exp",
        "expm1": "np.expm1",
        "log": "np.log",
        "log1p": "np.log1p",
        "sqrt": "np.sqrt",
        "sin": "np.sin",
        "cos": "np.cos",
        "tan": "np.tan",
        "abs": "np.abs",
        "sign": "np.sign",
        "real": "np.real",
        "imag": "np.imag",
        "sinc": "_sinc",
        "sinc_d1": "_sinc_d1",
        "gamma": "sps.gamma",
        "digamma": "sps.digamma",
        "polygamma": "sps.polygamma",
        "kve": "sps.kve",
        "kve_dnu": "_kve_dnu",
        "where": "np.where",
    },
    "jax": {
        "exp": "jnp.exp",
        "expm1": "jnp.expm1",
        "log": "jnp.log",
        "log1p": "jnp.log1p",
        "sqrt": "jnp.sqrt",
        "sin": "jnp.sin",
        "cos": "jnp.cos",
        "tan": "jnp.tan",
        "abs": "jnp.abs",
        "sign": "jnp.sign",
        "real": "jnp.real",
        "imag": "jnp.imag",
        "sinc": "_jsinc",
        "sinc_d1": "_jsinc_d1",
        "gamma": "jsp.gamma",
        "digamma": "jsp.digamma",
        "polygamma": "jsp.polygamma",
        "where": "jnp.where",
    },
}


def _array_expr(node: Expr, a: List[str], em: _Emitter, backend: str) -> str:
    op = node.op
    funcs = _ARRAY_FUNCS[backend]
    if op == "add":
        x, y = node.args
        # a + (-b) is printed as a - b (identical in IEEE arithmetic) when -b is inline
        if y.op == "neg" and _is_inline_neg(a[1]):
            return f"({a[0]} - {a[1][2:-1]})"
        if x.op == "neg" and _is_inline_neg(a[0]):
            return f"({a[1]} - {a[0][2:-1]})"
        return f"({a[0]} + {a[1]})"
    if op == "mul":
        return f"({a[0]} * {a[1]})"
    if op == "div":
        return f"({a[0]} / {a[1]})"
    if op == "pow":
        if backend == "numpy":
            # the ** operator special-cases some scalar exponents (0.5 -> sqrt, ...);
            # the ufunc does not, and is what the reference evaluator uses
            return f"np.power({a[0]}, {a[1]})"
        return f"({a[0]} ** {a[1]})"
    if op == "neg":
        return f"(-{a[0]})"
    if op == "ge":
        return f"({a[0]} >= {a[1]})"
    if op == "gt":
        return f"({a[0]} > {a[1]})"
    if op == "polygamma":
        return f"{funcs['polygamma']}({node.payload}, {a[0]})"
    if op == "call":
        if backend != "numpy":
            raise BackendUnsupported(f"opaque process exponents cannot run on {backend}")
        return f"_call({em.proc_ref(node)}, {a[0]})"
    fn = funcs.get(op)
    if fn is None:
        raise BackendUnsupported(f"{op} is not available on the {backend} backend")
    return f"{fn}({', '.join(a)})"


def _is_inline_neg(s: str) -> bool:
    """Whether ``s`` is the inline text ``(-X)`` emitted for a neg node."""
    return s.startswith("(-") and s.endswith(")")


def _emit_array(em: _Emitter, backend: str, fname: str) -> Tuple[List[str], int]:
    """Lines of the function body; returns (lines, number of hoisted statements)."""
    # parameter-only nodes first (they never depend on u-dependent ones)
    order = [n for n in em.nodes if not n.depends_u] + [n for n in em.nodes if n.depends_u]
    text: Dict[int, str] = {}
    depth: Dict[int, int] = {}
    lines: List[str] = []
    hoisted = 0
    tmp = itertools.count()
    for node in order:
        if node.op == "const":
            text[id(node)] = em.const_ref(node)
            depth[id(node)] = 0
            continue
        if node.op == "param":
            text[id(node)] = f"p{node.payload[0]}"
            depth[id(node)] = 0
            continue
        if node.op == "var":
            text[id(node)] = "u"
            depth[id(node)] = 0
            continue
        a = [text[id(x)] for x in node.args]
        expr = _array_expr(node, a, em, backend)
        d = 1 + max((depth[id(x)] for x in node.args), default=0)
        if em.uses.get(id(node), 0) > 1 or d > _MAX_INLINE_DEPTH:
            name = f"t{next(tmp)}"
            lines.append(f"{name} = {expr}")
            if not node.depends_u:
                hoisted += 1
            text[id(node)] = name
            depth[id(node)] = 0
        else:
            text[id(node)] = expr
            depth[id(node)] = d
    outs = ", ".join(text[id(o)] for o in em.outputs)
    lines.append(f"return ({outs},)")
    return lines, hoisted


def _array_program(outputs: Sequence[Expr], n_params: int, backend: str, name: str) -> Program:
    em = _Emitter(outputs, n_params)
    lines, hoisted = _emit_array(em, backend, name)
    args = ", ".join(["u"] + [f"p{k}" for k in range(n_params)])
    fname = f"_kernel_{name}"
    header = [f"def {fname}({args}):"]
    body: List[str]
    if backend == "numpy":
        # as in the reference evaluator: IEEE results (inf, nan) without warnings;
        # where() evaluates both branches and the discarded one may overflow
        body = ['    with np.errstate(all="ignore"):']
        body += ["        " + ln for ln in lines]
    else:
        body = ["    " + ln for ln in lines]
    src = "\n".join(header + body) + "\n"
    ns: Dict[str, Any] = dict(em.namespace)
    if backend == "numpy":
        import scipy.special as sps

        from spxa.compile import _numeric
        from spxa.compile.evaluate import call_exponent

        ns.update(
            np=np,
            sps=sps,
            _sinc=_numeric.sinc,
            _sinc_d1=_numeric.sinc_d1,
            _kve_dnu=_numeric.kve_dnu,
            _call=call_exponent,
        )
        fn = _exec(src, _register(name, src), ns, fname)
    else:
        jax, jnp, jsp = _import_jax()
        ns.update(jnp=jnp, jsp=jsp, _jsinc=_jax_sinc, _jsinc_d1=_jax_sinc_d1)
        raw = _exec(src, _register(name, src), ns, fname)
        fn = _jax_wrap(jax, raw)
    return Program(backend, name, src, fn, n_params, len(em.outputs), hoisted, ns)


# ── JAX helpers ─────────────────────────────────────────────────────────────


def _import_jax() -> Tuple[Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy.special as jsp
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise BackendUnavailable("the jax backend needs `pip install jax`") from exc
    return jax, jnp, jsp


def x64_context() -> Any:
    """A context manager enabling 64-bit JAX types (works across JAX versions)."""
    import jax

    ctx = getattr(jax, "enable_x64", None)
    if ctx is not None:
        return ctx(True)
    from jax.experimental import enable_x64  # pragma: no cover - older JAX

    return enable_x64()  # pragma: no cover


def _jax_sinc(z: Any) -> Any:
    import jax.numpy as jnp

    small = z == 0
    safe = jnp.where(small, 1.0, z)
    return jnp.where(small, 1.0, jnp.sin(safe) / safe)


def _jax_sinc_d1(z: Any) -> Any:
    import jax.numpy as jnp

    from spxa.compile._numeric import _SINC_D1_COEFFS, _SINC_SERIES_RADIUS

    small = jnp.abs(z) < _SINC_SERIES_RADIUS
    zs = jnp.where(small, z, 0.0)
    z2 = zs * zs
    acc = _SINC_D1_COEFFS[-1] + 0.0 * z2
    for c in _SINC_D1_COEFFS[-2::-1]:
        acc = c + z2 * acc
    zl = jnp.where(small, 1.0, z)
    direct = (jnp.cos(zl) - jnp.sin(zl) / zl) / zl
    return jnp.where(small, zs * acc, direct)


def _jax_wrap(jax: Any, raw: Callable[..., Any]) -> Callable[..., Any]:
    with x64_context():
        jitted = jax.jit(raw)

    def run(u: Any, *params: Any) -> Tuple[Any, ...]:
        with x64_context():
            out = jitted(u, *params)
            return tuple(np.asarray(o) for o in out)

    run.jitted = jitted  # type: ignore[attr-defined]
    run.raw = raw  # type: ignore[attr-defined]
    return run


# ── Numba backend ───────────────────────────────────────────────────────────

_NUMBA_OPS_C = {
    "exp": "_cexp",
    "log": "cmath.log",
    "sqrt": "cmath.sqrt",
    "sin": "cmath.sin",
    "cos": "cmath.cos",
    "tan": "cmath.tan",
    "expm1": "_cexpm1",
    "log1p": "_clog1p",
    "sinc": "_csinc",
    "sinc_d1": "_csinc_d1",
}
_NUMBA_OPS_R = {
    "exp": "math.exp",
    "log": "_rlog",
    "sqrt": "_rsqrt",
    "sin": "math.sin",
    "cos": "math.cos",
    "tan": "math.tan",
    "expm1": "math.expm1",
    "log1p": "_rlog1p",
    "sinc": "_rsinc",
    "sinc_d1": "_rsinc_d1",
}

_NUMBA_HELPERS = """
import cmath
import math


@nb.njit(cache=False, error_model="numpy")
def _cexpm1(z):
    x = z.real
    y = z.imag
    if y == 0.0:
        return complex(math.expm1(x), 0.0)
    s = math.sin(0.5 * y)
    return complex(math.expm1(x) * math.cos(y) - 2.0 * s * s, math.exp(x) * math.sin(y))


@nb.njit(cache=False, error_model="numpy")
def _cexp(z):
    if z.imag == 0.0:
        return complex(math.exp(z.real), z.imag)
    return cmath.exp(z)


@nb.njit(cache=False, error_model="numpy")
def _clog1p(z):
    if z.imag == 0.0:
        x = z.real
        if x >= -1.0:
            return complex(_rlog1p(x), z.imag)
        return complex(math.log(-1.0 - x), math.copysign(math.pi, z.imag))
    w = 1.0 + z
    if w == 1.0:
        return z
    return cmath.log(w) * (z / (w - 1.0))


@nb.njit(cache=False, error_model="numpy")
def _rlog(x):
    if x > 0.0:
        return math.log(x)
    if x == 0.0:
        return -math.inf
    return math.nan


@nb.njit(cache=False, error_model="numpy")
def _rsqrt(x):
    if x >= 0.0:
        return math.sqrt(x)
    return math.nan


@nb.njit(cache=False, error_model="numpy")
def _rlog1p(x):
    if x > -1.0:
        return math.log1p(x)
    if x == -1.0:
        return -math.inf
    return math.nan


@nb.njit(cache=False, error_model="numpy")
def _csinc(z):
    if z == 0:
        return complex(1.0, 0.0)
    return cmath.sin(z) / z


@nb.njit(cache=False, error_model="numpy")
def _rsinc(x):
    if x == 0.0:
        return 1.0
    return math.sin(x) / x


@nb.njit(cache=False, error_model="numpy")
def _csinc_d1(z):
    if abs(z) < @RADIUS@:
        z2 = z * z
        return z * @POLY@
    return (cmath.cos(z) - cmath.sin(z) / z) / z


@nb.njit(cache=False, error_model="numpy")
def _rsinc_d1(x):
    if abs(x) < @RADIUS@:
        z2 = x * x
        return x * @POLY@
    return (math.cos(x) - math.sin(x) / x) / x


@nb.njit(cache=False, error_model="numpy")
def _cdiv(a, b):
    # NumPy's convention for a zero divisor (numba raises ZeroDivisionError)
    if b.real == 0.0 and b.imag == 0.0:
        return complex(a.real / abs(b.real), a.imag / abs(b.real))
    return a / b


@nb.njit(cache=False, error_model="numpy")
def _sign(x):
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    if x == 0.0:
        return 0.0
    return math.nan
"""

_NUMBA_HELPER_NS: Optional[Dict[str, Any]] = None


def _horner(coeffs: Sequence[float], var: str) -> str:
    """Horner form ``(c0 + var * (c1 + var * (...)))`` with exact float literals."""
    out = repr(float(coeffs[-1]))
    for c in coeffs[-2::-1]:
        out = f"({float(c)!r} + {var} * {out})"
    return out


def _numba_helper_source() -> str:
    from spxa.compile._numeric import _SINC_D1_COEFFS, _SINC_SERIES_RADIUS

    return _NUMBA_HELPERS.replace("@RADIUS@", repr(float(_SINC_SERIES_RADIUS))).replace(
        "@POLY@", _horner(_SINC_D1_COEFFS, "z2")
    )


def _numba_helpers() -> Dict[str, Any]:
    global _NUMBA_HELPER_NS
    if _NUMBA_HELPER_NS is None:
        try:
            import numba as nb
        except ImportError as exc:
            raise BackendUnavailable("the numba backend needs `pip install numba`") from exc
        ns: Dict[str, Any] = {"nb": nb}
        src = _numba_helper_source()
        filename = _register("numba_helpers", src)
        exec(compile(src, filename, "exec"), ns)  # noqa: S102
        _NUMBA_HELPER_NS = ns
    return _NUMBA_HELPER_NS


def _numba_scalar(node: Expr, a: List[str]) -> str:
    op = node.op
    if op == "add":
        return f"({a[0]} + {a[1]})"
    if op == "mul":
        return f"({a[0]} * {a[1]})"
    if op == "div":
        if node.args[1].kind == I.COMPLEX:
            return f"_cdiv({a[0]}, {a[1]})"
        return f"({a[0]} / {a[1]})"
    if op == "pow":
        return f"({a[0]} ** {a[1]})"
    if op == "neg":
        return f"(-{a[0]})"
    if op == "ge":
        return f"({a[0]} >= {a[1]})"
    if op == "gt":
        return f"({a[0]} > {a[1]})"
    if op == "abs":
        return f"abs({a[0]})"
    if op == "sign":
        return f"_sign({a[0]})"
    if op == "real":
        return f"({a[0]}).real"
    if op == "imag":
        return f"({a[0]}).imag"
    if op == "where":
        if node.kind == I.COMPLEX:
            return f"(complex({a[1]}) if {a[0]} else complex({a[2]}))"
        return f"({a[1]} if {a[0]} else {a[2]})"
    table = _NUMBA_OPS_C if node.args and node.args[0].kind == I.COMPLEX else _NUMBA_OPS_R
    fn = table.get(op)
    if fn is None:
        raise BackendUnsupported(
            f"{op} of a u-dependent value is not supported by the numba backend "
            "(use backend='numpy')"
        )
    return f"{fn}({a[0]})"


def _numba_program(outputs: Sequence[Expr], n_params: int, name: str, parallel: bool) -> Program:
    helpers = _numba_helpers()
    nb = helpers["nb"]
    em = _Emitter(outputs, n_params)
    nodes = em.nodes
    # hoisted frontier: parameter-only, non-constant nodes used by u-dependent nodes or outputs
    frontier: List[Expr] = []
    seen: set = set()
    for n in nodes:
        if not n.depends_u:
            continue
        for x in n.args:
            if not x.depends_u and x.op != "const" and id(x) not in seen:
                seen.add(id(x))
                frontier.append(x)
    for o in outputs:
        if not o.depends_u and o.op != "const" and id(o) not in seen:
            seen.add(id(o))
            frontier.append(o)
    # prologue in NumPy: computes the frontier from the parameters
    pre = _array_program(frontier, n_params, "numpy", f"{name}_pre") if frontier else None
    # the u-dependent part as a scalar loop body
    fidx = {id(x): k for k, x in enumerate(frontier)}
    text: Dict[int, str] = {}
    body: List[str] = []
    tmp = itertools.count()
    for n in nodes:
        if id(n) in fidx:
            text[id(n)] = f"h{fidx[id(n)]}[b]"
            continue
        if n.op == "const":
            v = n.payload
            if isinstance(v, bool):
                text[id(n)] = repr(v)
            elif isinstance(v, complex):
                text[id(n)] = f"complex({v.real!r}, {v.imag!r})"
            else:
                text[id(n)] = repr(float(v)) if math.isfinite(v) else f"float('{v!r}')"
            continue
        if not n.depends_u:
            continue  # computed in the prologue (only frontier nodes are referenced)
        if n.op == "var":
            text[id(n)] = "ui"
            continue
        if n.op == "call":
            raise BackendUnsupported("opaque process exponents cannot run on numba")
        a = [text[id(x)] for x in n.args]
        expr = _numba_scalar(n, a)
        if em.uses.get(id(n), 0) > 1:
            nm = f"t{next(tmp)}"
            body.append(f"{nm} = {expr}")
            text[id(n)] = nm
        else:
            text[id(n)] = expr
    u_outputs = [k for k, o in enumerate(outputs) if o.depends_u]
    h_args = [f"h{k}" for k in range(len(frontier))]
    o_args = [f"o{k}" for k in u_outputs]
    rng = "nb.prange" if parallel else "range"
    loop = [
        f"@nb.njit(cache=False, error_model='numpy', parallel={parallel!r})",
        f"def _loop_{name}({', '.join(['uf', 'nbatch'] + h_args + o_args)}):",
        "    n = uf.shape[0]",
        f"    for k in {rng}(nbatch * n):",
        "        b = k // n",
        "        ui = uf[k - b * n]",
    ]
    loop += ["        " + ln for ln in body]
    for k in u_outputs:
        loop.append(f"        o{k}[k] = {text[id(outputs[k])]}")
    src = "\n".join(loop) + "\n"
    ns: Dict[str, Any] = dict(helpers)
    ns["nb"] = nb
    loop_fn = _exec(src, _register(name, src), ns, f"_loop_{name}")
    out_kinds = [o.kind for o in outputs]
    f_kinds = [x.kind for x in frontier]

    def run(u: Any, *params: Any) -> Tuple[Any, ...]:
        uu = np.asarray(u, dtype=complex)
        uf = np.ascontiguousarray(uu.ravel())
        pshape = np.broadcast_shapes(*(np.shape(p) for p in params)) if params else ()
        if len(pshape) > 1:
            raise CompileError("the numba backend takes scalar or 1-D (batched) parameters")
        bsz = pshape[0] if pshape else 1
        hs = pre(uu, *params) if pre is not None else ()
        harrs: List[np.ndarray] = []
        for h, kd in zip(hs, f_kinds):
            dt = {I.COMPLEX: complex, I.REAL: float, I.BOOL: bool}[kd]
            harrs.append(np.ascontiguousarray(np.broadcast_to(np.asarray(h, dtype=dt), (bsz,))))
        outs: List[Any] = [None] * len(outputs)
        o_bufs: List[np.ndarray] = []
        for k in u_outputs:
            buf = np.empty(bsz * uf.size, dtype=complex if out_kinds[k] == I.COMPLEX else float)
            o_bufs.append(buf)
        if u_outputs and uf.size:
            loop_fn(uf, bsz, *harrs, *o_bufs)
        oshape = (pshape[0],) + uu.shape if pshape else uu.shape
        for k, buf in zip(u_outputs, o_bufs):
            outs[k] = buf.reshape(oshape)
        for k, o in enumerate(outputs):
            if outs[k] is not None:
                continue
            if o.op == "const":
                outs[k] = np.broadcast_to(np.asarray(o.payload), oshape)
            else:
                v = np.asarray(hs[fidx[id(o)]])
                outs[k] = np.broadcast_to(v.reshape(v.shape + (1,) * uu.ndim), oshape)
        return tuple(outs)

    full_src = (pre.source if pre is not None else "") + "\n" + src
    return Program("numba", name, full_src, run, n_params, len(outputs), len(frontier), ns)


# ── entry point ─────────────────────────────────────────────────────────────


def generate(
    outputs: Sequence[Expr],
    n_params: int,
    backend: str = "numpy",
    name: str = "kernel",
    parallel: bool = False,
) -> Program:
    """
    Generate one function computing ``outputs`` from ``u`` and ``n_params`` parameters.

    Parameters
    ----------
    outputs :
        Expressions to compute (they share subexpressions).
    n_params :
        Number of positional parameters ``p0 .. p{n-1}``; every ``param``
        node's index must be below it.
    backend :
        ``"numpy"``, ``"numba"`` or ``"jax"``.
    name :
        Name used in the generated source and in tracebacks.
    parallel :
        Numba only: run the loop with ``prange``.
    """
    outputs = list(outputs)
    if not outputs:
        raise CompileError("nothing to generate")
    for n in I.postorder(outputs):
        if n.op == "param" and n.payload[0] >= n_params:
            raise CompileError(f"parameter index {n.payload[0]} >= n_params={n_params}")
    if not name.isidentifier():
        raise CompileError(f"kernel name {name!r} is not an identifier")
    if backend in ("numpy", "jax"):
        return _array_program(outputs, n_params, backend, name)
    if backend == "numba":
        return _numba_program(outputs, n_params, name, parallel)
    raise CompileError(f"unknown backend {backend!r}; choose from {BACKENDS}")
