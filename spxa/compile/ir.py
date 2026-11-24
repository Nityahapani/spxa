r"""
Expression IR for compiled characteristic exponents.

An :class:`Expr` is a node of a directed acyclic graph over one complex
variable :math:`u` (the frequency), real parameters :math:`\theta_k` and
constants. Nodes are *hash-consed*: building the same operation on the same
operands twice returns the same object, so common subexpressions are shared
automatically and identity is structural equality.

Every constructor simplifies as it builds:

* operations on constants are folded with the same NumPy routines the
  generated code uses, so folding never changes a value;
* the identities :math:`x + 0 = x`, :math:`x\cdot 1 = x`, :math:`x/1 = x`,
  :math:`x^1 = x`, :math:`-(-x) = x`, :math:`\Re x = x` for real :math:`x` are
  applied, all of which are exact in IEEE arithmetic;
* :math:`0\cdot x = 0`, :math:`0/x = 0` and :math:`x^0 = 1` are applied
  symbolically (they differ from IEEE only when :math:`x` is infinite or NaN);
* the operands of ``+`` and ``*`` are put in a canonical order, which is
  exact because IEEE addition and multiplication of two operands commute.

Associativity is never used, so a lowered formula is evaluated in the order
in which it was written. Kinds are tracked per node: ``"real"``, ``"complex"``
or ``"bool"``. A real node stays real in the generated code, which matters
for branch cuts: ``log`` of a negative real is NaN, as in NumPy, not
:math:`\log|x| + i\pi`.
"""

from __future__ import annotations

import weakref
import zlib
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Expr",
    "CompileError",
    "REAL",
    "COMPLEX",
    "BOOL",
    "const",
    "param",
    "var",
    "as_expr",
    "add",
    "sub",
    "mul",
    "neg",
    "div",
    "pow_",
    "exp",
    "expm1",
    "log",
    "log1p",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "sinc",
    "sinc_d1",
    "abs_",
    "sign",
    "real",
    "imag",
    "gamma",
    "digamma",
    "polygamma",
    "kve",
    "kve_dnu",
    "where",
    "ge",
    "gt",
    "le",
    "lt",
    "call",
    "postorder",
    "substitute",
    "rebuild",
    "to_str",
    "count_nodes",
    "U",
    "ZERO",
    "ONE",
]

REAL = "real"
COMPLEX = "complex"
BOOL = "bool"

Number = Union[int, float, complex, np.number]


class CompileError(ValueError):
    """A process or expression cannot be compiled as requested."""


#: Ops whose result kind follows their (single) argument.
_UNARY_SAME_KIND = ("exp", "expm1", "log", "log1p", "sqrt", "sin", "cos", "tan", "sinc", "sinc_d1")
#: Ops that require a real argument and give a real result.
_UNARY_REAL_ONLY = ("gamma", "digamma", "polygamma", "sign")
#: Commutative binary ops (operands are put in canonical order).
_COMMUTATIVE = ("add", "mul")


def _mix(*parts: int) -> int:
    """Deterministic 64-bit mixing of integers (FNV-1a over their bytes)."""
    h = 0xCBF29CE484222325
    for p in parts:
        for b in int(p & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"):
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _str_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf8"))


class Expr:
    """
    A node of the expression DAG. Build nodes with the module functions or
    with Python operators; never instantiate directly.

    Attributes
    ----------
    op :
        Operation name (``"add"``, ``"log"``, ``"param"``, ...).
    args :
        Operand nodes.
    payload :
        Extra data: the value of a constant, ``(index, name)`` of a
        parameter, the order of ``polygamma``, the process of a ``call``.
    kind :
        ``"real"``, ``"complex"`` or ``"bool"``.
    depends_u :
        Whether the node depends on the frequency :math:`u`.
    params :
        Indices of the parameters the node depends on.
    """

    __slots__ = ("op", "args", "payload", "kind", "depends_u", "params", "shash", "__weakref__")
    __array_ufunc__ = None  # NumPy scalars defer to Expr's reflected operators
    __array_priority__ = 1000.0

    op: str
    args: Tuple["Expr", ...]
    payload: Any
    kind: str
    depends_u: bool
    params: frozenset
    shash: int

    def __init__(self) -> None:  # pragma: no cover - guarded
        raise TypeError("build Expr nodes with the functions of spxa.compile.ir")

    # -- predicates ----------------------------------------------------------
    @property
    def is_const(self) -> bool:
        return self.op == "const"

    def const_value(self) -> Any:
        """The value of a constant node (raises for other nodes)."""
        if self.op != "const":
            raise CompileError(f"{self.op} node is not a constant")
        return self.payload

    def is_const_equal(self, v: Number) -> bool:
        return self.op == "const" and self.kind != BOOL and self.payload == v

    # -- operators -----------------------------------------------------------
    def __add__(self, o: Any) -> Expr:
        return add(self, as_expr(o))

    def __radd__(self, o: Any) -> Expr:
        return add(as_expr(o), self)

    def __sub__(self, o: Any) -> Expr:
        return sub(self, as_expr(o))

    def __rsub__(self, o: Any) -> Expr:
        return sub(as_expr(o), self)

    def __mul__(self, o: Any) -> Expr:
        return mul(self, as_expr(o))

    def __rmul__(self, o: Any) -> Expr:
        return mul(as_expr(o), self)

    def __truediv__(self, o: Any) -> Expr:
        return div(self, as_expr(o))

    def __rtruediv__(self, o: Any) -> Expr:
        return div(as_expr(o), self)

    def __pow__(self, o: Any) -> Expr:
        return pow_(self, as_expr(o))

    def __rpow__(self, o: Any) -> Expr:
        return pow_(as_expr(o), self)

    def __neg__(self) -> Expr:
        return neg(self)

    def __pos__(self) -> Expr:
        return self

    def __bool__(self) -> bool:
        raise CompileError(
            "an Expr has no truth value; branch with spxa.compile.ir.where instead of `if`"
        )

    def __float__(self) -> float:
        if self.op == "const" and self.kind == REAL:
            return float(self.payload)
        raise CompileError("only real constant expressions convert to float")

    def __repr__(self) -> str:
        return f"Expr({to_str(self)})"


_TABLE: "weakref.WeakValueDictionary[tuple, Expr]" = weakref.WeakValueDictionary()


def _payload_key(op: str, payload: Any) -> Any:
    if op == "const":
        if isinstance(payload, bool):
            return ("b", payload)
        if isinstance(payload, complex):
            return ("c", payload.real.hex(), payload.imag.hex())
        return ("r", float(payload).hex())
    if op == "call":
        return ("call", id(payload))
    return payload


def _payload_hash(op: str, payload: Any) -> int:
    if op == "call":
        return _str_hash(f"call:{type(payload).__name__}")
    return _str_hash(repr(_payload_key(op, payload)))


def _node(
    op: str,
    args: Tuple[Expr, ...],
    payload: Any,
    kind: str,
) -> Expr:
    key = (op, tuple(id(a) for a in args), _payload_key(op, payload))
    hit = _TABLE.get(key)
    if hit is not None:
        return hit
    e = object.__new__(Expr)
    e.op = op
    e.args = args
    e.payload = payload
    e.kind = kind
    if op == "var":
        e.depends_u = True
    elif op == "call":
        e.depends_u = True
    else:
        e.depends_u = any(a.depends_u for a in args)
    ps: frozenset = frozenset()
    for a in args:
        ps = ps | a.params
    if op == "param":
        ps = frozenset((payload[0],))
    e.params = ps
    e.shash = _mix(_str_hash(op), _payload_hash(op, payload), *(a.shash for a in args))
    _TABLE[key] = e
    return e


# ── leaves ──────────────────────────────────────────────────────────────────


def const(v: Any) -> Expr:
    """A constant. Python/NumPy complex values give complex nodes, bools give bool nodes."""
    if isinstance(v, Expr):
        return v
    if isinstance(v, (bool, np.bool_)):
        return _node("const", (), bool(v), BOOL)
    if isinstance(v, (complex, np.complexfloating)):
        return _node("const", (), complex(v), COMPLEX)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return _node("const", (), float(v), REAL)
    raise CompileError(f"cannot make a constant from {type(v).__name__}")


def param(index: int, name: str) -> Expr:
    """The real parameter :math:`\\theta_{index}`, labelled ``name``."""
    if int(index) < 0:
        raise CompileError("parameter indices are non-negative")
    return _node("param", (), (int(index), str(name)), REAL)


def var() -> Expr:
    """The complex frequency :math:`u`."""
    return _node("var", (), "u", COMPLEX)


def as_expr(x: Any) -> Expr:
    return x if isinstance(x, Expr) else const(x)


# ── folding ─────────────────────────────────────────────────────────────────


def _join(*kinds: str) -> str:
    if BOOL in kinds:
        raise CompileError("boolean operands are only allowed as the condition of where()")
    return COMPLEX if COMPLEX in kinds else REAL


def _fold(op: str, args: Tuple[Expr, ...], payload: Any, kind: str) -> Expr:
    from spxa.compile._numeric import apply_op

    vals = [a.payload for a in args]
    if kind == REAL:
        vals = [float(v) if not isinstance(v, bool) else v for v in vals]
    with np.errstate(all="ignore"):
        out = apply_op(op, vals, payload)
    out = np.asarray(out)[()]
    if kind == BOOL:
        return const(bool(out))
    if kind == COMPLEX:
        return const(complex(out))
    if np.iscomplexobj(out):  # pragma: no cover - kind inference guarantees real
        raise CompileError(f"folding {op} produced a complex value from real operands")
    return const(float(out))


def _all_const(args: Sequence[Expr]) -> bool:
    return all(a.op == "const" for a in args)


def _order(a: Expr, b: Expr) -> Tuple[Expr, Expr]:
    ka = (0 if a.op == "const" else 1, a.shash)
    kb = (0 if b.op == "const" else 1, b.shash)
    return (a, b) if ka <= kb else (b, a)


def _promote_ok(neutral: Expr, other: Expr) -> bool:
    """Dropping a neutral constant must not change the kind of the result."""
    return not (neutral.kind == COMPLEX and other.kind == REAL)


# ── arithmetic ──────────────────────────────────────────────────────────────


def add(a: Any, b: Any) -> Expr:
    a, b = as_expr(a), as_expr(b)
    kind = _join(a.kind, b.kind)
    if _all_const((a, b)):
        return _fold("add", (a, b), None, kind)
    if a.is_const_equal(0) and _promote_ok(a, b):
        return b
    if b.is_const_equal(0) and _promote_ok(b, a):
        return a
    a, b = _order(a, b)
    return _node("add", (a, b), None, kind)


def neg(a: Any) -> Expr:
    a = as_expr(a)
    kind = _join(a.kind)
    if a.op == "const":
        return _fold("neg", (a,), None, kind)
    if a.op == "neg":
        return a.args[0]
    return _node("neg", (a,), None, kind)


def sub(a: Any, b: Any) -> Expr:
    """:math:`a - b`, stored as :math:`a + (-b)` (identical in IEEE arithmetic)."""
    return add(a, neg(b))


def mul(a: Any, b: Any) -> Expr:
    a, b = as_expr(a), as_expr(b)
    kind = _join(a.kind, b.kind)
    if _all_const((a, b)):
        return _fold("mul", (a, b), None, kind)
    for x, y in ((a, b), (b, a)):
        if x.is_const_equal(0):
            return const(0j) if kind == COMPLEX else const(0.0)
        if x.is_const_equal(1) and _promote_ok(x, y):
            return y
        if x.is_const_equal(-1) and _promote_ok(x, y):
            return neg(y)
    a, b = _order(a, b)
    return _node("mul", (a, b), None, kind)


def div(a: Any, b: Any) -> Expr:
    a, b = as_expr(a), as_expr(b)
    kind = _join(a.kind, b.kind)
    if _all_const((a, b)):
        return _fold("div", (a, b), None, kind)
    if b.is_const_equal(1) and _promote_ok(b, a):
        return a
    if a.is_const_equal(0):
        return const(0j) if kind == COMPLEX else const(0.0)
    return _node("div", (a, b), None, kind)


def pow_(a: Any, b: Any) -> Expr:
    """:math:`a^b` with NumPy semantics (principal branch for complex bases)."""
    a, b = as_expr(a), as_expr(b)
    kind = _join(a.kind, b.kind)
    if _all_const((a, b)):
        return _fold("pow", (a, b), None, kind)
    if b.is_const_equal(1) and _promote_ok(b, a):
        return a
    if b.is_const_equal(0):
        return const(1 + 0j) if kind == COMPLEX else const(1.0)
    return _node("pow", (a, b), None, kind)


# ── elementary and special functions ────────────────────────────────────────


def _unary(op: str, a: Any, payload: Any = None) -> Expr:
    a = as_expr(a)
    if op in _UNARY_REAL_ONLY:
        if a.kind != REAL:
            raise CompileError(f"{op} needs a real argument")
        kind = REAL
    else:
        kind = _join(a.kind)
    if a.op == "const":
        return _fold(op, (a,), payload, kind)
    return _node(op, (a,), payload, kind)


def exp(a: Any) -> Expr:
    return _unary("exp", a)


def expm1(a: Any) -> Expr:
    return _unary("expm1", a)


def log(a: Any) -> Expr:
    return _unary("log", a)


def log1p(a: Any) -> Expr:
    return _unary("log1p", a)


def sqrt(a: Any) -> Expr:
    return _unary("sqrt", a)


def sin(a: Any) -> Expr:
    return _unary("sin", a)


def cos(a: Any) -> Expr:
    return _unary("cos", a)


def tan(a: Any) -> Expr:
    return _unary("tan", a)


def sinc(a: Any) -> Expr:
    r""":math:`\sin z / z`, equal to 1 at 0 (unnormalised, unlike ``np.sinc``)."""
    return _unary("sinc", a)


def sinc_d1(a: Any) -> Expr:
    r""":math:`\frac{d}{dz}\frac{\sin z}{z}`, evaluated without cancellation near 0."""
    return _unary("sinc_d1", a)


def gamma(a: Any) -> Expr:
    return _unary("gamma", a)


def digamma(a: Any) -> Expr:
    return _unary("digamma", a)


def polygamma(n: int, a: Any) -> Expr:
    if int(n) < 1:
        raise CompileError("polygamma order must be >= 1 (use digamma for 0)")
    return _unary("polygamma", a, int(n))


def sign(a: Any) -> Expr:
    return _unary("sign", a)


def abs_(a: Any) -> Expr:
    a = as_expr(a)
    if a.kind == BOOL:
        raise CompileError("abs of a boolean")
    if a.op == "const":
        return _fold("abs", (a,), None, REAL)
    return _node("abs", (a,), None, REAL)


def real(a: Any) -> Expr:
    a = as_expr(a)
    if a.kind == BOOL:
        raise CompileError("real part of a boolean")
    if a.kind == REAL:
        return a
    if a.op == "const":
        return _fold("real", (a,), None, REAL)
    return _node("real", (a,), None, REAL)


def imag(a: Any) -> Expr:
    a = as_expr(a)
    if a.kind == BOOL:
        raise CompileError("imaginary part of a boolean")
    if a.kind == REAL:
        return const(0.0)
    if a.op == "const":
        return _fold("imag", (a,), None, REAL)
    return _node("imag", (a,), None, REAL)


def kve(nu: Any, z: Any) -> Expr:
    r"""Exponentially scaled Bessel :math:`e^zK_\nu(z)` (real order :math:`\nu`)."""
    nu, z = as_expr(nu), as_expr(z)
    if nu.kind != REAL:
        raise CompileError("kve needs a real order")
    kind = _join(z.kind)
    if _all_const((nu, z)):
        return _fold("kve", (nu, z), None, kind)
    return _node("kve", (nu, z), None, kind)


def kve_dnu(nu: Any, z: Any) -> Expr:
    r""":math:`\partial_\nu\,e^zK_\nu(z)`."""
    nu, z = as_expr(nu), as_expr(z)
    if nu.kind != REAL:
        raise CompileError("kve_dnu needs a real order")
    kind = _join(z.kind)
    if _all_const((nu, z)):
        return _fold("kve_dnu", (nu, z), None, kind)
    return _node("kve_dnu", (nu, z), None, kind)


# ── comparisons and selection ───────────────────────────────────────────────


def _compare(op: str, a: Any, b: Any) -> Expr:
    a, b = as_expr(a), as_expr(b)
    if a.kind != REAL or b.kind != REAL:
        raise CompileError("comparisons need real operands (take real() or abs_() first)")
    if _all_const((a, b)):
        return _fold(op, (a, b), None, BOOL)
    return _node(op, (a, b), None, BOOL)


def ge(a: Any, b: Any) -> Expr:
    return _compare("ge", a, b)


def gt(a: Any, b: Any) -> Expr:
    return _compare("gt", a, b)


def le(a: Any, b: Any) -> Expr:
    return _compare("ge", b, a)


def lt(a: Any, b: Any) -> Expr:
    return _compare("gt", b, a)


def where(c: Any, a: Any, b: Any) -> Expr:
    """``a`` where ``c`` holds, else ``b`` (both branches are evaluated)."""
    c, a, b = as_expr(c), as_expr(a), as_expr(b)
    if c.kind != BOOL:
        raise CompileError("the condition of where() must be a comparison")
    kind = _join(a.kind, b.kind)
    if c.op == "const":
        chosen = a if c.payload else b
        if chosen.kind == kind:
            return chosen
        return add(chosen, const(0j))  # promote: keep the kind of the where
    if a is b:
        return a
    return _node("where", (c, a, b), None, kind)


def call(process: Any, arg: Any) -> Expr:
    r"""
    Opaque :math:`\psi_X(\text{arg})` for a process without a lowering rule.

    Evaluated by ``process._char_exponent``. It is not differentiable in the
    process's own parameters, and is differentiable in ``arg`` only through
    the cumulants of ``X`` (Taylor mode at :math:`\text{arg} = 0`).
    """
    arg = as_expr(arg)
    if arg.kind == BOOL:
        raise CompileError("call() needs a numeric argument")
    return _node("call", (arg,), process, COMPLEX)


U = var()
ZERO = const(0.0)
ONE = const(1.0)


# ── traversal and rewriting ─────────────────────────────────────────────────


def postorder(roots: Iterable[Expr]) -> List[Expr]:
    """All nodes reachable from ``roots``, each once, operands before users."""
    seen: set = set()
    out: List[Expr] = []
    for root in roots:
        if id(root) in seen:
            continue
        stack: List[Tuple[Expr, int]] = [(root, 0)]
        while stack:
            node, i = stack.pop()
            if i == 0 and id(node) in seen:
                continue
            if i < len(node.args):
                stack.append((node, i + 1))
                child = node.args[i]
                if id(child) not in seen:
                    stack.append((child, 0))
            else:
                if id(node) not in seen:
                    seen.add(id(node))
                    out.append(node)
    return out


_REBUILD: Dict[str, Callable[..., Expr]] = {
    "add": add,
    "mul": mul,
    "neg": neg,
    "div": div,
    "pow": pow_,
    "exp": exp,
    "expm1": expm1,
    "log": log,
    "log1p": log1p,
    "sqrt": sqrt,
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "sinc": sinc,
    "sinc_d1": sinc_d1,
    "gamma": gamma,
    "digamma": digamma,
    "sign": sign,
    "abs": abs_,
    "real": real,
    "imag": imag,
    "kve": kve,
    "kve_dnu": kve_dnu,
    "ge": ge,
    "gt": gt,
    "where": where,
}


def rebuild(node: Expr, args: Sequence[Expr]) -> Expr:
    """``node``'s operation applied to new operands (simplifying again)."""
    if not node.args:
        return node
    if node.op == "polygamma":
        return polygamma(node.payload, args[0])
    if node.op == "call":
        return call(node.payload, args[0])
    return _REBUILD[node.op](*args)


def substitute(roots: Union[Expr, Sequence[Expr]], mapping: Mapping[Expr, Any]) -> Any:
    """
    Replace nodes (typically parameters or :data:`U`) and re-simplify.

    ``mapping`` maps nodes to nodes or numbers. Returns one expression for one
    root, else a list.
    """
    single = isinstance(roots, Expr)
    rs: List[Expr] = [roots] if single else list(roots)  # type: ignore[list-item]
    table: Dict[int, Expr] = {id(k): as_expr(v) for k, v in mapping.items()}
    new: Dict[int, Expr] = {}
    for node in postorder(rs):
        if id(node) in table:
            new[id(node)] = table[id(node)]
            continue
        if not node.args:
            new[id(node)] = node
            continue
        args = [new[id(a)] for a in node.args]
        if all(x is y for x, y in zip(args, node.args)):
            new[id(node)] = node
        else:
            new[id(node)] = rebuild(node, args)
    out = [new[id(r)] for r in rs]
    return out[0] if single else out


def count_nodes(roots: Union[Expr, Sequence[Expr]]) -> int:
    """Number of distinct nodes (the size of the DAG after sharing)."""
    rs = [roots] if isinstance(roots, Expr) else list(roots)
    return len(postorder(rs))


_INFIX = {"add": "+", "mul": "*", "div": "/", "pow": "**", "ge": ">=", "gt": ">"}


def to_str(e: Expr, max_len: Optional[int] = 2000) -> str:
    """Infix rendering, for messages and debugging (shared nodes are repeated)."""
    memo: Dict[int, str] = {}
    for node in postorder([e]):
        a = [memo[id(x)] for x in node.args]
        op = node.op
        if op == "const":
            v = node.payload
            s = repr(v) if not isinstance(v, complex) else f"({v.real!r}{v.imag:+}j)"
        elif op == "param":
            s = node.payload[1]
        elif op == "var":
            s = "u"
        elif op in _INFIX:
            if op == "add" and node.args[1].op == "neg":
                s = f"({a[0]} - {memo[id(node.args[1].args[0])]})"
            elif op == "add" and node.args[0].op == "neg":
                s = f"({a[1]} - {memo[id(node.args[0].args[0])]})"
            else:
                s = f"({a[0]} {_INFIX[op]} {a[1]})"
        elif op == "neg":
            s = f"(-{a[0]})"
        elif op == "where":
            s = f"where({a[0]}, {a[1]}, {a[2]})"
        elif op == "polygamma":
            s = f"polygamma({node.payload}, {a[0]})"
        elif op == "call":
            s = f"psi[{type(node.payload).__name__}]({a[0]})"
        else:
            s = f"{op}({', '.join(a)})"
        memo[id(node)] = s
    out = memo[id(e)]
    if max_len is not None and len(out) > max_len:
        return out[: max_len - 3] + "..."
    return out
