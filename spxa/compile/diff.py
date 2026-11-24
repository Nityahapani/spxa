r"""
Symbolic differentiation of the IR.

:func:`derivative` returns new expressions for :math:`\partial e/\partial w`,
where :math:`w` is a parameter node or the frequency :data:`~spxa.compile.ir.U`.
The result is built from the same hash-consed nodes as ``e``, so derivative
expressions share every subexpression with the value (for example the
``sqrt`` of an inverse Gaussian exponent is computed once for
:math:`\psi` and :math:`\partial\psi/\partial\theta`), and a parameter that
only enters one leaf of a composition contributes zeros everywhere else,
which simplify away. This is what makes compiled gradients cheaper than
dense forward-mode jets for compositions with many parameters.

Derivatives with respect to a real parameter are taken for any operation.
Derivatives with respect to :math:`u` are taken only for operations that are
analytic in :math:`u` (``real``, ``imag``, ``abs`` and ``sign`` are not).
Opaque ``call`` nodes have no derivative except where their argument is
constant in :math:`w`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

from spxa.compile import ir as I
from spxa.compile.ir import CompileError, Expr

__all__ = ["derivative", "gradient"]


def _depends(node: Expr, w: Expr) -> bool:
    if w.op == "var":
        return node.depends_u
    return w.payload[0] in node.params


def derivative(roots: Union[Expr, Sequence[Expr]], w: Expr) -> Union[Expr, List[Expr]]:
    r"""
    :math:`\partial r/\partial w` for each root ``r``.

    Parameters
    ----------
    roots :
        One expression or a sequence of expressions.
    w :
        A ``param`` node or :data:`~spxa.compile.ir.U`.
    """
    if w.op not in ("param", "var"):
        raise CompileError("differentiate with respect to a parameter or U")
    single = isinstance(roots, Expr)
    rs: List[Expr] = [roots] if single else list(roots)  # type: ignore[list-item]
    wrt_u = w.op == "var"
    d: Dict[int, Expr] = {}
    zero = I.const(0.0)
    for node in I.postorder(rs):
        if not _depends(node, w) or node.kind == I.BOOL:
            d[id(node)] = zero
            continue
        d[id(node)] = _rule(node, [d[id(a)] for a in node.args], wrt_u)
    out = [d[id(r)] for r in rs]
    return out[0] if single else out


def gradient(expr: Expr, ws: Sequence[Expr]) -> List[Expr]:
    """``[derivative(expr, w) for w in ws]``."""
    return [derivative(expr, w) for w in ws]  # type: ignore[misc]


def _zero(e: Expr) -> bool:
    return e.is_const_equal(0)


def _rule(node: Expr, da: List[Expr], wrt_u: bool) -> Expr:
    op = node.op
    a = node.args
    if op in ("param", "var"):
        return I.const(1.0)
    if op == "add":
        return I.add(da[0], da[1])
    if op == "neg":
        return I.neg(da[0])
    if op == "mul":
        return I.add(I.mul(da[0], a[1]), I.mul(a[0], da[1]))
    if op == "div":
        # (a/b)' = (a' - (a/b) b') / b, reusing the quotient node
        if _zero(da[1]):
            return I.div(da[0], a[1])
        return I.div(I.sub(da[0], I.mul(node, da[1])), a[1])
    if op == "pow":
        x, y = a
        dx, dy = da
        if y.op == "const":
            yv = y.payload
            if yv == 2:
                return I.mul(I.mul(I.const(2.0), x), dx)
            return I.mul(I.mul(y, I.pow_(x, I.const(yv - 1))), dx)
        terms = I.mul(dy, I.log(x))
        if not _zero(dx):
            terms = I.add(terms, I.div(I.mul(y, dx), x))
        return I.mul(node, terms)
    if op == "exp":
        return I.mul(node, da[0])
    if op == "expm1":
        return I.mul(I.exp(a[0]), da[0])
    if op == "log":
        return I.div(da[0], a[0])
    if op == "log1p":
        return I.div(da[0], I.add(I.const(1.0), a[0]))
    if op == "sqrt":
        return I.div(da[0], I.mul(I.const(2.0), node))
    if op == "sin":
        return I.mul(I.cos(a[0]), da[0])
    if op == "cos":
        return I.neg(I.mul(I.sin(a[0]), da[0]))
    if op == "tan":
        return I.mul(I.add(I.const(1.0), I.mul(node, node)), da[0])
    if op == "sinc":
        return I.mul(I.sinc_d1(a[0]), da[0])
    if op == "gamma":
        return I.mul(I.mul(node, I.digamma(a[0])), da[0])
    if op == "digamma":
        return I.mul(I.polygamma(1, a[0]), da[0])
    if op == "polygamma":
        return I.mul(I.polygamma(node.payload + 1, a[0]), da[0])
    if op == "where":
        return I.where(a[0], da[1], da[2])
    if op == "kve":
        nu, z = a
        dnu, dz = da
        out = I.const(0.0)
        if not _zero(dz):
            dk = I.sub(
                node,
                I.mul(I.const(0.5), I.add(I.kve(I.sub(nu, 1.0), z), I.kve(I.add(nu, 1.0), z))),
            )
            out = I.add(out, I.mul(dk, dz))
        if not _zero(dnu):
            out = I.add(out, I.mul(I.kve_dnu(nu, z), dnu))
        return out
    if op in ("real", "imag", "abs", "sign"):
        if wrt_u:
            raise CompileError(f"{op} is not analytic in u; no complex derivative exists")
        if op == "real":
            return I.real(da[0])
        if op == "imag":
            return I.imag(da[0])
        if op == "sign":
            return I.const(0.0)
        x = a[0]
        if x.kind == I.REAL:
            return I.mul(I.sign(x), da[0])
        # d|x| = Re(conj(x) dx) / |x| for a real parameter
        num = I.add(I.mul(I.real(x), I.real(da[0])), I.mul(I.imag(x), I.imag(da[0])))
        return I.div(num, node)
    if op == "call":
        raise CompileError(
            f"the exponent of {type(node.payload).__name__} is opaque (no lowering rule), "
            "so it cannot be differentiated; register a rule or hold its inputs fixed"
        )
    raise CompileError(f"no derivative rule for {op!r}")
