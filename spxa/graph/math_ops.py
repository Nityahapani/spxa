"""
Mathematical operation nodes for graph evaluation, transformations, and gradients.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from scipy.special import erf, gamma
from scipy.stats import norm

from spxa.graph.node import Node, OperatorNode


# ── exact VJP helpers ─────────────────────────────────────────────────────────


def _pow_vjp(grad_out, a, b):  # type: ignore[no-untyped-def]
    r"""VJP of :math:`a^b`: :math:`b a^{b-1}` and :math:`a^b\ln a`.

    The exponent's cotangent is :math:`a^b\ln a` for :math:`a > 0`, its limit
    :math:`0` at :math:`a = 0` (where :math:`a^b = 0` for :math:`b > 0`), and
    NaN for :math:`a < 0` where :math:`a^b` is not differentiable in :math:`b`
    (it only matters if the exponent depends on a source).
    """
    a_arr = np.asarray(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        da = grad_out * b * np.power(a, b - 1)
        if np.iscomplexobj(a_arr):
            db = grad_out * np.power(a, b) * np.log(a)
        else:
            pos = a_arr > 0
            log_a = np.where(
                pos, np.log(np.where(pos, a_arr, 1.0)), np.where(a_arr == 0, 0.0, np.nan)
            )
            db = grad_out * np.power(a, b) * log_a
    return da, db


def _sqrt_vjp(grad_out, a):  # type: ignore[no-untyped-def]
    """VJP of ``sqrt(max(a, 0))``: 1/(2 sqrt a) for a > 0, 0 for a < 0, +inf at a = 0."""
    a_arr = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore"):
        d = np.where(
            a_arr > 0, 0.5 / np.sqrt(np.maximum(a_arr, 0.0)), np.where(a_arr < 0, 0.0, np.inf)
        )
    return (grad_out * d,)


def _expand_reduced(grad_out, a, axis):  # type: ignore[no-untyped-def]
    """Broadcast the cotangent of a reduction over ``axis`` back to ``a``'s shape."""
    shape = np.shape(a)
    g = np.asarray(grad_out)
    if axis is not None:
        axes = (axis,) if np.ndim(axis) == 0 else tuple(axis)
        axes = tuple(ax % len(shape) for ax in axes)
        for ax in sorted(axes):
            g = np.expand_dims(g, ax)
    return np.array(np.broadcast_to(g, shape))


def _reduced_count(a, axis):  # type: ignore[no-untyped-def]
    shape = np.shape(a)
    if axis is None:
        return float(np.prod(shape)) if shape else 1.0
    axes = (axis,) if np.ndim(axis) == 0 else tuple(axis)
    return float(np.prod([shape[ax] for ax in axes]))


def _matmul_vjp(grad_out, a, b):  # type: ignore[no-untyped-def]
    """VJP of ``matmul`` including the 1-d (vector) cases, unbroadcast by the tape."""
    a, b, g = np.asarray(a), np.asarray(b), np.asarray(grad_out)
    if a.ndim == 1 and b.ndim == 1:  # inner product
        return g * b, g * a
    a2 = a[None, :] if a.ndim == 1 else a
    b2 = b[:, None] if b.ndim == 1 else b
    g2 = g
    if a.ndim == 1:
        g2 = np.expand_dims(g2, -2)
    if b.ndim == 1:
        g2 = np.expand_dims(g2, -1)
    ga = np.matmul(g2, np.swapaxes(b2, -1, -2))
    gb = np.matmul(np.swapaxes(a2, -1, -2), g2)
    if a.ndim == 1:
        ga = ga.reshape(ga.shape[:-2] + (ga.shape[-1],)) if ga.ndim > 2 else ga[0]
    if b.ndim == 1:
        gb = gb[..., 0]
    return ga, gb


def _cholesky_vjp(grad_out, a):  # type: ignore[no-untyped-def]
    r"""
    Exact VJP of ``np.linalg.cholesky`` (which reads the lower triangle only).

    With :math:`A = LL^\top` and :math:`\Phi` taking the lower triangle with a
    halved diagonal, :math:`P = \Phi(L^\top\bar L)` and
    :math:`S = L^{-\top} P L^{-1}`; the cotangent of the lower triangle of
    :math:`A` is :math:`\operatorname{tril}(S + S^\top) - \operatorname{diag}(S)`
    and the strict upper triangle receives zero (Murray, 2016).
    """
    from scipy.linalg import solve_triangular

    a = np.asarray(a)
    Lb = np.asarray(grad_out)
    if a.ndim > 2:  # batched: one VJP per matrix
        Lb = np.broadcast_to(Lb, a.shape)
        out = np.empty(a.shape, dtype=np.result_type(a, Lb, float))
        for idx in np.ndindex(*a.shape[:-2]):
            out[idx] = _cholesky_vjp(Lb[idx], a[idx])[0]
        return (out,)
    L = np.linalg.cholesky(a)
    P = np.tril(L.T @ Lb)
    P = P - 0.5 * np.diag(np.diag(P))
    X = solve_triangular(L, P.T, lower=True, trans="T")  # X = L^{-T} P^T
    S = solve_triangular(L, X.T, lower=True, trans="T")  # S = L^{-T} P L^{-1}
    return (np.tril(S + S.T) - np.diag(np.diag(S)),)


class AddNode(OperatorNode):
    r"""Elementwise addition node :math:`z = x + y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a + b,
            inputs=[x, y],
            name=name or f"Add_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out, grad_out),
        )


class SubNode(OperatorNode):
    r"""Elementwise subtraction node :math:`z = x - y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a - b,
            inputs=[x, y],
            name=name or f"Sub_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out, -grad_out),
        )


class MulNode(OperatorNode):
    r"""Elementwise multiplication node :math:`z = x \cdot y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a * b,
            inputs=[x, y],
            name=name or f"Mul_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out * b, grad_out * a),
        )


class DivNode(OperatorNode):
    r"""Elementwise division node :math:`z = x / y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a / b,
            inputs=[x, y],
            name=name or f"Div_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out / b, -grad_out * a / (b**2)),
        )


class PowNode(OperatorNode):
    r"""Power node :math:`z = x^y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a**b,
            inputs=[x, y],
            name=name or f"Pow_{x.name}_{y.name}",
            vjp_fn=_pow_vjp,
        )


class NegNode(OperatorNode):
    r"""Unary negation node :math:`z = -x`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: -a,
            inputs=[x],
            name=name or f"Neg_{x.name}",
            vjp_fn=lambda grad_out, a: (-grad_out,),
        )


class ExpNode(OperatorNode):
    r"""Exponential node :math:`z = e^x`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.exp(a),
            inputs=[x],
            name=name or f"Exp_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * np.exp(a),),
        )


class LogNode(OperatorNode):
    r"""Natural logarithm node :math:`z = \ln(x)` with numerical floor."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.log(np.maximum(a, 1e-15)),
            inputs=[x],
            name=name or f"Log_{x.name}",
            # derivative of the floored log actually computed: 1/a above the floor, 0 below
            vjp_fn=lambda grad_out, a: (
                grad_out * np.where(np.asarray(a) > 1e-15, 1.0 / np.maximum(a, 1e-15), 0.0),
            ),
        )


class SqrtNode(OperatorNode):
    r"""Square root node :math:`z = \sqrt{x}`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.sqrt(np.maximum(a, 0.0)),
            inputs=[x],
            name=name or f"Sqrt_{x.name}",
            vjp_fn=_sqrt_vjp,
        )


class AbsNode(OperatorNode):
    r"""Absolute value node :math:`z = |x|`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.abs(a),
            inputs=[x],
            name=name or f"Abs_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * np.sign(a),),
        )


class SinNode(OperatorNode):
    r"""Sine node :math:`z = \sin(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.sin(a),
            inputs=[x],
            name=name or f"Sin_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * np.cos(a),),
        )


class CosNode(OperatorNode):
    r"""Cosine node :math:`z = \cos(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.cos(a),
            inputs=[x],
            name=name or f"Cos_{x.name}",
            vjp_fn=lambda grad_out, a: (-grad_out * np.sin(a),),
        )


class SinhNode(OperatorNode):
    r"""Hyperbolic sine node :math:`z = \sinh(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.sinh(a),
            inputs=[x],
            name=name or f"Sinh_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * np.cosh(a),),
        )


class CoshNode(OperatorNode):
    r"""Hyperbolic cosine node :math:`z = \cosh(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.cosh(a),
            inputs=[x],
            name=name or f"Cosh_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * np.sinh(a),),
        )


class TanhNode(OperatorNode):
    r"""Hyperbolic tangent node :math:`z = \tanh(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.tanh(a),
            inputs=[x],
            name=name or f"Tanh_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * (1.0 - np.tanh(a) ** 2),),
        )


class NormalCDFNode(OperatorNode):
    r"""Standard normal cumulative distribution function :math:`z = \Phi(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: norm.cdf(a),
            inputs=[x],
            name=name or f"NormCDF_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * norm.pdf(a),),
        )


class NormalPDFNode(OperatorNode):
    r"""Standard normal probability density function :math:`z = \phi(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: norm.pdf(a),
            inputs=[x],
            name=name or f"NormPDF_{x.name}",
            vjp_fn=lambda grad_out, a: (-grad_out * a * norm.pdf(a),),
        )


class ErfNode(OperatorNode):
    r"""Gauss error function node :math:`z = \text{erf}(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: erf(a),
            inputs=[x],
            name=name or f"Erf_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * (2.0 / np.sqrt(np.pi)) * np.exp(-(a**2)),),
        )


class GammaNode(OperatorNode):
    r"""Euler Gamma function node :math:`z = \Gamma(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        from scipy.special import polygamma

        super().__init__(
            op_fn=lambda a: gamma(a),
            inputs=[x],
            name=name or f"Gamma_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * gamma(a) * polygamma(0, a),),
        )


class SumNode(OperatorNode):
    r"""Array sum reduction node :math:`z = \sum x_i`."""

    def __init__(self, x: Node, axis: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.sum(a, axis=axis),
            inputs=[x],
            name=name or f"Sum_{x.name}",
            vjp_fn=lambda grad_out, a: (_expand_reduced(grad_out, a, axis),),
        )


class MeanNode(OperatorNode):
    r"""Array mean reduction node :math:`z = \frac{1}{N} \sum x_i`."""

    def __init__(self, x: Node, axis: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.mean(a, axis=axis),
            inputs=[x],
            name=name or f"Mean_{x.name}",
            vjp_fn=lambda grad_out, a: (
                _expand_reduced(grad_out, a, axis) / _reduced_count(a, axis),
            ),
        )


def _extreme_vjp(grad_out: Any, a: Any, b: Any, largest: bool) -> tuple[Any, Any]:
    """VJP of max/min: the larger (smaller) input takes the cotangent, ties split it.

    Half to each at a tie is the midpoint of the Clarke subdifferential, as in JAX
    and PyTorch, and matches the forward-mode jets.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    wins = (a > b) if largest else (a < b)
    wa = np.where(a == b, 0.5, wins.astype(float))
    return grad_out * wa, grad_out * (1.0 - wa)


class MaximumNode(OperatorNode):
    r"""Elementwise maximum node :math:`z = \max(x, y)`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: np.maximum(a, b),
            inputs=[x, y],
            name=name or f"Max_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: _extreme_vjp(grad_out, a, b, largest=True),
        )


class MinimumNode(OperatorNode):
    r"""Elementwise minimum node :math:`z = \min(x, y)`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: np.minimum(a, b),
            inputs=[x, y],
            name=name or f"Min_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: _extreme_vjp(grad_out, a, b, largest=False),
        )


class MatMulNode(OperatorNode):
    r"""Matrix multiplication node :math:`Z = X Y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: np.matmul(a, b),
            inputs=[x, y],
            name=name or f"MatMul_{x.name}_{y.name}",
            vjp_fn=_matmul_vjp,
        )


class TransposeNode(OperatorNode):
    r"""Matrix transpose node :math:`Z = X^\top`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.swapaxes(a, -1, -2) if np.ndim(a) >= 2 else a,
            inputs=[x],
            name=name or f"Transpose_{x.name}",
            vjp_fn=lambda grad_out, a: (
                np.swapaxes(grad_out, -1, -2) if np.ndim(grad_out) >= 2 else grad_out,
            ),
        )


class CholeskyNode(OperatorNode):
    r"""Cholesky decomposition node :math:`L = \text{chol}(\Sigma)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.linalg.cholesky(a),
            inputs=[x],
            name=name or f"Cholesky_{x.name}",
            vjp_fn=_cholesky_vjp,
        )
