"""
Calibration loss nodes, objective functions, and arbitrage penalty functions.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from spxa.graph.autodiff.jet import is_jet
from spxa.graph.node import Node, OperatorNode, PortDescriptor
from spxa.graph.types import DType


def _array(v: Any) -> Any:
    """``v`` as a float array, leaving forward-mode jets intact."""
    return v if is_jet(v) else np.asarray(v, dtype=float)


def _scalar_out(v: Any) -> Any:
    """A plain ``float`` for numeric results; jets pass through with their derivatives."""
    return v if is_jet(v) else float(np.asarray(v))


def _second_difference(n: int) -> np.ndarray:
    r"""The ``(n-2, n)`` matrix :math:`D` with :math:`(Dp)_i = p_i - 2p_{i+1} + p_{i+2}`."""
    d = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        d[i, i : i + 3] = (1.0, -2.0, 1.0)
    return d


class MSELossNode(OperatorNode):
    r"""
    Mean squared error :math:`L = \frac{1}{N} \sum_i (y_i - q_i)^2`, or the weighted mean
    :math:`L = \sum_i w_i (y_i - q_i)^2 / \sum_i w_i` when ``weights`` is given.

    The reverse-mode rule is exact, including the derivative in the weights,
    :math:`\partial L/\partial w_i = ((y_i - q_i)^2 - L)/\sum_j w_j`. The forward
    uses NumPy operations only, so it also runs on forward-mode jets and
    Hessians through the loss are exact.
    """

    def __init__(
        self,
        model_output: Node,
        market_quotes: Node,
        weights: Optional[Node] = None,
        name: str = "MSE_Loss",
    ) -> None:
        inputs = (
            [model_output, market_quotes]
            if weights is None
            else [model_output, market_quotes, weights]
        )

        def mse_fn(*vals: Any) -> Any:
            diff = _array(vals[0]) - _array(vals[1])
            if len(vals) > 2:
                w = _array(vals[2])
                return _scalar_out(np.sum(w * diff * diff) / np.sum(w))
            return _scalar_out(np.mean(diff * diff))

        def mse_vjp(grad_out: Any, *vals: Any) -> Any:
            y_mod = np.asarray(vals[0], dtype=float)
            y_mkt = np.asarray(vals[1], dtype=float)
            diff = y_mod - y_mkt
            if len(vals) > 2:
                w = np.asarray(vals[2], dtype=float)
                w_sum = np.sum(w)
                loss = np.sum(w * diff**2) / w_sum
                return (
                    grad_out * 2.0 * w * diff / w_sum,
                    -grad_out * 2.0 * w * diff / w_sum,
                    grad_out * (diff**2 - loss) / w_sum,
                )
            n = np.size(diff)
            return (grad_out * 2.0 * diff / n, -grad_out * 2.0 * diff / n)

        super().__init__(
            op_fn=mse_fn,
            inputs=inputs,
            name=name,
            output_descriptor=PortDescriptor(name="loss", dtype=DType.FLOAT64),
            vjp_fn=mse_vjp,
        )


class ArbitragePenaltyNode(OperatorNode):
    r"""
    Butterfly (convexity) penalty :math:`P = \lambda \sum_i \max(0, -(Dp)_i)^2` on prices
    :math:`p` ordered by strike, where :math:`(Dp)_i = p_i - 2p_{i+1} + p_{i+2}`.

    With :math:`v = \max(0, -Dp)` the gradient is :math:`\nabla_p P = -2\lambda D^\top v`.
    :math:`P` is continuously differentiable, since :math:`\max(0, x)^2` has derivative
    :math:`2\max(0, x)`, so this rule is exact everywhere, at the convexity
    boundary included. The forward uses NumPy operations only and runs on jets.
    Fewer than three prices carry no butterfly and give :math:`P = 0`.
    """

    def __init__(
        self, prices_node: Node, penalty_weight: float = 100.0, name: str = "ArbPenalty"
    ) -> None:
        self.penalty_weight = float(penalty_weight)
        lam = self.penalty_weight

        def penalty_fn(p_vals: Any) -> Any:
            p = _array(p_vals)
            if len(np.shape(p)) != 1:
                raise ValueError(f"{name}: prices must be one-dimensional, got shape {np.shape(p)}")
            if np.shape(p)[0] < 3:
                return 0.0
            butterfly = p[:-2] - 2.0 * p[1:-1] + p[2:]
            violations = np.maximum(-butterfly, 0.0)
            return _scalar_out(lam * np.sum(violations * violations))

        def penalty_vjp(grad_out: Any, p_vals: Any) -> Any:
            p = np.asarray(p_vals, dtype=float)
            if p.shape[0] < 3:
                return (np.zeros_like(p),)
            d = _second_difference(p.shape[0])
            v = np.maximum(-(d @ p), 0.0)
            return (-2.0 * lam * grad_out * (d.T @ v),)

        super().__init__(
            op_fn=penalty_fn,
            inputs=[prices_node],
            name=name,
            output_descriptor=PortDescriptor(name="penalty", dtype=DType.FLOAT64),
            vjp_fn=penalty_vjp,
        )
