"""
Calibration loss nodes, objective functions, and arbitrage penalty functions.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from spxa.graph.node import Node, OperatorNode, PortDescriptor
from spxa.graph.types import DType


class MSELossNode(OperatorNode):
    r"""Mean Squared Error (MSE) loss node :math:`L = \frac{1}{N} \sum (y_{\text{model}} - y_{\text{market}})^2`."""

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

        def mse_fn(*vals: Any) -> float:
            y_mod = np.asarray(vals[0], dtype=float)
            y_mkt = np.asarray(vals[1], dtype=float)
            diff = y_mod - y_mkt
            if len(vals) > 2:
                w = np.asarray(vals[2], dtype=float)
                return float(np.sum(w * (diff**2)) / np.sum(w))
            return float(np.mean(diff**2))

        def mse_vjp(grad_out: float, *vals: Any) -> Any:
            y_mod = np.asarray(vals[0], dtype=float)
            y_mkt = np.asarray(vals[1], dtype=float)
            n = float(np.size(y_mod))
            diff = y_mod - y_mkt
            if len(vals) > 2:
                w = np.asarray(vals[2], dtype=float)
                w_sum = float(np.sum(w))
                loss = float(np.sum(w * diff**2)) / w_sum
                return (
                    grad_out * 2.0 * w * diff / w_sum,
                    -grad_out * 2.0 * w * diff / w_sum,
                    grad_out * (diff**2 - loss) / w_sum,
                )
            return (grad_out * 2.0 * diff / n, -grad_out * 2.0 * diff / n)

        super().__init__(
            op_fn=mse_fn,
            inputs=inputs,
            name=name,
            output_descriptor=PortDescriptor(name="loss", dtype=DType.FLOAT64),
            vjp_fn=mse_vjp,
        )


class ArbitragePenaltyNode(OperatorNode):
    r"""Calendar/butterfly spread arbitrage penalty node :math:`P = \lambda \sum \max(0, - \Delta C)^2`."""

    def __init__(
        self, prices_node: Node, penalty_weight: float = 100.0, name: str = "ArbPenalty"
    ) -> None:
        def penalty_fn(p_vals: Any) -> float:
            p = np.asarray(p_vals, dtype=float)
            # Convexity / butterfly penalty: C(K_i-1) - 2 C(K_i) + C(K_i+1) >= 0
            if len(p) >= 3:
                butterfly = p[:-2] - 2.0 * p[1:-1] + p[2:]
                violations = np.maximum(0.0, -butterfly)
                return float(penalty_weight * np.sum(violations**2))
            return 0.0

        super().__init__(
            op_fn=penalty_fn,
            inputs=[prices_node],
            name=name,
            output_descriptor=PortDescriptor(name="penalty", dtype=DType.FLOAT64),
        )
