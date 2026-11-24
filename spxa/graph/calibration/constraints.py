"""
Parameter constraints and boundary condition penalties for stochastic model calibration.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spxa.graph.autodiff.jet import is_jet
from spxa.graph.node import Node, OperatorNode, PortDescriptor
from spxa.graph.types import DType


class FellerConstraintNode(OperatorNode):
    r"""
    Feller condition penalty :math:`P = \lambda \max(0, \sigma^2 - 2\kappa\theta)^2`.

    With :math:`u = \max(0, \sigma^2 - 2\kappa\theta)`,
    :math:`\nabla P = 2\lambda u\,(-2\theta, -2\kappa, 2\sigma)`. :math:`P` is continuously
    differentiable, so the rule is exact on the boundary too. The forward uses
    NumPy operations only and runs on jets, so Hessians are exact.
    """

    def __init__(
        self,
        kappa: Node,
        theta: Node,
        sigma: Node,
        penalty_weight: float = 1000.0,
        name: str = "FellerPenalty",
    ) -> None:
        self.penalty_weight = penalty_weight

        def feller_fn(k_val: Any, th_val: Any, sig_val: Any) -> Any:
            excess = np.maximum(sig_val * sig_val - 2.0 * k_val * th_val, 0.0)
            out = self.penalty_weight * excess * excess
            return out if is_jet(out) else float(out)

        def feller_vjp(grad_out: Any, k_val: Any, th_val: Any, sig_val: Any) -> Any:
            excess = max(float(sig_val) ** 2 - 2.0 * float(k_val) * float(th_val), 0.0)
            d_excess = 2.0 * self.penalty_weight * excess * grad_out
            return (
                -d_excess * 2.0 * float(th_val),
                -d_excess * 2.0 * float(k_val),
                d_excess * 2.0 * float(sig_val),
            )

        super().__init__(
            op_fn=feller_fn,
            inputs=[kappa, theta, sigma],
            name=name,
            output_descriptor=PortDescriptor(name="feller_penalty", dtype=DType.FLOAT64),
            vjp_fn=feller_vjp,
        )
