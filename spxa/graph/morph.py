"""
Computational graph nodes for differentiable model morphing.

Provides graph-native operations and reverse-mode VJP gradient rules:

    z = Morph(x, y; α) = (1 - α)x + α y

with vector-Jacobian products:
    ∂L/∂x = (1 - α) · ∂L/∂z
    ∂L/∂y = α · ∂L/∂z
    ∂L/∂α = (y - x) · ∂L/∂z
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from spxa.graph.node import Node, OperatorNode


class MorphNode(OperatorNode):
    r"""
    Computational graph node for model morphing homotopy :math:`z = (1 - \alpha) x + \alpha y`.

    Parameters
    ----------
    x :
        Base node (:math:`\alpha = 0`).
    y :
        Target node (:math:`\alpha = 1`).
    alpha :
        Homotopy weight node :math:`\alpha \in [0, 1]`.
    """

    def __init__(
        self,
        x: Node,
        y: Node,
        alpha: Node,
        name: Optional[str] = None,
    ) -> None:
        def op(val_x: Any, val_y: Any, val_a: Any) -> Any:
            return (1.0 - val_a) * val_x + val_a * val_y

        def vjp(grad_out: Any, val_x: Any, val_y: Any, val_a: Any) -> tuple[Any, Any, Any]:
            grad_x = grad_out * (1.0 - val_a)
            grad_y = grad_out * val_a
            grad_a = np.sum(grad_out * (val_y - val_x))
            return grad_x, grad_y, grad_a

        super().__init__(
            op_fn=op,
            inputs=[x, y, alpha],
            name=name or f"Morph_{x.name}_{y.name}",
            vjp_fn=vjp,
        )


class QuadraticVarianceMorphNode(OperatorNode):
    r"""
    Morphed volatility diffusion node :math:`\sigma = \sqrt{(1 - \alpha) \sigma_0^2 + \alpha \sigma_1^2}`.

    Reverse-mode gradient:
    .. math::
        \frac{\partial \sigma}{\partial \alpha} = \frac{\sigma_1^2 - \sigma_0^2}{2 \sigma}
    """

    def __init__(
        self,
        sig0: Node,
        sig1: Node,
        alpha: Node,
        name: Optional[str] = None,
    ) -> None:
        def op(s0: Any, s1: Any, val_a: Any) -> Any:
            var = (1.0 - val_a) * (s0**2) + val_a * (s1**2)
            return np.sqrt(np.maximum(1e-12, var))

        def vjp(grad_out: Any, s0: Any, s1: Any, val_a: Any) -> tuple[Any, Any, Any]:
            var = (1.0 - val_a) * (s0**2) + val_a * (s1**2)
            sig = np.sqrt(np.maximum(1e-12, var))
            live = np.asarray(var) > 1e-12  # the floor is flat: zero derivative below it
            # d_sig / d_s0 = (1-a)*s0 / sig
            grad_s0 = grad_out * np.where(live, (1.0 - val_a) * s0 / sig, 0.0)
            # d_sig / d_s1 = a*s1 / sig
            grad_s1 = grad_out * np.where(live, val_a * s1 / sig, 0.0)
            # d_sig / d_a = (s1^2 - s0^2) / (2 sig)
            grad_a = np.sum(grad_out * np.where(live, (s1**2 - s0**2) / (2.0 * sig), 0.0))
            return grad_s0, grad_s1, grad_a

        super().__init__(
            op_fn=op,
            inputs=[sig0, sig1, alpha],
            name=name or f"QuadVarMorph_{sig0.name}_{sig1.name}",
            vjp_fn=vjp,
        )
