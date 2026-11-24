"""
Derivative payoff nodes for graph-based pricing and risk engines.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.graph.node import Node, OperatorNode


class EuropeanCallPayoffNode(OperatorNode):
    r"""European Call option terminal payoff node :math:`(S_T - K)^+`."""

    def __init__(self, spot_terminal: Node, strike: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda s, k: np.maximum(np.asarray(s, dtype=float) - float(k), 0.0),
            inputs=[spot_terminal, strike],
            name=name or f"CallPayoff_{spot_terminal.name}",
            # H(0) = 1/2 at the strike: the midpoint of the subdifferential of the kink
            vjp_fn=lambda grad_out, s, k: (
                grad_out * np.heaviside(np.asarray(s, dtype=float) - float(k), 0.5),
                -grad_out * np.heaviside(np.asarray(s, dtype=float) - float(k), 0.5),
            ),
        )


class EuropeanPutPayoffNode(OperatorNode):
    r"""European Put option terminal payoff node :math:`(K - S_T)^+`."""

    def __init__(self, spot_terminal: Node, strike: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda s, k: np.maximum(float(k) - np.asarray(s, dtype=float), 0.0),
            inputs=[spot_terminal, strike],
            name=name or f"PutPayoff_{spot_terminal.name}",
            vjp_fn=lambda grad_out, s, k: (
                -grad_out * np.heaviside(float(k) - np.asarray(s, dtype=float), 0.5),
                grad_out * np.heaviside(float(k) - np.asarray(s, dtype=float), 0.5),
            ),
        )


class DigitalPayoffNode(OperatorNode):
    r"""Digital cash-or-nothing payoff node :math:`\mathbf{1}_{\{S_T \ge K\}}`."""

    def __init__(self, spot_terminal: Node, strike: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda s, k: (np.asarray(s, dtype=float) >= float(k)).astype(float),
            inputs=[spot_terminal, strike],
            name=name or f"DigitalPayoff_{spot_terminal.name}",
        )


class AsianPayoffNode(OperatorNode):
    r"""Arithmetic Asian average rate option payoff node :math:`(\frac{1}{N} \sum S_{t_i} - K)^+`."""

    def __init__(self, path_node: Node, strike: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda p, k: np.maximum(
                np.mean(np.asarray(p, dtype=float), axis=-1) - float(k), 0.0
            ),
            inputs=[path_node, strike],
            name=name or "AsianPayoff",
        )
