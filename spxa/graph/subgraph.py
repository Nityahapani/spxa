"""
Modular hierarchical DAG encapsulation via SubgraphNode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node, OperatorNode, PortDescriptor
from spxa.graph.types import DType


class SubgraphNode(OperatorNode):
    r"""Encapsulates an entire ComputationalGraph as a single hierarchical node in a parent DAG."""

    def __init__(
        self,
        inner_graph: ComputationalGraph,
        outer_inputs: List[Node],
        inner_input_nodes: List[Node],
        output_name: str,
        name: Optional[str] = None,
    ) -> None:
        if len(outer_inputs) != len(inner_input_nodes):
            raise ValueError("Length of outer_inputs must match inner_input_nodes")

        self.inner_graph = inner_graph
        self.inner_input_nodes = inner_input_nodes
        self.output_name = output_name
        self.tape = GradientTape(inner_graph)

        def subgraph_forward(*input_vals: Any) -> Any:
            feed_dict: Dict[Node, Any] = {}
            for inner_node, val in zip(self.inner_input_nodes, input_vals):
                feed_dict[inner_node] = val  # fed values override; nothing is mutated

            res = self.inner_graph.evaluate(feed_dict=feed_dict, return_outputs_only=True)
            return res[self.output_name]

        def subgraph_vjp(grad_out: Any, *input_vals: Any) -> Tuple[Any, ...]:
            feed_dict: Dict[Node, Any] = {}
            for inner_node, val in zip(self.inner_input_nodes, input_vals):
                feed_dict[inner_node] = val

            inner_grads = self.tape.gradient(
                self.output_name,
                sources=self.inner_input_nodes,
                feed_dict=feed_dict,
                grad_output=grad_out,
            )
            self.gradient_exactness = inner_grads.exactness
            return tuple(inner_grads.get(n, 0.0) for n in self.inner_input_nodes)

        super().__init__(
            op_fn=subgraph_forward,
            inputs=outer_inputs,
            name=name or f"Subgraph_{inner_graph.name}",
            output_descriptor=PortDescriptor(name=output_name, dtype=DType.FLOAT64),
            vjp_fn=subgraph_vjp,
        )
