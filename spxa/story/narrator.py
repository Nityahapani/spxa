"""
Derivation trace engine for .__story__().

Every Process carries a CompositionNode that records how it was built.
The narrator walks the tree and emits a human-readable (and LaTeX-renderable)
derivation of the process's distributional properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class NodeKind(Enum):
    PRIMITIVE = auto()
    ADDITION = auto()
    SCALING = auto()
    SUBORDINATION = auto()
    MOMENT_APPROX = auto()


@dataclass
class CompositionNode:
    """
    A node in the composition tree of a process.

    Parameters
    ----------
    kind :
        What operation produced this node.
    name :
        Human-readable name of the process or operation (e.g. 'VarianceGamma',
        '0.5 * NIG', 'X + Y').
    children :
        Child nodes (empty for primitives).
    math_note :
        LaTeX string explaining the key formula at this node.
    reference :
        Citation for the theorem or formula at this node.
    """

    kind: NodeKind
    name: str
    children: list[CompositionNode] = field(default_factory=list)
    math_note: str = ""
    reference: str = ""


def narrate(node: CompositionNode, depth: int = 0) -> str:
    """
    Recursively walk a CompositionNode tree and produce a derivation string.

    Parameters
    ----------
    node :
        Root of the composition tree.
    depth :
        Current indentation depth (used internally for recursion).
    """
    indent = "  " * depth
    lines: list[str] = []

    if node.kind == NodeKind.PRIMITIVE:
        lines.append(f"{indent}[Primitive] {node.name}")
        if node.math_note:
            lines.append(f"{indent}  Triplet: {node.math_note}")
        if node.reference:
            lines.append(f"{indent}  Ref: {node.reference}")

    elif node.kind == NodeKind.ADDITION:
        left, right = node.children[0], node.children[1]
        lines.append(f"{indent}[Addition] {node.name}")
        lines.append(
            f"{indent}  Rule: (b_X+b_Y, σ²_X+σ²_Y, ν_X+ν_Y) "
            f"[Sato 1999, Prop. 11.10]"
        )
        if node.math_note:
            lines.append(f"{indent}  {node.math_note}")
        lines.append(f"{indent}  Left operand:")
        lines.append(narrate(left, depth + 2))
        lines.append(f"{indent}  Right operand:")
        lines.append(narrate(right, depth + 2))

    elif node.kind == NodeKind.SCALING:
        child = node.children[0]
        lines.append(f"{indent}[Scaling] {node.name}")
        lines.append(
            f"{indent}  Rule: (cb + drift_correction, c²σ², ν(·/c)) "
            f"[Sato 1999, Prop. 11.10]"
        )
        if node.math_note:
            lines.append(f"{indent}  {node.math_note}")
        lines.append(f"{indent}  Operand:")
        lines.append(narrate(child, depth + 2))

    elif node.kind == NodeKind.SUBORDINATION:
        parent, sub = node.children[0], node.children[1]
        lines.append(f"{indent}[Subordination] {node.name}")
        lines.append(
            f"{indent}  Rule: ψ_Z(u) = -φ(-ψ_X(u)) where φ is the Laplace "
            f"exponent of T [Sato 1999, Thm. 30.1]"
        )
        if node.math_note:
            lines.append(f"{indent}  {node.math_note}")
        lines.append(f"{indent}  Parent process:")
        lines.append(narrate(parent, depth + 2))
        lines.append(f"{indent}  Subordinator:")
        lines.append(narrate(sub, depth + 2))

    elif node.kind == NodeKind.MOMENT_APPROX:
        lines.append(f"{indent}[Approximation — moment propagation] {node.name}")
        if node.math_note:
            lines.append(f"{indent}  {node.math_note}")
        for child in node.children:
            lines.append(narrate(child, depth + 2))

    return "\n".join(lines)
