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


def narrate_latex(node: CompositionNode, depth: int = 0) -> str:
    """
    Recursively walk a CompositionNode tree and produce a LaTeX derivation.

    Output is valid MathJax/LaTeX that renders in Jupyter notebooks via
    ``IPython.display.Math(narrate_latex(node))``.

    Parameters
    ----------
    node :
        Root of the composition tree.
    depth :
        Current nesting depth.
    """
    indent = r"\quad " * depth
    lines: list[str] = []

    if node.kind == NodeKind.PRIMITIVE:
        name_tex = node.name.replace("_", r"\_")
        lines.append(rf"{indent}\textbf{{[Primitive]}}\ \texttt{{{name_tex}}}")
        if node.math_note:
            note_tex = node.math_note.replace("_", r"\_")
            lines.append(rf"{indent}\quad \text{{Triplet: }} {note_tex}")
        if node.reference:
            lines.append(rf"{indent}\quad \text{{Ref: {node.reference}}}")

    elif node.kind == NodeKind.ADDITION:
        left, right = node.children[0], node.children[1]
        lines.append(rf"{indent}\textbf{{[Addition]}}")
        lines.append(
            rf"{indent}\quad"
            r"(b_Z,\sigma^2_Z,\nu_Z)=(b_X+b_Y,\,\sigma^2_X+\sigma^2_Y,\,\nu_X+\nu_Y)"
        )
        lines.append(rf"{indent}\quad \text{{[Sato 1999, Prop.\ 11.10]}}")
        lines.append(rf"{indent}\text{{Left:}}")
        lines.append(narrate_latex(left, depth + 1))
        lines.append(rf"{indent}\text{{Right:}}")
        lines.append(narrate_latex(right, depth + 1))

    elif node.kind == NodeKind.SCALING:
        child = node.children[0]
        lines.append(rf"{indent}\textbf{{[Scaling]}}")
        lines.append(
            rf"{indent}\quad"
            r"(b_Z,\sigma^2_Z,\nu_Z)=(cb+\Delta b,\,c^2\sigma^2,\,\nu(\cdot/c))"
        )
        lines.append(rf"{indent}\quad \text{{[Sato 1999, Prop.\ 11.10]}}")
        lines.append(rf"{indent}\text{{Operand:}}")
        lines.append(narrate_latex(child, depth + 1))

    elif node.kind == NodeKind.SUBORDINATION:
        parent, sub = node.children[0], node.children[1]
        lines.append(rf"{indent}\textbf{{[Subordination]}}")
        lines.append(
            rf"{indent}\quad"
            r"\psi_Z(u)=-\phi(-\psi_X(u)),\quad"
            r"\phi=\text{Laplace exponent of }T"
        )
        lines.append(rf"{indent}\quad \text{{[Sato 1999, Thm.\ 30.1]}}")
        lines.append(rf"{indent}\text{{Parent:}}")
        lines.append(narrate_latex(parent, depth + 1))
        lines.append(rf"{indent}\text{{Subordinator:}}")
        lines.append(narrate_latex(sub, depth + 1))

    elif node.kind == NodeKind.MOMENT_APPROX:
        lines.append(rf"{indent}\textbf{{[Moment Propagation]}}")
        if node.math_note:
            note_tex = node.math_note.replace("_", r"\_")
            lines.append(rf"{indent}\quad \text{{{note_tex}}}")
        for child in node.children:
            lines.append(narrate_latex(child, depth + 1))

    return "\n".join(lines)


def story_latex(process: object) -> str:
    """
    Return a full LaTeX derivation string for a process.

    Wraps the composition tree in ``\\begin{aligned}...\\end{aligned}``
    for display in a Jupyter notebook:

    .. code-block:: python

        from IPython.display import display, Math
        display(Math(story_latex(Z)))

    Parameters
    ----------
    process :
        Any spxa Process instance.
    """
    # Import here to avoid circular at module load time
    node = getattr(process, "_node", None)
    exactness = getattr(process, "exactness", None)
    props = process.properties  # type: ignore[union-attr]

    lines = [
        r"\begin{aligned}",
        r"&\textbf{spxa\ derivation\ trace}\\[4pt]",
        rf"&\text{{Exactness: }}\texttt{{{exactness.name if exactness else '?'}}}\\[6pt]",
        r"&\textbf{Composition\ tree:}\\",
    ]
    if node is not None:
        lines.append(narrate_latex(node))
    lines += [
        r"\\[6pt]",
        r"&\textbf{Properties:}\\",
        rf"&\quad\text{{Stationary increments: }}{props.has_stationary_increments}\\",
        rf"&\quad\text{{Independent increments: }}{props.has_independent_increments}\\",
        rf"&\quad\text{{Martingale: }}{props.is_martingale}\\",
        rf"&\quad\text{{Finite mean: }}{props.has_finite_mean}\\",
        rf"&\quad\text{{Finite variance: }}{props.has_finite_variance}\\",
    ]
    if props.tail_index is not None:
        lines.append(rf"&\quad\text{{Tail index: }}\alpha={props.tail_index}\\")
    if props.self_similarity_index is not None:
        lines.append(rf"&\quad\text{{Self-similarity: }}H={props.self_similarity_index}\\")
    if props.hurst_index is not None:
        lines.append(rf"&\quad\text{{Hurst index: }}H={props.hurst_index}\\")
    if props.notes:
        lines.append(r"&\quad\textit{Notes:}\\")
        for note in props.notes:
            note_tex = note.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            lines.append(rf"&\quad\quad\text{{{note_tex}}}\\")
    lines.append(r"\end{aligned}")
    return "\n".join(lines)
