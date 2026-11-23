r"""
Hessians of graph outputs.

The Hessian is computed exactly with second-order forward jets when every
node between the parameters and the target can run on
:class:`~spxa.graph.autodiff.jet.Jet` inputs. Otherwise it is assembled from
central differences of the (exact, reverse-mode) gradient, with relative step
``eps``, and labelled ``NUMERICAL``. Parameters are perturbed through
``feed_dict``, never by mutating the graph, so parameter bounds cannot clip a
perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.autodiff.exactness import (
    GradientExactness,
    GradientNote,
    check_strict,
    worst,
)
from spxa.graph.autodiff.tape import GradientTape, _ForwardUnavailable, forward_hessian
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node, ParameterNode

__all__ = ["HessianResult", "graph_hessian", "compute_graph_hessian"]


@dataclass
class HessianResult:
    """Value, gradient and Hessian of a scalar graph output, with exactness."""

    value: Any
    gradient: np.ndarray
    hessian: np.ndarray
    names: List[str]
    exactness: GradientExactness = GradientExactness.EXACT
    notes: List[GradientNote] = field(default_factory=list)


def graph_hessian(
    graph: ComputationalGraph,
    target_output: Union[str, Node],
    parameters: Optional[List[ParameterNode]] = None,
    eps: float = 1e-4,
    *,
    strict: bool = False,
    feed_dict: Optional[Dict[Node, Any]] = None,
) -> HessianResult:
    r"""
    Hessian :math:`H_{ij} = \partial^2 Y/\partial\theta_i\partial\theta_j` of ``sum(target)``.

    Exact (second-order jets) when possible; otherwise central differences of
    the exact gradient with step :math:`h_j = \mathrm{eps}\cdot\max(1, |\theta_j|)`
    (``NUMERICAL``; symmetrized). ``strict=True`` raises instead of returning
    a numerical Hessian.
    """
    if parameters is None:
        parameters = [p for p in graph.get_parameters().values() if p.is_trainable]
    names = [p.name for p in parameters]
    tape = GradientTape(graph)
    try:
        val, g, h = forward_hessian(tape, target_output, list(parameters), feed_dict)
        return HessianResult(val, g, h, names)
    except _ForwardUnavailable as exc:
        reason = f"node {exc} cannot run on jets"

    base: Dict[Node, Any] = dict(feed_dict or {})
    theta = [float(base.get(p, p.value)) for p in parameters]
    n = len(parameters)
    notes: List[GradientNote] = [
        GradientNote(
            str(target_output), GradientExactness.NUMERICAL, "differences of gradient", reason
        )
    ]
    g0 = tape.gradient(target_output, sources=list(parameters), feed_dict=base or None)
    level = worst([g0.exactness, GradientExactness.NUMERICAL])
    notes.extend(g0.notes)
    H = np.zeros((n, n))
    for j in range(n):
        h = eps * max(1.0, abs(theta[j]))
        cols = []
        for sgn in (1.0, -1.0):
            fd = dict(base)
            fd[parameters[j]] = theta[j] + sgn * h
            gr = tape.gradient(target_output, sources=list(parameters), feed_dict=fd)
            level = worst([level, gr.exactness])
            cols.append(np.array([float(np.real(gr[p])) for p in parameters]))
        H[:, j] = (cols[0] - cols[1]) / (2.0 * h)
    H = 0.5 * (H + H.T)
    vals = tape._forward(base or None)[1]
    tn = tape._resolve(target_output)
    out = HessianResult(
        vals[tn.node_id],
        np.array([float(np.real(g0[p])) for p in parameters]),
        H,
        names,
        level,
        notes,
    )
    check_strict(strict, level, notes, "hessian")
    return out


def compute_graph_hessian(
    graph: ComputationalGraph,
    target_output: str,
    parameters: Optional[List[ParameterNode]] = None,
    eps: float = 1e-4,
    *,
    strict: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    r"""
    Hessian matrix and parameter names (see :func:`graph_hessian` for details).

    Returns ``(H, names)`` with ``H`` of shape ``(n_params, n_params)``.
    """
    res = graph_hessian(graph, target_output, parameters, eps, strict=strict)
    return res.hessian, res.names
