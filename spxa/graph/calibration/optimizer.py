"""
Graph-based model calibration and numerical optimization with autodiff gradients.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from spxa.graph.autodiff.exactness import GradientExactness, GradientNote, GradientResult, worst
from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ParameterNode


def _scalar_loss(value: Any, name: str) -> float:
    """The loss as a ``float``; anything but a real scalar is an error, not summed."""
    v = np.asarray(value)
    if v.ndim != 0:
        raise ValueError(
            f"the loss output {name!r} must be a scalar, got shape {v.shape}; reduce it "
            "first (e.g. MSELossNode or SumNode)"
        )
    if np.iscomplexobj(v):
        raise TypeError(f"the loss output {name!r} is complex; calibration needs a real loss")
    return float(v)


def _gradient_vector(grads: GradientResult, parameters: List[ParameterNode]) -> np.ndarray:
    """Stack the tape's per-parameter gradients, which are scalars for scalar parameters."""
    out = np.empty(len(parameters))
    for i, p in enumerate(parameters):
        g = np.asarray(grads[p])
        if g.ndim != 0:
            raise ValueError(f"gradient for parameter {p.name!r} has shape {g.shape}, not ()")
        if np.iscomplexobj(g):
            raise TypeError(f"gradient for parameter {p.name!r} is complex: {g}")
        out[i] = float(g)
    return out


class GraphCalibrator:
    r"""
    Calibrates graph parameters by minimizing a scalar loss output.

    Gradients come from :class:`~spxa.graph.autodiff.tape.GradientTape`, which
    gives every scalar :class:`~spxa.graph.node.ParameterNode` a scalar gradient.
    The loss must be a real scalar. An array loss is rejected rather than
    summed. Each result reports ``gradient_exactness``, the worst label over all
    gradient evaluations (``EXACT`` unless some node fell back to central
    differences or a stochastic estimator), and ``gradient_notes``, which says
    which nodes and why. ``strict=True`` raises
    :class:`~spxa.graph.autodiff.exactness.InexactGradientError` at the first
    inexact gradient.
    """

    def __init__(
        self,
        calibration_graph: ComputationalGraph,
        loss_output_name: str = "loss",
    ) -> None:
        self.graph = calibration_graph
        self.loss_output_name = loss_output_name

    def _parameters(self, parameters: Optional[List[ParameterNode]]) -> List[ParameterNode]:
        if parameters is None:
            parameters = [p for p in self.graph.get_parameters().values() if p.is_trainable]
        if not parameters:
            raise ValueError("no parameters to calibrate")
        return list(parameters)

    def _loss(self) -> float:
        out = self.graph.evaluate(return_outputs_only=True)[self.loss_output_name]
        return _scalar_loss(out, self.loss_output_name)

    def fit(
        self,
        parameters: Optional[List[ParameterNode]] = None,
        method: str = "L-BFGS-B",
        max_iter: int = 100,
        tol: float = 1e-6,
        *,
        strict: bool = False,
    ) -> Dict[str, Any]:
        r"""Minimize the loss with SciPy, passing it the tape's gradient (``jac=True``)."""
        parameters = self._parameters(parameters)
        x0 = np.array([p.value for p in parameters])
        bounds = [(p.bounds.lower, p.bounds.upper) for p in parameters]
        tape = GradientTape(self.graph)
        self._loss()  # fail fast on a non-scalar loss
        level = GradientExactness.EXACT
        notes: List[GradientNote] = []

        def objective(x: np.ndarray) -> Tuple[float, np.ndarray]:
            nonlocal level
            for idx, p in enumerate(parameters):
                p.value = float(x[idx])
            loss_val = self._loss()
            grads = tape.gradient(self.loss_output_name, sources=parameters, strict=strict)
            level = worst([level, grads.exactness])
            notes.extend(n for n in grads.notes if n not in notes)
            return loss_val, _gradient_vector(grads, parameters)

        res = minimize(
            objective,
            x0=x0,
            jac=True,
            method=method,
            bounds=bounds if any(b != (None, None) for b in bounds) else None,
            options={"maxiter": max_iter},
            tol=tol,
        )

        # Set optimal parameters in graph
        for idx, p in enumerate(parameters):
            p.value = float(res.x[idx])

        return {
            "success": bool(res.success),
            "final_loss": float(res.fun),
            "iterations": int(res.nit),
            "optimal_parameters": {p.name: p.value for p in parameters},
            "message": str(res.message),
            "gradient_exactness": level,
            "gradient_notes": notes,
        }

    def fit_adam(
        self,
        parameters: Optional[List[ParameterNode]] = None,
        lr: float = 0.01,
        max_iter: int = 150,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        *,
        strict: bool = False,
    ) -> Dict[str, Any]:
        r"""
        Adam (Kingma & Ba 2015) on the tape's gradient.

        Each step is projected onto the parameter bounds: ``ParameterNode``
        clips values that leave them.
        """
        parameters = self._parameters(parameters)
        self._loss()  # fail fast on a non-scalar loss
        m = np.zeros(len(parameters))
        v = np.zeros(len(parameters))
        tape = GradientTape(self.graph)
        level = GradientExactness.EXACT
        notes: List[GradientNote] = []

        for step in range(1, max_iter + 1):
            grads = tape.gradient(self.loss_output_name, sources=parameters, strict=strict)
            level = worst([level, grads.exactness])
            notes.extend(n for n in grads.notes if n not in notes)
            g = _gradient_vector(grads, parameters)
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g**2
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            delta = lr * m_hat / (np.sqrt(v_hat) + eps)
            for i, p in enumerate(parameters):
                p.value = p.value - float(delta[i])

        return {
            "success": True,
            "final_loss": self._loss(),
            "iterations": max_iter,
            "optimal_parameters": {p.name: p.value for p in parameters},
            "gradient_exactness": level,
            "gradient_notes": notes,
        }
