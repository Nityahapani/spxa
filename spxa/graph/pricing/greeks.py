"""
Greeks of a pricing graph, with the exactness of every derivative reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from spxa.graph.autodiff.exactness import (
    GradientExactness,
    GradientNote,
    check_strict,
    worst,
)
from spxa.graph.autodiff.hessian import graph_hessian
from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph

__all__ = ["GreeksResult", "compute_graph_greeks", "graph_greeks"]


@dataclass
class GreeksResult:
    """Price and Greeks of a pricing graph, with their exactness.

    ``values`` has the keys ``price``, ``delta``, ``gamma``, ``vega``, ``theta``,
    ``rho``, ``vanna`` and ``volga``; ``result["delta"]`` reads it directly.
    """

    values: Dict[str, float]
    exactness: GradientExactness = GradientExactness.EXACT
    notes: List[GradientNote] = field(default_factory=list)

    def __getitem__(self, key: str) -> float:
        return self.values[key]


def _real(x: object) -> float:
    return float(np.real(np.asarray(x)).sum())


def graph_greeks(
    pricing_graph: ComputationalGraph,
    price_output_name: str = "price",
    spot_name: str = "s0",
    vol_name: str = "volatility",
    maturity_name: str = "maturity",
    rate_name: str = "rate",
    eps: float = 1e-4,
    *,
    strict: bool = False,
) -> GreeksResult:
    r"""
    Price, Greeks and their exactness for the parameters named ``spot_name`` etc.

    Delta, vega, rho and theta :math:`= -\partial V/\partial T` come from one reverse
    pass of the tape. Gamma, vanna and volga are the entries of the Hessian in
    (spot, volatility) from :func:`~spxa.graph.autodiff.hessian.graph_hessian`: exact
    second-order jets when every node on the way runs on jets, otherwise central
    differences of the exact gradient with step ``eps * max(1, |x|)``, reported as
    ``NUMERICAL``. ``strict=True`` raises instead of returning inexact Greeks. A
    parameter missing from the graph gives Greeks of 0.
    """
    params = pricing_graph.get_parameters()
    tape = GradientTape(pricing_graph)
    price = _real(pricing_graph.evaluate(return_outputs_only=True)[price_output_name])
    grads = tape.gradient(price_output_name)
    level = grads.exactness
    notes = list(grads.notes)

    def first(name: str) -> float:
        node = params.get(name)
        return _real(grads[node]) if node is not None and node in grads else 0.0

    spot, vol = params.get(spot_name), params.get(vol_name)
    second = [p for p in (spot, vol) if p is not None]
    gamma = vanna = volga = 0.0
    if second:
        hr = graph_hessian(pricing_graph, price_output_name, second, eps)
        level = worst([level, hr.exactness])
        notes.extend(n for n in hr.notes if n not in notes)
        pos = {id(p): i for i, p in enumerate(second)}
        H = np.real(hr.hessian)
        if spot is not None:
            gamma = float(H[pos[id(spot)], pos[id(spot)]])
        if vol is not None:
            volga = float(H[pos[id(vol)], pos[id(vol)]])
        if spot is not None and vol is not None:
            vanna = float(H[pos[id(spot)], pos[id(vol)]])
    out = GreeksResult(
        {
            "price": price,
            "delta": first(spot_name),
            "gamma": gamma,
            "vega": first(vol_name),
            "theta": -first(maturity_name),
            "rho": first(rate_name),
            "vanna": vanna,
            "volga": volga,
        },
        level,
        notes,
    )
    check_strict(strict, level, notes, "greeks")
    return out


def compute_graph_greeks(
    pricing_graph: ComputationalGraph,
    price_output_name: str = "price",
    spot_name: str = "s0",
    vol_name: str = "volatility",
    maturity_name: str = "maturity",
    rate_name: str = "rate",
    eps: float = 1e-4,
    *,
    strict: bool = False,
) -> Dict[str, float]:
    r"""Price and Greeks as a dict (see :func:`graph_greeks`, which also reports exactness).

    Keys: ``price``, ``delta``, ``gamma``, ``vega``, ``theta``, ``rho``, ``vanna``, ``volga``.
    """
    return graph_greeks(
        pricing_graph,
        price_output_name,
        spot_name,
        vol_name,
        maturity_name,
        rate_name,
        eps,
        strict=strict,
    ).values
