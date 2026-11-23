"""
Characteristic functions, exponents, cumulants, and moments graph nodes.

The process parameters are graph inputs of every node here (after the
explicit inputs ``u``/``t``/``theta``), so gradients with respect to them flow
through the tape. All formulas are written with NumPy operations only, so the
nodes run on :class:`~spxa.graph.autodiff.jet.Jet` inputs and differentiate
exactly. Moments of the Brownian and Merton processes come from closed-form
cumulants; for other processes they come from a Cauchy contour integral of the
characteristic function, whose (geometric) convergence is checked, and which
is labelled ``NUMERICAL``.
"""

from __future__ import annotations

from math import factorial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spxa.graph.autodiff.exactness import GradientExactness
from spxa.graph.autodiff.jet import Jet, JetError, is_jet, variables
from spxa.graph.node import Node, OperatorNode, PortDescriptor
from spxa.graph.stochastic_nodes import (
    BrownianMotionNode,
    HestonNode,
    MertonJumpDiffusionNode,
    StochasticProcessNode,
)
from spxa.graph.types import DType

_PARAMS: Dict[type, Tuple[str, ...]] = {
    BrownianMotionNode: ("mu", "sigma"),
    MertonJumpDiffusionNode: ("mu", "sigma", "lam", "mu_j", "sigma_j"),
    HestonNode: ("s0", "v0", "mu", "kappa", "theta", "xi", "rho"),
}


def _param_names(process_node: StochasticProcessNode, what: str) -> Tuple[str, ...]:
    for cls, names in _PARAMS.items():
        if type(process_node) is cls:
            return names
    raise NotImplementedError(f"{what} not implemented for {process_node.__class__.__name__}")


def _param_nodes(process_node: StochasticProcessNode, names: Sequence[str]) -> List[Node]:
    return [process_node.params[n] for n in names]


def _current(nodes: Sequence[Node]) -> List[Any]:
    return [n.forward() for n in nodes]


def _complex(u: Any) -> Any:
    return u + 0j if is_jet(u) else np.asarray(u, dtype=complex)


def _levy_exponent(process_node: StochasticProcessNode, u: Any, p: Sequence[Any]) -> Any:
    """psi(u) with E[exp(iuX_t)] = exp(t psi(u)) for the Lévy process nodes."""
    if type(process_node) is BrownianMotionNode:
        mu, sig = p
        return 1j * u * mu - 0.5 * (sig**2) * (u**2)
    if type(process_node) is MertonJumpDiffusionNode:
        mu, sig, lam, mu_j, sig_j = p
        diff_part = 1j * u * mu - 0.5 * (sig**2) * (u**2)
        return diff_part + lam * (np.exp(1j * u * mu_j - 0.5 * (sig_j**2) * (u**2)) - 1.0)
    raise NotImplementedError(
        f"Characteristic exponent not implemented for {process_node.__class__.__name__}"
    )


def _log_char(process_node: StochasticProcessNode, u: Any, t: Any, p: Sequence[Any]) -> Any:
    """log E[exp(iuX_t)] (continuous in u; no principal-branch jumps)."""
    if type(process_node) is HestonNode:
        s0, v0, mu, kappa, theta, xi, rho = p
        # Albrecher et al. ("little Heston trap") form: continuous in u for all t
        b = kappa - 1j * rho * xi * u
        d = np.sqrt(b**2 + (xi**2) * (u**2 + 1j * u))
        g = (b - d) / (b + d)
        e = np.exp(-d * t)
        C = 1j * u * (np.log(s0) + mu * t) + (kappa * theta / (xi**2)) * (
            (b - d) * t - 2.0 * np.log((1.0 - g * e) / (1.0 - g))
        )
        D = ((b - d) / (xi**2)) * ((1.0 - e) / (1.0 - g * e))
        return C + D * v0
    return t * _levy_exponent(process_node, u, p)


class CharacteristicExponentNode(OperatorNode):
    r"""Characteristic exponent node :math:`\psi(u)` for Lévy processes:

    .. math::
        \mathbb{E}[e^{i u X_t}] = e^{t \psi(u)}

    Inputs: ``u`` followed by the process parameters.
    """

    def __init__(
        self,
        process_node: StochasticProcessNode,
        u_node: Node,
        name: Optional[str] = None,
    ) -> None:
        self.process_node = process_node
        self.u_node = u_node
        names = _param_names(process_node, "Characteristic exponent")
        if type(process_node) is HestonNode:
            raise NotImplementedError("Characteristic exponent not implemented for HestonNode")
        self.param_nodes = _param_nodes(process_node, names)

        def char_exponent_fn(u_val: Any, *p: Any) -> Any:
            p = tuple(p) if p else tuple(_current(self.param_nodes))
            return _levy_exponent(process_node, _complex(u_val), p)

        super().__init__(
            op_fn=char_exponent_fn,
            inputs=[u_node, *self.param_nodes],
            name=name or f"CharExp_{process_node.name}",
            output_descriptor=PortDescriptor(name="psi", dtype=DType.COMPLEX128),
        )


class CharacteristicFunctionNode(OperatorNode):
    r"""Characteristic function node :math:`\phi(u, t) = \mathbb{E}[e^{i u X_t}]`.

    Inputs: ``u``, ``t``, then the process parameters. ``op_fn(u, t)`` called
    without parameter values uses the parameters' current values.
    """

    def __init__(
        self,
        process_node: StochasticProcessNode,
        u_node: Node,
        t_node: Node,
        name: Optional[str] = None,
    ) -> None:
        self.process_node = process_node
        self.u_node = u_node
        self.t_node = t_node
        names = _param_names(process_node, "Characteristic function")
        self.param_nodes = _param_nodes(process_node, names)

        def char_fn(u_val: Any, t_val: Any, *p: Any) -> Any:
            return np.exp(self.log_char(u_val, t_val, *p))

        super().__init__(
            op_fn=char_fn,
            inputs=[u_node, t_node, *self.param_nodes],
            name=name or f"CharFunc_{process_node.name}",
            output_descriptor=PortDescriptor(name="phi", dtype=DType.COMPLEX128),
        )

    def log_char(self, u_val: Any, t_val: Any, *p: Any) -> Any:
        r""":math:`\log\phi(u, t)`, continuous in ``u`` (parameters default to current values)."""
        p = tuple(p) if p else tuple(_current(self.param_nodes))
        return _log_char(self.process_node, _complex(u_val), t_val, p)

    def has_closed_cumulants(self) -> bool:
        return type(self.process_node) in (BrownianMotionNode, MertonJumpDiffusionNode)

    def cumulants(self, t_val: Any, *p: Any) -> List[Any]:
        """First four cumulants of :math:`X_t` in closed form (Brownian / Merton only)."""
        p = tuple(p) if p else tuple(_current(self.param_nodes))
        if type(self.process_node) is BrownianMotionNode:
            mu, sig = p
            return [mu * t_val, sig**2 * t_val, 0.0 * t_val, 0.0 * t_val]
        if type(self.process_node) is MertonJumpDiffusionNode:
            mu, sig, lam, mj, sj = p
            ej = [mj, mj**2 + sj**2, mj**3 + 3 * mj * sj**2, mj**4 + 6 * mj**2 * sj**2 + 3 * sj**4]
            return [
                (mu + lam * ej[0]) * t_val,
                (sig**2 + lam * ej[1]) * t_val,
                lam * ej[2] * t_val,
                lam * ej[3] * t_val,
            ]
        raise NotImplementedError("closed-form cumulants need a Brownian or Merton process")


class AffineCharacteristicNode(OperatorNode):
    r"""Affine characteristic transform node :math:`\phi(u, t, x) = \exp(A(t, u) + B(t, u) x)`."""

    def __init__(
        self,
        a_fn: Callable[[float, Any], Any],
        b_fn: Callable[[float, Any], Any],
        u_node: Node,
        t_node: Node,
        x_node: Node,
        name: str = "AffineCharFunc",
    ) -> None:
        def affine_eval(u_val: Any, t_val: Any, x_val: Any) -> Any:
            u = _complex(u_val)
            x = x_val if is_jet(x_val) else np.asarray(x_val, dtype=float)
            return np.exp(a_fn(t_val, u) + b_fn(t_val, u) * x)

        super().__init__(
            op_fn=affine_eval,
            inputs=[u_node, t_node, x_node],
            name=name,
            output_descriptor=PortDescriptor(name="phi", dtype=DType.COMPLEX128),
        )


class CumulantGeneratingFunctionNode(OperatorNode):
    r"""Cumulant generating function node :math:`K(\theta, t) = \ln \mathbb{E}[e^{\theta X_t}]`.

    Computed as :math:`\operatorname{Re}\log\phi(-i\theta, t)` from the continuous
    log-characteristic function (no ``log(exp(.))`` round trip, so no overflow or
    branch jumps). Inputs: ``theta``, ``t``, then the process parameters.
    """

    def __init__(
        self,
        char_func_node: CharacteristicFunctionNode,
        theta_node: Node,
        t_node: Node,
        name: str = "CGF",
    ) -> None:
        self.char_func_node = char_func_node

        def cgf_eval(theta_val: Any, t_val: Any, *p: Any) -> Any:
            return np.real(char_func_node.log_char(-1j * _complex(theta_val), t_val, *p))

        super().__init__(
            op_fn=cgf_eval,
            inputs=[theta_node, t_node, *char_func_node.param_nodes],
            name=name,
            output_descriptor=PortDescriptor(name="cgf", dtype=DType.FLOAT64),
        )


class GeneratorApplicationNode(OperatorNode):
    r"""Infinitesimal generator application node :math:`z = \mathcal{A} f(x) = b f' + \tfrac12\sigma^2 f''`.

    :math:`f'` and :math:`f''` are exact (second-order jets) when ``func`` is
    written with NumPy operations; otherwise central differences with step
    ``eps`` are used and the node is labelled ``NUMERICAL``. Inputs: ``x``,
    then the process parameters.
    """

    def __init__(
        self,
        process_node: StochasticProcessNode,
        func: Callable[[float], float],
        x_node: Node,
        name: Optional[str] = None,
        eps: float = 1e-4,
    ) -> None:
        names = list(process_node.params)
        self.param_nodes = [process_node.params[n] for n in names]

        def derivatives(x: Any) -> Tuple[Any, Any]:
            try:
                (xj,) = variables([x if not is_jet(x) else x.value], order=2)
                fx = func(xj)
                if not isinstance(fx, Jet):
                    raise JetError("func returned no derivatives")
                return fx.grad[..., 0], fx.hess[..., 0, 0]
            except (JetError, TypeError, ValueError):
                self.gradient_exactness = GradientExactness.NUMERICAL
                xf = float(x if not is_jet(x) else x.value)
                f_p, f_m, f_c = func(xf + eps), func(xf - eps), func(xf)
                return (f_p - f_m) / (2.0 * eps), (f_p - 2.0 * f_c + f_m) / (eps**2)

        def gen_eval(x_val: Any, *p: Any) -> Any:
            pv = dict(zip(names, p)) if p else {}
            xa = np.array([x_val.value if is_jet(x_val) else float(x_val)])
            b = process_node.drift(0.0, xa, **pv)[0]
            sig = process_node.diffusion(0.0, xa, **pv)[0]
            if is_jet(x_val):
                raise JetError("the generator is not differentiated in x")
            f1, f2 = derivatives(x_val)
            return b * f1 + 0.5 * (sig**2) * f2

        super().__init__(
            op_fn=gen_eval,
            inputs=[x_node, *self.param_nodes],
            name=name or f"Generator_{process_node.name}",
            output_descriptor=PortDescriptor(name="gen_val", dtype=DType.FLOAT64),
        )


def _moments_from_cumulants(k: Sequence[Any]) -> List[Any]:
    k1, k2, k3, k4 = k
    return [
        k1,
        k2 + k1**2,
        k3 + 3 * k2 * k1 + k1**3,
        k4 + 4 * k3 * k1 + 3 * k2**2 + 6 * k2 * k1**2 + k1**4,
    ]


def _contour_moment(phi: Callable[[Any], Any], order: int) -> float:
    r"""
    :math:`(-i)^k\phi^{(k)}(0)` by the trapezoid rule on circles :math:`|u| = r`.

    :math:`\phi^{(k)}(0) = \frac{k!}{2\pi r^k}\int_0^{2\pi}\phi(re^{i\vartheta})
    e^{-ik\vartheta}d\vartheta`; the trapezoid rule converges geometrically for
    analytic :math:`\phi`. Radii and node counts are refined until two estimates
    agree to 1e-12 relative; otherwise a ``RuntimeError`` is raised.
    """
    best = None
    for r in (0.5, 0.25, 1.0, 0.125):
        prev = None
        for n in (32, 64, 128, 256):
            th = 2.0 * np.pi * np.arange(n) / n
            vals = np.asarray(phi(r * np.exp(1j * th)), dtype=complex)
            dk = factorial(order) / r**order * np.mean(vals * np.exp(-1j * order * th))
            est = complex((-1j) ** order * dk)
            if prev is not None and abs(est - prev) <= 1e-12 * max(abs(est), 1e-300):
                return float(est.real)
            prev = est
        best = prev
    raise RuntimeError(f"contour moment of order {order} did not converge (last {best})")


class ProcessMomentNode(OperatorNode):
    r"""Process moment node :math:`m_k(t) = \mathbb{E}[X_t^k]` for orders :math:`k \in \{1, 2, 3, 4\}`.

    Brownian and Merton moments come from closed-form cumulants (exact, and
    exactly differentiable). Other processes use a converged Cauchy contour
    integral of :math:`\phi` (``NUMERICAL``). Inputs: ``t``, then the process
    parameters.
    """

    def __init__(
        self,
        char_func_node: CharacteristicFunctionNode,
        order: int,
        t_node: Node,
        name: Optional[str] = None,
        eps: float = 1e-4,
    ) -> None:
        if order not in (1, 2, 3, 4):
            raise ValueError("order must be 1, 2, 3 or 4")
        self.order = order
        closed = char_func_node.has_closed_cumulants()
        if not closed:
            self.gradient_exactness = GradientExactness.NUMERICAL
            self.jet_capable = False

        def moment_eval(t_val: Any, *p: Any) -> Any:
            if closed:
                return _moments_from_cumulants(char_func_node.cumulants(t_val, *p))[order - 1]
            return _contour_moment(lambda u: np.exp(char_func_node.log_char(u, t_val, *p)), order)

        super().__init__(
            op_fn=moment_eval,
            inputs=[t_node, *char_func_node.param_nodes],
            name=name or f"Moment_{order}_{char_func_node.name}",
            output_descriptor=PortDescriptor(name=f"moment_{order}", dtype=DType.FLOAT64),
        )
