r"""
:class:`CompiledProcess`: a process expression behind compiled kernels.

``process.compile()`` (or :func:`compile_process`) lowers a Lévy process
expression once and returns an object with the evaluation interface of
:class:`~spxa.graph.autodiff.process.DiffProcess` — exponent, gradient,
Hessian, cumulants and their Jacobian over the free parameters — backed by
generated code. Every method takes an optional ``theta``: one vector of
free-parameter values, or a ``(B, n_free)`` batch evaluated in one call.
Sampling (:meth:`CompiledProcess.simulate`, :meth:`~CompiledProcess.sample_at`,
:meth:`~CompiledProcess.paths`) runs the fused plan of :mod:`spxa.compile.sampler`.

.. code-block:: python

    from spxa.zoo.levy import BrownianMotion, GammaProcess, VarianceGamma

    proc = (BrownianMotion(0.05, 0.2) + VarianceGamma(0.2, 0.3, -0.1)) @ GammaProcess(4, 4)
    cp = proc.compile()                      # NumPy backend; "numba" or "jax" also
    psi, grad = cp.char_exponent_and_grad(u)  # one fused program
    kappa, jac = cp.cumulants_and_jacobian(4)
    paths = cp.simulate(252, 10_000, 1.0, rng)  # exact fused sampler
    prices, jac = cp.cos_price(100.0, strikes, 0.5, rate=0.03, grad=True)
    print(cp.source("grad"))                  # the generated code
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from spxa.compile.kernel import ExponentKernel
from spxa.compile.lower import Lowered, lower
from spxa.compile.sampler import Sampler

__all__ = ["CompiledProcess", "compile_process"]


class CompiledProcess:
    r"""
    A Lévy process expression compiled to fused kernels.

    Parameters
    ----------
    process :
        A zoo Lévy process or a ``+`` / scalar ``*`` / ``@`` composition of them.
    backend :
        ``"numpy"`` (default; always available), ``"numba"`` or ``"jax"``.
    free :
        Names (``"<Family>.<param>"``, as :attr:`all_names`) or indices of the
        parameters to expose; default all. The others are compiled in as constants.
    fixed :
        Values ``{name: value}`` overriding fixed (non-free) parameters.
    parallel :
        Numba backend: parallel loops.
    allow_opaque :
        Accept exact Lévy leaves without a lowering rule as opaque calls
        (NumPy backend only; not differentiable in their own parameters).
    """

    def __init__(
        self,
        process: Any,
        backend: str = "numpy",
        free: Optional[Sequence[Union[str, int]]] = None,
        fixed: Optional[Dict[str, float]] = None,
        parallel: bool = False,
        allow_opaque: bool = True,
    ) -> None:
        self.process = process
        self.lowered: Lowered = lower(process, allow_opaque=allow_opaque)
        self.kernel = ExponentKernel(self.lowered, backend, free, fixed, parallel)
        self.sampler = Sampler(self.lowered)
        self.backend = backend
        self.parallel = parallel

    # ── parameters ──────────────────────────────────────────────────────────
    @property
    def names(self) -> Tuple[str, ...]:
        """Names of the free parameters, in ``theta`` order."""
        return self.kernel.names

    @property
    def all_names(self) -> Tuple[str, ...]:
        """Names of every parameter of the expression."""
        return self.lowered.names

    @property
    def values(self) -> np.ndarray:
        """Current free-parameter values."""
        return self.kernel.theta0.copy()

    @property
    def n_params(self) -> int:
        return self.kernel.n_free

    @property
    def analytic(self) -> bool:
        """Whether :math:`\\psi` is analytic in ``u`` (cumulants and complex ``u`` exist)."""
        return self.lowered.analytic

    def full_theta(self, theta: Optional[Any] = None) -> np.ndarray:
        """All parameters (fixed ones filled in) for free values ``theta``."""
        return self.kernel.full_theta(theta)

    def to_process(self, theta: Optional[Sequence[float]] = None) -> Any:
        """The spxa process at free-parameter values ``theta`` (default: current)."""
        full = self.full_theta(None if theta is None else np.asarray(theta, dtype=float))
        if full.ndim != 1:
            raise ValueError("to_process takes one parameter vector, not a batch")
        return self.lowered.build(full)

    def with_values(self, theta: Sequence[float]) -> "CompiledProcess":
        """A compiled copy at new free-parameter values (recompiles)."""
        return CompiledProcess(
            self.to_process(theta), self.backend, list(self.kernel.free), None, self.parallel
        )

    # ── characteristic function ─────────────────────────────────────────────
    def char_exponent(self, u: Any, theta: Optional[Any] = None) -> np.ndarray:
        r""":math:`\psi(u;\theta)` with :math:`E[e^{iuX_t}] = e^{t\psi(u)}`."""
        return self.kernel.psi(u, theta)

    def char_exponent_and_grad(
        self, u: Any, theta: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r""":math:`\psi` and :math:`\partial\psi/\partial\theta` (last axis)."""
        return self.kernel.psi_and_grad(u, theta)

    def char_exponent_hessian(
        self, u: Any, theta: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r""":math:`\psi`, gradient and Hessian in :math:`\theta` (last two axes)."""
        return self.kernel.psi_grad_hess(u, theta)

    def log_char(self, u: Any, t: float = 1.0, theta: Optional[Any] = None) -> np.ndarray:
        r""":math:`\log E[e^{iuX_t}] = t\psi(u)` (continuous, no branch jumps)."""
        return float(t) * self.kernel.psi(u, theta)

    def char_func(self, u: Any, t: float = 1.0, theta: Optional[Any] = None) -> np.ndarray:
        r""":math:`E[e^{iuX_t}]`."""
        return np.exp(self.log_char(u, t, theta))

    def log_mgf(self, z: Any, theta: Optional[Any] = None) -> np.ndarray:
        r""":math:`\log E[e^{zX_1}] = \psi(-iz)`, inside the exponential-moment strip."""
        return self.kernel.psi(-1j * np.asarray(z, dtype=complex), theta)

    # ── cumulants ───────────────────────────────────────────────────────────
    def cumulants(self, order: int = 4, theta: Optional[Any] = None) -> np.ndarray:
        r""":math:`\kappa_1..\kappa_{order}` of :math:`X_1` (array; batch-leading)."""
        return self.kernel.cumulants(order, theta)

    def cumulants_and_jacobian(
        self, order: int = 4, theta: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Cumulants and :math:`\partial\kappa_n/\partial\theta_k`."""
        return self.kernel.cumulants_and_jacobian(order, theta)

    # ── sampling ────────────────────────────────────────────────────────────
    def _sampler_theta(self, theta: Optional[Any]) -> np.ndarray:
        return self.full_theta(None if theta is None else np.asarray(theta, dtype=float))

    def sample_at(
        self,
        times: Any,
        rng: Optional[np.random.Generator] = None,
        theta: Optional[Any] = None,
    ) -> np.ndarray:
        r"""Independent exact draws of :math:`X_{t_k}`, one per entry of ``times``."""
        return self.sampler.sample_at(times, rng, self._sampler_theta(theta))

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,  # noqa: N803
        rng: Optional[np.random.Generator] = None,
        theta: Optional[Any] = None,
    ) -> np.ndarray:
        """
        ``(n_paths, n_steps + 1)`` paths over ``[0, T]``, as ``Process.simulate``.

        A ``(B, n_free)`` batch of ``theta`` returns ``(B, n_paths, n_steps + 1)``.
        """
        return self.sampler.simulate(n_steps, n_paths, T, rng, self._sampler_theta(theta))

    def paths(
        self,
        times: Any,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
        theta: Optional[Any] = None,
    ) -> np.ndarray:
        """Paths observed at the non-decreasing ``times`` (any grid)."""
        return self.sampler.paths(times, n_paths, rng, self._sampler_theta(theta))

    def sampler_exact(self, theta: Optional[Any] = None) -> bool:
        """Whether every leaf is drawn by an exact sampler at ``theta``."""
        return self.sampler.exact(self._sampler_theta(theta))

    # ── pricing ─────────────────────────────────────────────────────────────
    def cos_price(
        self,
        spot: float,
        strikes: Any,
        maturities: Any,
        rate: float = 0.0,
        dividend: float = 0.0,
        call: Any = True,
        n_terms: int = 1024,
        width: float = 12.0,
        theta: Optional[Any] = None,
        grad: bool = False,
    ) -> Any:
        """European prices by COS (and their exact Jacobian); see :func:`spxa.compile.pricing.cos_price`."""
        from spxa.compile.pricing import cos_price

        return cos_price(
            self, spot, strikes, maturities, rate, dividend, call, n_terms, width, theta, grad
        )

    # ── introspection ───────────────────────────────────────────────────────
    def source(self, what: str = "psi", theta: Optional[Any] = None) -> str:
        """Generated source for ``"psi"``, ``"grad"`` or ``"hess"``."""
        return self.kernel.source(what, theta)

    def story(self, theta: Optional[Sequence[float]] = None) -> str:
        """The expression with parameter values, e.g. ``(BM(...) @ Gamma(...))``."""
        full = self.full_theta(None if theta is None else np.asarray(theta, dtype=float))
        return self.lowered.story(full)

    def __repr__(self) -> str:
        return (
            f"CompiledProcess({self.story()}, backend={self.backend!r}, "
            f"free={len(self.names)}/{len(self.all_names)})"
        )


def compile_process(
    process: Any,
    backend: str = "numpy",
    free: Optional[Sequence[Union[str, int]]] = None,
    fixed: Optional[Dict[str, float]] = None,
    parallel: bool = False,
    allow_opaque: bool = True,
) -> CompiledProcess:
    """Compile ``process``; see :class:`CompiledProcess`."""
    return CompiledProcess(process, backend, free, fixed, parallel, allow_opaque)
