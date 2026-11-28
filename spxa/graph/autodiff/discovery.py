r"""
Exact gradients and Jacobians for :mod:`spxa.discovery` fits.

:mod:`spxa.discovery` optimises in the unconstrained coordinates
:math:`x` of :meth:`~spxa.discovery.expr.Model.from_free`. This module
evaluates the same objectives with :math:`x` a forward-mode
:class:`~spxa.graph.autodiff.jet.Jet`, through the families' closed-form
:math:`K(z)` and the composition rules, so the gradients are exact
derivatives of the objective *as implemented*:

``options_residual_jacobian``
    Jacobian of the vega-weighted price residuals of
    :func:`~spxa.discovery.options.fit_options` (COS, same interval and
    terms). Where the discovery pricer floors a put at zero the row is zero,
    which is the derivative of the floored price.
``likelihood_value_and_grad``
    The binned COS likelihood of :func:`~spxa.discovery.fit.fit_structure`.
    Its probabilities are clipped to :math:`[0, 1]`, made monotone with a
    running maximum and floored; the gradient follows the active branch of
    each (zero where a clip or the floor is active, the running-maximum's
    source element otherwise), i.e. the derivative almost everywhere.
``likelihood_value_grad_hess``
    The same likelihood with its exact Hessian, from second-order jets. The
    Laplace evidence of :mod:`spxa.discovery.bayes` uses it.
``cumulant_value_and_grad``
    The cumulant objective, including the profiled drift.

Cumulants are exact Taylor coefficients of :math:`K(z)` with jet coefficients.
The discretisation (COS interval, bins) depends on parameter values only.
"""

from __future__ import annotations

from math import factorial
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from spxa.discovery.expr import Model, Node
from spxa.graph.autodiff.jet import Jet, is_jet, stack, value_of, variable_vector
from spxa.graph.autodiff.taylor import taylor_variable

__all__ = [
    "FreeModel",
    "options_residual_jacobian",
    "likelihood_value_and_grad",
    "likelihood_value_grad_hess",
    "cumulant_value_and_grad",
]


def _raw_params(structure: Node, x: Any) -> List[Tuple[Any, ...]]:
    """Natural parameters per leaf at free coordinates ``x`` (jets pass through)."""
    out: List[Tuple[Any, ...]] = []
    i = 0
    for lf in structure.leaves():
        n = lf.family.n_params
        xs = x[i : i + n]
        out.append(tuple(lf.family.from_free_fn(xs)))
        i += n
    return out


class FreeModel:
    """
    A discovery structure as a differentiable process in its free coordinates.

    Implements the interface the pricers of :mod:`spxa.graph.autodiff.fourier`
    need (``names``, ``values``, ``analytic``, ``char_exponent(u, theta)``,
    ``cumulants(order, theta)``).
    """

    analytic = True

    def __init__(self, structure: Node, x: Sequence[float]) -> None:
        self.structure = structure
        self.x = np.asarray(x, dtype=float).copy()
        Model.from_free(structure, self.x)  # validates
        names = []
        for k, lf in enumerate(structure.leaves()):
            names += [f"{lf.family.name}[{k}].free{j}" for j in range(lf.family.n_params)]
        self.names = tuple(names)

    @property
    def values(self) -> np.ndarray:
        return self.x.copy()

    def _theta(self, theta: Any) -> Any:
        if theta is None:
            return self.x
        if is_jet(theta):
            return theta
        return np.asarray(theta, dtype=float)

    def log_mgf(self, z: Any, theta: Any = None) -> Any:
        th = self._theta(theta)
        if not is_jet(th):
            return Model.from_free(self.structure, th).log_mgf(z)
        params = _raw_params(self.structure, th)
        zz = z if (is_jet(z) or hasattr(z, "coeffs")) else np.asarray(z, dtype=complex)
        return self.structure._log_mgf(zz, params)

    def char_exponent(self, u: Any, theta: Any = None) -> Any:
        uu = u if (is_jet(u) or hasattr(u, "coeffs")) else np.asarray(u, dtype=complex)
        return self.log_mgf(1j * uu, theta)

    def cumulants(self, order: int, theta: Any = None) -> Any:
        th = self._theta(theta)
        if not is_jet(th):
            return Model.from_free(self.structure, th).cumulants(order)
        z = taylor_variable(0.0, order)
        params = _raw_params(self.structure, th)
        K = self.structure._log_mgf(z, params)
        return stack([np.real(factorial(n) * K.coeffs[n]) for n in range(1, order + 1)])


# ── option fits ─────────────────────────────────────────────────────────────


def options_residual_jacobian(
    structure: Node, quotes: Any, x: Sequence[float], vega: np.ndarray, n_terms: int = 1024
) -> np.ndarray:
    """Exact ``d resid / d x`` for :func:`~spxa.discovery.options.fit_options` residuals."""
    from spxa.graph.autodiff.fourier import cos_price_surface

    fm = FreeModel(structure, x)
    th = variable_vector(fm.values)
    K, T, call = quotes.strikes, quotes.maturities, quotes.call
    p = cos_price_surface(
        fm, quotes.spot, K, T, quotes.rate, quotes.dividend, call, n_terms=n_terms, theta=th
    )
    # discovery floors the put at 0 before parity: zero derivative there
    forward = quotes.spot * np.exp(-quotes.dividend * T) - K * np.exp(-quotes.rate * T)
    put = np.real(value_of(p)) - np.where(call, forward, 0.0)
    g = np.real(p.grad) if isinstance(p, Jet) else np.zeros((len(quotes), fm.values.size))
    J = np.where((put < 0.0)[:, None], 0.0, g)
    return J / vega[:, None]


# ── likelihood ──────────────────────────────────────────────────────────────


def _running_max_source(F: np.ndarray) -> np.ndarray:
    """Index whose value ``np.maximum.accumulate`` reports at each position (latest tie)."""
    idx = np.arange(F.size)
    run = np.maximum.accumulate(F)
    is_src = F >= run
    return np.maximum.accumulate(np.where(is_src, idx, 0))


def _binned_cdf(
    structure: Node, binned: Any, x: Sequence[float], order: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Clipped, monotone COS bin CDF with its derivatives on the active branches.

    Returns the values at the bin edges (including the end points 0 and 1),
    their gradient ``(m, n)`` and, for ``order=2``, their Hessian ``(m, n, n)``.
    """
    fm = FreeModel(structure, x)
    if hasattr(binned, "cos_grid"):
        A, B, u, sine = binned.cos_grid(Model.from_free(structure, fm.values))
    else:
        A, B, u, sine = binned.a, binned.b, binned.u, binned.sine
    th = variable_vector(fm.values, order=order)
    psi = fm.char_exponent(u, th)
    phi = np.exp(binned.dt * psi) * np.exp(-0.5 * (binned.bandwidth * u) ** 2)
    coef = 2.0 / (B - A) * np.real(phi * np.exp(-1j * u * A))
    inner = binned.edges[1:-1]
    F_in = (inner - A) / (B - A) + sine @ coef[1:]
    n = fm.values.size
    Fv = np.concatenate([[0.0], np.real(value_of(F_in)), [1.0]])
    Fg = np.concatenate([np.zeros((1, n)), np.real(F_in.grad), np.zeros((1, n))])
    Fh = None
    if order == 2:
        Fh = np.concatenate([np.zeros((1, n, n)), np.real(F_in.hess), np.zeros((1, n, n))])
    clipped = np.clip(Fv, 0.0, 1.0)
    outside = (Fv < 0.0) | (Fv > 1.0)
    Fg = np.where(outside[:, None], 0.0, Fg)
    src = _running_max_source(clipped)
    if Fh is not None:
        Fh = np.where(outside[:, None, None], 0.0, Fh)[src]
    return clipped[src], Fg[src], Fh


def likelihood_value_and_grad(
    structure: Node, binned: Any, x: Sequence[float], floor: float = 1e-300
) -> Tuple[float, np.ndarray]:
    """Negative binned log-likelihood and its exact gradient in the free coordinates."""
    Fa, Fg, _ = _binned_cdf(structure, binned, x, order=1)
    p = np.diff(Fa)
    pg = np.diff(Fg, axis=0)
    floored = p <= floor
    p = np.maximum(p, floor)
    pg = np.where(floored[:, None], 0.0, pg)
    counts = binned.counts
    val = -float(counts @ np.log(p))
    grad = -(counts / p) @ pg
    return val, np.asarray(grad, dtype=float)


def likelihood_value_grad_hess(
    structure: Node, binned: Any, x: Sequence[float], floor: float = 1e-300
) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""Negative binned log-likelihood, its exact gradient and its exact Hessian.

    With bin probabilities :math:`p_j(x)` and smoothed counts :math:`c_j`, the
    objective is :math:`f = -\sum_j c_j \log p_j` and

    .. math::
        \nabla^2 f = \sum_j c_j \Bigl(\frac{\nabla p_j \nabla p_j^\top}{p_j^2}
        - \frac{\nabla^2 p_j}{p_j}\Bigr),

    with :math:`\nabla p_j` and :math:`\nabla^2 p_j` carried exactly by
    second-order jets through :math:`\psi`, the COS series and the bin
    differences. As for the gradient, a clipped, floored or running-maximum
    bin contributes the derivative of its active branch. The Hessian is
    symmetrised to remove rounding asymmetry.
    """
    Fa, Fg, Fh = _binned_cdf(structure, binned, x, order=2)
    assert Fh is not None
    p = np.diff(Fa)
    pg = np.diff(Fg, axis=0)
    ph = np.diff(Fh, axis=0)
    floored = p <= floor
    p = np.maximum(p, floor)
    pg = np.where(floored[:, None], 0.0, pg)
    ph = np.where(floored[:, None, None], 0.0, ph)
    counts = binned.counts
    val = -float(counts @ np.log(p))
    w = counts / p
    grad = -w @ pg
    hess = np.einsum("j,ja,jb->ab", w / p, pg, pg) - np.einsum("j,jab->ab", w, ph)
    hess = 0.5 * (hess + hess.T)
    return val, np.asarray(grad, dtype=float), np.asarray(hess, dtype=float)


# ── cumulant objective ────────────────────────────────────────────────────────


def cumulant_value_and_grad(
    structure: Node,
    target: np.ndarray,
    x: Sequence[float],
    dt: float,
    weights: Optional[np.ndarray] = None,
    drift_at: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """
    The cumulant objective of :func:`~spxa.discovery.fit.fit_structure` and its gradient.

    With ``drift_at`` the drift coordinate is profiled (set so that
    :math:`\\kappa_1` matches ``target[0]``), so its gradient entry is zero.
    """
    fm = FreeModel(structure, x)
    n = fm.values.size
    th = variable_vector(fm.values)
    if drift_at is not None:
        e = np.zeros(n)
        e[drift_at] = 1.0
        base = th * (1.0 - e)
        k0 = fm.cumulants(1, base)
        k1 = fm.cumulants(1, base + e)
        xd = (target[0] - k0[0]) / (k1[0] - k0[0])
        th = base + xd * e
    k = fm.cumulants(4, th)
    if weights is None:

        def std(c: Any) -> List[Any]:
            var = c[1] * dt
            return [var, c[2] * dt / var**1.5, c[3] * dt / var**2]

        m, t = std(k), std(np.asarray(target, dtype=float))
        scale = [abs(t[0]), max(abs(t[1]), 1.0), max(abs(t[2]), 1.0)]
        total: Any = sum(((m[i] - t[i]) / scale[i]) ** 2 for i in range(3))
    else:
        r = k - np.asarray(target, dtype=float)
        total = sum(weights[i] * r[i] ** 2 for i in range(1, 4))
    val = float(np.real(value_of(total)))
    grad = np.real(total.grad) if isinstance(total, Jet) else np.zeros(n)
    if drift_at is not None:
        grad = np.array(grad, dtype=float)
        grad[drift_at] = 0.0
    return val, np.asarray(grad, dtype=float)
