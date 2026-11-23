r"""
The quantities laws talk about, computed from a process.

Each function reads one well-defined quantity off a
:class:`~spxa.core.process.Process`:

- :func:`cumulants`, :func:`mean` and :func:`variance` of :math:`X_t`, from
  the process's own ``cumulants`` (per unit time, multiplied by
  :math:`t`).
- :func:`char_func`: :math:`\varphi_t(u) = E[e^{iuX_t}]`, from the process's
  own ``char_func``.
- :func:`char_exponent`: :math:`\psi(u)` with :math:`\varphi_t(u) = e^{t\psi(u)}`,
  computed from the Lévy–Khintchine triplet :math:`(b, \sigma^2, \nu)`
  (for a subordinated process, by composing exponents),

  .. math::
      \psi(u) = ibu - \tfrac12\sigma^2u^2
      + \int\big(e^{iux} - 1 - iux\mathbb 1_{|x|\le1}\big)\,\nu(dx).

  It is defined directly, not as :math:`\log\varphi`, so there is no
  branch ambiguity, and it accepts complex :math:`u`.
- :func:`laplace_exponent`: :math:`K(z) = \log E[e^{zX_1}] = \psi(-iz)`, for
  :math:`z` in the strip of finite exponential moments. For a
  subordinator this includes the whole half-plane
  :math:`\operatorname{Re} z \le 0`.

These are deliberately thin. A law that compares, say, :func:`char_func`
of a sum with the product of :func:`char_func` of its parts tests the
process implementations, not this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spxa.core.process import Process


def cumulants(X: Process, order: int = 4, t: float = 1.0) -> np.ndarray:
    r""":math:`\kappa_1, \dots, \kappa_{order}` of :math:`X_t`, as an array."""
    k = X.cumulants(order)
    return np.array([float(k[n]) for n in range(1, order + 1)]) * t


def mean(X: Process, t: float = 1.0) -> float:
    """:math:`E[X_t]`."""
    return float(cumulants(X, 1, t)[0])


def variance(X: Process, t: float = 1.0) -> float:
    r""":math:`\mathrm{Var}(X_t)`."""
    return float(cumulants(X, 2, t)[1])


def char_func(X: Process, u: Any, t: float = 1.0) -> np.ndarray:
    r""":math:`E[e^{iuX_t}]` at each frequency in ``u`` (always an array)."""
    u_arr = np.atleast_1d(np.asarray(u, dtype=float))
    return np.atleast_1d(np.asarray(X.char_func(u_arr, t), dtype=complex))


def char_exponent(X: Process, u: Any) -> np.ndarray:
    r""":math:`\psi(u)`, without a logarithm (complex ``u`` allowed).

    This comes from the triplet, or for subordinated processes from
    composing the exponents of the parts.
    """
    return X._char_exponent(np.atleast_1d(np.asarray(u, dtype=complex)))


def laplace_exponent(X: Process, z: Any) -> np.ndarray:
    r""":math:`K(z) = \log E[e^{zX_1}] = \psi(-iz)`."""
    return char_exponent(X, -1j * np.atleast_1d(np.asarray(z, dtype=complex)))


def mgf(X: Process, z: Any, t: float = 1.0) -> np.ndarray:
    r""":math:`E[e^{zX_t}] = e^{tK(z)}`."""
    return np.exp(t * laplace_exponent(X, z))


def exponential_moment_strip(X: Process) -> tuple[float, float]:
    r"""
    The open interval :math:`(z_-, z_+)` of real :math:`z` with :math:`E[e^{zX_1}] < \infty`.

    On this strip the cumulant generating function :math:`K(z) =
    \log E[e^{zX_1}]` is analytic. Its Taylor coefficients at 0 are
    :math:`\kappa_n/n!`, which is what makes cumulants recoverable from the
    characteristic function. The endpoints are closed forms for the zoo
    primitives:

    ===========================  ==============================================
    Drift, BM, Poisson, Merton,  :math:`(-\infty, \infty)`
    normal or uniform CP
    Gamma(a, b)                  :math:`(-\infty, b)`
    IG(mu, lam)                  :math:`(-\infty, \lambda/2\mu^2)`
    tempered stable (lam)        :math:`(-\infty, \lambda)`
    negative binomial (p)        :math:`(-\infty, -\log p)`
    VG                           the roots of :math:`1 - \theta\nu z - \sigma^2\nu z^2/2`
    NIG, GH                      :math:`(-\alpha-\beta, \alpha-\beta)`
    Meixner                      :math:`((-\pi-\beta)/\alpha, (\pi-\beta)/\alpha)`
    CGMY                         :math:`(-G, M)`
    Kou, double-exponential CP   :math:`(-\eta_2, \eta_1)`
    alpha-stable                 :math:`\{0\}`, returned as ``(0.0, 0.0)``
    ===========================  ==============================================

    Composites: sums intersect strips, :math:`cX` maps :math:`(z_-, z_+)` to
    :math:`(z_-/c, z_+/c)` (swapped for :math:`c < 0`). For :math:`X\circ T`,
    the strip is the interval around 0 where :math:`K_X(z) < z_+^T`. It is
    found by bisection, using the convexity of :math:`K_X`.

    Raises
    ------
    NotImplementedError
        For processes not listed here.
    """
    from spxa.core.process import _ComposedProcess, _ScaledProcess
    from spxa.zoo.levy import (
        CGMY,
        GH,
        NIG,
        AlphaStable,
        BrownianMotion,
        CompoundPoisson,
        Drift,
        GammaProcess,
        InverseGaussianProcess,
        KouJumpDiffusion,
        MeixnerProcess,
        MertonJumpDiffusion,
        NegativeBinomialProcess,
        PoissonProcess,
        TemperedStable,
        VarianceGamma,
    )
    from spxa.zoo.levy.compound_poisson import (
        DiscreteJump,
        DoubleExponentialJump,
        ExponentialJump,
        NormalJump,
        UniformJump,
    )

    inf = float("inf")
    if isinstance(X, _ScaledProcess):
        lo, hi = exponential_moment_strip(X._base)
        c = X._scale
        return (lo / c, hi / c) if c > 0 else (hi / c, lo / c)
    if isinstance(X, _ComposedProcess):
        if X._operation == "+":
            a, b = exponential_moment_strip(X._left), exponential_moment_strip(X._right)
            return max(a[0], b[0]), min(a[1], b[1])
        return _subordinated_strip(X._left, exponential_moment_strip(X._right)[1])
    if isinstance(X, (Drift, BrownianMotion, PoissonProcess, MertonJumpDiffusion)):
        return -inf, inf
    if isinstance(X, GammaProcess):
        return -inf, float(X.b)
    if isinstance(X, InverseGaussianProcess):
        return -inf, float(X.lam / (2 * X.mu**2))
    if isinstance(X, TemperedStable):
        return -inf, float(X.lam)
    if isinstance(X, NegativeBinomialProcess):
        return -inf, float(-np.log(X.p))
    if isinstance(X, VarianceGamma):
        s2n, tn = X.sigma**2 * X.nu, X.theta * X.nu
        root = float(np.sqrt(tn**2 + 2 * s2n))
        return (-tn - root) / s2n, (-tn + root) / s2n
    if isinstance(X, NIG):
        return -X.alpha - X.beta, X.alpha - X.beta
    if isinstance(X, GH):
        return -X._alpha - X._beta, X._alpha - X._beta
    if isinstance(X, MeixnerProcess):
        return (-np.pi - X.beta) / X.alpha, (np.pi - X.beta) / X.alpha
    if isinstance(X, CGMY):
        return -float(X.G), float(X.M)
    if isinstance(X, KouJumpDiffusion):
        return -float(X.eta2), float(X.eta1)
    if isinstance(X, AlphaStable):
        return 0.0, 0.0
    if isinstance(X, CompoundPoisson):
        d = X.jump_dist
        if isinstance(d, (NormalJump, UniformJump, DiscreteJump)):
            return -inf, inf
        if isinstance(d, ExponentialJump):
            return -inf, float(d.rate)
        if isinstance(d, DoubleExponentialJump):
            return -float(d.eta2), float(d.eta1)
    raise NotImplementedError(f"no exponential moment strip known for {X!r}")


def _subordinated_strip(X: Process, clock_upper: float) -> tuple[float, float]:
    """Interval around 0 inside X's strip where the real CGF of X is below ``clock_upper``."""
    lo, hi = exponential_moment_strip(X)
    if not np.isfinite(clock_upper):
        return lo, hi

    def k(z: float) -> float:
        return float(laplace_exponent(X, z)[0].real)

    def edge(bound: float, sign: float) -> float:
        # largest |z| on one side with K_X(z) < clock_upper (K_X convex, K_X(0) = 0)
        inner = 0.0
        if np.isfinite(bound):
            outer = bound * (1 - 1e-12)
            if k(outer) < clock_upper:
                return bound
        else:
            outer = sign
            while k(outer) < clock_upper:
                inner, outer = outer, 2 * outer
                if abs(outer) > 1e6:
                    return sign * float("inf")
        for _ in range(80):
            mid = 0.5 * (inner + outer)
            if k(mid) < clock_upper:
                inner = mid
            else:
                outer = mid
        return inner

    return edge(lo, -1.0), edge(hi, 1.0)
