r"""
Monte Carlo parameter gradients with standard errors (labelled STOCHASTIC).

Prices are :math:`V(\theta) = e^{-rT}E[f(S_T)]` with the risk-neutral
convention of :mod:`spxa.graph.autodiff.fourier`,
:math:`S_T = S_0\exp((r - q)T + \omega(\theta)T + X_T)`,
:math:`\omega = -\psi(-i)`. Three unbiased estimators are offered; each
returns per-path contributions, so the standard error is the sample standard
deviation over :math:`\sqrt{n}`.

Pathwise (reparameterisation), :func:`pathwise_gradient`
    :math:`X_T = g(\theta, \xi)` with :math:`\xi` independent of
    :math:`\theta`, differentiated per path with forward-mode jets. Gamma and
    inverse-Gaussian variates use *implicit reparameterisation* (Figurnov,
    Mohamed & Mnih 2018): for :math:`G = F^{-1}(U;\theta)`,
    :math:`\partial_\theta G = -\partial_\theta F(G;\theta)/f(G;\theta)`, with
    :math:`\partial_a P(a, x)` of the regularised incomplete gamma function
    from its positive series and the inverse-Gaussian CDF differentiated in
    closed form. Poisson counts are not differentiable in their rate; their
    rate enters through the score :math:`N/\lambda - T` (a hybrid
    pathwise/likelihood-ratio estimator, still unbiased). Needs a payoff that
    is Lipschitz in :math:`S_T` (calls, puts); payoffs marked
    ``lipschitz=False`` are refused.
Likelihood ratio, :func:`likelihood_ratio_gradient`
    With :math:`Y = X_T + \omega T` the log-return net of :math:`(r-q)T`,
    :math:`Y` has density :math:`p_T(y - \omega(\theta)T;\theta)`, so
    :math:`\partial_\theta V = e^{-rT}E[f(S_T)\,\partial_\theta\log p_T(Y - \omega T;\theta)]`;
    the score (martingale correction included) is exact, from jets through
    the closed densities of Brownian motion, Gamma, inverse Gaussian, variance
    gamma (Madan, Carr & Chang 1998) and NIG. No payoff derivative is
    needed, so digitals are fine.
Malliavin, :func:`malliavin_greeks`
    For :math:`X = \mu t + \sigma W_t + J_t` with :math:`J` independent of
    :math:`W` and :math:`\sigma`: delta weight :math:`W_T/(S_0\sigma T)` and
    vega weight :math:`W_T^2/(\sigma T) - W_T - 1/\sigma` (Fournié et al.
    1999; Benth, Di Nunno, Løkka, Øksendal & Proske 2003 for jumps). Works
    for discontinuous payoffs.

Common random numbers are drawn once per call from ``rng``; all estimators
are reproducible for a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.special as sps

from spxa.graph.autodiff.exactness import GradientExactness
from spxa.graph.autodiff.jet import Jet, is_jet, value_of, variable_vector
from spxa.graph.autodiff.process import (
    DiffProcess,
    UnsupportedProcessError,
    _Leaf,
    _Node,
    _Scale,
    _Subordinate,
    _Sum,
)

__all__ = [
    "MCGradient",
    "Payoff",
    "call",
    "put",
    "digital_call",
    "pathwise_gradient",
    "likelihood_ratio_gradient",
    "malliavin_greeks",
    "gamma_shape_derivative",
    "inverse_gaussian_reparam",
]


# ── payoffs ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Payoff:
    """A payoff :math:`f(S_T)` (vectorised, jet-safe) and whether it is Lipschitz."""

    fn: Callable[[Any], Any]
    lipschitz: bool
    label: str

    def __call__(self, s: Any) -> Any:
        return self.fn(s)


def call(strike: float) -> Payoff:
    return Payoff(lambda s: np.maximum(s - strike, 0.0), True, f"call({strike:g})")


def put(strike: float) -> Payoff:
    return Payoff(lambda s: np.maximum(strike - s, 0.0), True, f"put({strike:g})")


def digital_call(strike: float) -> Payoff:
    return Payoff(
        lambda s: (np.asarray(value_of(s)) > strike).astype(float), False, f"digital({strike:g})"
    )


@dataclass
class MCGradient:
    """A Monte Carlo price and gradient with standard errors."""

    value: float
    value_se: float
    grad: np.ndarray
    grad_se: np.ndarray
    names: Tuple[str, ...]
    method: str
    n_paths: int
    exactness: GradientExactness = GradientExactness.STOCHASTIC
    notes: Dict[str, Any] = field(default_factory=dict)

    def z_scores(self, reference: np.ndarray) -> np.ndarray:
        """``(grad - reference) / grad_se`` (zero where the standard error is zero)."""
        se = np.where(self.grad_se > 0, self.grad_se, 1.0)
        return np.where(self.grad_se > 0, (self.grad - reference) / se, 0.0)

    def __story__(self) -> str:
        return (
            f"{self.method} Monte Carlo gradient over {self.n_paths} paths: unbiased, "
            f"standard errors reported (STOCHASTIC); value {self.value:.6g} +- {self.value_se:.2g}"
        )


def _summarise(per_value: np.ndarray, per_grad: np.ndarray) -> Tuple[float, float, Any, Any]:
    n = per_value.shape[0]
    v = float(np.mean(per_value))
    vse = float(np.std(per_value, ddof=1) / np.sqrt(n))
    g = np.mean(per_grad, axis=0)
    gse = np.std(per_grad, axis=0, ddof=1) / np.sqrt(n)
    return v, vse, g, gse


# ── implicit reparameterisation ──────────────────────────────────────────────


def _dP_da(a: float, x: np.ndarray) -> np.ndarray:
    r"""
    :math:`\partial_a P(a, x)` of the regularised lower incomplete gamma function.

    From :math:`P(a, x) = \sum_{n\ge0} e^{-x}x^{a+n}/\Gamma(a+n+1)` (all terms
    positive), :math:`\partial_a P = \sum_n t_n(\log x - \psi(a+n+1))`. Terms
    are formed in log space and summed until they are below 1e-17 of the
    running total past the mode at :math:`n \approx x`.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    live = x > 0
    if not np.any(live):
        return out
    xl = x[live]
    res = np.empty_like(xl)
    order = np.argsort(xl)  # chunks of similar x share a short series
    for lo in range(0, xl.size, 8192):
        idx = order[lo : lo + 8192]
        xc = xl[idx]
        lx = np.log(xc)
        n_max = int(np.ceil(np.max(xc) + 12.0 * np.sqrt(np.max(xc)) + 60.0))
        n = np.arange(n_max + 1, dtype=float)[:, None]
        t = np.exp(-xc[None, :] + (a + n) * lx[None, :] - sps.gammaln(a + n + 1.0))
        res[idx] = np.sum(t * (lx[None, :] - sps.digamma(a + n + 1.0)), axis=0)
    out[live] = res
    return out


_GL_NODES, _GL_WEIGHTS = np.polynomial.laguerre.laggauss(96)


def _dG_da_upper(a: float, x: np.ndarray) -> np.ndarray:
    r"""
    :math:`\partial G/\partial a` from the upper tail, for :math:`x > a`.

    With :math:`f` the Gamma(a) density,
    :math:`\partial G/\partial a = \partial_a Q(a, x)/f(x)
    = \int_0^\infty (\log(x+s) - \psi(a))(1+s/x)^{a-1}e^{-s}\,ds`, an
    integral of positive terms with no cancellation. It is computed by
    Gauss--Laguerre after the change of variable :math:`s = u/r`,
    :math:`r = 1 - (a-1)/x`, which absorbs the exponential growth of
    :math:`(1+s/x)^{a-1}`.
    """
    out = np.empty_like(x)
    u = _GL_NODES[:, None]
    for lo in range(0, x.size, 8192):
        xc = x[lo : lo + 8192]
        r = 1.0 - (a - 1.0) / xc
        s = u / r[None, :]
        logw = (a - 1.0) * np.log1p(s / xc[None, :]) - s + u
        integrand = (np.log(xc[None, :] + s) - sps.digamma(a)) * np.exp(logw) / r[None, :]
        out[lo : lo + 8192] = np.sum(_GL_WEIGHTS[:, None] * integrand, axis=0)
    return out


def gamma_shape_derivative(a: float, g: np.ndarray) -> np.ndarray:
    r"""
    :math:`\partial G/\partial a` at fixed :math:`U = P(a, G)` for a unit-scale Gamma(a) variate.

    The lower series of :math:`\partial_a P` is used for
    :math:`G \le a + 1 + 4\sqrt{a}` and the upper-tail integral above (relative
    accuracy ~1e-11 throughout,
    checked against mpmath). :math:`G = 0` (underflow) gets derivative 0.
    """
    g = np.asarray(g, dtype=float)
    out = np.zeros_like(g)
    cut = a + 1.0 + 4.0 * np.sqrt(a)
    lo = (g > 0) & (g <= cut)
    hi = g > cut
    if np.any(lo):
        gl = g[lo]
        logpdf = (a - 1.0) * np.log(gl) - gl - sps.gammaln(a)
        out[lo] = -_dP_da(a, gl) / np.exp(logpdf)
    if np.any(hi):
        out[hi] = _dG_da_upper(a, g[hi])
    return out


def _gamma_variate(shape: Any, scale: Any, size: int, rng: np.random.Generator) -> Any:
    """Gamma(shape, scale) draws; jets in shape/scale get implicit-reparameterisation derivatives."""
    a = float(np.real(value_of(shape)))
    g0 = rng.gamma(a, 1.0, size)
    if is_jet(shape):
        d = gamma_shape_derivative(a, g0)
        g0j = Jet(g0, d[:, None] * np.asarray(np.real(shape.grad))[None, :])
        return g0j * scale
    return g0 * scale


def inverse_gaussian_reparam(mean: Any, shape: Any, x: np.ndarray) -> Any:
    r"""
    ``x`` (drawn from IG(mean, shape) at the parameter values) as a jet in the parameters.

    :math:`F(x) = \Phi(\sqrt{\lambda/x}(x/m - 1)) + e^{2\lambda/m}\Phi(-\sqrt{\lambda/x}(x/m + 1))`
    is differentiated with jets (the second term in log space), and
    :math:`\partial x = -\partial F / f(x)`.
    """
    if not (is_jet(mean) or is_jet(shape)):
        return x
    s = np.sqrt(shape / x)
    F = sps.ndtr(s * (x / mean - 1.0)) + np.exp(
        2.0 * shape / mean + sps.log_ndtr(-s * (x / mean + 1.0))
    )
    mv, lv = float(np.real(value_of(mean))), float(np.real(value_of(shape)))
    pdf = np.sqrt(lv / (2.0 * np.pi * x**3)) * np.exp(-lv * (x - mv) ** 2 / (2.0 * mv**2 * x))
    return Jet(x, -np.real(F.grad) / pdf[:, None])


def _poisson_variate(rate_T: Any, size: int, rng: np.random.Generator) -> Tuple[np.ndarray, Any]:
    """Counts and their log-probability (a jet in the rate) for the score term."""
    lam = float(np.real(value_of(rate_T)))
    N = rng.poisson(lam, size).astype(float)
    logp = N * np.log(rate_T) - rate_T - sps.gammaln(N + 1.0)
    return N, logp


# ── samplers over the DiffProcess tree ────────────────────────────────────────


def _sample(
    node: _Node, th: List[Any], T: float, n: int, rng: np.random.Generator
) -> Tuple[Any, Any]:
    """``X_T`` (jets for pathwise derivatives) and the log-probability of discrete draws."""
    if isinstance(node, _Sum):
        xa, la = _sample(node.a, th, T, n, rng)
        xb, lb = _sample(node.b, th, T, n, rng)
        return xa + xb, la + lb
    if isinstance(node, _Scale):
        x, lp = _sample(node.base, th, T, n, rng)
        return th[node.slot] * x, lp
    if isinstance(node, _Subordinate):
        clock, lp = _sample(node.clock, th, T, n, rng)
        return _at_random_time(node.x, th, clock, n, rng), lp
    if not isinstance(node, _Leaf):  # pragma: no cover - defensive
        raise UnsupportedProcessError(type(node).__name__)
    p = node._p(th)
    name = node.rule.cls_name
    if name == "BrownianMotion":
        mu, sig = p
        return mu * T + sig * np.sqrt(T) * rng.standard_normal(n), 0.0
    if name == "Drift":
        return p[0] * T + np.zeros(n), 0.0
    if name == "GammaProcess":
        a, b = p
        return _gamma_variate(a * T, 1.0 / b, n, rng), 0.0
    if name == "InverseGaussianProcess":
        mu, lam = p
        m, s = mu * T, lam * T * T
        x = rng.wald(float(np.real(value_of(m))), float(np.real(value_of(s))), n)
        return inverse_gaussian_reparam(m, s, x), 0.0
    if name == "VarianceGamma":
        sig, nu, theta = p
        G = _gamma_variate(T / nu, nu, n, rng)
        return theta * G + sig * np.sqrt(G) * rng.standard_normal(n), 0.0
    if name == "NIG":
        alpha, beta, delta, mu = p
        gam = np.sqrt(alpha**2 - beta**2)
        m, s = delta * T / gam, (delta * T) ** 2
        x = rng.wald(float(np.real(value_of(m))), float(np.real(value_of(s))), n)
        Z = inverse_gaussian_reparam(m, s, x)
        return mu * T + beta * Z + np.sqrt(Z) * rng.standard_normal(n), 0.0
    if name == "MertonJumpDiffusion":
        drift, sig, rate, jm, js = p
        N, logp = _poisson_variate(rate * T, n, rng)
        jumps = N * jm + js * np.sqrt(N) * rng.standard_normal(n)
        return drift * T + sig * np.sqrt(T) * rng.standard_normal(n) + jumps, logp
    if name == "PoissonProcess":
        rate, c = p
        N, logp = _poisson_variate(rate * T, n, rng)
        return c * N, logp
    raise UnsupportedProcessError(f"no Monte Carlo sampler for {name}")


def _at_random_time(
    node: _Node, th: List[Any], clock: Any, n: int, rng: np.random.Generator
) -> Any:
    if isinstance(node, _Sum):
        return _at_random_time(node.a, th, clock, n, rng) + _at_random_time(
            node.b, th, clock, n, rng
        )
    if isinstance(node, _Leaf) and node.rule.cls_name == "BrownianMotion":
        mu, sig = node._p(th)
        return mu * clock + sig * np.sqrt(clock) * rng.standard_normal(n)
    if isinstance(node, _Leaf) and node.rule.cls_name == "Drift":
        return node._p(th)[0] * clock
    raise UnsupportedProcessError("subordination is sampled for Brownian motion (with drift) only")


def _setup(dp: DiffProcess, theta: Any) -> Tuple[Jet, List[Any], Any]:
    th = variable_vector(dp.values if theta is None else theta)
    full = dp._full(th)
    omega = -np.real(dp.char_exponent(np.array(-1j), th))
    return th, full, omega


# ── estimators ──────────────────────────────────────────────────────────────


def pathwise_gradient(
    dp: DiffProcess,
    payoff: Payoff,
    spot: float,
    maturity: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    n_paths: int = 100_000,
    rng: Optional[np.random.Generator] = None,
    theta: Any = None,
) -> MCGradient:
    """Pathwise (+ Poisson score) gradient of ``e^{-rT} E[payoff(S_T)]``."""
    if not payoff.lipschitz:
        raise ValueError(
            f"{payoff.label} is not Lipschitz: the pathwise estimator would be biased; "
            "use likelihood_ratio_gradient or malliavin_greeks"
        )
    rng = rng if rng is not None else np.random.default_rng(0)
    T = float(maturity)
    th, full, omega = _setup(dp, theta)
    X, logp = _sample(dp._tree, full, T, int(n_paths), rng)
    S = spot * np.exp((rate - dividend) * T + omega * T + X)
    disc = np.exp(-rate * T)
    V = disc * payoff(S)
    n = th.n
    vv = np.asarray(np.real(value_of(V)), float) * np.ones(n_paths)
    g = np.asarray(np.real(V.grad), float) if isinstance(V, Jet) else np.zeros((n_paths, n))
    g = np.broadcast_to(g, (n_paths, n)).copy()
    if isinstance(logp, Jet):
        g += vv[:, None] * np.asarray(np.real(logp.grad), float)
    v, vse, gm, gse = _summarise(vv, g)
    return MCGradient(v, vse, gm, gse, dp.names, "pathwise", int(n_paths))


def _log_density(name: str, p: Tuple[Any, ...], x: np.ndarray, T: float) -> Any:
    """Exact log-density of ``X_T`` for single families (jets in the parameters)."""
    if name == "BrownianMotion":
        mu, sig = p
        v = sig * sig * T
        return -0.5 * np.log(2.0 * np.pi * v) - (x - mu * T) ** 2 / (2.0 * v)
    if name == "GammaProcess":
        a, b = p
        k = a * T
        return k * np.log(b) - sps.gammaln(k) + (k - 1.0) * np.log(x) - b * x
    if name == "InverseGaussianProcess":
        mu, lam = p
        m, s = mu * T, lam * T * T
        return 0.5 * np.log(s / (2.0 * np.pi * x**3)) - s * (x - m) ** 2 / (2.0 * m * m * x)
    if name == "VarianceGamma":
        # Madan, Carr & Chang (1998), eq. (23)
        sig, nu, theta = p
        k = T / nu
        c2 = 2.0 * sig * sig / nu + theta * theta
        r = np.sqrt(x * x * c2) / (sig * sig)
        order = k - 0.5
        logK = np.log(sps.kve(order, r)) - r
        return (
            np.log(2.0)
            + theta * x / (sig * sig)
            - k * np.log(nu)
            - 0.5 * np.log(2.0 * np.pi)
            - np.log(sig)
            - sps.gammaln(k)
            + (0.5 * k - 0.25) * np.log(x * x / c2)
            + logK
        )
    if name == "NIG":
        alpha, beta, delta, mu = p
        d, m = delta * T, mu * T
        gam = np.sqrt(alpha**2 - beta**2)
        q = np.sqrt(d * d + (x - m) ** 2)
        return (
            np.log(alpha * d / np.pi)
            + np.log(sps.kve(1.0, alpha * q))
            - alpha * q
            - np.log(q)
            + d * gam
            + beta * (x - m)
        )
    raise UnsupportedProcessError(f"no closed density for {name}")


def likelihood_ratio_gradient(
    dp: DiffProcess,
    payoff: Payoff,
    spot: float,
    maturity: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    n_paths: int = 100_000,
    rng: Optional[np.random.Generator] = None,
    theta: Any = None,
) -> MCGradient:
    """Likelihood-ratio gradient (any payoff) for single families with closed densities."""
    tree = dp._tree
    if not isinstance(tree, _Leaf):
        raise UnsupportedProcessError("the likelihood ratio needs a single family (closed density)")
    rng = rng if rng is not None else np.random.default_rng(0)
    T = float(maturity)
    th, full, omega = _setup(dp, theta)
    vals = [float(np.real(value_of(v))) for v in full]
    X, _ = _sample(tree, vals, T, int(n_paths), rng)  # plain draws at the values
    X = np.asarray(X, float)
    w0 = float(np.real(value_of(omega)))
    Y = X + w0 * T  # log-return net of (r - q)T; its density is p_T(y - omega T)
    S = spot * np.exp((rate - dividend) * T + Y)
    V = np.exp(-rate * T) * np.asarray(payoff(S), float)
    logp = _log_density(tree.rule.cls_name, tree._p(full), Y - omega * T, T)
    score = np.asarray(np.real(logp.grad), float) if isinstance(logp, Jet) else 0.0
    g = V[:, None] * score
    v, vse, gm, gse = _summarise(V, np.broadcast_to(g, (V.size, th.n)))
    return MCGradient(v, vse, gm, gse, dp.names, "likelihood_ratio", int(n_paths))


def malliavin_greeks(
    dp: DiffProcess,
    payoff: Payoff,
    spot: float,
    maturity: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    n_paths: int = 100_000,
    rng: Optional[np.random.Generator] = None,
) -> MCGradient:
    r"""
    Delta and vega (w.r.t. the Brownian :math:`\sigma`) by Malliavin weights.

    ``dp`` must be a sum whose Brownian part is a single
    :class:`~spxa.zoo.levy.brownian.BrownianMotion` leaf or the diffusion of
    a Merton jump diffusion; the rest is sampled independently. Returns
    ``grad = [delta, vega]`` with ``names = ("delta", "vega")``. The vega is
    :math:`\partial V/\partial\sigma` with the martingale correction adjusted
    (:math:`\partial\omega/\partial\sigma = -\sigma`).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    T = float(maturity)
    leaves = _top_leaves(dp._tree)
    bm = [lf for lf in leaves if lf.rule.cls_name in ("BrownianMotion", "MertonJumpDiffusion")]
    if len(bm) != 1:
        raise UnsupportedProcessError("need exactly one Brownian component at the top level")
    vals = [float(v) for v in dp._full(None)]
    lf = bm[0]
    sig = vals[lf.slots[1]]
    if sig <= 0:
        raise ValueError("the Brownian volatility must be positive")
    n = int(n_paths)
    W = np.sqrt(T) * rng.standard_normal(n)
    # the rest: everything except the Brownian increment
    rest = np.zeros(n)
    for other in leaves:
        if other is lf:
            continue
        x, _ = _sample(other, vals, T, n, rng)
        rest = rest + np.asarray(x, float)
    p = [vals[i] for i in lf.slots]
    if lf.rule.cls_name == "BrownianMotion":
        rest = rest + p[0] * T
    else:  # Merton: drift + jumps, without its diffusion
        drift, _, lam, jm, js = p
        N = rng.poisson(lam * T, n).astype(float)
        rest = rest + drift * T + N * jm + js * np.sqrt(N) * rng.standard_normal(n)
    omega = float(-np.real(dp.char_exponent(np.array(-1j))))
    S = spot * np.exp((rate - dividend) * T + omega * T + sig * W + rest)
    V = np.exp(-rate * T) * np.asarray(payoff(S), float)
    delta = V * W / (spot * sig * T)
    vega = V * (W * W / (sig * T) - W - 1.0 / sig)
    v, vse, gm, gse = _summarise(V, np.stack([delta, vega], axis=1))
    return MCGradient(v, vse, gm, gse, ("delta", "vega"), "malliavin", n)


def _top_leaves(node: _Node) -> List[_Leaf]:
    if isinstance(node, _Sum):
        return _top_leaves(node.a) + _top_leaves(node.b)
    if isinstance(node, _Leaf):
        return [node]
    raise UnsupportedProcessError("Malliavin weights need a plain sum of families")
