r"""Kalman-filter likelihood and Gaussian quasi-maximum-likelihood fitting of CARMA processes.

A CARMA(p, q) process (:class:`~spxa.zoo.beyond.carma.CARMAProcess`) observed
at times :math:`t_0 < t_1 < \dots` is a linear state-space model:

.. math::
    X_{k+1} = F_k X_k + W_k, \qquad Y_k = \mu + b^\top X_k, \qquad
    F_k = e^{A\Delta_k}, \quad \mathrm{Var}(W_k) = \sigma^2\big(\Sigma - F_k\Sigma F_k^\top\big),

with :math:`\sigma^2 = \kappa_2(L_1)`, :math:`\Sigma` the unit stationary state
covariance and :math:`X_0` drawn from the stationary law. The Kalman filter
computes the one-step innovations :math:`e_k` and their variances
:math:`S_k`. The Gaussian log-likelihood is

.. math::
    \ell = -\tfrac12\sum_k\Big(\log(2\pi S_k) + e_k^2/S_k\Big),

which is the exact likelihood when the driver is a Brownian motion. For any
other Lévy driver with finite variance it is a quasi-likelihood, and its
maximiser is consistent and asymptotically normal (Schlemm & Stelzer 2012).
The second-order structure, which is all the filter uses, depends on the
driver only through :math:`\sigma^2`.

**Speed.** On an even grid, the covariance recursion does not depend on the
data and converges geometrically to a steady state. :func:`carma_kalman_filter`
runs the time-varying filter only until then. After that, the filter is a
fixed linear recursion. Its modes are run with ``scipy.signal.lfilter``, so one
likelihood evaluation costs :math:`O(np)` vectorised operations.

**Fitting.** :func:`fit_carma` profiles out :math:`\sigma^2` and :math:`\mu`
exactly (the innovations are linear in :math:`\mu` and scale with :math:`\sigma`)
and optimises over the zeros only. :math:`a` and :math:`b` are parametrised as
products of factors :math:`z^2 + c_1z + c_2` and :math:`z + c_0` with positive
coefficients. A real polynomial has all its zeros in :math:`\mathrm{Re}\,z<0`
exactly when it factors this way, so every point of the search space is
causal, and the MA part is minimum-phase (``b_q = 1``). This removes the
reflection ambiguity :math:`\mu \leftrightarrow -\bar\mu` of the MA zeros,
which leave the spectral density unchanged.

References
----------
Brockwell, P. J., Davis, R. A. & Yang, Y. (2011). Estimation for non-negative
Lévy-driven CARMA processes. *J. Bus. Econ. Statist.* 29.

Jones, R. H. (1981). Fitting a continuous time autoregression to discrete
data. In *Applied Time Series Analysis II*, Academic Press.

Schlemm, E. & Stelzer, R. (2012). Quasi maximum likelihood estimation for
strongly mixing state space models and multivariate Lévy-driven CARMA
processes. *Electron. J. Statist.* 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
from scipy import linalg, optimize, signal

from spxa.zoo.beyond.carma import CARMAProcess
from spxa.zoo.levy.brownian import BrownianMotion

__all__ = [
    "CARMAFit",
    "KalmanInnovations",
    "carma_kalman_filter",
    "carma_log_likelihood",
    "fit_carma",
    "select_carma_order",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KalmanInnovations:
    """One-step prediction errors of a CARMA model on a series.

    Attributes
    ----------
    innovations :
        :math:`e_k = y_k - E[y_k \\mid y_{<k}]`.
    variances :
        :math:`S_k = \\mathrm{Var}(e_k)`.
    log_likelihood :
        The Gaussian log-likelihood :math:`-\\frac12\\sum(\\log 2\\pi S_k + e_k^2/S_k)`.
    steady_from :
        Index from which the steady-state gain was used (``n`` if never).
    """

    innovations: np.ndarray = field(repr=False)
    variances: np.ndarray = field(repr=False)
    log_likelihood: float
    steady_from: int

    @property
    def standardized(self) -> np.ndarray:
        """:math:`e_k/\\sqrt{S_k}`: i.i.d. N(0, 1) for a correct Gaussian model."""
        return self.innovations / np.sqrt(self.variances)


def _gaps(y: np.ndarray, dt: Optional[float], times: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """None for an even grid, else the positive gaps between observation times."""
    if (dt is None) == (times is None):
        raise ValueError("give exactly one of dt and times")
    if dt is not None:
        if not dt > 0:
            raise ValueError("dt must be positive")
        return None
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError("times must have one entry per observation")
    gaps = np.diff(t)
    if np.any(gaps <= 0):
        raise ValueError("times must be strictly increasing")
    return gaps


def _unit_filter(
    A: np.ndarray,
    b: np.ndarray,
    sigma_unit: np.ndarray,
    y: np.ndarray,
    dt: Optional[float],
    gaps: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Innovations and variances for a unit-variance driver (variances scale with sigma^2).

    ``y`` may have shape ``(n,)`` or ``(n, m)``: the columns share gains and variances.
    """
    n, p = y.shape[0], A.shape[0]
    e = np.empty(y.shape)
    s = np.empty(n)
    x = np.zeros((p,) + y.shape[1:])
    P = sigma_unit.copy()
    if gaps is not None:
        Fs, Qs = _transitions(A, sigma_unit, gaps)
        for k in range(n):
            Pb = P @ b
            s[k] = b @ Pb
            e[k] = y[k] - b @ x
            if k == n - 1:
                break
            F, Q = Fs[k], Qs[k]
            K = F @ Pb / s[k]
            x = F @ x + np.multiply.outer(K, e[k])
            P = F @ P @ F.T + Q - np.outer(K, K) * s[k]
            P = 0.5 * (P + P.T)
        return e, s, n

    assert dt is not None
    F = linalg.expm(A * dt)
    Q = sigma_unit - F @ sigma_unit @ F.T
    steady = n
    K = np.zeros(p)
    for k in range(n):
        Pb = P @ b
        s[k] = b @ Pb
        e[k] = y[k] - b @ x
        K = F @ Pb / s[k]
        x = F @ x + np.multiply.outer(K, e[k])
        P_next = F @ P @ F.T + Q - np.outer(K, K) * s[k]
        P_next = 0.5 * (P_next + P_next.T)
        if np.max(np.abs(P_next - P)) <= 1e-15 * np.max(np.abs(P)):
            steady = k + 1
            P = P_next
            break
        P = P_next
    if steady >= n:
        return e, s, n
    # steady state: x_{k+1} = G x_k + K y_k, e_k = y_k - b'x_k, constant S
    Pb = P @ b
    s_inf = float(b @ Pb)
    K = F @ Pb / s_inf
    G = F - np.outer(K, b)
    s[steady:] = s_inf
    rest = y[steady:]
    if rest.ndim == 1:
        e[steady:] = rest - _run_linear(G, K, b, x, rest)
    else:
        for j in range(rest.shape[1]):
            e[steady:, j] = rest[:, j] - _run_linear(G, K, b, x[:, j], rest[:, j])
    return e, s, steady


def _transitions(
    A: np.ndarray, sigma_unit: np.ndarray, gaps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """e^{A d} and Sigma - e^{Ad} Sigma e^{A'd} for every gap d, batched through A's eigenvectors."""
    lam, V = np.linalg.eig(A)
    if np.linalg.cond(V) < 1e8:
        Vinv = np.linalg.inv(V)
        Fs = np.real(np.einsum("ij,kj,jl->kil", V, np.exp(np.outer(gaps, lam)), Vinv))
    else:
        Fs = np.stack([linalg.expm(A * d) for d in gaps])
    Qs = sigma_unit[None] - Fs @ sigma_unit @ np.transpose(Fs, (0, 2, 1))
    return Fs, 0.5 * (Qs + np.transpose(Qs, (0, 2, 1)))


def _run_linear(
    G: np.ndarray, K: np.ndarray, b: np.ndarray, x0: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """b'x_k for x_{k+1} = G x_k + K u_k from x_0, k = 0..len(u)-1, by modal lfilter."""
    m = u.size
    d, V = np.linalg.eig(G)
    if np.linalg.cond(V) < 1e8 and np.all(np.abs(d) < 1.0):
        Vinv = np.linalg.inv(V)
        z0 = Vinv @ x0
        k_modal = Vinv @ K
        c = b @ V
        out = np.zeros(m, dtype=complex)
        for i in range(d.size):
            # w_j = z_{j+1} = d z_j + k u_j; z_0 given
            w = signal.lfilter([k_modal[i]], [1.0, -d[i]], u.astype(complex), zi=[d[i] * z0[i]])[0]
            z = np.concatenate([[z0[i]], w[:-1]])
            out += c[i] * z
        return np.real(out)
    x = x0.copy()
    out_r = np.empty(m)
    for j in range(m):
        out_r[j] = b @ x
        x = G @ x + K * u[j]
    return out_r


def carma_kalman_filter(
    process: CARMAProcess,
    y: Sequence[float] | np.ndarray,
    dt: Optional[float] = None,
    *,
    times: Optional[Sequence[float] | np.ndarray] = None,
) -> KalmanInnovations:
    """Kalman innovations of ``process`` on observations ``y``, given ``dt`` (even grid) or ``times``.

    The filter starts from the stationary law. The model's mean is
    :attr:`~spxa.zoo.beyond.carma.CARMAProcess.stationary_mean`, and its noise
    variance is the driver's :math:`\\kappa_2(L_1)`.
    """
    yy = np.asarray(y, dtype=float).reshape(-1)
    if yy.size < 1 or not np.all(np.isfinite(yy)):
        raise ValueError("y must be a non-empty finite series")
    gaps = _gaps(yy, dt, None if times is None else np.asarray(times, dtype=float))
    sigma2 = float(process.driver.cumulants(2)[2])
    sigma_unit = process.stationary_state_covariance / sigma2
    e, s, steady = _unit_filter(
        process.state_matrix,
        process.output_vector,
        sigma_unit,
        yy - process.stationary_mean,
        dt,
        gaps,
    )
    s = s * sigma2
    ll = -0.5 * float(np.sum(_LOG_2PI + np.log(s) + e**2 / s))
    return KalmanInnovations(innovations=e, variances=s, log_likelihood=ll, steady_from=steady)


def carma_log_likelihood(
    process: CARMAProcess,
    y: Sequence[float] | np.ndarray,
    dt: Optional[float] = None,
    *,
    times: Optional[Sequence[float] | np.ndarray] = None,
) -> float:
    """Gaussian (quasi-)log-likelihood of ``y`` under ``process``; see :func:`carma_kalman_filter`."""
    return carma_kalman_filter(process, y, dt, times=times).log_likelihood


# ---------------------------------------------------------------------------
# parametrisation: Hurwitz polynomials from positive factor coefficients
# ---------------------------------------------------------------------------


def _hurwitz_poly(log_c: np.ndarray) -> np.ndarray:
    """Monic polynomial (descending) of degree len(log_c) with all zeros in Re z < 0."""
    c = np.exp(log_c)
    poly = np.array([1.0])
    for i in range(0, c.size - 1, 2):
        poly = np.polymul(poly, [1.0, c[i], c[i + 1]])
    if c.size % 2 == 1:
        poly = np.polymul(poly, [1.0, c[-1]])
    return poly


def _factor_hurwitz(roots: np.ndarray) -> np.ndarray:
    """Log factor coefficients reproducing the given zeros (all with Re < 0)."""
    roots = np.asarray(roots, dtype=complex)
    cplx = [r for r in roots if r.imag > 1e-12]
    real = sorted(float(r.real) for r in roots if abs(r.imag) <= 1e-12)
    coefs: list[float] = []
    for r in cplx:
        coefs += [-2.0 * r.real, abs(r) ** 2]
    while len(real) >= 2:
        r1, r2 = real.pop(), real.pop()
        coefs += [-(r1 + r2), r1 * r2]
    if real:
        coefs.append(-real[0])
    return np.log(np.maximum(np.asarray(coefs, dtype=float), 1e-12))


def _coefficients(theta: np.ndarray, p: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    """(ar, ma) from free coordinates: AR factors, then MA factors; b_q = 1."""
    ar = _hurwitz_poly(theta[:p])[1:]
    ma = _hurwitz_poly(theta[p : p + q])[::-1] if q > 0 else np.array([1.0])
    return ar, ma


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CARMAFit:
    """A Gaussian QML fit of a CARMA(p, q) model.

    ``process`` is the fitted :class:`~spxa.zoo.beyond.carma.CARMAProcess`
    with a Brownian driver of drift ``mean * a_p / b_0`` and volatility
    ``sigma``. For a non-Gaussian driver, these are its first two cumulants.
    """

    process: CARMAProcess
    p: int
    q: int
    ar: np.ndarray
    ma: np.ndarray
    sigma: float
    mean: float
    log_likelihood: float
    n_obs: int
    converged: bool
    filter: KalmanInnovations = field(repr=False)
    theta: np.ndarray = field(repr=False)

    @property
    def n_params(self) -> int:
        """Free parameters: p AR coefficients, q MA coefficients, sigma and the mean."""
        return self.p + self.q + 2

    @property
    def aic(self) -> float:
        return float(-2.0 * self.log_likelihood + 2.0 * self.n_params)

    @property
    def bic(self) -> float:
        return float(-2.0 * self.log_likelihood + np.log(self.n_obs) * self.n_params)

    @property
    def ar_roots(self) -> np.ndarray:
        return self.process.ar_roots

    @property
    def ma_roots(self) -> np.ndarray:
        return self.process.ma_roots

    @property
    def residuals(self) -> np.ndarray:
        """Standardised innovations; i.i.d. N(0, 1) if the Gaussian model holds."""
        return self.filter.standardized

    def summary(self) -> str:
        lines = [
            f"CARMA({self.p},{self.q}) Gaussian QML fit, n = {self.n_obs}",
            f"  log-likelihood {self.log_likelihood:.4f}   AIC {self.aic:.2f}   BIC {self.bic:.2f}",
            f"  ar  {np.array2string(self.ar, precision=5)}   zeros {np.array2string(self.ar_roots, precision=4)}",
            f"  ma  {np.array2string(self.ma, precision=5)}",
            f"  sigma {self.sigma:.6g}   mean {self.mean:.6g}   converged {self.converged}",
        ]
        return "\n".join(lines)


def _profiled(
    theta: np.ndarray,
    p: int,
    q: int,
    y: np.ndarray,
    dt: Optional[float],
    gaps: Optional[np.ndarray],
) -> tuple[float, float, float, np.ndarray, np.ndarray, int]:
    """Profile log-likelihood over (sigma^2, mean); returns (ll, sigma2, mean, e, s, steady)."""
    ar, ma = _coefficients(theta, p, q)
    proc = CARMAProcess(ar, ma)
    A, b = proc.state_matrix, proc.output_vector
    sig_unit = proc.stationary_state_covariance
    both, s, steady = _unit_filter(A, b, sig_unit, np.column_stack([y, np.ones_like(y)]), dt, gaps)
    e0, c = both[:, 0], both[:, 1]
    mean = float(np.sum(e0 * c / s) / np.sum(c * c / s))
    e = e0 - mean * c
    n = y.size
    sigma2 = float(np.mean(e * e / s))
    ll = -0.5 * (n * (_LOG_2PI + np.log(sigma2) + 1.0) + float(np.sum(np.log(s))))
    return ll, sigma2, mean, e, s * sigma2, steady


def _start_from_ar(y: np.ndarray, step: float, p: int) -> Optional[np.ndarray]:
    """AR zeros mapped from a discrete Yule-Walker AR(p) fit at spacing ``step``."""
    x = y - y.mean()
    n = x.size
    if n <= 2 * p + 2:
        return None
    acov = np.array([x[: n - k] @ x[k:] / n for k in range(p + 1)])
    if acov[0] <= 0:
        return None
    try:
        phi = linalg.solve_toeplitz(acov[:p], acov[1 : p + 1])
    except (linalg.LinAlgError, ValueError):
        return None
    z = np.roots(np.concatenate([[1.0], -phi]))
    lam = []
    for zi in z:
        r = min(max(abs(zi), 1e-6), 0.999)
        # a real (possibly negative) discrete root maps to a real rate; pairs keep their angle
        angle = np.angle(zi) if abs(zi.imag) > 1e-12 else 0.0
        lam.append((np.log(r) + 1j * angle) / step)
    return _factor_hurwitz(np.asarray(lam))


def fit_carma(
    y: Sequence[float] | np.ndarray,
    p: int,
    q: int = 0,
    dt: Optional[float] = None,
    *,
    times: Optional[Sequence[float] | np.ndarray] = None,
    n_starts: int = 4,
    rng: Optional[np.random.Generator] = None,
    maxiter: int = 500,
) -> CARMAFit:
    r"""Gaussian QML fit of a CARMA(p, q) model to ``y`` observed on step ``dt`` or at ``times``.

    The optimiser searches the zeros of :math:`a` and :math:`b` over the
    Hurwitz factorisation (see the module docstring), with :math:`\sigma^2`
    and the mean profiled out. Starts are a discrete Yule–Walker AR(p) fit
    mapped to continuous time, plus ``n_starts - 1`` random perturbations of
    it. Each runs L-BFGS-B followed by Nelder–Mead polishing, and the best
    profile likelihood wins.
    """
    yy = np.asarray(y, dtype=float).reshape(-1)
    if not (p >= 1 and 0 <= q < p):
        raise ValueError("need p >= 1 and 0 <= q < p")
    if yy.size < 2 * (p + q + 2) or not np.all(np.isfinite(yy)):
        raise ValueError("y must be finite with more observations than parameters")
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
    gaps = _gaps(yy, dt, None if times is None else np.asarray(times, dtype=float))
    step = float(dt) if dt is not None else float(np.median(gaps))  # type: ignore[arg-type]
    rng = rng or np.random.default_rng(0)

    base_ar = _start_from_ar(yy, step, p)
    if base_ar is None:
        base_ar = np.log(np.full(p, 1.0 / step / 10.0))
    # MA zeros start well inside the band: between the slowest AR rate and the sampling rate
    base_ma = np.empty(0)
    if q > 0:
        base_ma = _factor_hurwitz(-np.geomspace(1.0 / step, 0.2 / step, q).astype(complex))
    base = np.concatenate([base_ar, base_ma])

    def objective(theta: np.ndarray) -> float:
        if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 40:
            return 1e300
        try:
            ll = _profiled(theta, p, q, yy, dt, gaps)[0]
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e300
        return -ll if np.isfinite(ll) else 1e300

    best: Optional[optimize.OptimizeResult] = None
    for k in range(n_starts):
        x0 = base if k == 0 else base + rng.normal(0.0, 0.7, base.size)
        with np.errstate(all="ignore"):
            r1 = optimize.minimize(objective, x0, method="L-BFGS-B", options={"maxiter": maxiter})
            r2 = optimize.minimize(
                objective,
                r1.x,
                method="Nelder-Mead",
                options={"maxiter": 200 * base.size, "xatol": 1e-8, "fatol": 1e-10},
            )
        r = r2 if r2.fun <= r1.fun else r1
        r.converged = bool(r1.success or r2.success)
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None
    ll, sigma2, mean, e, s, steady = _profiled(best.x, p, q, yy, dt, gaps)
    ar, ma = _coefficients(best.x, p, q)
    sigma = float(np.sqrt(sigma2))
    drift = mean * ar[-1] / ma[0]
    process = CARMAProcess(ar, ma, driver=BrownianMotion(drift, sigma))
    kal = KalmanInnovations(innovations=e, variances=s, log_likelihood=ll, steady_from=steady)
    return CARMAFit(
        process=process,
        p=p,
        q=q,
        ar=ar,
        ma=ma,
        sigma=sigma,
        mean=mean,
        log_likelihood=float(ll),
        n_obs=yy.size,
        converged=bool(best.converged),
        filter=kal,
        theta=np.asarray(best.x),
    )


def select_carma_order(
    y: Sequence[float] | np.ndarray,
    dt: Optional[float] = None,
    *,
    times: Optional[Sequence[float] | np.ndarray] = None,
    max_p: int = 3,
    criterion: Literal["bic", "aic"] = "bic",
    n_starts: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> list[CARMAFit]:
    """Fit every CARMA(p, q) with ``1 <= p <= max_p`` and ``q < p``, ranked by ``criterion``."""
    if criterion not in ("bic", "aic"):
        raise ValueError("criterion must be 'bic' or 'aic'")
    if max_p < 1:
        raise ValueError("max_p must be at least 1")
    rng = rng or np.random.default_rng(0)
    fits = [
        fit_carma(y, p, q, dt, times=times, n_starts=n_starts, rng=rng)
        for p in range(1, max_p + 1)
        for q in range(p)
    ]
    return sorted(fits, key=lambda f: f.bic if criterion == "bic" else f.aic)
