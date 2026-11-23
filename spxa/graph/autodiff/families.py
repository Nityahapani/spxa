r"""
Closed-form characteristic exponents of the zoo Lévy families, differentiable.

Each :class:`FamilyRule` gives :math:`\psi(u;\theta)` with
:math:`E[e^{iuX_t}] = e^{t\psi(u)}` as NumPy code in the family's own
constructor parameters, so that

* the parameters may be :class:`~spxa.graph.autodiff.jet.Jet` objects, which
  yields the exact :math:`\partial\psi/\partial\theta` and
  :math:`\partial^2\psi/\partial\theta^2`;
* ``u`` may be complex (inside the exponential-moment strip) or a
  :class:`~spxa.graph.autodiff.taylor.TSeries`, which yields the cumulants
  :math:`\kappa_n = n!\,[z^n]\,\psi(-iz)` exactly.

The formulas are the ones the zoo itself evaluates (same branches, same
overflow-free rearrangements); ``tests/graph/test_diff_process.py`` checks
value agreement with every zoo class. :class:`~spxa.zoo.levy.stable.AlphaStable`
is not analytic in ``u`` (it has no exponential moments); its exponent is
differentiable in the parameters for real ``u \ne 0`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy.special as sps

from spxa.core.process import Process
from spxa.graph.autodiff.jet import value_of
from spxa.graph.autodiff.taylor import TSeries, TaylorError

__all__ = ["FamilyRule", "rule_for", "RULES", "register_rule"]


@dataclass(frozen=True)
class FamilyRule:
    """Differentiable description of one zoo Lévy family."""

    cls_name: str
    params: Tuple[str, ...]
    getter: Callable[[Process], Tuple[float, ...]]
    psi: Callable[[Any, Tuple[Any, ...]], Any]
    build: Callable[[Tuple[float, ...]], Process]
    analytic: bool = True
    reference: str = ""


RULES: Dict[str, FamilyRule] = {}


def register_rule(rule: FamilyRule) -> FamilyRule:
    """Register (or replace) the rule for ``rule.cls_name``."""
    RULES[rule.cls_name] = rule
    return rule


def rule_for(process: Process) -> Optional[FamilyRule]:
    """The rule for ``process``'s exact class, or ``None``."""
    return RULES.get(type(process).__name__)


def _is_series(u: Any) -> bool:
    return isinstance(u, TSeries)


def _u(u: Any) -> Any:
    if _is_series(u) or hasattr(u, "grad"):
        return u
    return np.asarray(u, dtype=complex)


# ── diffusions and drifts ─────────────────────────────────────────────────────


def _bm_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    mu, sigma = p
    u = _u(u)
    return 1j * mu * u - 0.5 * sigma**2 * u * u


def _drift_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    (mu,) = p
    return 1j * mu * _u(u)


def _build(module: str, cls: str) -> Callable[..., Any]:
    def make(*args: Any, **kwargs: Any) -> Process:
        import importlib

        return getattr(importlib.import_module(module), cls)(*args, **kwargs)  # type: ignore[no-any-return]

    return make


_LEVY = "spxa.zoo.levy"

register_rule(
    FamilyRule(
        "BrownianMotion",
        ("mu", "sigma"),
        lambda P: (P.mu, P.sigma),  # type: ignore[attr-defined]
        _bm_psi,
        lambda p: _build(f"{_LEVY}.brownian", "BrownianMotion")(mu=p[0], sigma=p[1]),
        reference="Sato (1999) Ex. 8.3",
    )
)
register_rule(
    FamilyRule(
        "Drift",
        ("mu",),
        lambda P: (P.mu,),  # type: ignore[attr-defined]
        _drift_psi,
        lambda p: _build(f"{_LEVY}.drift", "Drift")(p[0]),
    )
)

# ── subordinators ───────────────────────────────────────────────────────────


def _gamma_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    a, b = p
    return -a * np.log(1.0 - 1j * _u(u) / b)


def _ig_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    mu, lam = p
    return (lam / mu) * (1.0 - np.sqrt(1.0 - 2j * mu**2 * _u(u) / lam))


def _ts_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    alpha, C, lam = p
    return C * sps.gamma(-alpha) * ((lam - 1j * _u(u)) ** alpha - lam**alpha)


def _negbin_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    r, q = p
    return r * (np.log(1.0 - q) - np.log(1.0 - q * np.exp(1j * _u(u))))


def _poisson_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    rate, c = p
    return rate * (np.exp(1j * c * _u(u)) - 1.0)


register_rule(
    FamilyRule(
        "GammaProcess",
        ("a", "b"),
        lambda P: (P.a, P.b),  # type: ignore[attr-defined]
        _gamma_psi,
        lambda p: _build(f"{_LEVY}.gamma", "GammaProcess")(a=p[0], b=p[1]),
        reference="Cont & Tankov (2004) Table 4.2",
    )
)
register_rule(
    FamilyRule(
        "InverseGaussianProcess",
        ("mu", "lam"),
        lambda P: (P.mu, P.lam),  # type: ignore[attr-defined]
        _ig_psi,
        lambda p: _build(f"{_LEVY}.inverse_gaussian", "InverseGaussianProcess")(mu=p[0], lam=p[1]),
    )
)
register_rule(
    FamilyRule(
        "TemperedStable",
        ("alpha", "C", "lam"),
        lambda P: (P.alpha, P.C, P.lam),  # type: ignore[attr-defined]
        _ts_psi,
        lambda p: _build(f"{_LEVY}.tempered_stable", "TemperedStable")(
            alpha=p[0], C=p[1], lam=p[2]
        ),
        reference="Rosiński (2007) eq. (1.3)",
    )
)
register_rule(
    FamilyRule(
        "NegativeBinomialProcess",
        ("r", "p"),
        lambda P: (P.r, P.p),  # type: ignore[attr-defined]
        _negbin_psi,
        lambda p: _build(f"{_LEVY}.negative_binomial", "NegativeBinomialProcess")(r=p[0], p=p[1]),
    )
)
register_rule(
    FamilyRule(
        "PoissonProcess",
        ("rate", "jump_size"),
        lambda P: (P.rate, P.jump_size),  # type: ignore[attr-defined]
        _poisson_psi,
        lambda p: _build(f"{_LEVY}.poisson", "PoissonProcess")(rate=p[0], jump_size=p[1]),
    )
)

# ── normal mean-variance mixtures ───────────────────────────────────────────


def _vg_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    sigma, nu, theta = p
    u = _u(u)
    return -np.log(1.0 - 1j * u * theta * nu + 0.5 * sigma**2 * nu * u * u) / nu


def _nig_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    alpha, beta, delta, mu = p
    u = _u(u)
    gam = np.sqrt(alpha**2 - beta**2)
    return 1j * mu * u + delta * (gam - np.sqrt(alpha**2 - (beta + 1j * u) ** 2))


def _nts_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    alpha, lambd, delta, theta, sigma = p
    u = _u(u)
    z = lambd - 1j * theta * u + 0.5 * sigma**2 * u * u
    c = delta * sps.gamma(1.0 - alpha) / alpha
    return -c * (z**alpha - lambd**alpha)


def _logcosh(w: Any) -> Any:
    r""":math:`\log\cosh w = s w + \log(1 + e^{-2sw}) - \log 2`, :math:`s = \operatorname{sign}\operatorname{Re} w`."""
    if _is_series(w):
        # one analytic branch near the expansion point
        s = 1.0 if np.real(value_of(w.coeffs[0])) >= 0 else -1.0
        return s * w + np.log(1.0 + np.exp(-2.0 * s * w)) - np.log(2.0)
    s = np.where(np.real(value_of(w)) >= 0, 1.0, -1.0)
    return s * w + np.log(1.0 + np.exp(-2.0 * s * w)) - np.log(2.0)


def _meixner_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    alpha, beta, delta, m = p
    u = _u(u)
    w = (alpha * u - 1j * beta) / 2.0
    return 2.0 * delta * (np.log(np.cos(beta / 2.0)) - _logcosh(w)) + 1j * m * u


def _gh_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    lam, alpha, beta, delta, mu = p
    u = _u(u)
    zeta = np.sqrt(alpha**2 - (beta + 1j * u) ** 2)
    gam = np.sqrt(alpha**2 - beta**2)
    k_base = sps.kve(lam, delta * gam)
    k_zeta = sps.kve(lam, delta * zeta)
    return 1j * mu * u + lam * np.log(gam / zeta) + np.log(k_zeta / k_base) - delta * (zeta - gam)


register_rule(
    FamilyRule(
        "VarianceGamma",
        ("sigma", "nu", "theta"),
        lambda P: (P.sigma, P.nu, P.theta),  # type: ignore[attr-defined]
        _vg_psi,
        lambda p: _build(f"{_LEVY}.vg", "VarianceGamma")(sigma=p[0], nu=p[1], theta=p[2]),
        reference="Madan, Carr & Chang (1998) eq. (4)",
    )
)
register_rule(
    FamilyRule(
        "NIG",
        ("alpha", "beta", "delta", "mu"),
        lambda P: (P.alpha, P.beta, P.delta, P.mu),  # type: ignore[attr-defined]
        _nig_psi,
        lambda p: _build(f"{_LEVY}.nig", "NIG")(alpha=p[0], beta=p[1], delta=p[2], mu=p[3]),
        reference="Barndorff-Nielsen (1997) eq. (2.2)",
    )
)
register_rule(
    FamilyRule(
        "NormalTemperedStableProcess",
        ("alpha", "lambd", "delta", "theta", "sigma"),
        lambda P: (P.alpha, P.lambd, P.delta, P.theta, P.sigma),  # type: ignore[attr-defined]
        _nts_psi,
        lambda p: _build(f"{_LEVY}.nts", "NormalTemperedStableProcess")(
            alpha=p[0], lambd=p[1], delta=p[2], theta=p[3], sigma=p[4]
        ),
    )
)
register_rule(
    FamilyRule(
        "MeixnerProcess",
        ("alpha", "beta", "delta", "m"),
        lambda P: (P.alpha, P.beta, P.delta, P.m),  # type: ignore[attr-defined]
        _meixner_psi,
        lambda p: _build(f"{_LEVY}.meixner", "MeixnerProcess")(
            alpha=p[0], beta=p[1], delta=p[2], m=p[3]
        ),
        reference="Schoutens & Teugels (1998) eq. (3)",
    )
)
register_rule(
    FamilyRule(
        "GeneralizedHyperbolicProcess",
        ("lambda_param", "alpha", "beta", "delta", "mu"),
        lambda P: (P._lambda, P._alpha, P._beta, P._delta, P._mu),  # type: ignore[attr-defined]
        _gh_psi,
        lambda p: _build(f"{_LEVY}.gh", "GeneralizedHyperbolicProcess")(
            lambda_param=p[0], alpha=p[1], beta=p[2], delta=p[3], mu=p[4]
        ),
        reference="Eberlein (2001)",
    )
)

# ── CGMY, with exact handling of the Gamma(-Y) poles at Y = 0, 1 ─────────────

_CGMY_SERIES_WIDTH = 5e-2
_CGMY_SERIES_TERMS = 24


def _cgmy_bracket_derivs(G: Any, M: Any, u: Any, k: int, j: int) -> Any:
    r""":math:`\partial_Y^j B(Y)|_{Y=k}`, :math:`B = (M-iu)^Y - M^Y + (G+iu)^Y - G^Y`."""
    a, b = M - 1j * u, G + 1j * u
    la, lm, lb, lg = np.log(a), np.log(M), np.log(b), np.log(G)
    if k == 0:
        return la**j - lm**j + lb**j - lg**j
    return a * la**j - M * lm**j + b * lb**j - G * lg**j


def _cgmy_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    r"""
    :math:`C\Gamma(-Y)\big((M-iu)^Y - M^Y + (G+iu)^Y - G^Y\big)` (Carr et al. 2002, eq. 5).

    Near the poles :math:`Y = k \in \{0, 1\}` (:math:`|Y - k| < 0.05`) it is
    evaluated as :math:`C\,A_k(\varepsilon)\sum_{j\ge1}B^{(j)}(k)\,\varepsilon^{j-1}/j!`
    with :math:`\varepsilon = Y - k`, :math:`A_0 = -\Gamma(1-\varepsilon)` and
    :math:`A_1 = \Gamma(1-\varepsilon)/(1+\varepsilon)` (so that
    :math:`\Gamma(-Y) = A_k/\varepsilon`). This is analytic in :math:`Y` across the
    pole, so derivatives in :math:`Y` are exact there too. Twenty-four terms bound
    the truncation by :math:`(0.05\,|\log|)^{24}/24!` (below 1e-16 for
    :math:`|u| < 10^8`); outside the band the direct form loses at most
    :math:`\epsilon_{mach}/0.05` to cancellation.
    """
    C, G, M, Y = p
    u = _u(u)
    yv = float(np.real(value_of(Y)))
    for k in (0, 1):
        if abs(yv - k) < _CGMY_SERIES_WIDTH:
            eps = Y - k
            A = -sps.gamma(1.0 - eps) if k == 0 else sps.gamma(1.0 - eps) / (1.0 + eps)
            total: Any = 0.0
            powe: Any = 1.0
            fact = 1.0
            for j in range(1, _CGMY_SERIES_TERMS + 1):
                fact *= j
                total = total + _cgmy_bracket_derivs(G, M, u, k, j) * (powe / fact)
                powe = powe * eps
            return C * A * total
    return C * sps.gamma(-Y) * ((M - 1j * u) ** Y - M**Y + (G + 1j * u) ** Y - G**Y)


register_rule(
    FamilyRule(
        "CGMY",
        ("C", "G", "M", "Y"),
        lambda P: (P.C, P.G, P.M, P.Y),  # type: ignore[attr-defined]
        _cgmy_psi,
        lambda p: _build(f"{_LEVY}.cgmy", "CGMY")(C=p[0], G=p[1], M=p[2], Y=p[3]),
        reference="Carr, Geman, Madan & Yor (2002) eq. (5)",
    )
)

# ── jump diffusions ───────────────────────────────────────────────────────────


def _merton_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    drift, sigma, rate, jm, js = p
    u = _u(u)
    return (
        1j * drift * u
        - 0.5 * sigma**2 * u * u
        + rate * (np.exp(1j * jm * u - 0.5 * js**2 * u * u) - 1.0)
    )


def _kou_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    drift, sigma, rate, q, eta1, eta2 = p
    u = _u(u)
    jump = q * eta1 / (eta1 - 1j * u) + (1.0 - q) * eta2 / (eta2 + 1j * u) - 1.0
    return 1j * drift * u - 0.5 * sigma**2 * u * u + rate * jump


register_rule(
    FamilyRule(
        "MertonJumpDiffusion",
        ("drift", "sigma", "rate", "jump_mean", "jump_std"),
        lambda P: (P._drift, P._sigma, P._rate, P._jump_mean, P._jump_std),  # type: ignore[attr-defined]
        _merton_psi,
        lambda p: _build(f"{_LEVY}.merton", "MertonJumpDiffusion")(
            drift=p[0], sigma=p[1], rate=p[2], jump_mean=p[3], jump_std=p[4]
        ),
        reference="Merton (1976)",
    )
)
register_rule(
    FamilyRule(
        "KouJumpDiffusion",
        ("drift", "sigma", "rate", "p", "eta1", "eta2"),
        lambda P: (P.drift, P.sigma, P.rate, P.p, P.eta1, P.eta2),  # type: ignore[attr-defined]
        _kou_psi,
        lambda p: _build(f"{_LEVY}.kou", "KouJumpDiffusion")(
            drift=p[0], sigma=p[1], rate=p[2], p=p[3], eta1=p[4], eta2=p[5]
        ),
        reference="Kou (2002)",
    )
)

# ── alpha-stable: differentiable in the parameters, not analytic in u ────────


def _stable_psi(u: Any, p: Tuple[Any, ...]) -> Any:
    alpha, beta, sigma, mu = p
    if _is_series(u):
        raise TaylorError("the alpha-stable exponent is not analytic at u = 0 (no moments)")
    uv = np.asarray(value_of(u))
    if np.iscomplexobj(uv) and np.any(np.imag(uv) != 0):
        raise ValueError("the alpha-stable exponent is defined for real u only")
    u = np.real(u) if hasattr(u, "grad") else np.real(uv)
    au = np.abs(u)
    sgn = np.sign(value_of(u))
    live = np.asarray(value_of(au)) > 0
    safe = np.where(live, au, 1.0) if not hasattr(au, "grad") else au + np.where(live, 0.0, 1.0)
    av = float(np.real(value_of(alpha)))
    if abs(av - 1.0) > 1e-12:
        core = (
            -(sigma**alpha)
            * np.exp(alpha * np.log(safe))
            * (1.0 - 1j * beta * sgn * np.tan(np.pi * alpha / 2.0))
        )
    else:
        core = -sigma * safe * (1.0 + 1j * beta * (2.0 / np.pi) * sgn * np.log(safe))
    return np.where(live, core, 0.0) + 1j * mu * u


register_rule(
    FamilyRule(
        "AlphaStable",
        ("alpha", "beta", "sigma", "mu"),
        lambda P: (P.alpha, P.beta, P.sigma, P.mu),  # type: ignore[attr-defined]
        _stable_psi,
        lambda p: _build(f"{_LEVY}.stable", "AlphaStable")(
            alpha=p[0], beta=p[1], sigma=p[2], mu=p[3]
        ),
        analytic=False,
        reference="Nolan (2020) Def. 1.6",
    )
)
