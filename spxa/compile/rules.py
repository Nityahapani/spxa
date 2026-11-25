r"""
Lowering rules: zoo Lévy families → IR characteristic exponents.

Each rule maps one process instance to a :class:`LeafSpec`: its parameter
names and values, a builder for the IR of :math:`\psi(u;\theta)`, a builder
that reconstructs the process at new parameter values, and the key of its
sampling rule (:mod:`spxa.compile.sampler`).

The formulas are the closed forms the zoo evaluates, with the same branches
and the same overflow-free rearrangements, written *branch-free*: a branch
on a parameter value (the :math:`\Gamma(-Y)` poles of CGMY, the
:math:`\alpha = 1` case of the stable law) is a ``where`` node whose
condition depends on the parameters only. Kernels resolve such guards per
parameter set and compile one specialised program per pattern, so a batch of
parameters may straddle a branch. Parameter names follow
:class:`spxa.graph.autodiff.process.DiffProcess` (``"<Family>.<param>"``),
so the two are interchangeable.

Three rules improve on the textbook form without changing the value:

* Poisson and Merton jump parts use ``expm1``, which is accurate at small
  :math:`u` where :math:`e^{x} - 1` cancels;
* the Kou and exponential jump parts are written as
  :math:`iu\,p/(\eta_1 - iu) - iu\,q/(\eta_2 + iu)` instead of
  :math:`p\eta_1/(\eta_1 - iu) + q\eta_2/(\eta_2 + iu) - 1`, which is the
  same function with no cancellation near :math:`u = 0`;
* uniform jumps use :math:`e^{ium}\,\mathrm{sinc}(uw)`, which is analytic at
  :math:`u = 0` (the zoo's quotient form is singular there).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spxa.compile import ir as I
from spxa.compile.ir import CompileError, Expr

__all__ = ["LeafSpec", "leaf_rule", "spec_for", "supported_families", "LEAF_RULES"]


@dataclass(frozen=True)
class LeafSpec:
    """Compilation recipe for one primitive process."""

    family: str
    names: Tuple[str, ...]
    values: Tuple[float, ...]
    psi: Callable[[Expr, Sequence[Expr]], Expr]
    build: Callable[[Sequence[float]], Any]
    sampler: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    reference: str = ""


LEAF_RULES: Dict[str, Callable[[Any], LeafSpec]] = {}


def leaf_rule(cls_name: str) -> Callable[[Callable[[Any], LeafSpec]], Callable[[Any], LeafSpec]]:
    """Register a rule for processes whose class is named ``cls_name``."""

    def deco(fn: Callable[[Any], LeafSpec]) -> Callable[[Any], LeafSpec]:
        LEAF_RULES[cls_name] = fn
        return fn

    return deco


def spec_for(process: Any) -> Optional[LeafSpec]:
    """The :class:`LeafSpec` of ``process``'s exact class, or ``None``."""
    fn = LEAF_RULES.get(type(process).__name__)
    return None if fn is None else fn(process)


def _f(*xs: Any) -> Tuple[float, ...]:
    out = []
    for x in xs:
        v = float(x)
        if not np.isfinite(v):
            raise CompileError(f"parameter value {x!r} is not finite")
        out.append(v)
    return tuple(out)


def _family_build(cls_name: str) -> Callable[[Sequence[float]], Any]:
    from spxa.graph.autodiff.families import RULES

    rule = RULES[cls_name]
    return lambda p: rule.build(tuple(float(x) for x in p))


def _family_values(cls_name: str, process: Any) -> Tuple[float, ...]:
    from spxa.graph.autodiff.families import RULES

    return _f(*RULES[cls_name].getter(process))


def _family_names(cls_name: str) -> Tuple[str, ...]:
    from spxa.graph.autodiff.families import RULES

    return RULES[cls_name].params


def _family_spec(
    cls_name: str,
    process: Any,
    psi: Callable[[Expr, Sequence[Expr]], Expr],
    sampler: Optional[str],
    reference: str = "",
) -> LeafSpec:
    return LeafSpec(
        family=cls_name,
        names=_family_names(cls_name),
        values=_family_values(cls_name, process),
        psi=psi,
        build=_family_build(cls_name),
        sampler=sampler,
        reference=reference,
    )


J = 1j

# ── diffusions and drifts ───────────────────────────────────────────────────


def _bm(u: Expr, p: Sequence[Expr]) -> Expr:
    mu, sigma = p
    return J * mu * u - 0.5 * sigma**2 * u * u


def _drift(u: Expr, p: Sequence[Expr]) -> Expr:
    (mu,) = p
    return J * mu * u


@leaf_rule("BrownianMotion")
def _r_bm(P: Any) -> LeafSpec:
    return _family_spec("BrownianMotion", P, _bm, "gaussian", "Sato (1999) Ex. 8.3")


@leaf_rule("Drift")
def _r_drift(P: Any) -> LeafSpec:
    return _family_spec("Drift", P, _drift, "drift")


# ── subordinators ───────────────────────────────────────────────────────────


def _gamma_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    a, b = p
    return -a * I.log(1.0 - J * u / b)


def _ig_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    mu, lam = p
    return (lam / mu) * (1.0 - I.sqrt(1.0 - 2j * mu**2 * u / lam))


def _ts_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    alpha, C, lam = p
    return C * I.gamma(-alpha) * ((lam - J * u) ** alpha - lam**alpha)


def _negbin_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    r, q = p
    return r * (I.log(1.0 - q) - I.log(1.0 - q * I.exp(J * u)))


def _poisson_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    rate, c = p
    return rate * I.expm1(J * c * u)


@leaf_rule("GammaProcess")
def _r_gamma(P: Any) -> LeafSpec:
    return _family_spec("GammaProcess", P, _gamma_psi, "gamma", "Cont & Tankov (2004) Table 4.2")


@leaf_rule("InverseGaussianProcess")
def _r_ig(P: Any) -> LeafSpec:
    return _family_spec("InverseGaussianProcess", P, _ig_psi, "inverse_gaussian")


@leaf_rule("TemperedStable")
def _r_ts(P: Any) -> LeafSpec:
    return _family_spec("TemperedStable", P, _ts_psi, None, "Rosiński (2007) eq. (1.3)")


@leaf_rule("NegativeBinomialProcess")
def _r_negbin(P: Any) -> LeafSpec:
    return _family_spec("NegativeBinomialProcess", P, _negbin_psi, "negative_binomial")


@leaf_rule("PoissonProcess")
def _r_poisson(P: Any) -> LeafSpec:
    return _family_spec("PoissonProcess", P, _poisson_psi, "poisson")


# ── normal mean-variance mixtures ───────────────────────────────────────────


def _vg_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    sigma, nu, theta = p
    return -I.log(1.0 - J * u * theta * nu + 0.5 * sigma**2 * nu * u * u) / nu


def _nig_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    alpha, beta, delta, mu = p
    gam = I.sqrt(alpha**2 - beta**2)
    return J * mu * u + delta * (gam - I.sqrt(alpha**2 - (beta + J * u) ** 2))


def _nts_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    alpha, lambd, delta, theta, sigma = p
    z = lambd - J * theta * u + 0.5 * sigma**2 * u * u
    c = delta * I.gamma(1.0 - alpha) / alpha
    return -c * (z**alpha - lambd**alpha)


def _logcosh(w: Expr) -> Expr:
    r""":math:`\log\cosh w = sw + \log(1 + e^{-2sw}) - \log 2`, :math:`s = \operatorname{sign}\operatorname{Re}w`."""
    s = I.where(I.ge(I.real(w), 0.0), 1.0, -1.0)
    return s * w + I.log1p(I.exp(-2.0 * s * w)) - float(np.log(2.0))


def _meixner_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    alpha, beta, delta, m = p
    w = (alpha * u - J * beta) / 2.0
    return 2.0 * delta * (I.log(I.cos(beta / 2.0)) - _logcosh(w)) + J * m * u


def _gh_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    lam, alpha, beta, delta, mu = p
    zeta = I.sqrt(alpha**2 - (beta + J * u) ** 2)
    gam = I.sqrt(alpha**2 - beta**2)
    k_base = I.kve(lam, delta * gam)
    k_zeta = I.kve(lam, delta * zeta)
    return J * mu * u + lam * I.log(gam / zeta) + I.log(k_zeta / k_base) - delta * (zeta - gam)


@leaf_rule("VarianceGamma")
def _r_vg(P: Any) -> LeafSpec:
    return _family_spec("VarianceGamma", P, _vg_psi, "variance_gamma", "Madan, Carr & Chang (1998)")


@leaf_rule("NIG")
def _r_nig(P: Any) -> LeafSpec:
    return _family_spec("NIG", P, _nig_psi, "nig", "Barndorff-Nielsen (1997) eq. (2.2)")


@leaf_rule("NormalTemperedStableProcess")
def _r_nts(P: Any) -> LeafSpec:
    return _family_spec("NormalTemperedStableProcess", P, _nts_psi, None)


@leaf_rule("MeixnerProcess")
def _r_meixner(P: Any) -> LeafSpec:
    return _family_spec("MeixnerProcess", P, _meixner_psi, None, "Schoutens & Teugels (1998)")


@leaf_rule("GeneralizedHyperbolicProcess")
def _r_gh(P: Any) -> LeafSpec:
    return _family_spec("GeneralizedHyperbolicProcess", P, _gh_psi, None, "Eberlein (2001)")


# ── CGMY across the Gamma(-Y) poles ─────────────────────────────────────────

#: Same band and number of terms as spxa.graph.autodiff.families (see there).
_CGMY_SERIES_WIDTH = 5e-2
_CGMY_SERIES_TERMS = 24


def _cgmy_series(C: Expr, G: Expr, M: Expr, Y: Expr, u: Expr, k: int) -> Expr:
    r"""
    :math:`C\,A_k(\varepsilon)\sum_{j\ge1}B^{(j)}(k)\,\varepsilon^{j-1}/j!`, :math:`\varepsilon = Y - k`.

    :math:`\Gamma(-Y) = A_k/\varepsilon` with :math:`A_0 = -\Gamma(1-\varepsilon)`,
    :math:`A_1 = \Gamma(1-\varepsilon)/(1+\varepsilon)`; :math:`B^{(j)}` are the
    :math:`Y`-derivatives of the bracket at the pole. Analytic across it.
    """
    a, b = M - J * u, G + J * u
    la, lm, lb, lg = I.log(a), I.log(M), I.log(b), I.log(G)
    eps = Y - float(k)
    A = -I.gamma(1.0 - eps) if k == 0 else I.gamma(1.0 - eps) / (1.0 + eps)
    total: Expr = I.const(0j)
    powe: Expr = I.const(1.0)
    pa, pm, pb, pg = la, lm, lb, lg
    fact = 1.0
    for j in range(1, _CGMY_SERIES_TERMS + 1):
        fact *= j
        if k == 0:
            bj = pa - pm + pb - pg
        else:
            bj = a * pa - M * pm + b * pb - G * pg
        total = total + bj * (powe / fact)
        powe = powe * eps
        pa, pm, pb, pg = pa * la, pm * lm, pb * lb, pg * lg
    return C * A * total


def _cgmy_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    C, G, M, Y = p
    direct = C * I.gamma(-Y) * ((M - J * u) ** Y - M**Y + (G + J * u) ** Y - G**Y)
    near0 = I.lt(I.abs_(Y), _CGMY_SERIES_WIDTH)
    near1 = I.lt(I.abs_(Y - 1.0), _CGMY_SERIES_WIDTH)
    return I.where(
        near0,
        _cgmy_series(C, G, M, Y, u, 0),
        I.where(near1, _cgmy_series(C, G, M, Y, u, 1), direct),
    )


@leaf_rule("CGMY")
def _r_cgmy(P: Any) -> LeafSpec:
    return _family_spec("CGMY", P, _cgmy_psi, None, "Carr, Geman, Madan & Yor (2002) eq. (5)")


# ── jump diffusions ─────────────────────────────────────────────────────────


def _merton_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    drift, sigma, rate, jm, js = p
    return J * drift * u - 0.5 * sigma**2 * u * u + rate * I.expm1(J * jm * u - 0.5 * js**2 * u * u)


def _double_exp_jump(u: Expr, q: Expr, eta1: Expr, eta2: Expr) -> Expr:
    r""":math:`E[e^{iuY}] - 1` for double-exponential jumps, without cancellation at 0."""
    iu = J * u
    return iu * (q / (eta1 - iu) - (1.0 - q) / (eta2 + iu))


def _kou_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    drift, sigma, rate, q, eta1, eta2 = p
    return J * drift * u - 0.5 * sigma**2 * u * u + rate * _double_exp_jump(u, q, eta1, eta2)


@leaf_rule("MertonJumpDiffusion")
def _r_merton(P: Any) -> LeafSpec:
    return _family_spec("MertonJumpDiffusion", P, _merton_psi, "merton", "Merton (1976)")


@leaf_rule("KouJumpDiffusion")
def _r_kou(P: Any) -> LeafSpec:
    return _family_spec("KouJumpDiffusion", P, _kou_psi, "kou", "Kou (2002)")


# ── alpha-stable: differentiable in the parameters, real u only ─────────────


def _stable_psi(u: Expr, p: Sequence[Expr]) -> Expr:
    alpha, beta, sigma, mu = p
    x = I.real(u)
    ax = I.abs_(x)
    sgn = I.sign(x)
    live = I.gt(ax, 0.0)
    safe = I.where(live, ax, 1.0)
    general = (
        -(sigma**alpha)
        * I.exp(alpha * I.log(safe))
        * (1.0 - J * beta * sgn * I.tan(float(np.pi) * alpha / 2.0))
    )
    at_one = -sigma * safe * (1.0 + J * beta * (2.0 / float(np.pi)) * sgn * I.log(safe))
    core = I.where(I.gt(I.abs_(alpha - 1.0), 1e-12), general, at_one)
    return I.where(live, core, 0j) + J * mu * x


@leaf_rule("AlphaStable")
def _r_stable(P: Any) -> LeafSpec:
    return _family_spec("AlphaStable", P, _stable_psi, None, "Nolan (2020) Def. 1.6")


# ── compound Poisson with the zoo's jump laws ───────────────────────────────


def _cp_build(kind: str, extra: Dict[str, Any]) -> Callable[[Sequence[float]], Any]:
    def build(p: Sequence[float]) -> Any:
        from spxa.zoo.levy import compound_poisson as cp

        rate = float(p[0])
        if kind == "normal":
            return cp.CompoundPoisson(rate, cp.NormalJump(mean=float(p[1]), std=float(p[2])))
        if kind == "exponential":
            return cp.CompoundPoisson(rate, cp.ExponentialJump(rate=float(p[1])))
        if kind == "double_exponential":
            return cp.CompoundPoisson(
                rate, cp.DoubleExponentialJump(p=float(p[1]), eta1=float(p[2]), eta2=float(p[3]))
            )
        if kind == "uniform":
            return cp.CompoundPoisson(rate, cp.UniformJump(low=float(p[1]), high=float(p[2])))
        return cp.CompoundPoisson(
            rate, cp.DiscreteJump(values=list(extra["values"]), probs=list(extra["probs"]))
        )

    return build


@leaf_rule("CompoundPoisson")
def _r_compound_poisson(P: Any) -> LeafSpec:
    from spxa.zoo.levy import compound_poisson as cp

    jd = P.jump_dist
    t = type(jd)
    if t is cp.NormalJump:
        kind, names, vals = "normal", ("rate", "jump_mean", "jump_std"), _f(P.rate, jd.mean, jd.std)

        def psi(u: Expr, p: Sequence[Expr]) -> Expr:
            rate, m, s = p
            return rate * I.expm1(J * m * u - 0.5 * s**2 * u * u)

    elif t is cp.ExponentialJump:
        kind, names, vals = "exponential", ("rate", "jump_rate"), _f(P.rate, jd.rate)

        def psi(u: Expr, p: Sequence[Expr]) -> Expr:
            rate, eta = p
            iu = J * u
            return rate * iu / (eta - iu)

    elif t is cp.DoubleExponentialJump:
        kind = "double_exponential"
        names = ("rate", "jump_p", "jump_eta1", "jump_eta2")
        vals = _f(P.rate, jd.p, jd.eta1, jd.eta2)

        def psi(u: Expr, p: Sequence[Expr]) -> Expr:
            rate, q, e1, e2 = p
            return rate * _double_exp_jump(u, q, e1, e2)

    elif t is cp.UniformJump:
        kind, names, vals = (
            "uniform",
            ("rate", "jump_low", "jump_high"),
            _f(P.rate, jd.low, jd.high),
        )

        def psi(u: Expr, p: Sequence[Expr]) -> Expr:
            rate, lo, hi = p
            mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            return rate * (I.exp(J * mid * u) * I.sinc(half * u) - 1.0)

    elif t is cp.DiscreteJump:
        kind, names, vals = "discrete", ("rate",), _f(P.rate)
        values = tuple(float(v) for v in jd.values)
        probs = tuple(float(q) for q in jd.probs)

        def psi(u: Expr, p: Sequence[Expr]) -> Expr:
            (rate,) = p
            phi_m1: Expr = I.const(0j)
            for v, q in zip(values, probs):
                if q != 0.0:
                    phi_m1 = phi_m1 + q * I.expm1(J * v * u)
            return rate * phi_m1

    else:
        raise CompileError(f"no compound-Poisson rule for jump law {t.__name__}")
    extra: Dict[str, Any] = {"jump": kind}
    if kind == "discrete":
        extra.update(values=values, probs=probs)
    return LeafSpec(
        family="CompoundPoisson",
        names=names,
        values=vals,
        psi=psi,
        build=_cp_build(kind, extra),
        sampler="compound_poisson",
        extra=extra,
        reference="Sato (1999) Ex. 8.4",
    )


def supported_families() -> List[str]:
    """Class names with a lowering rule."""
    return sorted(LEAF_RULES)
