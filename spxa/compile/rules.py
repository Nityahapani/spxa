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


def supported_families() -> List[str]:
    """Class names with a lowering rule."""
    return sorted(LEAF_RULES)
