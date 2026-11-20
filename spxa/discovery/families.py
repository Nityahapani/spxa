r"""
Primitive families: the alphabet of process discovery.

A :class:`Family` describes one parametric Lévy primitive through its
*cumulant exponent*

.. math:: K(z) = \log E[e^{zX_1}], \qquad z \in \mathbb C,

which is defined on a vertical strip :math:`\{\operatorname{Re} z \in (z_-, z_+)\}`
containing the imaginary axis. The characteristic exponent is
:math:`\psi(u) = K(iu)`. For a subordinator, :math:`K` is also needed on
:math:`\operatorname{Re} z \le 0`, where the subordination identity
:math:`K_{X \circ T} = K_T \circ K_X` evaluates it. Every family supplies:

- its cumulants :math:`\kappa_n` in closed form, to any order;
- :math:`K(z)` in closed form on the principal branch;
- the exponential-moment strip :math:`(z_-, z_+)`;
- a smooth bijection between its parameters and :math:`\mathbb R^d` for
  unconstrained optimisation;
- the spxa zoo process it stands for.

Families carry *no location parameter*. Location is one global
:data:`DRIFT` term in a composition, which keeps compositions identifiable:
``VG(θ) + NIG(μ)`` would otherwise have two interchangeable drifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Callable, Sequence

import numpy as np

from spxa.core.process import Process

Params = tuple[float, ...]


def _double_factorial(n: int) -> int:
    return 1 if n <= 0 else n * _double_factorial(n - 2)


def _normal_raw_moments(mean: float, std: float, order: int) -> np.ndarray:
    """E[Y^n], n = 1..order, for Y ~ N(mean, std²), by the recursion m_n = μ m_{n-1} + (n-1)σ² m_{n-2}."""
    m = [1.0, mean]
    for n in range(2, order + 1):
        m.append(mean * m[n - 1] + (n - 1) * std**2 * m[n - 2])
    return np.array(m[1 : order + 1])


@dataclass(frozen=True)
class Family:
    """A parametric Lévy primitive described analytically.

    Attributes
    ----------
    name :
        Short display name, e.g. ``"VG"``.
    process_name :
        The spxa zoo class it stands for, e.g. ``"VarianceGamma"``.
    param_names :
        Natural parameter names in order.
    cumulants_fn :
        ``(params, order) -> array`` of :math:`\\kappa_1..\\kappa_{order}`.
    log_mgf_fn :
        ``(z, params) -> K(z)`` for complex ``z`` in the strip.
    strip_fn :
        ``params -> (z_-, z_+)``, the open exponential-moment interval.
    to_free_fn, from_free_fn :
        The parameter bijection with :math:`\\mathbb R^d`.
    process_fn :
        ``params -> Process``.
    is_subordinator :
        Whether every member has non-decreasing paths.
    validate_fn :
        ``params -> None``; raises ``ValueError`` outside the domain.
    rescale_fn :
        ``(params, c) -> params`` of :math:`cX` for :math:`c > 0`. Every family
        here is closed under positive scaling.
    unit_scale_fn :
        ``params -> (c, unit_params)`` with :math:`X = c\\,X_{unit}`, where
        the unit member fixes the family's scale (used to write a component
        as ``c * Primitive(...)``).
    typical_fn :
        ``std -> params``, a member with standard deviation about ``std``,
        used to seed searches.
    """

    name: str
    process_name: str
    param_names: tuple[str, ...]
    cumulants_fn: Callable[[Params, int], np.ndarray] = field(repr=False)
    log_mgf_fn: Callable[[np.ndarray, Params], np.ndarray] = field(repr=False)
    strip_fn: Callable[[Params], tuple[float, float]] = field(repr=False)
    to_free_fn: Callable[[Params], np.ndarray] = field(repr=False)
    from_free_fn: Callable[[np.ndarray], Params] = field(repr=False)
    process_fn: Callable[[Params], Process] = field(repr=False)
    is_subordinator: bool
    validate_fn: Callable[[Params], None] = field(repr=False)
    rescale_fn: Callable[[Params, float], Params] = field(repr=False)
    unit_scale_fn: Callable[[Params], tuple[float, Params]] = field(repr=False)
    typical_fn: Callable[[float], Params] = field(repr=False)
    reference: str = ""

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def validate(self, params: Sequence[float]) -> Params:
        p = tuple(float(x) for x in params)
        if len(p) != self.n_params:
            raise ValueError(
                f"{self.name} takes {self.n_params} parameters {self.param_names}, got {len(p)}"
            )
        if not all(np.isfinite(p)):
            raise ValueError(f"{self.name} parameters must be finite, got {p}")
        self.validate_fn(p)
        return p

    def cumulants(self, params: Sequence[float], order: int) -> np.ndarray:
        """Closed-form :math:`\\kappa_1, \\dots, \\kappa_{order}` of :math:`X_1`."""
        if order < 1:
            raise ValueError("order must be >= 1")
        return np.asarray(self.cumulants_fn(self.validate(params), order), dtype=float)

    def log_mgf(self, z: complex | np.ndarray, params: Sequence[float]) -> np.ndarray:
        """:math:`K(z) = \\log E[e^{zX_1}]` on the principal branch."""
        return self.log_mgf_fn(np.asarray(z, dtype=complex), self.validate(params))

    def char_exponent(self, u: float | np.ndarray, params: Sequence[float]) -> np.ndarray:
        """:math:`\\psi(u) = K(iu)`."""
        return self.log_mgf(1j * np.asarray(u, dtype=float), params)

    def strip(self, params: Sequence[float]) -> tuple[float, float]:
        """Open interval :math:`(z_-, z_+)` of real :math:`z` with :math:`E[e^{zX_1}] < \\infty`."""
        return self.strip_fn(self.validate(params))

    def to_free(self, params: Sequence[float]) -> np.ndarray:
        return np.asarray(self.to_free_fn(self.validate(params)), dtype=float)

    def from_free(self, x: Sequence[float]) -> Params:
        return self.validate(self.from_free_fn(np.asarray(x, dtype=float)))

    def to_process(self, params: Sequence[float]) -> Process:
        return self.process_fn(self.validate(params))

    def rescale(self, params: Sequence[float], c: float) -> Params:
        if not c > 0:
            raise ValueError(f"rescale needs c > 0, got {c}")
        return self.validate(self.rescale_fn(self.validate(params), float(c)))

    def unit_scale(self, params: Sequence[float]) -> tuple[float, Params]:
        c, unit = self.unit_scale_fn(self.validate(params))
        return float(c), self.validate(unit)

    def typical(self, std: float = 1.0) -> Params:
        return self.validate(self.typical_fn(float(std)))

    def format(self, params: Sequence[float], digits: int = 4) -> str:
        p = self.validate(params)
        body = ", ".join(f"{n}={v:.{digits}g}" for n, v in zip(self.param_names, p))
        return f"{self.process_name}({body})"


# ── transforms ────────────────────────────────────────────────────────────────


def _logit(p: float, lo: float, hi: float) -> float:
    s = (p - lo) / (hi - lo)
    return float(np.log(s) - np.log1p(-s))


def _expit(x: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) / (1.0 + float(np.exp(-np.clip(x, -700.0, 700.0))))


def _pos(*names: str) -> Callable[[Params], None]:
    def check(p: Params) -> None:
        for n, v in zip(names, p):
            if n and v <= 0:
                raise ValueError(f"{n} must be positive, got {v}")

    return check


INF = float("inf")


# ── Drift ─────────────────────────────────────────────────────────────────────


def _drift_process(p: Params) -> Process:
    from spxa.zoo.levy.drift import Drift

    return Drift(p[0])


DRIFT = Family(
    name="Drift",
    process_name="Drift",
    param_names=("mu",),
    cumulants_fn=lambda p, N: np.array([p[0]] + [0.0] * (N - 1)),
    log_mgf_fn=lambda z, p: p[0] * z,
    strip_fn=lambda p: (-INF, INF),
    to_free_fn=lambda p: np.array([p[0]]),
    from_free_fn=lambda x: (float(x[0]),),
    process_fn=_drift_process,
    is_subordinator=False,
    validate_fn=lambda p: None,
    rescale_fn=lambda p, c: (c * p[0],),
    unit_scale_fn=lambda p: (1.0, p),
    typical_fn=lambda s: (0.0,),
    reference="Sato (1999) Ex. 8.3",
)


# ── Brownian motion ───────────────────────────────────────────────────────────


def _bm_process(p: Params) -> Process:
    from spxa.zoo.levy.brownian import BrownianMotion

    return BrownianMotion(mu=0.0, sigma=p[0])


BROWNIAN = Family(
    name="BM",
    process_name="BrownianMotion",
    param_names=("sigma",),
    cumulants_fn=lambda p, N: np.array([0.0, p[0] ** 2, *[0.0] * (N - 2)])[:N],
    log_mgf_fn=lambda z, p: 0.5 * p[0] ** 2 * z**2,
    strip_fn=lambda p: (-INF, INF),
    to_free_fn=lambda p: np.log([p[0]]),
    from_free_fn=lambda x: (float(np.exp(x[0])),),
    process_fn=_bm_process,
    is_subordinator=False,
    validate_fn=_pos("sigma"),
    rescale_fn=lambda p, c: (c * p[0],),
    unit_scale_fn=lambda p: (p[0], (1.0,)),
    typical_fn=lambda s: (s,),
    reference="Wiener (1923)",
)


# ── Gamma subordinator ────────────────────────────────────────────────────────


def _gamma_process(p: Params) -> Process:
    from spxa.zoo.levy.gamma import GammaProcess

    return GammaProcess(a=p[0], b=p[1])


GAMMA = Family(
    name="Gamma",
    process_name="GammaProcess",
    param_names=("a", "b"),
    # κ_n = a (n-1)! / b^n
    cumulants_fn=lambda p, N: np.array(
        [p[0] * factorial(n - 1) / p[1] ** n for n in range(1, N + 1)]
    ),
    # K(z) = -a log(1 - z/b)
    log_mgf_fn=lambda z, p: -p[0] * np.log(1.0 - z / p[1]),
    strip_fn=lambda p: (-INF, p[1]),
    to_free_fn=lambda p: np.log(p),
    from_free_fn=lambda x: (float(np.exp(x[0])), float(np.exp(x[1]))),
    process_fn=_gamma_process,
    is_subordinator=True,
    validate_fn=_pos("a", "b"),
    rescale_fn=lambda p, c: (p[0], p[1] / c),
    unit_scale_fn=lambda p: (1.0 / p[1], (p[0], 1.0)),
    # mean a/b = s, variance a/b² = s²: a = 1, b = 1/s
    typical_fn=lambda s: (1.0, 1.0 / s),
    reference="Madan, Carr & Chang (1998)",
)


# ── Inverse Gaussian subordinator ─────────────────────────────────────────────


def _ig_cumulants(p: Params, N: int) -> np.ndarray:
    mu, lam = p
    # κ_1 = μ, κ_n = (2n-3)!! μ^{2n-1} / λ^{n-1}
    return np.array(
        [mu]
        + [
            _double_factorial(2 * n - 3) * mu ** (2 * n - 1) / lam ** (n - 1)
            for n in range(2, N + 1)
        ]
    )


def _ig_process(p: Params) -> Process:
    from spxa.zoo.levy.inverse_gaussian import InverseGaussianProcess

    return InverseGaussianProcess(mu=p[0], lam=p[1])


INVERSE_GAUSSIAN = Family(
    name="IG",
    process_name="InverseGaussianProcess",
    param_names=("mu", "lam"),
    cumulants_fn=_ig_cumulants,
    # K(z) = (λ/μ)(1 - sqrt(1 - 2μ²z/λ))
    log_mgf_fn=lambda z, p: (p[1] / p[0]) * (1.0 - np.sqrt(1.0 - 2.0 * p[0] ** 2 * z / p[1])),
    strip_fn=lambda p: (-INF, p[1] / (2.0 * p[0] ** 2)),
    to_free_fn=lambda p: np.log(p),
    from_free_fn=lambda x: (float(np.exp(x[0])), float(np.exp(x[1]))),
    process_fn=_ig_process,
    is_subordinator=True,
    validate_fn=_pos("mu", "lam"),
    # cX ~ IG(cμ, cλ)
    rescale_fn=lambda p, c: (c * p[0], c * p[1]),
    unit_scale_fn=lambda p: (p[0], (1.0, p[1] / p[0])),
    # mean s, variance s³/λ = s²
    typical_fn=lambda s: (s, s),
    reference="Barndorff-Nielsen (1997)",
)


#: Every family, keyed by name.
FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        DRIFT,
        BROWNIAN,
        GAMMA,
        INVERSE_GAUSSIAN,
    )
}


def get_family(name: str) -> Family:
    """Look a family up by its short name."""
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown family {name!r}; known: {sorted(FAMILIES)}") from None
