r"""
Fitting one composition structure to data.

Two objectives are available.

``"likelihood"``
    A binned multinomial likelihood of the increments at the sampling
    interval :math:`\Delta t`. Data and model are both smoothed by a narrow
    Gaussian kernel (see :class:`BinnedData`), and the bin probabilities come
    from a COS expansion of the smoothed law. Models with atoms and laws
    whose short-horizon densities are unbounded are handled the same way.
    The bins and the kernel are fixed once per data set, which makes
    log-likelihoods, AIC and BIC comparable across structures. Goodness of
    fit is a Wald-type test on the smoothed counts that accounts for their
    covariance (:meth:`BinnedData.gof_statistic`).

``"cumulants"``
    A weighted distance between the model's per-unit-time cumulants
    :math:`\kappa_2, \kappa_3, \kappa_4` and their estimates, weighted by the
    inverse sampling variances when known. Otherwise it is the sum of
    squared :attr:`FitResult.moment_errors` (variance, skewness, excess
    kurtosis over :math:`\Delta t`).
    It is the only option when the data are summary statistics.

Location comes from a :data:`~spxa.discovery.families.DRIFT` leaf, which
:func:`fit_structure` appends when ``location=True`` and the structure lacks
one. The cumulant objective *profiles* it: its value is set so that
:math:`\kappa_1` matches the estimated mean exactly. The likelihood estimates
it with the other parameters, starting from that mean-matching value.
Mean matching is not efficient for location at short horizons, where
pure-jump laws put a near-atom at the drift.
Optimisation is multi-start L-BFGS-B in the unconstrained coordinates of
:meth:`~spxa.discovery.expr.Model.from_free`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
from scipy import optimize, stats

from spxa.discovery.empirical import EmpiricalSummary
from spxa.discovery.expr import Leaf, Model, Node, Scale, Subordinate, Sum
from spxa.discovery.families import DRIFT, GAMMA, INVERSE_GAUSSIAN, POISSON

DEGENERATE_BOUNDARY = 9.0  # see FitResult.degeneracies
DEGENERATE_SHARE = 1e-4

Method = Literal["likelihood", "cumulants"]
_PENALTY = 1e12
# models whose certified mass outside the COS interval exceeds this are outside
# the domain where the binned likelihood is valid (BinnedData.outside_mass_bound)
_MAX_OUTSIDE_MASS = 1e-2


@dataclass(frozen=True)
class BinnedData:
    r"""Increments on fixed bins, with a fixed COS grid for the model's bin probabilities.

    At short horizons many pure-jump laws have unbounded densities. VG with
    :math:`\Delta t/\nu < 1/2` is one example. Their characteristic functions
    then decay like a small power of :math:`|u|`, and no Fourier inversion of
    the raw law converges at a usable rate. Both sides are therefore smoothed
    by the same Gaussian kernel of bandwidth :math:`h`:

    - each observation contributes :math:`\Phi((e_{j+1}-x)/h) - \Phi((e_j-x)/h)`
      to bin :math:`j`, rather than a unit count;
    - the model's bin probabilities are those of :math:`X_{\Delta t} + h Z`,
      whose characteristic function :math:`\varphi(u)e^{-h^2u^2/2}` decays fast.

    The Gaussian factor never vanishes, so the smoothed law still identifies the
    raw one, and the contrast stays consistent. The bandwidth is
    :math:`\max(0.02\,\hat\sigma_{\Delta t}, 9(b-a)/(\pi N_{max}))`, a few
    percent of a standard deviation in practice. The expansion is truncated at
    :math:`u = 6/h`, where the kernel factor is below :math:`1.6\times10^{-8}`. The COS interval :math:`[a, b]`
    is fixed per data set from the empirical cumulants, so the sine matrix
    is computed once and each model evaluation costs one characteristic-function
    call and one matrix product.

    Bin edges are empirical quantiles at levels evenly spaced on the logit
    scale, from :math:`q_0` to :math:`1-q_0` with
    :math:`q_0 = \min(\texttt{tail\_count}/n, 1/(2\,\texttt{n\_bins}))`. The
    outermost bins therefore hold about ``tail_count`` observations. Equal
    quantile bins put a probability mass of :math:`1/\texttt{n\_bins}` in each
    tail bin, and the likelihood then cannot see the tails. Fits to sums of
    pure-jump laws then had kurtosis several times too large. Pass
    ``tail_count=None`` for equal quantile bins.
    """

    edges: np.ndarray
    counts: np.ndarray
    raw_counts: np.ndarray
    dt: float
    bandwidth: float
    a: float
    b: float
    u: np.ndarray = field(repr=False)
    sine: np.ndarray = field(repr=False)
    kernel_covariance: np.ndarray = field(repr=False)
    sample: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return int(round(self.raw_counts.sum()))

    @property
    def n_terms(self) -> int:
        return int(self.u.size)

    @classmethod
    def from_sample(
        cls,
        x: np.ndarray,
        dt: float,
        n_bins: Optional[int] = None,
        max_terms: int = 4096,
        tail_count: Optional[float] = 10.0,
    ) -> BinnedData:
        x = np.asarray(x, dtype=float)
        if n_bins is None:
            n_bins = int(np.clip(x.size // 50, 10, 60))
        d = x - x.mean()
        c2 = float(np.mean(d**2))
        c4 = max(float(np.mean(d**4)) - 3 * c2**2, 0.0)
        half = max(12.0 * np.sqrt(c2 + np.sqrt(c4)), 2.0 * float(np.max(np.abs(d))))
        a, b = float(x.mean() - half), float(x.mean() + half)
        h = max(0.02 * np.sqrt(c2), 9.0 * (b - a) / (np.pi * max_terms))

        # quantile edges, merged so that no bin is narrower than 4h: smoothing
        # then barely couples neighbouring bins, and the smoothed counts behave
        # like multinomial counts (short-horizon pure-jump data has a near-atom
        # at the drift, where quantile bins would otherwise be ~1e-11 wide)
        inner_list: list[float] = []
        if tail_count is None:
            levels = np.arange(1, n_bins) / n_bins
        else:
            q0 = min(tail_count / x.size, 0.5 / n_bins)
            g = np.linspace(np.log(q0 / (1 - q0)), -np.log(q0 / (1 - q0)), n_bins - 1)
            levels = 1.0 / (1.0 + np.exp(-g))
        for e in np.quantile(x, levels):
            if not inner_list or e - inner_list[-1] >= 4.0 * h:
                inner_list.append(float(e))
        inner = np.array(inner_list)
        edges = np.concatenate([[-np.inf], inner, [np.inf]])
        raw, _ = np.histogram(x, bins=edges)
        # the kernel factor e^{-h²u²/2} is below 1.6e-8 beyond u = 6/h, where the
        # series is cut; the minimum bandwidth keeps that within max_terms
        n_terms = int(np.clip(np.ceil(6.0 * (b - a) / (np.pi * h)), 256, max_terms))
        k = np.arange(1, n_terms)
        u = np.arange(n_terms) * np.pi / (b - a)
        sine = np.sin(np.outer(np.pi * (inner - a) / (b - a), k)) * ((b - a) / (k * np.pi))

        z = (edges[None, :] - x[:, None]) / h
        kernel = np.diff(stats.norm.cdf(z), axis=1)
        smoothed = kernel.sum(axis=0)
        centred = kernel - kernel.mean(axis=0)
        kernel_cov = centred.T @ centred / x.size
        return cls(
            edges=edges,
            counts=smoothed,
            raw_counts=raw.astype(float),
            dt=float(dt),
            bandwidth=float(h),
            a=a,
            b=b,
            u=u,
            sine=sine,
            kernel_covariance=kernel_cov,
            sample=x,
        )

    def kernel_matrix(self) -> np.ndarray:
        r"""Per-observation kernel weights :math:`K_{ij}`, an ``(n, n_bins)`` array.

        Row :math:`i` holds :math:`\Phi((e_{j+1}-x_i)/h) - \Phi((e_j-x_i)/h)` and
        sums to one; the smoothed counts are its column sums. It is recomputed on
        each call rather than stored.
        """
        if self.sample is None:
            raise ValueError("these bins were built without keeping the sample")
        z = (self.edges[None, :] - self.sample[:, None]) / self.bandwidth
        return np.diff(stats.norm.cdf(z), axis=1)

    def pointwise_loglik(self, model: Model) -> np.ndarray:
        r"""Contributions :math:`\ell_i = \sum_j K_{ij}\log\tilde p_j` of each observation."""
        return self.kernel_matrix() @ np.log(self.probabilities(model))

    def gof_statistic(self, probabilities: np.ndarray, n_params: int) -> tuple[float, float, int]:
        r"""Wald-type goodness-of-fit test for the smoothed counts.

        The smoothed counts :math:`\tilde c = \sum_i K(x_i)` are a sum of i.i.d.
        vectors. Under the model their mean is :math:`n\tilde p` and their
        covariance is :math:`n\Sigma_K`, which is estimated from the per-observation
        kernel vectors. The statistic
        :math:`J = (\tilde c - n\tilde p)^\top (n\hat\Sigma_K)^{+} (\tilde c - n\tilde p)`
        is asymptotically :math:`\chi^2` with :math:`\mathrm{rank}(\hat\Sigma_K) - k`
        degrees of freedom. The pseudo-inverse discards the near-null directions
        created by smoothing across narrow bins.

        Returns ``(J, p_value, dof)``.
        """
        n = float(self.counts.sum())
        resid = self.counts - n * probabilities
        w, v = np.linalg.eigh(self.kernel_covariance)
        keep = w > 1e-9 * w.max()
        proj = v[:, keep].T @ resid
        J = float(np.sum(proj**2 / (n * w[keep])))
        dof = max(int(keep.sum()) - n_params, 1)
        return J, float(stats.chi2.sf(J, dof)), dof

    def outside_mass_bound(self, model: Model) -> float:
        r"""Certified bound on the mass of :math:`Y = X_{\Delta t} + hZ` outside :math:`[a, b]`.

        The cosine expansion on :math:`[a, b]` sees the law only through its
        even :math:`2(b-a)`-periodic extension: mass outside the interval folds
        back onto it, so a law and its translate by :math:`2(b-a)` give the same
        bin probabilities. The probabilities are exact, up to the series
        truncation, only when that mass is negligible. With :math:`c` the centre
        and :math:`w` the half-width of the interval, Markov's inequality on the
        fourth moment about :math:`c` bounds it by

        .. math:: P(|Y - c| \ge w) \le \frac{E[(Y-c)^4]}{w^4},\qquad
                  E[(Y-c)^4] = \kappa_4 + 3\kappa_2^2 + 4d\kappa_3 + 6d^2\kappa_2 + d^4,

        where :math:`d = \kappa_1 - c` and :math:`\kappa_j` are the exact
        cumulants of :math:`Y` (:math:`\kappa_2` includes the kernel's
        :math:`h^2`). Returns ``inf`` when a cumulant is not finite.
        """
        k = np.asarray(model.cumulants(4, t=self.dt), dtype=float)
        if not np.all(np.isfinite(k)):
            return float("inf")
        c, w = 0.5 * (self.a + self.b), 0.5 * (self.b - self.a)
        d = k[0] - c
        k2 = k[1] + self.bandwidth**2
        m4 = k[3] + 3.0 * k2**2 + 4.0 * d * k[2] + 6.0 * d**2 * k2 + d**4
        return float(m4 / w**4)

    def probabilities(self, model: Model, floor: float = 1e-300) -> np.ndarray:
        r"""Bin probabilities of :math:`X_{\Delta t} + hZ` under ``model``."""
        phi = model.char_func(self.u, t=self.dt) * np.exp(-0.5 * (self.bandwidth * self.u) ** 2)
        coef = 2.0 / (self.b - self.a) * np.real(phi * np.exp(-1j * self.u * self.a))
        inner = self.edges[1:-1]
        F = (inner - self.a) / (self.b - self.a) + self.sine @ coef[1:]
        F = np.maximum.accumulate(np.clip(np.concatenate([[0.0], F, [1.0]]), 0.0, 1.0))
        return np.maximum(np.diff(F), floor)


@dataclass(frozen=True)
class FitResult:
    """A fitted structure and its fit-quality measures."""

    model: Model
    method: str
    objective: float
    n_obs: Optional[int]
    loglik: Optional[float] = None
    g_statistic: Optional[float] = None  # Wald statistic J
    g_pvalue: Optional[float] = None
    ks_distance: Optional[float] = None
    target_cumulants: Optional[np.ndarray] = None
    converged: bool = True
    dt: Optional[float] = None
    target_standard_errors: Optional[np.ndarray] = None
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        """Free parameters, including the drift."""
        return self.model.n_params

    @property
    def aic(self) -> Optional[float]:
        return None if self.loglik is None else 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self) -> Optional[float]:
        if self.loglik is None or self.n_obs is None:
            return None
        return self.n_params * np.log(self.n_obs) - 2 * self.loglik

    @property
    def model_cumulants(self) -> np.ndarray:
        return self.model.cumulants(4)

    @property
    def cumulant_relative_errors(self) -> Optional[np.ndarray]:
        r""":math:`|\kappa_j^{model} - \hat\kappa_j| / |\hat\kappa_j|` for j = 1..4."""
        if self.target_cumulants is None:
            return None
        t = self.target_cumulants
        return np.abs(self.model_cumulants - t) / np.maximum(np.abs(t), 1e-300)

    @property
    def max_relative_error(self) -> Optional[float]:
        """Largest relative error over :math:`\\kappa_2..\\kappa_4` (the mean is matched exactly)."""
        e = self.cumulant_relative_errors
        return None if e is None else float(np.max(e[1:]))

    @property
    def moment_errors(self) -> Optional[np.ndarray]:
        r"""Errors in the variance, skewness and excess kurtosis of the increment over ``dt``.

        The returned values are, in order:

        - the relative error of the variance;
        - the error of the skewness, divided by :math:`\max(|\hat s|, 1)`;
        - the error of the excess kurtosis, divided by :math:`\max(|\hat k|, 1)`.

        So these are relative errors when the statistic is large, and
        absolute errors when it is small. Unlike raw cumulant relative errors,
        they stay meaningful when a target such as the skewness is close to zero.

        Read them next to :attr:`cumulant_z_scores`. With heavy tails, the
        sample skewness and kurtosis are dominated by a handful of
        observations. A likelihood fit then legitimately misses them by tens
        of percent, while every cumulant still lies within one or two standard
        errors of its estimate. For example, on 5000 weekly VG + Gamma returns
        (population excess kurtosis about 94), fits with moment errors of
        30-50% have all four cumulant z-scores below 1.
        """
        if self.target_cumulants is None or self.dt is None:
            return None
        return _moment_errors(self.model_cumulants, self.target_cumulants, self.dt)

    @property
    def match_error(self) -> Optional[float]:
        """Largest of :attr:`moment_errors`: "matches your data within x%"."""
        e = self.moment_errors
        return None if e is None else float(np.max(e))

    @property
    def cumulant_z_scores(self) -> Optional[np.ndarray]:
        r""":math:`(\kappa_j^{model} - \hat\kappa_j)/\mathrm{se}(\hat\kappa_j)` for j = 1..4."""
        if self.target_cumulants is None or self.target_standard_errors is None:
            return None
        return (self.model_cumulants - self.target_cumulants) / np.maximum(
            self.target_standard_errors, 1e-300
        )

    @property
    def structure_key(self) -> str:
        return self.model.structure.key()

    def degeneracies(
        self, boundary: float = DEGENERATE_BOUNDARY, share: float = DEGENERATE_SHARE
    ) -> list[str]:
        r"""Reasons to read this fit as a limiting case rather than as its nominal structure.

        Two signs are checked:

        - **boundary**: an unconstrained coordinate (log or logit of a
          parameter; see :meth:`Family.to_free`) exceeds ``boundary`` in
          absolute value. The default 9 means a parameter more than
          :math:`e^9 \approx 8100` times away from 1, or within about 1e-4
          of an interval bound. Such fits sit at an edge of the family.
          ``NIG(alpha=9e5, beta=9e5, ...)`` is the one-sided limit and
          ``CGMY(Y=5e-5)`` is the variance-gamma-like limit. Drift
          coordinates are exempt.
        - **vanishing summand**: a top-level summand supplies less than
          ``share`` of the variance of the sum, so the structure reduces to
          its parent. ``BM(sigma=1e-10) + VG(...)`` is an example.

        The boundary check depends on units: parameters are per unit time
        and returns are in natural units (decimals, not percent). An empty
        list means no sign of degeneracy was found.
        """
        from spxa.discovery.inference import parameter_names

        out: list[str] = []
        names = parameter_names(self.model)
        x = self.model.to_free()
        i = 0
        for lf in self.model.structure.leaves():
            k = lf.family.n_params
            if lf.family is not DRIFT:
                for j in range(i, i + k):
                    if not np.isfinite(x[j]) or abs(x[j]) > boundary:
                        out.append(
                            f"{names[j]} at a boundary of its family (free coordinate {x[j]:.3g})"
                        )
            i += k
        s = self.model.structure
        if isinstance(s, Sum):
            parts, i = [], 0
            for child in s.children:
                n = len(child.leaves())
                if not all(lf.family is DRIFT for lf in child.leaves()):
                    sub = Model(child, self.model.params[i : i + n])
                    parts.append((child.key(), float(sub.cumulants(2)[1])))
                i += n
            total = sum(v for _, v in parts)
            # an infinite part (a rate that underflowed) is already at a boundary
            if len(parts) > 1 and 0 < total < np.inf:
                for key, v in parts:
                    if v / total < share:
                        out.append(f"{key} carries {100 * v / total:.2g}% of the variance")
        return out

    @property
    def degenerate(self) -> bool:
        """``True`` if :meth:`degeneracies` finds anything at the default thresholds."""
        return bool(self.degeneracies())

    def __str__(self) -> str:
        parts = [self.model.render()]
        if self.g_pvalue is not None:
            parts.append(f"goodness-of-fit p={self.g_pvalue:.3g}")
        if self.ks_distance is not None:
            parts.append(f"max CDF gap {100 * self.ks_distance:.2f}%")
        if self.match_error is not None:
            parts.append(f"moment error {100 * self.match_error:.2f}%")
        return " | ".join(parts)


def _standardized(k: np.ndarray, dt: float) -> np.ndarray:
    """Variance, skewness and excess kurtosis over ``dt`` from per-unit-time κ1..κ4."""
    var = k[1] * dt
    return np.array([var, k[2] * dt / var**1.5, k[3] * dt / var**2])


def _moment_errors(model: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
    m, t = _standardized(model, dt), _standardized(target, dt)
    scale = np.array([abs(t[0]), max(abs(t[1]), 1.0), max(abs(t[2]), 1.0)])
    return np.abs(m - t) / scale


def _has_drift(structure: Node) -> bool:
    return any(lf.family is DRIFT for lf in structure.leaves())


def with_location(structure: Node) -> Node:
    """``structure + Drift`` unless a drift leaf is already present."""
    return structure if _has_drift(structure) else structure + Leaf(DRIFT)


def _clock_init(family_name: str) -> tuple[float, ...]:
    if family_name == GAMMA.name:
        return (2.0, 2.0)
    if family_name == INVERSE_GAUSSIAN.name:
        return (1.0, 2.0)
    if family_name == POISSON.name:
        return (1.0, 1.0)
    raise KeyError(family_name)


def initial_params(structure: Node, std: float) -> list[tuple[float, ...]]:
    """Deterministic starting values: total standard deviation ``std`` split over the summands."""

    def walk(node: Node, s: float, clock: bool) -> list[tuple[float, ...]]:
        if isinstance(node, Leaf):
            if node.family is DRIFT:
                return [(0.0,)]
            if clock:
                try:
                    return [_clock_init(node.family.name)]
                except KeyError:
                    pass
            return [node.family.typical(s)]
        if isinstance(node, Scale):
            return walk(node.child, s / abs(node.c), clock)
        if isinstance(node, Sum):
            k = sum(1 for c in node.children if not (isinstance(c, Leaf) and c.family is DRIFT))
            out: list[tuple[float, ...]] = []
            for c in node.children:
                out += walk(c, s / np.sqrt(max(k, 1)), clock)
            return out
        if isinstance(node, Subordinate):
            return walk(node.base, s, False) + walk(node.clock, s, True)
        raise TypeError(type(node).__name__)

    return walk(structure, std, False)


def _drift_index(structure: Node) -> Optional[int]:
    """Position of the first drift parameter in the free vector."""
    i = 0
    for lf in structure.leaves():
        if lf.family is DRIFT:
            return i
        i += lf.family.n_params
    return None


class _Objective:
    def __init__(
        self,
        structure: Node,
        summary: EmpiricalSummary,
        method: Method,
        binned: Optional[BinnedData],
    ):
        self.structure = structure
        self.summary = summary
        self.method = method
        self.binned = binned
        # κ_1 is affine in the drift parameter wherever the drift sits (a
        # scale multiplies it by c, a clock by E[T_1]), so it can be profiled
        self.drift_at = _drift_index(structure)
        target = summary.cumulants
        if summary.covariance is not None:
            se = np.sqrt(np.diag(summary.covariance))
            self.weights: Optional[np.ndarray] = 1.0 / np.maximum(se, 1e-300) ** 2
        else:
            # no sampling variances: least squares on the moment errors, which
            # stay finite when a target (typically the skewness) is zero
            self.weights = None
        self.target = target

    def model(self, x: np.ndarray) -> Model:
        if self.method == "cumulants":
            return self.matched(x)
        return Model.from_free(self.structure, x)

    def matched(self, x: np.ndarray) -> Model:
        """The model at ``x`` with the drift set so that κ_1 equals the target mean."""
        x = np.array(x, dtype=float)
        if self.drift_at is not None:
            x[self.drift_at] = 0.0
            base = Model.from_free(self.structure, x)
            slope = (
                Model.from_free(self.structure, _unit(x, self.drift_at)).cumulants(1)[0]
                - base.cumulants(1)[0]
            )
            x[self.drift_at] = (self.target[0] - base.cumulants(1)[0]) / slope
        return Model.from_free(self.structure, x)

    def __call__(self, x: np.ndarray) -> float:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = self.model(x)
                if self.method == "cumulants":
                    k = m.cumulants(4)
                    if self.weights is None:
                        e = _moment_errors(k, self.target, self.summary.dt)
                        return float(np.sum(e**2))
                    r = k - self.target
                    return float(np.sum(self.weights[1:] * r[1:] ** 2))
                assert self.binned is not None
                # the cosine series folds mass outside [a, b] back onto it, which
                # only ever raises the likelihood: a law that leaves the interval
                # (a translate by 2(b - a), a vanishing-rate Gamma spreading mass
                # over the whole line) can look better than the truth
                if not self.binned.outside_mass_bound(m) <= _MAX_OUTSIDE_MASS:
                    return _PENALTY
                val = -float(self.binned.counts @ np.log(self.binned.probabilities(m)))
            return val if np.isfinite(val) else _PENALTY
        except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
            return _PENALTY

    def value_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """The objective and its exact gradient (zero on the penalty plateau)."""
        from spxa.graph.autodiff import discovery as ad

        val = self(x)
        if val >= _PENALTY:
            return val, np.zeros_like(x)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.method == "cumulants":
                    _, g = ad.cumulant_value_and_grad(
                        self.structure,
                        self.target,
                        x,
                        self.summary.dt,
                        self.weights,
                        self.drift_at,
                    )
                else:
                    assert self.binned is not None
                    _, g = ad.likelihood_value_and_grad(self.structure, self.binned, x)
        except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError, TypeError):
            g = np.full_like(x, np.nan)
        if not np.all(np.isfinite(g)):
            # the objective is finite but a derivative is not representable here
            # (e.g. at a clip boundary); fall back to differences for this step
            g = optimize.approx_fprime(x, self, 1.4901161193847656e-08)
        return val, g


def _unit(x: np.ndarray, i: int) -> np.ndarray:
    y = np.array(x, dtype=float)
    y[i] = 1.0
    return y


def fit_structure(
    structure: Node,
    summary: EmpiricalSummary,
    method: Method = "likelihood",
    *,
    location: bool = True,
    n_starts: int = 4,
    rng: Optional[np.random.Generator] = None,
    binned: Optional[BinnedData] = None,
    maxiter: int = 400,
    initial: Optional[Sequence[Sequence[float]]] = None,
    gradient: str = "exact",
) -> FitResult:
    """Fit ``structure`` to ``summary``.

    Parameters
    ----------
    structure :
        The composition to fit. A drift leaf is appended when ``location`` is
        true and none is present.
    summary :
        Targets from :func:`~spxa.discovery.empirical.summarize` or
        :func:`~spxa.discovery.empirical.from_statistics`.
    method :
        ``"likelihood"`` needs ``summary.sample``; ``"cumulants"`` always works.
    n_starts :
        Starting points: the deterministic initial values plus
        ``n_starts - 1`` random perturbations from ``rng``.
    binned :
        Pre-computed bins. :func:`~spxa.discovery.search.discover` shares one
        set across all candidates so that likelihoods are comparable.
    initial :
        Natural parameters per leaf of the (location-augmented) structure,
        used as the first starting point instead of the generic initial
        values. Beam search passes the parent's fit with the new atom's
        typical member.
    gradient :
        ``"exact"`` (default) passes L-BFGS-B the exact gradient of the
        objective (:mod:`spxa.graph.autodiff.discovery`); ``"2-point"`` lets
        SciPy difference it, as in 1.2.
    """
    if gradient not in ("exact", "2-point"):
        raise ValueError("gradient must be 'exact' or '2-point'")
    if location:
        structure = with_location(structure)
    if method == "likelihood":
        if binned is None:
            if summary.sample is None:
                raise ValueError(
                    "the likelihood needs a sample; use method='cumulants' for summary statistics"
                )
            binned = BinnedData.from_sample(summary.sample, summary.dt)
    elif method != "cumulants":
        raise ValueError(f"unknown method {method!r}")
    rng = rng if rng is not None else np.random.default_rng(0)
    obj = _Objective(structure, summary, method, binned)

    std = float(np.sqrt(max(summary.variance, 1e-300)))
    x0 = Model(
        structure, initial if initial is not None else initial_params(structure, std)
    ).to_free()
    raw = [x0] + [x0 + rng.normal(scale=0.7, size=x0.size) for _ in range(max(n_starts, 1) - 1)]
    starts = [x0] if initial is not None else []
    for x in raw[1:] if initial is not None else raw:
        try:
            starts.append(obj.matched(x).to_free())
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
    best: Optional[optimize.OptimizeResult] = None
    for start in starts:
        if obj(start) >= _PENALTY:
            continue
        with warnings.catch_warnings():
            # finite differences straddling the penalty wall produce inf - inf
            warnings.simplefilter("ignore", RuntimeWarning)
            res = optimize.minimize(
                obj.value_and_grad if gradient == "exact" else obj,
                start,
                jac=gradient == "exact",
                method="L-BFGS-B",
                options={"maxiter": maxiter},
            )
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise RuntimeError(f"no admissible starting point for {structure.key()}")

    model = obj.model(best.x)
    loglik = g = g_p = ks = None
    if binned is not None:
        p = binned.probabilities(model)
        counts = binned.counts
        loglik = float(counts @ np.log(p))
        g, g_p, _ = binned.gof_statistic(p, model.n_params)
        ks = float(np.max(np.abs(np.cumsum(counts) / counts.sum() - np.cumsum(p))))
    return FitResult(
        model=model,
        method=method,
        objective=float(best.fun),
        n_obs=summary.n,
        loglik=loglik,
        g_statistic=g,
        g_pvalue=g_p,
        ks_distance=ks,
        target_cumulants=summary.cumulants.copy(),
        converged=bool(best.success),
        dt=summary.dt,
        target_standard_errors=None
        if summary.covariance is None
        else np.sqrt(np.diag(summary.covariance)),
    )
