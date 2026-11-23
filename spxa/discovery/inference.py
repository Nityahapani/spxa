r"""
Uncertainty of fitted parameters and tests between candidate structures.

Standard errors
---------------
The likelihood is built from smoothed counts, which are not multinomial, so
the inverse Fisher information is not the right covariance. The estimator
:math:`\hat x` (free coordinates) solves :math:`\sum_i s_i(\hat x) = 0` with
per-observation scores :math:`s_i = \sum_j K_{ij}\nabla\log\tilde p_j`. Its
covariance is the sandwich (Huber 1967; White 1982)

.. math::
    V_x = A^{-1} B A^{-1}, \qquad
    A = \sum_j \tilde c_j \nabla\log\tilde p_j\nabla\log\tilde p_j^\top, \qquad
    B = \sum_i s_i s_i^\top .

:math:`A` is the negative Hessian. The second-derivative term
:math:`\sum_j \tilde c_j\nabla^2\tilde p_j/\tilde p_j` vanishes in expectation
because the probabilities sum to one, so only first derivatives are needed.
For indicator kernels, i.e. no smoothing, :math:`B = A` and the sandwich
reduces to the inverse Fisher information.

For the cumulant method, :math:`\hat x` minimises :math:`r^\top W r` with
:math:`r = \kappa(x) - \hat\kappa`. The GMM sandwich
:math:`(G^\top W G)^{-1} G^\top W \Sigma W G (G^\top W G)^{-1}`, with
:math:`G = \partial\kappa/\partial x` and :math:`\Sigma` the sampling
covariance of :math:`\hat\kappa`, applies. The mean is matched exactly,
and this is represented by a weight on :math:`\kappa_1` far above the
others. With more parameters than the four cumulants, the parameters
are not identified and the covariance is ``NaN``. These standard errors are
asymptotic. With heavy tails the sampling distribution of :math:`\hat\kappa_4` is
skewed, and they are optimistic in small samples. In a 100-sample study of
weekly VG(0.2, 0.25, -0.3), the median GMM standard error of :math:`\nu` was:

- at n = 2000, about half the spread of the estimates (IQR/1.349);
- at n = 50000, within 15% of it.

The sandwich errors of the likelihood fit matched the spread within 15%
already at n = 2000.

Natural-parameter covariances follow by the delta method through
:meth:`~spxa.discovery.expr.Model.from_free`. All derivatives are central
finite differences.

Comparing structures
--------------------
:func:`vuong_test` compares two non-nested structures fitted to the same bins
(Vuong 1989). The per-observation differences
:math:`d_i = \ell_i^A - \ell_i^B` are i.i.d. The statistic
:math:`z = (\sum_i d_i - \text{penalty})/(\sqrt n\,\mathrm{sd}(d))` is
asymptotically standard normal when both are equally close to the truth. For
nested structures whose extra component is switched off, :math:`d_i \to 0`
and the test does not apply.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy import stats

from spxa.discovery.empirical import EmpiricalSummary
from spxa.discovery.expr import Model
from spxa.discovery.fit import BinnedData, FitResult

_KAPPA1_WEIGHT = 1e8


def _flat(model: Model) -> np.ndarray:
    return np.array([v for p in model.params for v in p], dtype=float)


def parameter_names(model: Model) -> list[str]:
    """``Family.param`` labels for the flattened parameters, with ``[k]`` on repeated families."""
    leaves = model.structure.leaves()
    counts: dict[str, int] = {}
    for lf in leaves:
        counts[lf.family.name] = counts.get(lf.family.name, 0) + 1
    seen: dict[str, int] = {}
    out: list[str] = []
    for lf in leaves:
        name = lf.family.name
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            name = f"{name}[{seen[name]}]"
        out += [f"{name}.{p}" for p in lf.family.param_names]
    return out


def _steps(x: np.ndarray, rel: float) -> np.ndarray:
    return rel * np.maximum(1.0, np.abs(x))


def _jacobian(f, x: np.ndarray, rel: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    h = _steps(x, rel)
    cols = []
    for k in range(x.size):
        e = np.zeros_like(x)
        e[k] = h[k]
        cols.append((np.asarray(f(x + e)) - np.asarray(f(x - e))) / (2 * h[k]))
    return np.stack(cols, axis=-1)


def _sandwich(L: np.ndarray, omega: np.ndarray) -> np.ndarray:
    r""":math:`A^{-1}BA^{-1}` with :math:`A = L^\top L`, :math:`B = L^\top\Omega L`, via QR.

    With :math:`L = QR`, the product is :math:`R^{-1}Q^\top\Omega QR^{-\top}`. Large
    weights, such as the one that enforces mean matching, then cause no
    cancellation. Unidentified directions (a column-scaled :math:`R` with
    condition number above :math:`10^{12}`) give ``NaN``.
    """
    k = L.shape[1]
    if not np.all(np.isfinite(L)) or L.shape[0] < k:
        return np.full((k, k), np.nan)
    norms = np.linalg.norm(L, axis=0)
    if np.any(norms == 0):
        return np.full((k, k), np.nan)
    Q, R = np.linalg.qr(L / norms)
    if np.linalg.cond(R) > 1e12:
        return np.full((k, k), np.nan)
    M = np.linalg.solve(R, Q.T) / norms[:, None]
    return M @ omega @ M.T


@dataclass(frozen=True)
class ParameterUncertainty:
    """Covariance and standard errors of a fit's natural parameters.

    Attributes
    ----------
    names :
        ``Family.param`` labels, in the order of the flattened parameters.
    estimates :
        The flattened natural parameters.
    covariance :
        Their covariance matrix; ``NaN`` where not identified.
    method :
        ``"sandwich"`` (likelihood) or ``"gmm"`` (cumulants).
    """

    names: list[str]
    estimates: np.ndarray
    covariance: np.ndarray
    method: str

    @property
    def standard_errors(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    def confidence_intervals(self, level: float = 0.95) -> np.ndarray:
        """Wald intervals, an ``(n_params, 2)`` array."""
        z = stats.norm.ppf(0.5 + level / 2)
        se = self.standard_errors
        return np.column_stack([self.estimates - z * se, self.estimates + z * se])

    def table(self, digits: int = 4) -> str:
        se = self.standard_errors
        width = max(len(n) for n in self.names)
        rows = [f"{'parameter':<{width}}  {'estimate':>12}  {'std err':>12}"]
        for n, v, s in zip(self.names, self.estimates, se):
            rows.append(f"{n:<{width}}  {v:>12.{digits}g}  {s:>12.{digits}g}")
        return "\n".join(rows)

    def __str__(self) -> str:
        return self.table()


def parameter_uncertainty(
    result: FitResult,
    summary: EmpiricalSummary,
    binned: Optional[BinnedData] = None,
    rel_step: float = 1e-5,
) -> ParameterUncertainty:
    """Standard errors of ``result``'s parameters (see the module docstring).

    ``binned`` must be the bins the fit used when they were shared, as in
    :func:`~spxa.discovery.search.discover`. Otherwise bins are rebuilt from
    ``summary.sample``.
    """
    model = result.model
    structure = model.structure
    x0 = model.to_free()
    theta = _flat(model)
    g = _jacobian(lambda x: _flat(Model.from_free(structure, x)), x0, rel_step)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if result.method == "likelihood":
            if binned is None:
                if summary.sample is None:
                    raise ValueError("the likelihood fit needs the sample or its bins")
                binned = BinnedData.from_sample(summary.sample, summary.dt)
            D = _jacobian(
                lambda x: np.log(binned.probabilities(Model.from_free(structure, x))),
                x0,
                rel_step,
            )
            root = np.sqrt(binned.counts)
            L = root[:, None] * D
            KC = binned.kernel_matrix() / np.where(root > 0, root, 1.0)
            omega = KC.T @ KC
            method = "sandwich"
        else:
            if summary.covariance is None:
                raise ValueError("cumulant standard errors need the sampling covariance (pass n=)")
            G = _jacobian(lambda x: Model.from_free(structure, x).cumulants(4), x0, rel_step)
            se = np.sqrt(np.diag(summary.covariance))
            wh = 1.0 / se
            wh[0] *= np.sqrt(_KAPPA1_WEIGHT)
            L = wh[:, None] * G
            omega = wh[:, None] * summary.covariance * wh[None, :]
            method = "gmm"

    cov_x = _sandwich(L, omega)
    cov = g @ cov_x @ g.T
    return ParameterUncertainty(parameter_names(model), theta, cov, method)


@dataclass(frozen=True)
class VuongResult:
    """Outcome of :func:`vuong_test`.

    ``z > 0`` favours the first model. ``preferred`` is ``"first"``,
    ``"second"`` or ``"neither"`` at the requested level.
    """

    z: float
    p_value: float
    preferred: str
    log_likelihood_ratio: float


def vuong_test(
    first: FitResult,
    second: FitResult,
    binned: BinnedData,
    correction: Literal["none", "aic", "bic"] = "bic",
    level: float = 0.05,
) -> VuongResult:
    """Vuong's (1989) test of two non-nested structures fitted to the same bins."""
    d = binned.pointwise_loglik(first.model) - binned.pointwise_loglik(second.model)
    n = d.size
    lr = float(d.sum())
    dk = first.n_params - second.n_params
    penalty = {"none": 0.0, "aic": float(dk), "bic": 0.5 * dk * np.log(n)}[correction]
    sd = float(np.std(d, ddof=1))
    if sd == 0.0:
        raise ValueError("identical pointwise log-likelihoods: the models coincide on these data")
    z = (lr - penalty) / (np.sqrt(n) * sd)
    p = float(2 * stats.norm.sf(abs(z)))
    preferred = "neither" if p >= level else ("first" if z > 0 else "second")
    return VuongResult(z=float(z), p_value=p, preferred=preferred, log_likelihood_ratio=lr)
