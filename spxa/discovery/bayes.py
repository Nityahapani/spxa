r"""
Model evidence for discovered structures.

:func:`~spxa.discovery.search.discover` ranks structures by BIC, which is a
large-sample approximation of :math:`-2\log p(D\mid M)`. Here the evidence

.. math::
    p(D \mid M) = \int p(D \mid x, M)\,\pi(x \mid M)\,dx

is computed directly, for a proper prior :math:`\pi` on the structure's free
coordinates :math:`x` (those of :meth:`~spxa.discovery.expr.Model.from_free`):

:func:`laplace_evidence`
    The Laplace approximation at the posterior mode :math:`\hat x`,

    .. math::
        \log p(D\mid M) \approx \ell(\hat x) + \log\pi(\hat x)
        + \tfrac{k}{2}\log 2\pi - \tfrac12\log\det H,

    where :math:`H = -\nabla^2[\ell + \log\pi](\hat x)`. The likelihood's
    Hessian is exact
    (:func:`~spxa.graph.autodiff.discovery.likelihood_value_grad_hess`), so
    the only approximation is Laplace's own. Its relative error is
    :math:`O(1/n)`.
:func:`importance_evidence`
    An unbiased importance-sampling estimate of :math:`p(D\mid M)`, with a
    multivariate-:math:`t` proposal centred at :math:`\hat x` with scale
    :math:`H^{-1}`. It returns a Monte Carlo standard error and an effective
    sample size, and checks the Laplace value where the posterior is skewed.

The likelihood :math:`\ell` is the smoothed binned likelihood of
:func:`~spxa.discovery.fit.fit_structure`,
:math:`\ell(x) = \sum_j \tilde c_j \log \tilde p_j(x)`. It omits the
multinomial coefficient, which depends on the data only. Evidences are
therefore defined up to one additive constant per set of bins, and they are
comparable across structures fitted to the same :class:`~spxa.discovery.fit.BinnedData`.
That is what posterior probabilities over structures need.

Priors
------
:class:`GaussianPrior` is a product of independent normals on the free
coordinates. :func:`default_prior` centres it at the deterministic starting
member of :func:`~spxa.discovery.fit.initial_params`, which has the data's
standard deviation spread over the summands. Its standard deviations are:

- ``scale`` (default 2) for coordinates that are logarithms, logits or
  ``arctanh`` of natural parameters. One standard deviation is then a factor of
  :math:`e^2 \approx 7.4`.
- ``scale`` times :math:`\sigma_1` for the three untransformed coordinates:
  the drift :math:`\mu`, the VG skew :math:`\theta` and the normal jump mean.
  :math:`\sigma_1` is the data's standard deviation per unit time.

It depends on the data only through :math:`\sigma_1`, like the starting
values do. Bayes factors depend on the prior widths of the parameters that
two structures do not share (Lindley's paradox), so pass a
:class:`GaussianPrior` of your own when you have real prior information.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np
from scipy import optimize

from spxa.discovery.expr import Model, Node
from spxa.discovery.fit import BinnedData, FitResult, initial_params

__all__ = [
    "REAL_COORDINATES",
    "GaussianPrior",
    "default_prior",
    "Evidence",
    "laplace_evidence",
]

#: Free coordinates that are natural parameters themselves (not logs, logits
#: or ``arctanh``), by family name. They carry the data's units.
REAL_COORDINATES: dict[str, tuple[int, ...]] = {
    "Drift": (0,),
    "VG": (2,),
    "NormalJumps": (1,),
}

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class GaussianPrior:
    """Independent normal priors :math:`x_i \\sim N(m_i, s_i^2)` on the free coordinates."""

    mean: np.ndarray
    sd: np.ndarray

    def __post_init__(self) -> None:
        m = np.atleast_1d(np.asarray(self.mean, dtype=float))
        s = np.atleast_1d(np.asarray(self.sd, dtype=float))
        if m.ndim != 1 or s.shape != m.shape:
            raise ValueError(f"mean and sd must be vectors of one length, got {m.shape}, {s.shape}")
        if not (np.all(np.isfinite(m)) and np.all(np.isfinite(s)) and np.all(s > 0)):
            raise ValueError("prior means must be finite and standard deviations positive")
        object.__setattr__(self, "mean", m)
        object.__setattr__(self, "sd", s)

    @property
    def n(self) -> int:
        return int(self.mean.size)

    def log_density(self, x: Sequence[float]) -> float:
        z = (np.asarray(x, dtype=float) - self.mean) / self.sd
        return float(-0.5 * z @ z - np.sum(np.log(self.sd)) - 0.5 * self.n * _LOG_2PI)

    def grad(self, x: Sequence[float]) -> np.ndarray:
        return -(np.asarray(x, dtype=float) - self.mean) / self.sd**2

    def hess(self) -> np.ndarray:
        return -np.diag(1.0 / self.sd**2)

    def sample(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return self.mean + self.sd * rng.standard_normal((size, self.n))


def default_prior(structure: Node, std: float, scale: float = 2.0) -> GaussianPrior:
    """The default prior for ``structure`` (see the module docstring).

    Parameters
    ----------
    structure :
        The structure as fitted, i.e. including its drift leaf.
    std :
        The data's standard deviation per unit time, ``sqrt(summary.variance)``.
    scale :
        Prior standard deviation of transformed coordinates, and the multiple of
        ``std`` for untransformed ones.
    """
    if not (std > 0 and np.isfinite(std)):
        raise ValueError(f"std must be positive and finite, got {std}")
    if not scale > 0:
        raise ValueError(f"scale must be positive, got {scale}")
    mean = Model(structure, initial_params(structure, std)).to_free()
    sd: list[float] = []
    for lf in structure.leaves():
        real = REAL_COORDINATES.get(lf.family.name, ())
        sd += [scale * std if j in real else scale for j in range(lf.family.n_params)]
    return GaussianPrior(mean, np.array(sd))


def _sample_std(binned: BinnedData) -> float:
    if binned.sample is None:
        raise ValueError("these bins were built without the sample; pass a prior explicitly")
    return float(np.std(binned.sample, ddof=1) / np.sqrt(binned.dt))


# ── Laplace ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Evidence:
    r"""The Laplace approximation of one structure's evidence.

    Attributes
    ----------
    structure :
        The fitted structure (with its drift leaf).
    prior :
        The prior on its free coordinates.
    mode :
        The posterior mode :math:`\hat x` in free coordinates.
    log_evidence :
        :math:`\log p(D\mid M)`, up to the constant shared by all structures
        fitted to the same bins.
    log_likelihood, log_prior :
        :math:`\ell(\hat x)` and :math:`\log\pi(\hat x)`.
    hessian :
        :math:`H`, the Hessian of :math:`-[\ell + \log\pi]` at the mode.
    converged :
        Whether the mode search met its gradient tolerance.
    n_obs :
        Number of observations in the bins.
    """

    structure: Node
    prior: GaussianPrior
    mode: np.ndarray
    log_evidence: float
    log_likelihood: float
    log_prior: float
    hessian: np.ndarray = field(repr=False)
    converged: bool
    n_obs: int

    @property
    def n_params(self) -> int:
        return int(self.mode.size)

    @property
    def model(self) -> Model:
        """The structure at its posterior mode."""
        return Model.from_free(self.structure, self.mode)

    @property
    def covariance(self) -> np.ndarray:
        """:math:`H^{-1}`, the Laplace posterior covariance of the free coordinates."""
        return np.asarray(np.linalg.inv(self.hessian))

    @property
    def log_det_hessian(self) -> float:
        return float(np.linalg.slogdet(self.hessian)[1])

    @property
    def log_occam_factor(self) -> float:
        r""":math:`\log p(D\mid M) - \ell(\hat x)`: what the parameters cost."""
        return self.log_evidence - self.log_likelihood

    @property
    def bic_equivalent(self) -> float:
        """:math:`-2\\log p(D\\mid M)`, on BIC's scale."""
        return -2.0 * self.log_evidence

    def draws(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """``n`` draws of the free coordinates from the Laplace posterior :math:`N(\\hat x, H^{-1})`."""
        rng = rng if rng is not None else np.random.default_rng()
        L = np.linalg.cholesky(self.covariance)
        return self.mode + rng.standard_normal((n, self.n_params)) @ L.T

    def __str__(self) -> str:
        return (
            f"{self.model.render(4)}: log evidence {self.log_evidence:.3f} "
            f"(log-likelihood {self.log_likelihood:.3f}, Occam factor {self.log_occam_factor:.3f})"
        )


class _Posterior:
    """``-[ℓ + log π]`` with its exact gradient and Hessian, cached per point."""

    def __init__(self, structure: Node, binned: BinnedData, prior: GaussianPrior) -> None:
        self.structure = structure
        self.binned = binned
        self.prior = prior
        self._x: Optional[np.ndarray] = None
        self._out: tuple[float, np.ndarray, np.ndarray] = (np.inf, np.zeros(0), np.zeros((0, 0)))

    def evaluate(self, x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        from spxa.graph.autodiff.discovery import likelihood_value_grad_hess

        x = np.asarray(x, dtype=float)
        if self._x is not None and np.array_equal(x, self._x):
            return self._out
        n = x.size
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v, g, H = likelihood_value_grad_hess(self.structure, self.binned, x)
            v -= self.prior.log_density(x)
            g = g - self.prior.grad(x)
            H = H - self.prior.hess()
            if not (np.isfinite(v) and np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
                raise FloatingPointError
        except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
            v, g, H = np.inf, np.zeros(n), np.eye(n)
        self._x, self._out = x.copy(), (float(v), g, H)
        return self._out

    def fun(self, x: np.ndarray) -> float:
        return self.evaluate(x)[0]

    def jac(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)[1]

    def hess(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)[2]


def laplace_evidence(
    fit: Union[FitResult, Node],
    binned: BinnedData,
    *,
    prior: Optional[GaussianPrior] = None,
    x0: Optional[Sequence[float]] = None,
    gtol: float = 1e-8,
    maxiter: int = 200,
) -> Evidence:
    """Laplace approximation of the evidence of a structure.

    Parameters
    ----------
    fit :
        A likelihood :class:`~spxa.discovery.fit.FitResult`, whose structure is
        used and whose maximum-likelihood point starts the mode search, or a
        bare structure (with its drift leaf) together with ``x0``.
    binned :
        The bins the structure was fitted on.
    prior :
        Prior on the free coordinates; :func:`default_prior` by default.
    x0 :
        Starting point in free coordinates (required for a bare structure).
    gtol, maxiter :
        Tolerance on the gradient norm and iteration cap of the trust-region
        Newton search (``scipy.optimize.minimize(method="trust-exact")``).

    Raises
    ------
    ValueError
        If the Hessian at the mode is not positive definite, so that no Laplace
        approximation exists there.
    """
    if isinstance(fit, FitResult):
        structure = fit.model.structure
        start = fit.model.to_free() if x0 is None else np.asarray(x0, dtype=float)
    else:
        if x0 is None:
            raise ValueError("a bare structure needs a starting point x0")
        structure, start = fit, np.asarray(x0, dtype=float)
    if prior is None:
        prior = default_prior(structure, _sample_std(binned))
    if prior.n != start.size:
        raise ValueError(f"the prior has {prior.n} coordinates, the structure {start.size}")
    post = _Posterior(structure, binned, prior)
    if not np.isfinite(post.fun(start)):
        raise ValueError(f"the posterior is not finite at the starting point of {structure.key()}")
    res = optimize.minimize(
        post.fun,
        start,
        jac=post.jac,
        hess=post.hess,
        method="trust-exact",
        options={"gtol": gtol, "maxiter": maxiter},
    )
    x = np.asarray(res.x, dtype=float)
    v, g, H = post.evaluate(x)
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"the posterior Hessian of {structure.key()} is not positive definite at its mode"
        ) from exc
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    k = x.size
    log_prior = prior.log_density(x)
    log_lik = -v - log_prior  # v = -(ℓ + log π)
    return Evidence(
        structure=structure,
        prior=prior,
        mode=x,
        log_evidence=float(-v + 0.5 * k * _LOG_2PI - 0.5 * log_det),
        log_likelihood=float(log_lik),
        log_prior=float(log_prior),
        hessian=H,
        converged=bool(np.max(np.abs(g)) <= max(gtol, 1e-6 * max(1.0, abs(v)))),
        n_obs=binned.n,
    )


def _coerce_prior(prior: Any, structure: Node, binned: BinnedData) -> GaussianPrior:
    """A prior for ``structure``: given, built by a callable, or the default."""
    if prior is None:
        return default_prior(structure, _sample_std(binned))
    if isinstance(prior, GaussianPrior):
        return prior
    if callable(prior):
        out = prior(structure)
        if not isinstance(out, GaussianPrior):
            raise TypeError("a prior callable must return a GaussianPrior")
        return out
    raise TypeError(f"prior must be a GaussianPrior, a callable or None, got {type(prior)}")
