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

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from spxa.discovery.expr import Model, Node
from spxa.discovery.fit import BinnedData, initial_params

__all__ = [
    "REAL_COORDINATES",
    "GaussianPrior",
    "default_prior",
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
