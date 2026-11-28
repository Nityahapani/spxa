r"""
Sequential Monte Carlo evidence for discovered structures.

:func:`~spxa.discovery.bayes.importance_evidence` is unbiased, but it relies on
one proposal, the Laplace-centred multivariate :math:`t`. When a posterior is
far from Gaussian, for example a weakly identified jump part, only a few
percent of its draws carry weight. Then the estimate and its standard error
both rest on those few draws.

:func:`smc_evidence` reaches the posterior gradually instead. It starts from
the same normalised reference :math:`q` and moves a population of particles
through the geometric bridge

.. math::
    \gamma_\beta(x) = q(x)^{1-\beta}\,\big[L(x)\,\pi(x)\big]^{\beta},
    \qquad 0 = \beta_0 < \beta_1 < \dots < \beta_T = 1

(Neal 2001; Del Moral, Doucet & Jasra 2006). Each step reweights the
particles by :math:`(L\pi/q)^{\beta_{t}-\beta_{t-1}}`, resamples when the
effective sample size drops below half, and moves them with random-walk
Metropolis kernels that leave :math:`\gamma_{\beta_t}` invariant. The product
of the mean incremental weights estimates
:math:`Z = \int L\pi\,dx`, because :math:`q` is normalised.

- **Temperatures** are chosen adaptively. The next :math:`\beta` keeps the
  conditional effective sample size
  :math:`N\,(\sum_i W_i u_i)^2 / \sum_i W_i u_i^2` at a fraction ``cess`` of
  :math:`N`, where :math:`u_i` are the incremental weights (Zhou, Johansen &
  Aston 2016).
- **The proposal covariance** is the particles' weighted covariance. Its scale
  starts at :math:`2.38/\sqrt d` (Roberts & Rosenthal 2001) and is adapted
  towards an acceptance rate of 0.3.
- **The standard error** comes from ``n_runs`` independent runs. Their
  estimates :math:`\hat Z_k` are averaged on the natural scale, and
  :math:`\operatorname{se}(\log\hat Z) = \operatorname{sd}(\hat Z_k)/(\sqrt K\,\bar Z)`.
  With :math:`K = 4` the standard error is itself only good to about 40%.

Every evaluation of :math:`L` costs one binned-likelihood call, about a
millisecond. A run costs about :math:`N(1 + \text{n\_moves}\cdot T)` calls,
where :math:`T` is the number of temperatures. :math:`T` is small when
:math:`q` is close to the posterior and grows as they part.

Like any method started from the Laplace reference, this one finds posterior
modes far from :math:`\hat x` only through :math:`q`'s tails.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import optimize, stats
from scipy.special import logsumexp

from spxa.discovery.bayes import Evidence, _log_target
from spxa.discovery.fit import BinnedData

__all__ = ["SMCRun", "tempered_smc", "SMCEvidence", "smc_evidence"]

_TARGET_ACCEPTANCE = 0.3


@dataclass(frozen=True)
class SMCRun:
    """One run of :func:`tempered_smc`.

    Attributes
    ----------
    log_evidence :
        :math:`\\log\\hat Z` of the run.
    temperatures :
        :math:`\\beta_0 = 0, \\dots, \\beta_T = 1`.
    acceptance :
        Mean Metropolis acceptance rate at each temperature after the first.
    particles, log_weights :
        The final weighted particles, which approximate the posterior. The
        weights are normalised: ``logsumexp(log_weights) == 0``.
    n_evaluations :
        Number of target evaluations.
    """

    log_evidence: float
    temperatures: np.ndarray
    acceptance: np.ndarray
    particles: np.ndarray = field(repr=False)
    log_weights: np.ndarray = field(repr=False)
    n_evaluations: int


def _systematic(log_w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = log_w.size
    c = np.cumsum(np.exp(log_w - logsumexp(log_w)))
    c[-1] = 1.0
    return np.searchsorted(c, (rng.random() + np.arange(n)) / n)


def _log_cess(log_w: np.ndarray, log_ratio: np.ndarray, delta: float) -> float:
    """log of N (sum W u)^2 / sum W u^2 for u = exp(delta * log_ratio), W normalised."""
    with np.errstate(invalid="ignore"):
        lu = np.where(np.isneginf(log_ratio), -np.inf, delta * log_ratio)
    return float(np.log(log_w.size) + 2 * logsumexp(log_w + lu) - logsumexp(log_w + 2 * lu))


def tempered_smc(
    log_target: Callable[[np.ndarray], float],
    loc: np.ndarray,
    shape: np.ndarray,
    *,
    df: float = 5.0,
    n_particles: int = 400,
    n_moves: int = 3,
    cess: float = 0.95,
    max_steps: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> SMCRun:
    r"""Estimate :math:`\log\int e^{f(x)}dx` by SMC from a multivariate-:math:`t` reference.

    Parameters
    ----------
    log_target :
        :math:`f`, the unnormalised log density, which returns ``-inf`` where
        the density is zero.
    loc, shape, df :
        The reference :math:`q = t_{df}(\text{loc}, \text{shape})`.
    n_particles, n_moves :
        Population size, and Metropolis sweeps per temperature.
    cess :
        Conditional effective-sample-size fraction kept between temperatures.
    max_steps :
        Cap on the number of temperatures; exceeding it raises.
    """
    if n_particles < 2 or n_moves < 1 or max_steps < 1:
        raise ValueError("n_particles must be at least 2, n_moves and max_steps at least 1")
    if not 0 < cess < 1:
        raise ValueError("cess must lie in (0, 1)")
    rng = rng if rng is not None else np.random.default_rng(0)
    loc = np.atleast_1d(np.asarray(loc, dtype=float))
    d = loc.size
    shape = np.asarray(shape, dtype=float).reshape(d, d)
    q = stats.multivariate_t(loc=loc, shape=0.5 * (shape + shape.T), df=df, seed=rng)

    def log_q(x: np.ndarray) -> np.ndarray:
        return np.asarray(q.logpdf(x), dtype=float).reshape(-1)

    def f(x: np.ndarray) -> float:
        v = float(log_target(x))
        return v if np.isfinite(v) else -np.inf

    n = int(n_particles)
    x = np.asarray(q.rvs(size=n), dtype=float).reshape(n, d)
    lq = log_q(x)
    lp = np.array([f(xi) for xi in x])
    evals = n
    if not np.any(np.isfinite(lp)):
        raise ValueError("the target is zero at every reference draw")
    log_w = np.full(n, -np.log(n))
    beta, log_z = 0.0, 0.0
    betas, accs = [0.0], []
    scale = 2.38 / np.sqrt(d)
    target = np.log(cess * n)
    while beta < 1.0:
        if len(betas) > max_steps:
            raise RuntimeError(f"SMC needed more than {max_steps} temperatures")
        lr = lp - lq
        # particles where the target is zero drop out at any positive step; aim
        # the conditional ESS at a fraction of the survivors
        goal = min(target, _log_cess(log_w, lr, 0.0) + np.log(cess))
        if _log_cess(log_w, lr, 1.0 - beta) >= goal:
            delta = 1.0 - beta
        else:
            delta = optimize.brentq(
                lambda t: _log_cess(log_w, lr, t) - goal, 0.0, 1.0 - beta, xtol=1e-12
            )
            delta = max(delta, 1e-12)
        with np.errstate(invalid="ignore"):
            inc = np.where(np.isneginf(lr), -np.inf, delta * lr)
        log_z += float(logsumexp(log_w + inc))
        log_w = log_w + inc
        log_w -= logsumexp(log_w)
        beta = 1.0 if 1.0 - (beta + delta) < 1e-12 else beta + delta
        betas.append(beta)
        if 1.0 / np.sum(np.exp(2 * log_w)) < 0.5 * n:
            idx = _systematic(log_w, rng)
            x, lq, lp = x[idx], lq[idx], lp[idx]
            log_w = np.full(n, -np.log(n))
        # random-walk Metropolis moves invariant for gamma_beta
        w = np.exp(log_w)
        cov = np.atleast_2d(np.cov(x.T, aweights=w)) if n > 1 else np.eye(d)
        cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(d) * max(np.trace(cov) / d, 1e-300)
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-300)))
        accepted = 0
        for _ in range(n_moves):
            prop = x + scale * rng.standard_normal((n, d)) @ chol.T
            lq_p = log_q(prop)
            lp_p = np.array([f(xi) for xi in prop])
            evals += n
            with np.errstate(invalid="ignore"):
                cur = (1 - beta) * lq + np.where(np.isneginf(lp), -np.inf, beta * lp)
                new = (1 - beta) * lq_p + np.where(np.isneginf(lp_p), -np.inf, beta * lp_p)
                # -inf - (-inf) is nan and compares False; a finite proposal from a
                # zero-density particle has ratio +inf and is accepted
                ok = np.log(rng.random(n)) < new - cur
            ok &= np.isfinite(new)
            x[ok], lq[ok], lp[ok] = prop[ok], lq_p[ok], lp_p[ok]
            rate = float(np.mean(ok))
            accepted += int(ok.sum())
            scale *= float(np.exp(rate - _TARGET_ACCEPTANCE))
        accs.append(accepted / (n * n_moves))
    return SMCRun(
        log_evidence=float(log_z),
        temperatures=np.array(betas),
        acceptance=np.array(accs),
        particles=x,
        log_weights=log_w,
        n_evaluations=evals,
    )


@dataclass(frozen=True)
class SMCEvidence:
    """SMC estimate of one structure's evidence.

    Attributes
    ----------
    log_evidence :
        :math:`\\log\\bar Z`, the log of the mean of the runs' estimates.
    standard_error :
        :math:`\\operatorname{sd}(\\hat Z_k)/(\\sqrt K\\,\\bar Z)` over the runs.
    runs :
        The :class:`SMCRun` of each independent run.
    laplace :
        The Laplace fit that supplied the reference.
    """

    log_evidence: float
    standard_error: float
    runs: list[SMCRun] = field(repr=False)
    laplace: Evidence = field(repr=False)

    @property
    def run_log_evidences(self) -> np.ndarray:
        return np.array([r.log_evidence for r in self.runs])

    @property
    def laplace_error(self) -> float:
        """Laplace minus SMC, in nats."""
        return self.laplace.log_evidence - self.log_evidence

    @property
    def n_evaluations(self) -> int:
        return sum(r.n_evaluations for r in self.runs)

    @property
    def n_temperatures(self) -> np.ndarray:
        return np.array([r.temperatures.size - 1 for r in self.runs])

    def weighted_particles(self) -> tuple[np.ndarray, np.ndarray]:
        r"""All runs' particles with normalised weights.

        Run :math:`k` enters in proportion to :math:`\hat Z_k`, which keeps the
        pooled measure consistent for the posterior (island particle schemes).
        """
        z = self.run_log_evidences
        x = np.concatenate([r.particles for r in self.runs])
        lw = np.concatenate([r.log_weights + zk for r, zk in zip(self.runs, z)])
        return x, np.exp(lw - logsumexp(lw))

    def draws(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """``n`` posterior draws in free coordinates, resampled from the particles."""
        rng = rng if rng is not None else np.random.default_rng(0)
        x, w = self.weighted_particles()
        return x[rng.choice(w.size, size=n, p=w)]


def smc_evidence(
    evidence: Evidence,
    binned: BinnedData,
    *,
    n_particles: int = 400,
    n_runs: int = 4,
    n_moves: int = 3,
    cess: float = 0.95,
    df: float = 5.0,
    inflate: float = 1.2,
    max_steps: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> SMCEvidence:
    r"""SMC estimate of :math:`\log p(D\mid M)` from a Laplace fit (see the module docstring).

    The reference is the multivariate :math:`t_{df}` at the mode with scale
    :math:`\text{inflate}^2 H^{-1}`, the proposal of
    :func:`~spxa.discovery.bayes.importance_evidence`. The target is
    :math:`\ell + \log\pi` under the prior of ``evidence``, on the same bins.
    """
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2 for a standard error")
    if not (df > 0 and inflate > 0):
        raise ValueError("df and inflate must be positive")
    rng = rng if rng is not None else np.random.default_rng(0)
    target = _log_target(evidence, binned)
    shape = inflate**2 * evidence.covariance
    runs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_runs):
            runs.append(
                tempered_smc(
                    target,
                    evidence.mode,
                    shape,
                    df=df,
                    n_particles=n_particles,
                    n_moves=n_moves,
                    cess=cess,
                    max_steps=max_steps,
                    rng=rng,
                )
            )
    z = np.array([r.log_evidence for r in runs])
    log_z = float(logsumexp(z) - np.log(n_runs))
    rel = np.exp(z - log_z)  # Z_k / Z_bar
    se = float(np.std(rel, ddof=1) / np.sqrt(n_runs))
    return SMCEvidence(log_evidence=log_z, standard_error=se, runs=runs, laplace=evidence)
