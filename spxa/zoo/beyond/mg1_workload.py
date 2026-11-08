r"""
M/G/1 workload process: compound Poisson input with unit drain, reflected at zero.

Jobs arrive as a Poisson process :math:`N_t` with rate :math:`\lambda`, bring
i.i.d. amounts of work :math:`S_1, S_2, \dots \ge 0`, and the server drains the
backlog at unit rate. With the free (net-input) process

.. math::

    X_t = \sum_{i=1}^{N_t} S_i - t,

the workload (virtual waiting time) started from :math:`W_0 = w_0 \ge 0` is the
Skorokhod reflection of :math:`w_0 + X` at zero,

.. math::

    W_t = w_0 + X_t + \sup_{s \le t} \bigl(-(w_0 + X_s)\bigr)^+ ,

which for :math:`w_0 = 0` is :math:`W_t = X_t + \sup_{s \le t}(-X_s)^+`.

Paths are right-continuous, so :math:`W(a_k)` denotes the workload immediately
after the :math:`k`-th arrival at epoch :math:`a_k` (its job :math:`S_k`
included), with :math:`a_0 = 0` and :math:`W(a_0) = w_0`. Between arrivals the
workload decreases linearly until it hits zero, so at the arrival epochs
:math:`a_1 < a_2 < \dots` it obeys the Lindley recursion

.. math::

    W(a_k) = \max\bigl(W(a_{k-1}) - (a_k - a_{k-1}),\, 0\bigr) + S_k,

and at any time :math:`t` with last arrival :math:`a_k \le t`,
:math:`W_t = \max(W(a_k) - (t - a_k), 0)`.

Stationary law
--------------
With traffic intensity :math:`\rho = \lambda E[S] < 1` the workload has a unique
stationary law :math:`W_\infty` (Pollaczek, 1930; Khinchine, 1932) with

.. math::

    P(W_\infty = 0) &= 1 - \rho, \\
    E[W_\infty] &= \frac{\lambda E[S^2]}{2(1-\rho)}, \\
    E[W_\infty^2] &= 2 E[W_\infty]^2 + \frac{\lambda E[S^3]}{3(1-\rho)}.

Higher stationary cumulants and all transient cumulants are not provided in
closed form and are returned as ``nan``.

Exactness Level: MOMENT_PROPAGATION. The reflection term depends on the whole
past of :math:`X`, so :math:`W` has neither independent nor stationary
increments and has no Lévy–Khintchine triplet.

References
----------
Pollaczek, F. (1930). Über eine Aufgabe der Wahrscheinlichkeitstheorie I.
*Mathematische Zeitschrift*, 32, 64–100.

Khinchine, A. Y. (1932). Mathematical theory of a stationary queue.
*Matematicheskii Sbornik*, 39(4), 73–84.

Lindley, D. V. (1952). The theory of queues with a single server.
*Mathematical Proceedings of the Cambridge Philosophical Society*, 48(2), 277–289.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class JobSizeLaw(Protocol):
    """Interface for a job-size law: a sampler plus raw moments ``E[S^n]``."""

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw ``size`` i.i.d. job sizes."""
        ...

    def moment(self, n: int) -> float:
        """Raw moment ``E[S^n]``."""
        ...


@dataclass(frozen=True)
class JobSizeDistribution:
    r"""
    Job-size law given by a sampler and its first three raw moments.

    Any object with ``sample(rng, size)`` and ``moment(n)`` methods, such as
    :class:`~spxa.zoo.levy.compound_poisson.ExponentialJump`, can be passed to
    :class:`MG1WorkloadProcess` directly; this class covers custom laws.

    Parameters
    ----------
    sampler : callable
        ``sampler(rng, size)`` returns ``size`` i.i.d. non-negative job sizes.
    m1, m2, m3 : float
        Raw moments :math:`E[S]`, :math:`E[S^2]` and :math:`E[S^3]`. ``m2`` and
        ``m3`` may be ``inf`` for heavy-tailed sizes.

    Examples
    --------
    Deterministic jobs of size 0.5 (M/D/1):

    >>> d = 0.5
    >>> jobs = JobSizeDistribution(lambda rng, n: np.full(n, d), d, d**2, d**3)
    """

    sampler: Callable[[np.random.Generator, int], np.ndarray]
    m1: float
    m2: float
    m3: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw ``size`` i.i.d. job sizes from ``sampler``."""
        return np.asarray(self.sampler(rng, size), dtype=np.float64)

    def moment(self, n: int) -> float:
        """Return :math:`E[S^n]` for ``n`` in 1, 2, 3."""
        moments = {1: self.m1, 2: self.m2, 3: self.m3}
        if n not in moments:
            raise ValueError(f"JobSizeDistribution only provides moments 1-3, got n={n}")
        return float(moments[n])


def _check_moment_consistency(m1: float, m2: float, m3: float) -> None:
    r"""
    Reject raw moments that no law on :math:`[0, \infty)` can have.

    Cauchy–Schwarz gives :math:`E[S]^2 \le E[S^2]` and
    :math:`E[S^2]^2 = E[S^{1/2} S^{3/2}]^2 \le E[S]\,E[S^3]`. Equality holds for a
    point mass, so a small relative tolerance absorbs rounding.
    """
    tol = 1e-12
    if m2 < m1**2 * (1.0 - tol):
        raise ValueError(
            f"Inconsistent job-size moments: E[S^2] = {m2} < E[S]^2 = {m1**2}. "
            "Pass the raw second moment, not the variance."
        )
    if math.isinf(m2) and math.isfinite(m3):
        raise ValueError("Inconsistent job-size moments: E[S^3] is finite but E[S^2] is infinite.")
    if math.isfinite(m3) and m1 * m3 < m2**2 * (1.0 - tol):
        raise ValueError(
            f"Inconsistent job-size moments: E[S] E[S^3] = {m1 * m3} < E[S^2]^2 = {m2**2}. "
            "Pass raw moments E[S^n]."
        )


class MG1WorkloadProcess(Process):
    r"""
    Workload of an M/G/1 queue (reflected compound Poisson process).

    Parameters
    ----------
    lam : float
        Poisson arrival rate :math:`\lambda > 0`.
    job_size : JobSizeLaw
        Job-size law. Any object with ``sample(rng, size)`` and ``moment(n)``
        for ``n = 1, 2, 3``, for example :class:`JobSizeDistribution` or
        :class:`~spxa.zoo.levy.compound_poisson.ExponentialJump`. Sizes must be
        non-negative with :math:`0 < E[S] < \infty`.
    w0 : float, optional
        Initial workload :math:`W_0 \ge 0`. Default 0.

    Notes
    -----
    The process may be built with :math:`\rho \ge 1` and simulated, but its
    stationary quantities then raise ``ValueError`` because the queue is not
    positive recurrent.

    ``cumulants(order)`` returns cumulants of the stationary law
    :math:`W_\infty`, not of :math:`W_t` at a finite time and not per unit time
    as for Lévy processes: :math:`\kappa_1 = E[W_\infty]`,
    :math:`\kappa_2 = \operatorname{Var}(W_\infty)` and ``nan`` for orders 3 and
    above. Transient cumulants of :math:`W_t` have no closed form here and are
    not provided; estimate them from :meth:`simulate`.

    The job-size moments are checked against the inequalities every law on
    :math:`[0, \infty)` satisfies, :math:`E[S^2] \ge E[S]^2` and
    :math:`E[S]\,E[S^3] \ge E[S^2]^2` (Cauchy–Schwarz), so a variance passed in
    place of :math:`E[S^2]`, for example, is usually caught.

    References
    ----------
    Pollaczek, F. (1930). Über eine Aufgabe der Wahrscheinlichkeitstheorie I.
    *Mathematische Zeitschrift*, 32, 64–100.

    Khinchine, A. Y. (1932). Mathematical theory of a stationary queue.
    *Matematicheskii Sbornik*, 39(4), 73–84.

    Lindley, D. V. (1952). The theory of queues with a single server.
    *Mathematical Proceedings of the Cambridge Philosophical Society*, 48(2), 277–289.

    Examples
    --------
    >>> from spxa.zoo.levy.compound_poisson import ExponentialJump
    >>> q = MG1WorkloadProcess(lam=0.5, job_size=ExponentialJump(rate=1.0))
    >>> q.rho
    0.5
    >>> q.stationary_mean
    1.0
    """

    def __init__(self, lam: float, job_size: JobSizeLaw, w0: float = 0.0) -> None:
        if not (math.isfinite(lam) and lam > 0):
            raise ValueError(f"lam must be finite and strictly positive, got {lam}")
        if not (math.isfinite(w0) and w0 >= 0):
            raise ValueError(f"w0 must be finite and non-negative, got {w0}")
        m1, m2, m3 = (float(job_size.moment(n)) for n in (1, 2, 3))
        if not (math.isfinite(m1) and m1 > 0):
            raise ValueError(f"E[S] must be finite and strictly positive, got {m1}")
        if not (m2 > 0 and m3 > 0):
            raise ValueError(f"E[S^2] and E[S^3] must be strictly positive, got {m2}, {m3}")
        _check_moment_consistency(m1, m2, m3)

        self.lam = float(lam)
        self.job_size = job_size
        self.w0 = float(w0)
        self._m1 = m1
        self._m2 = m2
        self._m3 = m3

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"MG1WorkloadProcess(λ={self.lam}, E[S]={m1:.4g}, w0={self.w0})",
            math_note=(
                "W_t = X_t + sup_{s≤t}(-X_s)^+ with X_t = Σ_{i≤N_t} S_i - t; "
                + (
                    f"ρ = {self.rho:.4g}, stationary mean {self.stationary_mean:.4g}"
                    if self.rho < 1.0
                    else f"ρ = {self.rho:.4g} ≥ 1, no stationary law"
                )
            ),
            reference="Pollaczek (1930); Khinchine (1932); Lindley (1952)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    @property
    def rho(self) -> float:
        r"""Traffic intensity :math:`\rho = \lambda E[S]`."""
        return self.lam * self._m1

    @property
    def has_stationary_distribution(self) -> bool:
        r"""True if :math:`\rho < 1` (the queue is positive recurrent)."""
        return self.rho < 1.0

    def _require_stable(self) -> None:
        if self.rho >= 1.0:
            raise ValueError(
                f"The M/G/1 workload has no stationary law for rho = lam*E[S] = {self.rho:.6g} "
                ">= 1; the Pollaczek-Khinchine formulas require rho < 1."
            )

    @property
    def idle_probability(self) -> float:
        r"""
        Stationary probability of an empty system, :math:`P(W_\infty = 0) = 1 - \rho`.

        Raises
        ------
        ValueError
            If :math:`\rho \ge 1`.
        """
        self._require_stable()
        return 1.0 - self.rho

    @property
    def stationary_mean(self) -> float:
        r"""
        Pollaczek–Khinchine mean :math:`E[W_\infty] = \lambda E[S^2] / (2(1-\rho))`.

        Raises
        ------
        ValueError
            If :math:`\rho \ge 1`.
        """
        self._require_stable()
        return self.lam * self._m2 / (2.0 * (1.0 - self.rho))

    @property
    def stationary_second_moment(self) -> float:
        r"""
        Stationary second moment
        :math:`E[W_\infty^2] = 2E[W_\infty]^2 + \lambda E[S^3] / (3(1-\rho))`.

        Raises
        ------
        ValueError
            If :math:`\rho \ge 1`.
        """
        mean = self.stationary_mean
        return 2.0 * mean**2 + self.lam * self._m3 / (3.0 * (1.0 - self.rho))

    @property
    def stationary_variance(self) -> float:
        r"""
        Stationary variance :math:`E[W_\infty^2] - E[W_\infty]^2`.

        Raises
        ------
        ValueError
            If :math:`\rho \ge 1`.
        """
        mean = self.stationary_mean
        return self.stationary_second_moment - mean**2

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "MG1WorkloadProcess is the reflection at 0 of a compound Poisson process "
                "with drift -1. The regulator sup_{s<=t}(-X_s)^+ depends on the whole past "
                "path, so increments of W are neither independent nor stationary (an "
                "increment depends on the current workload). No Lévy–Khintchine triplet "
                "exists. Use CompoundPoisson for the free input process X, or the "
                "stationary Pollaczek–Khinchine moments of this class."
            ),
        )

    def _properties(self) -> ProcessProperties:
        stable = self.rho < 1.0
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=stable and math.isfinite(self._m3),
            has_finite_mean=stable and math.isfinite(self._m2),
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "M/G/1 workload: compound Poisson input with unit drain, reflected at 0.",
                "Reflection breaks independent increments, so there is no Lévy triplet.",
                f"Traffic intensity rho = {self.rho:.6g}; stationary mean, variance and "
                "idle probability are exact for rho < 1 (Pollaczek–Khinchine).",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        r"""
        Cumulants of the stationary workload :math:`W_\infty`.

        :math:`\kappa_1` and :math:`\kappa_2` are the Pollaczek–Khinchine mean and
        variance; orders 3 and above are ``nan``. These are not cumulants of
        :math:`W_t` at a finite time, which are not available in closed form.

        Raises
        ------
        ValueError
            If :math:`\rho \ge 1`.
        """
        result: dict[int, float] = {}
        if order >= 1:
            result[1] = self.stationary_mean
        if order >= 2:
            result[2] = self.stationary_variance
        for n in range(3, order + 1):
            result[n] = float("nan")
        return result

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate workload paths exactly on an even time grid.

        Arrival epochs on :math:`[0, T]` are drawn as a Poisson number of sorted
        uniforms, the workload at each arrival follows the Lindley recursion, and
        each grid value is the workload after linear drain from the last arrival.
        There is no time-discretisation error.

        Parameters
        ----------
        n_steps : int
            Number of grid steps; the grid is ``linspace(0, T, n_steps + 1)``.
        n_paths : int, optional
            Number of independent paths. Default 1.
        T : float, optional
            Terminal time. Default 1.
        rng : numpy.random.Generator, optional
            Random number generator. Defaults to ``numpy.random.default_rng()``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_paths, n_steps + 1)``; ``paths[:, 0] == w0``.

        Raises
        ------
        ValueError
            If ``n_steps < 1``, ``n_paths < 1``, ``T <= 0`` or a sampled job size
            is negative.

        References
        ----------
        Lindley, D. V. (1952). The theory of queues with a single server.
        *Mathematical Proceedings of the Cambridge Philosophical Society*,
        48(2), 277–289.
        """
        if n_steps < 1 or n_paths < 1:
            raise ValueError(f"n_steps and n_paths must be >= 1, got {n_steps}, {n_paths}")
        if not (math.isfinite(T) and T > 0):
            raise ValueError(f"T must be finite and strictly positive, got {T}")
        if rng is None:
            rng = np.random.default_rng()

        times = np.linspace(0.0, T, n_steps + 1)
        counts = rng.poisson(self.lam * T, size=n_paths)
        width = int(np.max(counts)) + 1

        # Slot 0 is time 0 with workload w0. Unused slots sit at time T with zero
        # work, so the recursion carries the drained workload through them.
        has_job = np.arange(1, width)[None, :] <= counts[:, None]
        arrivals = np.full((n_paths, width), T, dtype=np.float64)
        arrivals[:, 0] = 0.0
        arrivals[:, 1:] = np.where(has_job, rng.uniform(0.0, T, size=(n_paths, width - 1)), T)
        arrivals.sort(axis=1)

        sizes = np.zeros((n_paths, width), dtype=np.float64)
        n_jobs = int(counts.sum())
        drawn = np.asarray(self.job_size.sample(rng, n_jobs), dtype=np.float64).reshape(-1)
        if drawn.size != n_jobs:
            raise ValueError(f"job_size.sample returned {drawn.size} values, expected {n_jobs}.")
        if not np.all(np.isfinite(drawn) & (drawn >= 0.0)):
            raise ValueError("Sampled job sizes must be finite and non-negative.")
        sizes[:, 1:][has_job] = drawn

        workload = np.empty((n_paths, width), dtype=np.float64)
        workload[:, 0] = self.w0
        for k in range(1, width):
            gap = arrivals[:, k] - arrivals[:, k - 1]
            workload[:, k] = np.maximum(workload[:, k - 1] - gap, 0.0) + sizes[:, k]

        paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        for p in range(n_paths):
            last = np.searchsorted(arrivals[p, : counts[p] + 1], times, side="right") - 1
            paths[p] = np.maximum(workload[p, last] - (times - arrivals[p, last]), 0.0)
        paths[:, 0] = self.w0
        return paths


__all__ = ["JobSizeDistribution", "JobSizeLaw", "MG1WorkloadProcess"]
