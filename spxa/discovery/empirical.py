r"""
Empirical summaries of a return series: the targets of process discovery.

For increments :math:`x_1, \dots, x_n` of a Lévy process sampled every
:math:`\Delta t`, the increments are i.i.d. with cumulants
:math:`\kappa_j \Delta t`. The unbiased *k-statistics* :math:`k_j` (Fisher 1930)
estimate them, and :math:`k_j/\Delta t` estimates the per-unit-time cumulants
that a model must match.

Their sampling covariance follows from the influence functions of the
cumulants (with :math:`d = x - \mu` and :math:`m_r` the central moments):

.. math::
   \mathrm{IF}_1 = d, \quad
   \mathrm{IF}_2 = d^2 - m_2, \quad
   \mathrm{IF}_3 = d^3 - m_3 - 3m_2 d, \quad
   \mathrm{IF}_4 = d^4 - m_4 - 4m_3 d - 6m_2(d^2 - m_2),

so :math:`\mathrm{Cov}(\hat\kappa) \approx \frac1n \mathrm{Cov}(\mathrm{IF})`. A
Bartlett (Newey–West) long-run covariance replaces the plain covariance when
``hac_lags > 0``. A Lévy model also implies zero autocorrelation of returns
*and* of squared returns. :func:`summarize` reports both, with Ljung–Box
tests, so that discovery can flag data that no Lévy composition can explain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats


def k_statistics(x: Sequence[float]) -> np.ndarray:
    """Unbiased estimators :math:`k_1..k_4` of the first four cumulants (Fisher 1930)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}")
    d = x - x.mean()
    m2, m3, m4 = (np.mean(d**r) for r in (2, 3, 4))
    k2 = n / (n - 1) * m2
    k3 = n**2 / ((n - 1) * (n - 2)) * m3
    k4 = n**2 * ((n + 1) * m4 - 3 * (n - 1) * m2**2) / ((n - 1) * (n - 2) * (n - 3))
    return np.array([x.mean(), k2, k3, k4])


def _influence(x: np.ndarray) -> np.ndarray:
    d = x - x.mean()
    m2, m3, m4 = (np.mean(d**r) for r in (2, 3, 4))
    return np.column_stack(
        [
            d,
            d**2 - m2,
            d**3 - m3 - 3 * m2 * d,
            d**4 - m4 - 4 * m3 * d - 6 * m2 * (d**2 - m2),
        ]
    )


def cumulant_covariance(x: Sequence[float], hac_lags: int = 0) -> np.ndarray:
    """Asymptotic covariance of :math:`(\\hat\\kappa_1, .., \\hat\\kappa_4)` per observation.

    Parameters
    ----------
    x :
        Observations.
    hac_lags :
        Bartlett-kernel lags for a long-run (Newey–West) covariance. 0 means
        i.i.d. increments.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    f = _influence(x)
    f = f - f.mean(axis=0)
    cov = f.T @ f / n
    for lag in range(1, hac_lags + 1):
        w = 1.0 - lag / (hac_lags + 1.0)
        g = f[lag:].T @ f[:-lag] / n
        cov += w * (g + g.T)
    return np.asarray(cov / n)


def acf(x: Sequence[float], max_lag: int) -> np.ndarray:
    """Sample autocorrelations :math:`\\hat\\rho_1..\\hat\\rho_{max\\_lag}`."""
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    denom = float(d @ d)
    if denom == 0:
        return np.zeros(max_lag)
    return np.array([float(d[lag:] @ d[:-lag]) / denom for lag in range(1, max_lag + 1)])


def ljung_box(x: Sequence[float], max_lag: int) -> tuple[float, float]:
    """Ljung–Box statistic :math:`Q = n(n+2)\\sum_k \\hat\\rho_k^2/(n-k)` and its :math:`\\chi^2_{max\\_lag}` p-value."""
    x = np.asarray(x, dtype=float)
    n = x.size
    r = acf(x, max_lag)
    q = n * (n + 2) * float(np.sum(r**2 / (n - np.arange(1, max_lag + 1))))
    return q, float(stats.chi2.sf(q, max_lag))


@dataclass(frozen=True)
class EmpiricalSummary:
    r"""What discovery fits to: per-unit-time cumulants and their uncertainty.

    Attributes
    ----------
    cumulants :
        Estimated :math:`\kappa_1..\kappa_4` *per unit time*.
    covariance :
        Their 4x4 sampling covariance (per unit time). ``None`` when the
        summary was built from user-supplied statistics without a sample size.
    n :
        Number of increments, if known.
    dt :
        Sampling interval in model time units.
    acf_returns, acf_squared :
        Autocorrelations of returns and of squared returns, if a sample was
        given.
    ljung_box_returns, ljung_box_squared :
        ``(Q, p-value)`` pairs.
    """

    cumulants: np.ndarray
    covariance: Optional[np.ndarray]
    dt: float
    n: Optional[int] = None
    acf_returns: Optional[np.ndarray] = None
    acf_squared: Optional[np.ndarray] = None
    ljung_box_returns: Optional[tuple[float, float]] = None
    ljung_box_squared: Optional[tuple[float, float]] = None
    sample: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def mean(self) -> float:
        return float(self.cumulants[0])

    @property
    def variance(self) -> float:
        return float(self.cumulants[1])

    def skewness(self, horizon: Optional[float] = None) -> float:
        """Skewness of the increment over ``horizon`` (default: the sampling interval)."""
        h = self.dt if horizon is None else horizon
        return float(self.cumulants[2] * h / (self.cumulants[1] * h) ** 1.5)

    def excess_kurtosis(self, horizon: Optional[float] = None) -> float:
        """Excess kurtosis over ``horizon`` (default: the sampling interval); it decays like 1/horizon."""
        h = self.dt if horizon is None else horizon
        return float(self.cumulants[3] * h / (self.cumulants[1] * h) ** 2)

    @property
    def standard_errors(self) -> Optional[np.ndarray]:
        return None if self.covariance is None else np.sqrt(np.diag(self.covariance))

    def is_levy_consistent(self, level: float = 0.01) -> bool:
        """False if returns or squared returns are significantly autocorrelated at ``level``."""
        for test in (self.ljung_box_returns, self.ljung_box_squared):
            if test is not None and test[1] < level:
                return False
        return True


def summarize(
    returns: Sequence[float],
    dt: float = 1.0,
    max_lag: int = 10,
    hac_lags: int = 0,
) -> EmpiricalSummary:
    """Summarise a series of increments sampled every ``dt`` time units."""
    x = np.asarray(returns, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < max(8, max_lag + 2):
        raise ValueError(f"need at least {max(8, max_lag + 2)} finite observations, got {x.size}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    k = k_statistics(x) / dt
    cov = cumulant_covariance(x, hac_lags=hac_lags) / dt**2
    return EmpiricalSummary(
        cumulants=k,
        covariance=cov,
        dt=float(dt),
        n=int(x.size),
        acf_returns=acf(x, max_lag),
        acf_squared=acf(x**2, max_lag),
        ljung_box_returns=ljung_box(x, max_lag),
        ljung_box_squared=ljung_box(x**2, max_lag),
        sample=x,
    )


def from_statistics(
    mean: float,
    variance: float,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    dt: float = 1.0,
    n: Optional[int] = None,
) -> EmpiricalSummary:
    r"""Build a summary from moments of increments over ``dt``.

    The moments describe the increment :math:`X_{\Delta t}`, as a returns
    table would quote them. They are converted to per-unit-time cumulants
    :math:`\kappa_1 = \text{mean}/\Delta t`, :math:`\kappa_2 = \text{var}/\Delta t`,
    :math:`\kappa_3 = \text{skew}\cdot\text{var}^{3/2}/\Delta t` and
    :math:`\kappa_4 = \text{exkurt}\cdot\text{var}^2/\Delta t`. If ``n`` is
    given, a Gaussian-reference covariance is attached:
    :math:`\mathrm{Var}(k_j) \approx j!\,\kappa_2^j/n` for the sample cumulant of
    order :math:`j`.
    """
    if variance <= 0:
        raise ValueError(f"variance must be positive, got {variance}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    k = np.array(
        [mean, variance, skewness * variance**1.5, excess_kurtosis * variance**2],
        dtype=float,
    )
    cov = None
    if n is not None:
        cov = np.diag([variance, 2 * variance**2, 6 * variance**3, 24 * variance**4]) / n / dt**2
    return EmpiricalSummary(cumulants=k / dt, covariance=cov, dt=float(dt), n=n)
