r"""
Live monitoring of a fitted model: change detection with guaranteed false-alarm rates.

A discovered model describes the law of each increment. Monitoring asks, as
new increments arrive, whether that law still holds. :class:`PageCUSUM`
answers this against a specified alternative, such as volatility up 30% or
jumps appearing, with Page's (1954) CUSUM of log-likelihood ratios

.. math::
    W_t = \max\big(0,\; W_{t-1} + \lambda(x_t)\big), \qquad W_0 = 0,

and it raises an alarm when :math:`W_t \ge h`.

The likelihood ratio is that of the smoothed binned likelihood used for
fitting. A new increment :math:`x` gets the same kernel weights :math:`K_j(x)`
as the data, and

.. math::
    \lambda(x) = \sum_j K_j(x)\,\log\frac{\tilde p_j(\theta_1)}{\tilde p_j(\theta_0)} .

**False alarms are controlled exactly.** The weights are non-negative and sum
to one, so the weighted AM–GM inequality gives
:math:`e^{\lambda(x)} \le \sum_j K_j(x)\,\tilde p_j(\theta_1)/\tilde p_j(\theta_0)`.
Under the null, :math:`E_0[K_j(X)] = \tilde p_j(\theta_0)`, because the model's
smoothed probabilities use the same kernel. Hence

.. math::
    E_0\big[e^{\lambda(X)}\big] \le \sum_j \tilde p_j(\theta_1) = 1 .

Each partial product :math:`\prod_{i=k+1}^t e^{\lambda(x_i)}` is therefore a
non-negative supermartingale under the null. Ville's inequality bounds the
chance that it ever reaches :math:`e^h` by :math:`e^{-h}`, and :math:`W_t \ge h`
exactly when one of them does. Two consequences follow:

- **average run length:** :math:`E_0[N] \ge e^h` (Lorden 1971), so ``arl0=A``
  sets :math:`h = \log A`;
- **finite horizon:** the chance of a false alarm within :math:`n`
  observations is at most :math:`n e^{-h}`, by a union over start times.

The bounds hold for any sample size and are not asymptotic. Their only
inputs are that the increments are i.i.d. under the null, which holds for a
Lévy model, and the COS bin probabilities, which are accurate to about
:math:`10^{-8}`.

**Detection.** After a change to :math:`\theta_1`, the drift of :math:`W_t` is
exactly :math:`E_1[\lambda(X)] = \mathrm{KL}(\tilde p(\theta_1)\,\|\,\tilde p(\theta_0))`,
because :math:`\lambda` is linear in :math:`K`. The expected delay is about
:math:`h/\mathrm{KL}` for large :math:`h` (Lorden 1971). The alarm also
estimates the change point: the CUSUM's excursion began just after the last
time :math:`W` was zero.

These guarantees are conditional on the null model, with its parameters
treated as known. Parameters estimated from a short history shorten the
in-control run length; refit on enough data before monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from spxa.discovery.expr import Model
from spxa.discovery.fit import BinnedData

__all__ = ["Alarm", "MonitorResult", "PageCUSUM"]

# observations per block in :meth:`PageCUSUM.increments`, bounding its memory
_CHUNK = 8192


@dataclass(frozen=True)
class Alarm:
    """One alarm of a change detector.

    Attributes
    ----------
    time :
        Index of the observation at which the statistic reached the threshold
        (0-based, counted from the start of monitoring).
    change_point :
        Estimated index of the first post-change observation.
    statistic :
        The statistic's value at the alarm.
    """

    time: int
    change_point: int
    statistic: float

    @property
    def delay(self) -> int:
        """Observations from the estimated change point to the alarm, inclusive."""
        return self.time - self.change_point + 1


@dataclass(frozen=True)
class MonitorResult:
    """The path of a detector over a batch of observations."""

    statistic: np.ndarray = field(repr=False)
    increments: np.ndarray = field(repr=False)
    alarms: list[Alarm]
    threshold: float

    @property
    def first_alarm(self) -> Optional[Alarm]:
        return self.alarms[0] if self.alarms else None


class PageCUSUM:
    r"""Page's CUSUM of smoothed binned log-likelihood ratios (see the module docstring).

    Parameters
    ----------
    null, alternative :
        The in-control law :math:`\theta_0` (typically a discovered fit) and the
        change to detect, :math:`\theta_1`. The two may have different
        structures; only their bin probabilities enter.
    binned :
        The bins and kernel, normally those the null was fitted on.
    threshold, arl0 :
        Give exactly one of them. ``arl0`` sets :math:`h = \log(\text{arl0})`,
        which guarantees an in-control average run length of at least ``arl0``.
    """

    def __init__(
        self,
        null: Model,
        alternative: Model,
        binned: BinnedData,
        *,
        threshold: Optional[float] = None,
        arl0: Optional[float] = None,
    ) -> None:
        if (threshold is None) == (arl0 is None):
            raise ValueError("give exactly one of threshold and arl0")
        if arl0 is not None:
            if not arl0 > 1:
                raise ValueError("arl0 must exceed 1")
            threshold = float(np.log(arl0))
        assert threshold is not None
        if not (np.isfinite(threshold) and threshold > 0):
            raise ValueError("the threshold must be positive and finite")
        self.null, self.alternative, self.binned = null, alternative, binned
        self.threshold = float(threshold)
        p0 = binned.probabilities(null)
        p1 = binned.probabilities(alternative)
        self.null_probabilities, self.alternative_probabilities = p0, p1
        self.log_ratio = np.log(p1) - np.log(p0)
        self.reset()

    # -- properties --------------------------------------------------------

    @property
    def kl_alternative(self) -> float:
        r""":math:`E_1[\lambda] = \mathrm{KL}(\tilde p_1\|\tilde p_0)`, the post-change drift."""
        return float(self.alternative_probabilities @ self.log_ratio)

    @property
    def kl_null(self) -> float:
        r""":math:`-E_0[\lambda] = \mathrm{KL}(\tilde p_0\|\tilde p_1)`, the in-control pull to zero."""
        return float(-(self.null_probabilities @ self.log_ratio))

    @property
    def guaranteed_arl0(self) -> float:
        """Lower bound :math:`e^h` on the in-control average run length."""
        return float(np.exp(self.threshold))

    @property
    def expected_delay(self) -> float:
        """First-order detection delay :math:`h/\\mathrm{KL}` (Lorden 1971)."""
        kl = self.kl_alternative
        return float(self.threshold / kl) if kl > 0 else np.inf

    def false_alarm_bound(self, n: int) -> float:
        """Upper bound :math:`\\min(1, n e^{-h})` on P(false alarm within ``n`` observations)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return float(min(1.0, n * np.exp(-self.threshold)))

    # -- running -----------------------------------------------------------

    def increments(self, x: np.ndarray) -> np.ndarray:
        r""":math:`\lambda(x)` for each observation in ``x`` (any shape)."""
        x = np.asarray(x, dtype=float)
        flat = x.reshape(-1)
        out = np.empty(flat.size)
        for i in range(0, flat.size, _CHUNK):
            out[i : i + _CHUNK] = self.binned.kernel_weights(flat[i : i + _CHUNK]) @ self.log_ratio
        return out.reshape(x.shape)

    def reset(self) -> None:
        """Restart the streaming state (statistic zero, clock zero)."""
        self.statistic = 0.0
        self.time = 0
        self._start = 0

    def update(self, x: float) -> Optional[Alarm]:
        """Process one observation; return an :class:`Alarm` if it triggers one.

        After an alarm the statistic restarts at zero; the clock keeps counting.
        """
        lam = float(self.increments(np.array([x]))[0])
        alarm = self._step(lam)
        self.time += 1
        return alarm

    def _step(self, lam: float) -> Optional[Alarm]:
        w = self.statistic + lam
        if w <= 0.0:
            self.statistic = 0.0
            self._start = self.time + 1
            return None
        self.statistic = w
        if w >= self.threshold:
            alarm = Alarm(time=self.time, change_point=self._start, statistic=w)
            self.statistic = 0.0
            self._start = self.time + 1
            return alarm
        return None

    def run(self, x: np.ndarray) -> MonitorResult:
        """Run on a batch from a fresh state; the streaming state is left at the end."""
        lam = self.increments(np.asarray(x, dtype=float).reshape(-1))
        self.reset()
        path = np.empty(lam.size)
        alarms: list[Alarm] = []
        for i, v in enumerate(lam):
            a = self._step(float(v))
            path[i] = a.statistic if a is not None else self.statistic
            if a is not None:
                alarms.append(a)
            self.time += 1
        return MonitorResult(
            statistic=path, increments=lam, alarms=alarms, threshold=self.threshold
        )
