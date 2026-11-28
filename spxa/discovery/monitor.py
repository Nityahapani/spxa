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
from typing import Optional, Sequence

import numpy as np
from scipy import optimize

from spxa.discovery.density import model_distribution
from spxa.discovery.expr import Model
from spxa.discovery.fit import BinnedData

__all__ = [
    "Alarm",
    "MonitorResult",
    "PageCUSUM",
    "ShiryaevRoberts",
    "average_run_length",
    "local_alternatives",
]

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
    alternative :
        For a :class:`ShiryaevRoberts` mixture, the index of the alternative
        contributing most to the statistic at the alarm; ``None`` otherwise.
    """

    time: int
    change_point: int
    statistic: float
    alternative: Optional[int] = None

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
        self.threshold = _threshold(threshold, arl0)
        self.null, self.alternative, self.binned = null, alternative, binned
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


def _threshold(threshold: Optional[float], arl0: Optional[float]) -> float:
    if (threshold is None) == (arl0 is None):
        raise ValueError("give exactly one of threshold and arl0")
    if arl0 is not None:
        if not arl0 > 1:
            raise ValueError("arl0 must exceed 1")
        threshold = float(np.log(arl0))
    assert threshold is not None
    if not (np.isfinite(threshold) and threshold > 0):
        raise ValueError("the threshold must be positive and finite")
    return float(threshold)


class ShiryaevRoberts:
    r"""Mixture Shiryaev–Roberts detector for a change to any of several alternatives.

    For alternatives :math:`\theta_1, \dots, \theta_J` with weights
    :math:`w_j` (summing to one), each with its smoothed binned
    log-likelihood ratio :math:`\lambda_j`,

    .. math::
        R^{(j)}_t = \big(1 + R^{(j)}_{t-1}\big)\,e^{\lambda_j(x_t)}, \qquad
        R_t = \sum_j w_j R^{(j)}_t ,

    and the alarm is at :math:`R_t \ge e^h`. :math:`R^{(j)}_t` is the sum over
    candidate change points of the likelihood ratios, so the detector needs no
    guess of when the change happened, nor, through the mixture, of which
    alternative it was.

    **Guarantee.** :math:`E_0[e^{\lambda_j}] \le 1` for every :math:`j` (see
    the module docstring), so :math:`E_0[R_t \mid \mathcal F_{t-1}] \le 1 + R_{t-1}`
    and :math:`R_t - t` is a supermartingale. Optional stopping at the alarm
    time :math:`N` gives :math:`E_0[N] \ge E_0[R_N] \ge e^h`: the in-control
    average run length is at least :math:`e^h = ` ``arl0``, whatever the number
    of alternatives (Pollak 1987). The finite-horizon bound of
    :class:`PageCUSUM` holds for each start time as well, by Markov's inequality,
    so P(false alarm within :math:`n`) :math:`\le n e^{-h}`.

    With one alternative, Shiryaev–Roberts and CUSUM have nearly equal delays;
    the mixture pays roughly :math:`\log(1/w_j)` extra in threshold to cover
    alternative :math:`j`. :func:`local_alternatives` builds a set of
    alternatives around a fitted law when no single one is specified.

    The alarm's ``change_point`` is the CUSUM estimate (the index after the
    last zero) for the alternative contributing most to :math:`R_t`.
    """

    def __init__(
        self,
        null: Model,
        alternatives: Sequence[Model],
        binned: BinnedData,
        *,
        weights: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
        arl0: Optional[float] = None,
    ) -> None:
        if len(alternatives) == 0:
            raise ValueError("give at least one alternative")
        self.threshold = _threshold(threshold, arl0)
        w = np.full(len(alternatives), 1.0 / len(alternatives))
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.shape != (len(alternatives),) or np.any(w < 0) or not w.sum() > 0:
                raise ValueError("weights must be non-negative, one per alternative")
            w = w / w.sum()
        self.null, self.alternatives, self.binned = null, list(alternatives), binned
        self.weights = w
        self._log_w = np.log(np.maximum(w, 1e-300))
        p0 = binned.probabilities(null)
        self.null_probabilities = p0
        self.log_ratios = np.stack(
            [np.log(binned.probabilities(m)) - np.log(p0) for m in alternatives]
        )
        self.reset()

    @property
    def guaranteed_arl0(self) -> float:
        """Lower bound :math:`e^h` on the in-control average run length."""
        return float(np.exp(self.threshold))

    def false_alarm_bound(self, n: int) -> float:
        """Upper bound :math:`\\min(1, n e^{-h})` on P(false alarm within ``n`` observations)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return float(min(1.0, n * np.exp(-self.threshold)))

    def increments(self, x: np.ndarray) -> np.ndarray:
        r""":math:`\lambda_j(x)`, shape ``x.shape + (J,)``."""
        x = np.asarray(x, dtype=float)
        flat = x.reshape(-1)
        out = np.empty((flat.size, len(self.alternatives)))
        for i in range(0, flat.size, _CHUNK):
            out[i : i + _CHUNK] = (
                self.binned.kernel_weights(flat[i : i + _CHUNK]) @ self.log_ratios.T
            )
        return out.reshape(x.shape + (len(self.alternatives),))

    @property
    def statistic(self) -> float:
        r""":math:`\log R_t`, compared with the threshold :math:`h`."""
        return float(np.logaddexp.reduce(self._log_w + self._log_r))

    def reset(self) -> None:
        """Restart the streaming state (all :math:`R^{(j)} = 0`, clock zero)."""
        self._log_r = np.full(len(self.alternatives), -np.inf)
        self._cusum = np.zeros(len(self.alternatives))
        self._starts = np.zeros(len(self.alternatives), dtype=int)
        self.time = 0

    def _step(self, lam: np.ndarray) -> Optional[Alarm]:
        self._log_r = np.logaddexp(0.0, self._log_r) + lam
        w = self._cusum + lam
        reset = w <= 0.0
        self._cusum = np.where(reset, 0.0, w)
        self._starts = np.where(reset, self.time + 1, self._starts)
        stat = self.statistic
        if stat >= self.threshold:
            j = int(np.argmax(self._log_w + self._log_r))
            alarm = Alarm(
                time=self.time, change_point=int(self._starts[j]), statistic=stat, alternative=j
            )
            self._log_r[:] = -np.inf
            self._cusum[:] = 0.0
            self._starts[:] = self.time + 1
            return alarm
        return None

    def update(self, x: float) -> Optional[Alarm]:
        """Process one observation; return an :class:`Alarm` if it triggers one.

        After an alarm the statistic restarts; the clock keeps counting.
        """
        alarm = self._step(self.increments(np.array([x]))[0])
        self.time += 1
        return alarm

    def run(self, x: np.ndarray) -> MonitorResult:
        """Run on a batch from a fresh state. ``statistic`` is the path of :math:`\\log R_t`."""
        lam = self.increments(np.asarray(x, dtype=float).reshape(-1))
        self.reset()
        path = np.empty(lam.shape[0])
        alarms: list[Alarm] = []
        for i in range(lam.shape[0]):
            a = self._step(lam[i])
            path[i] = a.statistic if a is not None else self.statistic
            if a is not None:
                alarms.append(a)
            self.time += 1
        return MonitorResult(
            statistic=path, increments=lam, alarms=alarms, threshold=self.threshold
        )


# ---------------------------------------------------------------------------
# alternatives around a fitted law
# ---------------------------------------------------------------------------


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(p @ (np.log(p) - np.log(q)))


def local_alternatives(
    null: Model,
    binned: BinnedData,
    *,
    kl: float = 0.02,
    step: float = 1e-5,
) -> list[Model]:
    r"""Alternatives in every direction around ``null``, each at divergence ``kl`` per observation.

    The directions are :math:`\pm v_k`, the eigenvectors of the Fisher
    information :math:`I = \sum_j \tilde p_j \nabla\log\tilde p_j \nabla\log\tilde p_j^\top`
    of the binned law in the free coordinates. Along each, the step is chosen
    so that :math:`\mathrm{KL}(\tilde p(\theta_1)\,\|\,\tilde p(\theta_0))` equals
    ``kl`` exactly (by root finding, starting from the quadratic
    approximation :math:`\tfrac12 s^2 v^\top I v`). Directions with no
    information, such as a non-identified combination, are skipped.

    This gives :math:`2d` alternatives for :math:`d` free coordinates. A
    detector tuned to divergence ``kl`` has delay about :math:`h/\mathrm{kl}`
    against changes of that size. Smaller changes are detected later, and
    larger ones sooner.
    """
    if not kl > 0:
        raise ValueError("kl must be positive")
    structure = null.structure
    x0 = null.to_free()
    p0 = binned.probabilities(null)
    grads = np.empty((x0.size, p0.size))
    for k in range(x0.size):
        e = np.zeros_like(x0)
        e[k] = step
        up = np.log(binned.probabilities(Model.from_free(structure, x0 + e)))
        down = np.log(binned.probabilities(Model.from_free(structure, x0 - e)))
        grads[k] = (up - down) / (2 * step)
    info = (grads * p0) @ grads.T
    values, vectors = np.linalg.eigh(info)
    out: list[Model] = []
    for value, v in zip(values, vectors.T):
        if not value > 1e-12 * max(values.max(), 1e-300):
            continue
        s0 = np.sqrt(2.0 * kl / value)
        for sign in (1.0, -1.0):

            def gap(s: float, direction: np.ndarray = sign * v) -> float:
                try:
                    q = binned.probabilities(Model.from_free(structure, x0 + s * direction))
                except (ValueError, FloatingPointError, OverflowError):
                    return np.inf
                d = _kl(q, p0)
                return d - kl if np.isfinite(d) else np.inf

            hi = s0
            while gap(hi) < 0.0 and hi < 1e3 * s0:
                hi *= 2.0
            if not gap(hi) >= 0.0 or not np.isfinite(gap(hi)):
                # shrink to the largest finite step; skip if the target is unreachable
                lo_ok = s0
                while lo_ok > 1e-6 * s0 and not np.isfinite(gap(lo_ok)):
                    lo_ok /= 2.0
                if not np.isfinite(gap(lo_ok)) or gap(lo_ok) < 0.0:
                    continue
                hi = lo_ok
            s = optimize.brentq(gap, 0.0, hi, xtol=1e-12 * max(hi, 1.0), rtol=1e-10)
            out.append(Model.from_free(structure, x0 + s * sign * v))
    return out


# ---------------------------------------------------------------------------
# run lengths
# ---------------------------------------------------------------------------


def average_run_length(
    detector: PageCUSUM,
    law: Optional[Model] = None,
    *,
    n_states: Optional[int] = None,
    n_grid: int = 20001,
) -> float:
    r"""Average run length of a :class:`PageCUSUM` when increments follow ``law``.

    The default ``law`` is the detector's null, giving the in-control ARL. The
    computation follows Brook & Evans (1972). It takes the law of
    :math:`\lambda(X)` with :math:`X \sim` ``law`` over one step (from its COS
    density), discretises :math:`[0, h)` into ``n_states`` cells, and solves
    :math:`(I - P)L = \mathbf 1` for the expected time to absorption beyond
    :math:`h`, starting from :math:`W_0 = 0`. The discretisation error falls
    like ``1/n_states**2`` and grows with :math:`h`; the default,
    ``max(400, 150 h)`` states, keeps it near 0.2% for :math:`h \le 10`.
    """
    if n_states is None:
        n_states = max(400, int(np.ceil(150 * detector.threshold)))
    if n_states < 2:
        raise ValueError("n_states must be at least 2")
    law = detector.null if law is None else law
    dist = model_distribution(law, detector.binned.dt)
    x = np.linspace(dist.a, dist.b, n_grid)
    dens = np.concatenate([dist.pdf(x[i : i + 2048]) for i in range(0, x.size, 2048)])
    dens = np.maximum(dens, 0.0)
    lam = detector.increments(x)
    mass = dens * np.gradient(x)
    mass /= mass.sum()
    order = np.argsort(lam)
    lam_sorted, cum = lam[order], np.cumsum(mass[order])

    def cdf(v: np.ndarray) -> np.ndarray:
        return np.interp(v, lam_sorted, cum, left=0.0, right=1.0)

    h = detector.threshold
    delta = 2.0 * h / (2 * n_states - 1)
    states = np.arange(n_states) * delta
    diff = states[None, :] - states[:, None]
    P = cdf(diff + delta / 2) - cdf(diff - delta / 2)
    P[:, 0] = cdf(-states + delta / 2)
    L = np.linalg.solve(np.eye(n_states) - P, np.ones(n_states))
    return float(L[0])
