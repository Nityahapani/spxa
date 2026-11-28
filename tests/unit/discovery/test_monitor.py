"""Page's CUSUM on smoothed binned likelihood ratios: guarantees and behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from spxa.discovery import families as F
from spxa.discovery.expr import Model, leaf
from spxa.discovery.fit import BinnedData
from spxa.discovery.monitor import Alarm, PageCUSUM

DT = 1.0 / 252.0
BM = leaf(F.BROWNIAN) + leaf(F.DRIFT)
JUMPS = leaf(F.BROWNIAN) + leaf(F.NORMAL_JUMPS) + leaf(F.DRIFT)
NULL = Model(BM, [(0.2,), (0.05,)])
VOL_UP = Model(BM, [(0.26,), (0.05,)])
JUMPY = Model(JUMPS, [(0.18,), (10.0, -0.02, 0.03), (0.05,)])


def _increments(model: Model, n: int, seed: int, n_paths: int = 1) -> np.ndarray:
    path = model.to_process().simulate(n, n_paths, T=n * DT, rng=np.random.default_rng(seed))
    return np.diff(path, axis=1)


@pytest.fixture(scope="module")
def binned() -> BinnedData:
    return BinnedData.from_sample(_increments(NULL, 2000, 0)[0], DT)


def _null_grid() -> tuple[np.ndarray, np.ndarray]:
    """A fine grid over the null's Gaussian increment law and its density."""
    mean, sd = 0.05 * DT, 0.2 * np.sqrt(DT)
    grid = np.linspace(mean - 12 * sd, mean + 12 * sd, 100_001)
    return grid, stats.norm.pdf(grid, mean, sd)


# ---------------------------------------------------------------------------
# the identities behind the guarantee
# ---------------------------------------------------------------------------


def test_kernel_weights_are_the_rows_of_the_kernel_matrix(binned: BinnedData) -> None:
    np.testing.assert_array_equal(binned.kernel_weights(binned.sample), binned.kernel_matrix())
    w = binned.kernel_weights(np.array([[0.0, 0.01, -0.5], [1.0, -1.0, 0.003]]))
    assert w.shape == (2, 3, binned.edges.size - 1)
    assert np.all(w >= 0.0)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-12)


def test_null_mean_of_the_kernel_weights_is_the_smoothed_bin_probability(
    binned: BinnedData,
) -> None:
    """E_0[K(X)] = p~(theta_0), checked by quadrature against the exact Gaussian law."""
    grid, density = _null_grid()
    mean_weights = np.trapezoid(binned.kernel_weights(grid) * density[:, None], grid, axis=0)
    np.testing.assert_allclose(mean_weights, binned.probabilities(NULL), rtol=0, atol=1e-10)


@pytest.mark.parametrize("alternative", [VOL_UP, JUMPY], ids=["vol-up", "jumps"])
def test_exponentiated_increment_has_null_mean_at_most_one(
    binned: BinnedData, alternative: Model
) -> None:
    """E_0[exp(lambda(X))] <= 1, so the likelihood ratios are supermartingales."""
    cusum = PageCUSUM(NULL, alternative, binned, threshold=3.0)
    grid, density = _null_grid()
    lam = cusum.increments(grid)
    mean = np.trapezoid(np.exp(lam) * density, grid)
    assert mean <= 1.0 + 1e-9
    assert mean > 0.99  # AM-GM is nearly tight: the kernel is narrow against the law
    # the drifts: positive after the change, negative before
    assert cusum.kl_alternative > 0.0 and cusum.kl_null > 0.0
    assert np.trapezoid(lam * density, grid) == pytest.approx(-cusum.kl_null, rel=1e-6)


def test_post_change_drift_is_the_kl_divergence(binned: BinnedData) -> None:
    cusum = PageCUSUM(NULL, VOL_UP, binned, threshold=3.0)
    lam = cusum.increments(_increments(VOL_UP, 200_000, 7)[0])
    se = lam.std() / np.sqrt(lam.size)
    assert abs(lam.mean() - cusum.kl_alternative) < 4 * se


# ---------------------------------------------------------------------------
# false alarms
# ---------------------------------------------------------------------------


def test_ville_bound_on_crossing_from_a_fixed_start(binned: BinnedData) -> None:
    """P_0(sup_t sum lambda >= h) <= exp(-h) for the random walk started once."""
    h = 2.0
    cusum = PageCUSUM(NULL, VOL_UP, binned, threshold=h)
    lam = cusum.increments(_increments(NULL, 400, 11, n_paths=3000))
    crossed = (np.cumsum(lam, axis=1) >= h).any(axis=1).mean()
    assert 0.0 < crossed <= np.exp(-h)


def test_in_control_run_length_exceeds_its_guarantee(binned: BinnedData) -> None:
    cusum = PageCUSUM(NULL, VOL_UP, binned, arl0=20.0)
    assert cusum.threshold == pytest.approx(np.log(20.0))
    assert cusum.guaranteed_arl0 == pytest.approx(20.0)
    result = cusum.run(_increments(NULL, 100_000, 5)[0])
    assert len(result.alarms) > 50
    assert 100_000 / len(result.alarms) >= cusum.guaranteed_arl0


def test_finite_horizon_false_alarm_rate(binned: BinnedData) -> None:
    n, cusum = 250, PageCUSUM(NULL, JUMPY, binned, threshold=np.log(1000.0))
    paths = _increments(NULL, n, 13, n_paths=1000)
    alarmed = np.mean([cusum.run(p).first_alarm is not None for p in paths])
    assert alarmed <= cusum.false_alarm_bound(n) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_detection_delay_is_about_threshold_over_kl(binned: BinnedData) -> None:
    cusum = PageCUSUM(NULL, VOL_UP, binned, threshold=4.0)
    result = cusum.run(_increments(VOL_UP, 100_000, 17)[0])
    arl1 = 100_000 / len(result.alarms)
    # Lorden's first-order delay, corrected only by the overshoot and the restarts
    assert 0.7 * cusum.expected_delay < arl1 < 1.3 * cusum.expected_delay
    # and the same detector in control runs far longer
    in_control = cusum.run(_increments(NULL, 100_000, 18)[0])
    assert 100_000 / max(len(in_control.alarms), 1) > 10 * arl1


def test_alarm_locates_the_change() -> None:
    """The last zero of the statistic estimates the change point far better than the alarm time."""
    bd = BinnedData.from_sample(_increments(NULL, 1000, 19)[0], DT)
    shifted = Model(BM, [(0.3,), (0.05,)])
    cusum = PageCUSUM(NULL, shifted, bd, arl0=10_000.0)
    n_streams, change = 200, 500
    before = _increments(NULL, change, 20, n_paths=n_streams)
    after = _increments(shifted, 300, 21, n_paths=n_streams)
    errors, delays, false_alarms = [], [], 0
    for b, a in zip(before, after):
        alarm = cusum.run(np.concatenate([b, a])).first_alarm
        assert alarm is not None  # 300 post-change steps are about 6 expected delays
        if alarm.time < change:
            false_alarms += 1
            continue
        assert alarm.statistic >= cusum.threshold
        assert alarm.delay == alarm.time - alarm.change_point + 1
        errors.append(alarm.change_point - change)
        delays.append(alarm.time - change + 1)
    assert false_alarms / n_streams <= cusum.false_alarm_bound(change)
    errors_abs = np.abs(errors)
    assert np.median(errors_abs) <= 0.2 * np.mean(delays)
    assert np.mean(delays) < 1.2 * cusum.expected_delay


def test_null_and_alternative_may_have_different_structures(binned: BinnedData) -> None:
    cusum = PageCUSUM(NULL, JUMPY, binned, arl0=500.0)
    result = cusum.run(_increments(JUMPY, 20_000, 23)[0])
    assert len(result.alarms) > 0
    assert cusum.expected_delay < 500.0


# ---------------------------------------------------------------------------
# mechanics
# ---------------------------------------------------------------------------


def test_streaming_matches_batch(binned: BinnedData) -> None:
    x = np.concatenate([_increments(NULL, 300, 30)[0], _increments(VOL_UP, 300, 31)[0]])
    batch = PageCUSUM(NULL, VOL_UP, binned, threshold=2.5)
    result = batch.run(x)
    stream = PageCUSUM(NULL, VOL_UP, binned, threshold=2.5)
    alarms: list[Alarm] = []
    path = []
    for v in x:
        a = stream.update(float(v))
        if a is not None:
            alarms.append(a)
        path.append(a.statistic if a is not None else stream.statistic)
    # one observation at a time rounds differently from the batched product (~1e-15)
    assert len(alarms) == len(result.alarms) >= 2
    for a, b in zip(alarms, result.alarms):
        assert (a.time, a.change_point) == (b.time, b.change_point)
        assert a.statistic == pytest.approx(b.statistic, abs=1e-12)
    np.testing.assert_allclose(path, result.statistic, rtol=0, atol=1e-12)
    assert stream.time == x.size == result.statistic.size
    # the statistic is the reflected walk of the increments, restarted at alarms
    w = 0.0
    for lam, s in zip(result.increments, result.statistic):
        w = max(0.0, w + lam)
        assert s == pytest.approx(w, abs=1e-12)
        if w >= result.threshold:
            w = 0.0


def test_validation(binned: BinnedData) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PageCUSUM(NULL, VOL_UP, binned)
    with pytest.raises(ValueError, match="exactly one"):
        PageCUSUM(NULL, VOL_UP, binned, threshold=2.0, arl0=100.0)
    with pytest.raises(ValueError, match="arl0"):
        PageCUSUM(NULL, VOL_UP, binned, arl0=1.0)
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="threshold"):
            PageCUSUM(NULL, VOL_UP, binned, threshold=bad)
    cusum = PageCUSUM(NULL, VOL_UP, binned, threshold=np.log(100.0))
    assert cusum.false_alarm_bound(0) == 0.0
    assert cusum.false_alarm_bound(10) == pytest.approx(0.1)
    assert cusum.false_alarm_bound(10**6) == 1.0
    with pytest.raises(ValueError):
        cusum.false_alarm_bound(-1)
    # identical laws: nothing to detect
    same = PageCUSUM(NULL, NULL, binned, threshold=1.0)
    assert same.kl_alternative == 0.0 and same.expected_delay == np.inf
    assert same.run(_increments(NULL, 500, 40)[0]).alarms == []
