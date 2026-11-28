"""Composite monitoring: mixture Shiryaev–Roberts, local alternatives, exact run lengths."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.discovery import families as F
from spxa.discovery.expr import Model, leaf
from spxa.discovery.fit import BinnedData
from spxa.discovery.monitor import (
    Alarm,
    PageCUSUM,
    ShiryaevRoberts,
    average_run_length,
    local_alternatives,
)

DT = 1.0 / 252.0
BM = leaf(F.BROWNIAN) + leaf(F.DRIFT)
JUMPS = leaf(F.BROWNIAN) + leaf(F.NORMAL_JUMPS) + leaf(F.DRIFT)
NULL = Model(BM, [(0.2,), (0.05,)])
VOL_UP = Model(BM, [(0.26,), (0.05,)])


def _increments(model: Model, n: int, seed: int) -> np.ndarray:
    path = model.to_process().simulate(n, 1, T=n * DT, rng=np.random.default_rng(seed))
    return np.diff(path[0])


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(p @ (np.log(p) - np.log(q)))


@pytest.fixture(scope="module")
def binned() -> BinnedData:
    return BinnedData.from_sample(_increments(NULL, 2000, 0), DT)


def _cycle_lengths(cusum: PageCUSUM, x: np.ndarray) -> np.ndarray:
    times = np.array([a.time for a in cusum.run(x).alarms])
    return np.diff(np.concatenate([[-1], times]))


# ---------------------------------------------------------------------------
# Brook-Evans run lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("law", "n", "seed"),
    [(NULL, 1_000_000, 1), (VOL_UP, 200_000, 2)],
    ids=["in-control", "after-change"],
)
def test_average_run_length_matches_simulation(
    binned: BinnedData, law: Model, n: int, seed: int
) -> None:
    cusum = PageCUSUM(NULL, VOL_UP, binned, threshold=2.0)
    exact = average_run_length(cusum, law)
    cycles = _cycle_lengths(cusum, _increments(law, n, seed))
    se = cycles.std() / np.sqrt(cycles.size)
    assert cycles.size > 3000 and se / cycles.mean() < 0.015
    assert abs(cycles.mean() - exact) < 4 * se


def test_average_run_length_converges_and_respects_the_guarantee(binned: BinnedData) -> None:
    cusum = PageCUSUM(NULL, VOL_UP, binned, threshold=3.0)
    coarse, fine = average_run_length(cusum, n_states=200), average_run_length(cusum, n_states=800)
    assert abs(coarse - fine) < 5e-3 * fine
    assert fine >= cusum.guaranteed_arl0
    after = average_run_length(cusum, VOL_UP)
    assert 0.6 * cusum.expected_delay < after < cusum.expected_delay
    with pytest.raises(ValueError):
        average_run_length(cusum, n_states=1)


def test_run_length_grows_with_the_threshold(binned: BinnedData) -> None:
    """ARL_0 stays above e^h and grows; ARL_1 grows like h / KL (Wald)."""
    arl0, arl1 = [], []
    for h in (3.0, 4.0, 5.0, 6.0):
        cusum = PageCUSUM(NULL, VOL_UP, binned, threshold=h)
        arl0.append(average_run_length(cusum))
        arl1.append(average_run_length(cusum, VOL_UP))
        assert arl0[-1] >= cusum.guaranteed_arl0
    assert np.all(np.diff(np.log(arl0)) > 0.9)
    kl = PageCUSUM(NULL, VOL_UP, binned, threshold=1.0).kl_alternative
    np.testing.assert_allclose(np.diff(arl1), 1.0 / kl, rtol=0.05)


# ---------------------------------------------------------------------------
# local alternatives
# ---------------------------------------------------------------------------


def test_local_alternatives_sit_at_the_requested_divergence(binned: BinnedData) -> None:
    p0 = binned.probabilities(NULL)
    alternatives = local_alternatives(NULL, binned, kl=0.02)
    assert len(alternatives) == 2 * NULL.to_free().size
    for alt in alternatives:
        assert alt.structure.key() == NULL.structure.key()
        assert _kl(binned.probabilities(alt), p0) == pytest.approx(0.02, rel=1e-8)
    # pairs lie on opposite sides of the null along each direction
    x0 = NULL.to_free()
    for a, b in zip(alternatives[::2], alternatives[1::2]):
        da, db = a.to_free() - x0, b.to_free() - x0
        assert da @ db < 0
        assert abs(da @ db) == pytest.approx(np.linalg.norm(da) * np.linalg.norm(db), rel=1e-8)
    # the volatility pair moves sigma both ways
    sigmas = sorted(a.params[0][0] for a in alternatives)
    assert sigmas[0] < 0.2 < sigmas[-1]


def test_local_alternatives_of_a_jump_structure(binned: BinnedData) -> None:
    null = Model(JUMPS, [(0.2,), (5.0, -0.03, 0.02), (0.05,)])
    p0 = binned.probabilities(null)
    alternatives = local_alternatives(null, binned, kl=0.02)
    assert 2 <= len(alternatives) <= 2 * null.to_free().size
    for alt in alternatives:
        assert _kl(binned.probabilities(alt), p0) == pytest.approx(0.02, rel=1e-8)
    with pytest.raises(ValueError):
        local_alternatives(null, binned, kl=0.0)


# ---------------------------------------------------------------------------
# mixture Shiryaev-Roberts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixture(binned: BinnedData) -> ShiryaevRoberts:
    return ShiryaevRoberts(NULL, local_alternatives(NULL, binned, kl=0.02), binned, arl0=200.0)


def test_mixture_in_control_run_length_exceeds_its_guarantee(mixture: ShiryaevRoberts) -> None:
    result = mixture.run(_increments(NULL, 300_000, 5))
    k = len(result.alarms)
    assert k > 1000
    assert 300_000 / k >= mixture.guaranteed_arl0
    assert mixture.false_alarm_bound(100) == pytest.approx(0.5)


def test_mixture_detects_and_names_the_change(mixture: ShiryaevRoberts) -> None:
    result = mixture.run(_increments(VOL_UP, 50_000, 6))
    arl1 = 50_000 / len(result.alarms)
    assert arl1 < 0.5 * mixture.guaranteed_arl0
    # the volatility-up alternative carries most alarms
    named = np.bincount([a.alternative for a in result.alarms], minlength=len(mixture.alternatives))
    sigma = [alt.params[0][0] for alt in mixture.alternatives]
    assert int(np.argmax(named)) == int(np.argmax(sigma))
    assert named.max() > 0.8 * named.sum()


def test_shiryaev_roberts_dominates_cusum(binned: BinnedData) -> None:
    """log R_t >= W_t when W_t > 0 (a sum over change points beats their maximum), so at equal h
    SR alarms no later; its guarantee e^h is nearly tight where CUSUM's is loose."""
    h = 4.0
    x = _increments(VOL_UP, 2000, 7)
    sr_path = ShiryaevRoberts(NULL, [VOL_UP], binned, threshold=1e9).run(x).statistic
    cusum_path = PageCUSUM(NULL, VOL_UP, binned, threshold=1e9).run(x).statistic
    # W_t = max(0, max_k S_k,t) and R_t >= exp(max_k S_k,t), so log R_t >= W_t wherever W_t > 0
    positive = cusum_path > 0
    assert positive.mean() > 0.5
    assert np.all(sr_path[positive] >= cusum_path[positive] - 1e-12)
    sr = ShiryaevRoberts(NULL, [VOL_UP], binned, threshold=h)
    cusum = PageCUSUM(NULL, VOL_UP, binned, threshold=h)
    y = _increments(NULL, 200_000, 8)
    sr_arl0 = y.size / len(sr.run(y).alarms)
    assert np.exp(h) <= sr_arl0 < 2 * np.exp(h)
    assert average_run_length(cusum) > 10 * sr_arl0


def test_shiryaev_roberts_recursion_and_streaming(binned: BinnedData) -> None:
    alts = [VOL_UP, Model(BM, [(0.15,), (0.05,)])]
    weights = [0.7, 0.3]
    x = np.concatenate([_increments(NULL, 300, 30), _increments(VOL_UP, 300, 31)])
    batch = ShiryaevRoberts(NULL, alts, binned, weights=weights, threshold=3.0)
    result = batch.run(x)
    # direct recursion
    r = np.zeros(2)
    for lam, s in zip(result.increments, result.statistic):
        r = (1.0 + r) * np.exp(lam)
        log_r = np.log(0.7 * r[0] + 0.3 * r[1])
        assert s == pytest.approx(log_r, abs=1e-10)
        if log_r >= 3.0:
            r = np.zeros(2)
    stream = ShiryaevRoberts(NULL, alts, binned, weights=weights, threshold=3.0)
    alarms: list[Alarm] = [a for v in x if (a := stream.update(float(v))) is not None]
    assert len(alarms) == len(result.alarms) >= 1
    for a, b in zip(alarms, result.alarms):
        assert (a.time, a.change_point, a.alternative) == (b.time, b.change_point, b.alternative)
        assert a.statistic == pytest.approx(b.statistic, abs=1e-12)


def test_zero_weight_alternatives_do_not_contribute(binned: BinnedData) -> None:
    x = _increments(VOL_UP, 1000, 40)
    jumpy = Model(JUMPS, [(0.18,), (10.0, -0.02, 0.03), (0.05,)])
    with_zero = ShiryaevRoberts(NULL, [VOL_UP, jumpy], binned, weights=[1.0, 0.0], threshold=50.0)
    alone = ShiryaevRoberts(NULL, [VOL_UP], binned, threshold=50.0)
    np.testing.assert_allclose(with_zero.run(x).statistic, alone.run(x).statistic, atol=1e-12)


def test_shiryaev_roberts_validation(binned: BinnedData) -> None:
    with pytest.raises(ValueError, match="at least one"):
        ShiryaevRoberts(NULL, [], binned, arl0=100.0)
    with pytest.raises(ValueError, match="exactly one"):
        ShiryaevRoberts(NULL, [VOL_UP], binned)
    with pytest.raises(ValueError, match="weights"):
        ShiryaevRoberts(NULL, [VOL_UP], binned, weights=[1.0, 1.0], arl0=100.0)
    with pytest.raises(ValueError, match="weights"):
        ShiryaevRoberts(NULL, [VOL_UP, VOL_UP], binned, weights=[1.0, -1.0], arl0=100.0)
    sr = ShiryaevRoberts(NULL, [VOL_UP, VOL_UP], binned, weights=[3.0, 1.0], arl0=100.0)
    np.testing.assert_allclose(sr.weights, [0.75, 0.25])
    assert sr.statistic == -np.inf
    with pytest.raises(ValueError):
        sr.false_alarm_bound(-1)


def test_report_monitor_watches_the_discovered_law() -> None:
    from spxa.discovery import discover, grammar_from_names

    x = np.random.default_rng(4).normal(0.08 * DT, 0.2 * np.sqrt(DT), 1500)
    report = discover(x, dt=DT, grammar=grammar_from_names(["BM", "VG"], max_terms=1), n_starts=2)
    best = report.best.model
    mixture = report.monitor(arl0=500.0)
    assert isinstance(mixture, ShiryaevRoberts)
    assert mixture.null is best and mixture.binned is report.binned
    assert mixture.guaranteed_arl0 == pytest.approx(500.0)
    assert len(mixture.alternatives) == 2 * best.to_free().size
    sigma = float(np.sqrt(best.cumulants(2, t=1.0)[1]))
    louder = np.random.default_rng(5).normal(0.08 * DT, 1.5 * sigma * np.sqrt(DT), 2000)
    assert mixture.run(louder).first_alarm is not None
    cusum = report.monitor(Model(BM, [(1.5 * sigma,), (0.08,)]), arl0=500.0)
    assert isinstance(cusum, PageCUSUM) and cusum.null is best
    alarm = cusum.run(louder).first_alarm
    assert alarm is not None and alarm.time < 100
