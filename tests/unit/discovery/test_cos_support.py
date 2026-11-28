"""The COS interval of the binned likelihood covers each candidate law.

A cosine series on ``[a, b]`` folds the mass of a law outside the interval
back inside it. With ``[a, b]`` fixed by the data, a jump part with jumps of
-85% on a sample whose moves stay within ±16% had reflections next to zero,
and it beat the true Merton model by 3 log-likelihood units. These tests pin
the fix: :meth:`BinnedData.cos_grid` widens the interval to the law's range.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from spxa.discovery import families as F
from spxa.discovery.empirical import summarize
from spxa.discovery.expr import Model, leaf
from spxa.discovery.fit import (
    _PENALTY,
    MAX_WIDENING,
    BinnedData,
    _Objective,
    fit_structure,
)
from spxa.graph.autodiff.discovery import likelihood_value_grad_hess

DT = 1 / 252
MERTON = Model(
    leaf(F.BROWNIAN) + leaf(F.NORMAL_JUMPS) + leaf(F.DRIFT),
    [(0.15,), (20.0, -0.02, 0.04), (0.05,)],
)
# the maximiser of the binned likelihood on a fixed interval for seed 2
ALIASED = Model(
    MERTON.structure,
    [
        (0.1526981119031147,),
        (16.659174481675244, -0.8489056430700951, 0.04231234175254787),
        (-0.04885772937600351,),
    ],
)
# admissible but wide (8.5 times the data's interval): 3.3% of each step's law
# lies outside [a, b], while every bin keeps a probability above 1e-5
WIDE = Model(MERTON.structure, [(0.15,), (20.0, -0.4, 0.2), (0.05,)])


def _merton_data(seed: int) -> Any:
    rng = np.random.default_rng(seed)
    x = np.diff(MERTON.to_process().simulate(3000, 1, T=3000 * DT, rng=rng)[0])
    summ = summarize(x, dt=DT)
    return summ, BinnedData.from_sample(summ.sample, summ.dt)


@pytest.fixture(scope="module")
def seed2() -> Any:
    return _merton_data(2)


def _loglik(binned: BinnedData, model: Model) -> float:
    return float(binned.counts @ np.log(binned.probabilities(model)))


def test_in_range_laws_use_the_data_grid(seed2: Any) -> None:
    _, bd = seed2
    A, B, u, sine = bd.cos_grid(MERTON)
    assert (A, B) == (bd.a, bd.b)
    assert u is bd.u and sine is bd.sine


def test_wide_laws_get_a_wider_grid_with_the_same_cut_off(seed2: Any) -> None:
    _, bd = seed2
    A, B, u, sine = bd.cos_grid(WIDE)
    step = 0.25 * (bd.b - bd.a)
    assert A < bd.a and B > bd.b
    assert B - A <= MAX_WIDENING * (bd.b - bd.a)
    for gap in (bd.a - A, B - bd.b):
        assert gap / step == pytest.approx(round(gap / step), abs=1e-9)
    # the law's 12-sd range fits inside the new interval
    k = WIDE.cumulants(4, t=DT)
    half = 12 * np.sqrt(k[1] + bd.bandwidth**2 + np.sqrt(k[3]))
    assert A <= k[0] - half and k[0] + half <= B
    # the frequency cut-off stays put and the grid is reused
    assert u[1] == pytest.approx(np.pi / (B - A))
    assert u[-1] == pytest.approx(bd.u[-1], rel=2 * (B - A) / (bd.b - bd.a) / bd.n_terms)
    assert sine.shape == (bd.edges.size - 2, u.size - 1)
    assert bd.cos_grid(WIDE)[2] is u


def test_the_aliased_law_no_longer_beats_the_truth(
    seed2: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bd = seed2
    # its 12-sd range is 13 times the data's interval, so it is rejected ...
    with pytest.raises(ValueError, match="times"):
        bd.probabilities(ALIASED)
    # ... and an expansion wide enough to hold it scores it far below the truth,
    # where on the fixed [a, b] it scored 3 units above
    monkeypatch.setattr("spxa.discovery.fit.MAX_WIDENING", 16.0)
    wide = BinnedData.from_sample(bd.sample, DT)
    assert _loglik(wide, ALIASED) < _loglik(wide, MERTON) - 1000


def test_wide_grid_probabilities_match_monte_carlo(seed2: Any) -> None:
    _, bd = seed2
    n = 200_000
    y = np.diff(WIDE.to_process().simulate(n, 1, T=n * DT, rng=np.random.default_rng(0))[0])
    y = y + bd.bandwidth * np.random.default_rng(1).normal(size=n)
    mc = np.histogram(y, bins=bd.edges)[0] / n
    p = bd.probabilities(WIDE)
    assert bd.cos_grid(WIDE)[0] < bd.a
    sd = np.sqrt(np.maximum(p * (1 - p), 1e-12) / n)
    assert np.all(np.abs(mc - p) <= 5 * sd + 2e-5)
    # the same series on the data's fixed interval is off by about 20 sd
    phi = WIDE.char_func(bd.u, t=DT) * np.exp(-0.5 * (bd.bandwidth * bd.u) ** 2)
    coef = 2 / (bd.b - bd.a) * np.real(phi * np.exp(-1j * bd.u * bd.a))
    F = (bd.edges[1:-1] - bd.a) / (bd.b - bd.a) + bd.sine @ coef[1:]
    fixed = np.diff(np.maximum.accumulate(np.clip(np.r_[0.0, F, 1.0], 0, 1)))
    assert np.max(np.abs(mc - fixed) / sd) > 10
    # the end bins absorb the 3.3% of the law that lies outside [a, b]
    outside = np.mean((y < bd.a) | (y > bd.b))
    assert outside > 0.03
    assert p.sum() == pytest.approx(1.0)  # the end bins absorb the tails


def test_probabilities_are_continuous_where_the_grid_switches(seed2: Any) -> None:
    # the interval grows in quarter steps; at the switch the two expansions
    # must agree to the series' accuracy, not jump by an aliasing error
    _, bd = seed2

    def model(m: float) -> Model:
        return Model(MERTON.structure, [(0.15,), (20.0, m, 0.2), (0.05,)])

    lo, hi = -0.02, -0.4
    assert bd.cos_grid(model(lo))[0] != bd.cos_grid(model(hi))[0]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bd.cos_grid(model(mid))[0] == bd.cos_grid(model(lo))[0]:
            lo = mid
        else:
            hi = mid
    assert bd.cos_grid(model(lo))[0] != bd.cos_grid(model(hi))[0]
    jump = np.abs(bd.probabilities(model(lo)) - bd.probabilities(model(hi)))
    assert np.max(jump) < 1e-11


def test_laws_far_wider_than_the_data_are_rejected(seed2: Any) -> None:
    summ, bd = seed2
    wide = Model(MERTON.structure, [(0.15,), (20.0, -50.0, 0.04), (0.05,)])
    with pytest.raises(ValueError, match="times"):
        bd.cos_grid(wide)
    with pytest.raises(ValueError):
        bd.probabilities(wide)
    # the optimiser sees the penalty plateau, not an exception
    obj = _Objective(MERTON.structure, summ, "likelihood", binned=bd)
    assert obj(wide.to_free()) == _PENALTY
    v, g = obj.value_and_grad(wide.to_free())
    assert v == _PENALTY and np.all(g == 0)


def test_exact_derivatives_on_a_wide_grid(seed2: Any) -> None:
    _, bd = seed2
    x0 = WIDE.to_free()
    A0 = bd.cos_grid(WIDE)[0]
    assert A0 < bd.a
    v, g, H = likelihood_value_grad_hess(WIDE.structure, bd, x0)
    assert v == pytest.approx(-_loglik(bd, WIDE), rel=1e-12)
    eps = 1e-5
    g_fd = np.zeros_like(x0)
    H_fd = np.zeros((x0.size, x0.size))
    for i in range(x0.size):
        e = np.zeros_like(x0)
        e[i] = eps
        for s in (1, -1):
            assert bd.cos_grid(Model.from_free(WIDE.structure, x0 + s * e))[0] == A0
        vp, gp, _ = likelihood_value_grad_hess(WIDE.structure, bd, x0 + e)
        vm, gm, _ = likelihood_value_grad_hess(WIDE.structure, bd, x0 - e)
        g_fd[i] = (vp - vm) / (2 * eps)
        H_fd[:, i] = (gp - gm) / (2 * eps)
    scale = np.max(np.abs(g)) + 1.0
    assert np.max(np.abs(g - g_fd)) < 1e-6 * scale
    assert np.max(np.abs(H - H_fd)) < 1e-6 * (np.max(np.abs(H)) + 1.0)


@pytest.mark.parametrize("seed", [2, 8])
def test_fits_on_previously_aliased_samples_find_the_true_jumps(seed: int) -> None:
    # both samples used to send the jump mean to about -0.8 from every start
    summ, bd = _merton_data(seed)
    truth = _loglik(bd, MERTON)
    for r in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fr = fit_structure(
                leaf(F.BROWNIAN) + leaf(F.NORMAL_JUMPS),
                summ,
                binned=bd,
                n_starts=3,
                rng=np.random.default_rng(r),
            )
        (sigma,), (rate, jump_mean, jump_std), _ = fr.model.params
        assert abs(jump_mean + 0.02) < 0.02, fr.model
        assert abs(sigma - 0.15) < 0.02 and abs(jump_std - 0.04) < 0.015
        assert 5 < rate < 60
        # it is the maximiser: at least as good as the parameters that made the data
        assert fr.loglik >= truth - 1e-6
