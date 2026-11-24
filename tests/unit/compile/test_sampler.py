"""Fused exact samplers: laws against the compiled characteristic function."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from spxa.compile.lower import lower
from spxa.compile.sampler import Sampler
from spxa.compile.kernel import ExponentKernel
from spxa.compile.sampler import LEAF_SAMPLERS, _jump_sums, _segment_sums
from spxa.zoo.levy import (
    CGMY,
    NIG,
    BrownianMotion,
    Drift,
    GammaProcess,
    InverseGaussianProcess,
    KouJumpDiffusion,
    MertonJumpDiffusion,
    NegativeBinomialProcess,
    PoissonProcess,
    TemperedStable,
    VarianceGamma,
)
from spxa.zoo.levy import compound_poisson as cp

N = 60_000
U = np.array([0.3, 1.0, 2.5])


def _cases() -> dict:
    return {
        "bm": BrownianMotion(0.1, 0.3),
        "gamma": GammaProcess(2.0, 3.0),
        "ig": InverseGaussianProcess(1.0, 2.0),
        "negbin": NegativeBinomialProcess(2.0, 0.4),
        "poisson": PoissonProcess(3.0),
        "vg": VarianceGamma(0.2, 0.3, -0.1),
        "nig": NIG(10.0, -2.0, 0.5, 0.03),
        "merton": MertonJumpDiffusion(0.05, 0.2, 1.5, -0.1, 0.15),
        "kou": KouJumpDiffusion(0.05, 0.2, 2.0, 0.4, 8.0, 5.0),
        "cp_normal": cp.CompoundPoisson(2.0, cp.NormalJump(0.1, 0.2)),
        "cp_exp": cp.CompoundPoisson(2.0, cp.ExponentialJump(3.0)),
        "cp_dexp": cp.CompoundPoisson(2.0, cp.DoubleExponentialJump(0.3, 5.0, 4.0)),
        "cp_uniform": cp.CompoundPoisson(2.0, cp.UniformJump(-0.2, 0.5)),
        "cp_discrete": cp.CompoundPoisson(2.0, cp.DiscreteJump([-1.0, 0.5, 2.0], [0.2, 0.5, 0.3])),
        "cgmy_fallback": CGMY(1.0, 5.0, 6.0, 0.5),
        "ts_fallback": TemperedStable(0.5, 1.0, 2.0),
        "composite": (BrownianMotion(0.05, 0.2) + 0.5 * VarianceGamma(0.2, 0.3, -0.1))
        @ GammaProcess(4.0, 4.0)
        + 2.0 * Drift(0.1),
        "nig_at_gamma": NIG(10.0, -2.0, 0.5) @ GammaProcess(5.0, 5.0),
        "nested": (BrownianMotion(0.0, 1.0) @ InverseGaussianProcess(1.0, 2.0))
        @ GammaProcess(3.0, 3.0),
        "bm_at_poisson": BrownianMotion(0.1, 0.5) @ PoissonProcess(2.0),
        "gauss_merge": BrownianMotion(0.1, 0.2)
        + BrownianMotion(0.0, 0.3)
        - 1.5 * Drift(0.2)
        + MertonJumpDiffusion(0.05, 0.2, 1.5, -0.1, 0.15),
    }


def _cf_error(x: np.ndarray, kernel: ExponentKernel, t: float, theta=None) -> float:
    """``max_u |empirical cf - exact cf| * sqrt(n)``; O(1) under the right law."""
    emp = np.exp(1j * np.outer(U, x)).mean(axis=1)
    exact = np.exp(t * kernel.psi(U, theta))
    return float(np.abs(emp - exact).max() * np.sqrt(x.size))


@pytest.mark.parametrize("key", sorted(_cases()))
@pytest.mark.parametrize("t", [1.0, 0.02])
def test_law_matches_characteristic_function(key: str, t: float) -> None:
    lw = lower(_cases()[key])
    sampler = Sampler(lw)
    kernel = ExponentKernel(lw, "numpy", None, None, False)
    x = sampler.sample_at(np.full(N, t), np.random.default_rng(zlib.crc32(f"{key}{t}".encode())))
    assert x.shape == (N,) and np.all(np.isfinite(x))
    assert _cf_error(x, kernel, t) < 4.5
    assert sampler.exact()


def test_gaussian_terms_are_merged_into_one_draw() -> None:
    lw = lower(_cases()["gauss_merge"])
    plan = Sampler(lw).plan
    assert len(plan.gauss) == 4 and len(plan.jumps) == 1
    # drift alone draws no normal: the result is deterministic
    drift = Sampler(lower(Drift(0.3) + 2.0 * Drift(-0.1)))
    np.testing.assert_allclose(drift.sample_at([0.0, 0.5, 2.0]), [0.0, 0.05, 0.2], rtol=1e-15)


def test_zero_times_give_zero() -> None:
    for key, proc in _cases().items():
        x = Sampler(lower(proc)).sample_at(np.zeros(50), np.random.default_rng(1))
        assert np.all(x == 0.0), key


def test_batched_theta() -> None:
    proc = (
        BrownianMotion(0.05, 0.2) + 0.5 * KouJumpDiffusion(0.05, 0.2, 2.0, 0.4, 8.0, 5.0)
    ) @ GammaProcess(4.0, 4.0) + CGMY(1.0, 5.0, 6.0, 0.5)
    lw = lower(proc)
    sampler, kernel = Sampler(lw), ExponentKernel(lw, "numpy", None, None, False)
    theta = np.tile(lw.values, (3, 1))
    theta[:, lw.names.index("BrownianMotion.sigma")] = [0.1, 0.25, 0.4]
    theta[:, lw.names.index("CGMY.C")] = [0.5, 1.0, 2.0]
    # small rows: vectorised over the batch; one large row per call otherwise
    for shape in [(2000,), (N,)]:
        x = sampler.sample_at(np.full(shape, 0.5), np.random.default_rng(5), theta)
        assert x.shape == (3,) + shape
        if shape == (N,):
            for b in range(3):
                assert _cf_error(x[b], kernel, 0.5, theta[b]) < 4.5
    small = sampler.sample_at(np.full((3000, 4), 0.25), np.random.default_rng(6), theta)
    for b in range(3):
        assert _cf_error(small[b].ravel(), kernel, 0.25, theta[b]) < 4.5
    assert sampler.exact(theta)


def test_reproducible_with_a_seed() -> None:
    s = Sampler(lower(_cases()["composite"]))
    a = s.simulate(20, 30, 1.0, np.random.default_rng(9))
    b = s.simulate(20, 30, 1.0, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)


def test_fallback_and_opaque_leaves() -> None:
    from spxa.zoo.levy import MeixnerProcess

    proc = MeixnerProcess(0.3, -0.5, 1.0, 0.0) + BrownianMotion(0.0, 0.1)
    lw = lower(proc)
    s = Sampler(lw)
    assert s.fallback_leaves == 1
    assert isinstance(s.exact(), bool)
    x = s.sample_at(np.full(N, 1.0), np.random.default_rng(14))
    kernel = ExponentKernel(lw, "numpy", None, None, False)
    assert _cf_error(x, kernel, 1.0) < 4.5
    assert "fallback_leaves=1" in repr(s)


def test_jump_sum_aggregation_is_exact() -> None:
    rng = np.random.default_rng(15)
    n = np.repeat(np.arange(5), 20_000)
    # uniform sums: Irwin-Hall moments
    s = _segment_sums(n, rng.random)
    for k in range(5):
        block = s[n == k]
        assert block.mean() == pytest.approx(k / 2, abs=0.02)
        assert block.var() == pytest.approx(k / 12, abs=0.01)
    d = _jump_sums("double_exponential", n, (0.3, 5.0, 4.0), rng, {})
    mean1 = 0.3 / 5.0 - 0.7 / 4.0
    assert d[n == 4].mean() == pytest.approx(4 * mean1, abs=0.01)
    assert np.all(d[n == 0] == 0.0)
    with pytest.raises(ValueError, match="unknown jump law"):
        _jump_sums("bogus", np.ones(3, dtype=int), (), rng, {})


def test_every_rule_sampler_is_known() -> None:
    from spxa.compile.rules import LEAF_RULES

    named = {"gaussian", "drift", "compound_poisson"}
    assert named <= set(LEAF_SAMPLERS)
    for case in _cases().values():
        for leaf in _leaves(lower(case).tree):
            kind = getattr(getattr(leaf, "spec", None), "sampler", None)
            assert kind is None or kind in LEAF_SAMPLERS
    assert len(LEAF_RULES) >= 17


def _leaves(node):
    for attr in ("a", "b", "base", "x", "clock"):
        child = getattr(node, attr, None)
        if child is not None:
            yield from _leaves(child)
    if not any(hasattr(node, a) for a in ("a", "base", "x")):
        yield node


def test_input_validation() -> None:
    s = Sampler(lower(BrownianMotion(0.0, 1.0)))
    with pytest.raises(ValueError, match="non-negative"):
        s.sample_at([-1.0])
    with pytest.raises(ValueError, match="non-negative"):
        s.sample_at([np.inf])
    with pytest.raises(ValueError, match="theta must have shape"):
        s.sample_at([1.0], theta=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-decreasing"):
        s.paths([0.5, 0.2])
    with pytest.raises(ValueError, match="1-D"):
        s.paths([])
    with pytest.raises(ValueError, match="n_steps"):
        s.simulate(0)
    clock = Sampler(lower(BrownianMotion(0.0, 1.0) @ (2.0 * GammaProcess(1.0, 1.0))))
    theta = clock.lowered.values.copy()
    assert clock.sample_at(np.ones(3), np.random.default_rng(0), theta).shape == (3,)
    theta[clock.lowered.names.index("scale")] = -1.0  # a clock running backwards
    with pytest.raises(ValueError, match="negative time"):
        clock.sample_at(np.ones(10), np.random.default_rng(0), theta)


def test_paths_on_a_grid() -> None:
    s = Sampler(lower(_cases()["nig_at_gamma"]))
    q = s.paths([0.0, 0.1, 0.1, 0.35, 1.0], 500, np.random.default_rng(17))
    assert q.shape == (500, 5)
    assert np.all(q[:, 0] == 0.0) and np.all(q[:, 2] == q[:, 1])
    theta = np.tile(s.lowered.values, (2, 1))
    assert s.paths([0.5, 1.0], 3, np.random.default_rng(18), theta).shape == (2, 3, 2)
