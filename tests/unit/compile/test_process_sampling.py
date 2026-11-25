"""CompiledProcess sampling: simulate, sample_at, paths, free and fixed parameters."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.compile import CompiledProcess
from spxa.zoo.levy import BrownianMotion, GammaProcess, PoissonProcess

from .test_sampler import N, U, _cases


@pytest.mark.parametrize("key", ["composite", "merton", "nig_at_gamma", "cp_discrete"])
def test_moments_match_compiled_cumulants(key: str) -> None:
    cpr = CompiledProcess(_cases()[key])
    k = cpr.cumulants(2)
    x = cpr.sample_at(np.full(4 * N, 0.5), np.random.default_rng(3))
    mean, var = 0.5 * k[0], 0.5 * k[1]
    assert abs(x.mean() - mean) < 5 * np.sqrt(var / x.size)
    assert x.var() == pytest.approx(var, rel=0.05)


def test_simulate_and_paths_shapes_and_increments() -> None:
    cpr = CompiledProcess(_cases()["nig_at_gamma"])
    rng = np.random.default_rng(11)
    p = cpr.simulate(252, 4000, 1.0, rng)
    assert p.shape == (4000, 253) and np.all(p[:, 0] == 0.0) and np.all(np.isfinite(p))
    k = cpr.cumulants(2)
    assert abs(p[:, -1].mean() - k[0]) < 5 * np.sqrt(k[1] / 4000)
    assert cpr.paths([0.0, 0.1, 0.1, 0.35, 1.0], 500, rng).shape == (500, 5)
    theta = np.tile(cpr.values, (2, 1))
    assert cpr.simulate(10, 7, 2.0, rng, theta).shape == (2, 7, 11)
    assert cpr.paths([0.5, 1.0], 3, rng, theta).shape == (2, 3, 2)


def test_independent_increments_on_an_uneven_grid() -> None:
    cpr = CompiledProcess(BrownianMotion(0.0, 1.0) @ GammaProcess(2.0, 2.0))
    grid = np.array([0.05, 0.3, 1.2])
    q = cpr.paths(grid, 40_000, np.random.default_rng(12))
    incr = np.diff(q, axis=1, prepend=0.0)
    # Var X_t = t * Var(N(0,1) at a mean-1 clock) = t
    np.testing.assert_allclose(incr.var(axis=0), np.diff(grid, prepend=0.0), rtol=0.05)
    assert abs(np.corrcoef(incr[:, 1], incr[:, 2])[0, 1]) < 0.03


def test_free_and_fixed_parameters() -> None:
    proc = BrownianMotion(0.0, 0.2) + PoissonProcess(1.0)
    cpr = CompiledProcess(proc, free=["BrownianMotion.sigma"], fixed={"PoissonProcess.rate": 0.0})
    x = cpr.sample_at(np.full(20_000, 1.0), np.random.default_rng(13), theta=[0.5])
    assert x.std() == pytest.approx(0.5, rel=0.03)  # no jumps: rate fixed to 0


def test_compiled_process_sampler_methods() -> None:
    cpr = CompiledProcess(_cases()["composite"])
    assert cpr.sampler_exact()
    x = cpr.sample_at(np.full(N, 1.0), np.random.default_rng(16))
    kernel = cpr.kernel
    emp = np.exp(1j * np.outer(U, x)).mean(axis=1)
    assert np.abs(emp - np.exp(kernel.psi(U))).max() * np.sqrt(N) < 4.5


def test_process_compile_samples_and_exports_sampler() -> None:
    import spxa.compile as sc

    assert "Sampler" in sc.__all__ and sc.Sampler is type(CompiledProcess(_cases()["bm"]).sampler)
    proc = BrownianMotion(0.1, 0.3) @ GammaProcess(2.0, 2.0)
    x = proc.compile().sample_at(np.full(N, 1.0), np.random.default_rng(19))
    # Var = E[T] sigma^2 + Var(T) mu^2 = 0.09 + 0.5 * 0.01
    assert x.var() == pytest.approx(0.095, rel=0.04)
