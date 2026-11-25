"""
Benchmarks for spxa.compile against the uncompiled paths.

Each pair times the same quantity through the zoo / DiffProcess and through a
compiled kernel; ``python benchmarks/compile_report.py`` prints the ratios.

Baselines are the fastest uncompiled route: a leaf family's closed-form
``char_exponent``, and a composite's ``_char_exponent`` (the path that sums,
scalings and subordination evaluate).
"""

from __future__ import annotations

import numpy as np
import pytest

from spxa.graph.autodiff.fourier import price_and_gradient
from spxa.graph.autodiff.process import DiffProcess
from spxa.zoo.levy import (
    CGMY,
    NIG,
    BrownianMotion,
    GammaProcess,
    InverseGaussianProcess,
    KouJumpDiffusion,
    MertonJumpDiffusion,
    VarianceGamma,
)

U = np.linspace(-50.0, 50.0, 4096)
STRIKES = np.linspace(80.0, 120.0, 9)


def _composite():
    """Includes Kou, whose exponent has no vectorised zoo override."""
    return (
        BrownianMotion(0.05, 0.2) + 0.5 * KouJumpDiffusion(0.05, 0.2, 2.0, 0.4, 8.0, 5.0)
    ) @ GammaProcess(4.0, 4.0) + CGMY(1.0, 5.0, 6.0, 0.5)


def _vectorised():
    """Every family here has a vectorised zoo exponent."""
    return (
        (NIG(10.0, -2.0, 0.5) + VarianceGamma(0.2, 0.3, -0.1)) @ GammaProcess(5.0, 5.0)
        + 0.5 * NIG(8.0, 1.0, 0.3)
    ) @ InverseGaussianProcess(1.0, 2.0)


@pytest.fixture(scope="module")
def composite():
    proc = _composite()
    cp = proc.compile()
    cp.char_exponent_and_grad(U)  # generate once
    return proc, DiffProcess(proc), cp


@pytest.fixture(scope="module")
def vectorised():
    proc = _vectorised()
    cp = proc.compile()
    cp.char_exponent_and_grad(U)
    return proc, DiffProcess(proc), cp


def test_benchmark_psi_merton_zoo(benchmark) -> None:
    proc = MertonJumpDiffusion(0.05, 0.2, 1.0, -0.05, 0.1)
    benchmark(lambda: proc.char_exponent(U))


def test_benchmark_psi_merton_compiled(benchmark) -> None:
    cp = MertonJumpDiffusion(0.05, 0.2, 1.0, -0.05, 0.1).compile()
    cp.char_exponent(U)
    benchmark(lambda: cp.char_exponent(U))


def test_benchmark_psi_composite_zoo(benchmark, composite) -> None:
    proc, _, _ = composite
    benchmark(lambda: proc._char_exponent(U))


def test_benchmark_psi_composite_compiled(benchmark, composite) -> None:
    _, _, cp = composite
    benchmark(lambda: cp.char_exponent(U))


def test_benchmark_psi_vectorised_zoo(benchmark, vectorised) -> None:
    proc, _, _ = vectorised
    benchmark(lambda: proc._char_exponent(U))


def test_benchmark_psi_vectorised_compiled(benchmark, vectorised) -> None:
    _, _, cp = vectorised
    benchmark(lambda: cp.char_exponent(U))


def test_benchmark_grad_composite_diffprocess(benchmark, composite) -> None:
    _, dp, _ = composite
    benchmark(lambda: dp.char_exponent_and_grad(U))


def test_benchmark_grad_composite_compiled(benchmark, composite) -> None:
    _, _, cp = composite
    benchmark(lambda: cp.char_exponent_and_grad(U))


def test_benchmark_grad_vectorised_diffprocess(benchmark, vectorised) -> None:
    _, dp, _ = vectorised
    benchmark(lambda: dp.char_exponent_and_grad(U))


def test_benchmark_grad_vectorised_compiled(benchmark, vectorised) -> None:
    _, _, cp = vectorised
    benchmark(lambda: cp.char_exponent_and_grad(U))


def test_benchmark_grad_composite_batch64_compiled(benchmark, composite) -> None:
    _, _, cp = composite
    theta = np.tile(cp.values, (64, 1))
    u = U[::8]
    cp.char_exponent_and_grad(u, theta)
    benchmark(lambda: cp.char_exponent_and_grad(u, theta))


def test_benchmark_cos_gradient_diffprocess(benchmark, composite) -> None:
    _, dp, _ = composite
    benchmark(lambda: price_and_gradient(dp, 100.0, STRIKES, 0.5, method="cos"))


def test_benchmark_cos_gradient_compiled(benchmark, composite) -> None:
    _, _, cp = composite
    cp.cos_price(100.0, STRIKES, 0.5, grad=True)
    benchmark(lambda: cp.cos_price(100.0, STRIKES, 0.5, grad=True))


def test_benchmark_cos_gradient_batch64_compiled(benchmark, composite) -> None:
    _, _, cp = composite
    theta = np.tile(cp.values, (64, 1))
    cp.cos_price(100.0, STRIKES, 0.5, theta=theta, grad=True)
    benchmark(lambda: cp.cos_price(100.0, STRIKES, 0.5, theta=theta, grad=True))


def test_benchmark_simulate_nig_gamma_zoo(benchmark) -> None:
    proc = NIG(10.0, -2.0, 0.5) @ GammaProcess(5.0, 5.0)
    rng = np.random.default_rng(0)
    res = benchmark(lambda: proc.simulate(252, 500, 1.0, rng))
    assert res.shape == (500, 253)


def test_benchmark_simulate_nig_gamma_compiled(benchmark) -> None:
    cp = (NIG(10.0, -2.0, 0.5) @ GammaProcess(5.0, 5.0)).compile()
    rng = np.random.default_rng(0)
    res = benchmark(lambda: cp.simulate(252, 500, 1.0, rng))
    assert res.shape == (500, 253)
