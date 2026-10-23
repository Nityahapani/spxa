"""Reproducible real workloads; run with pytest benchmarks/ --benchmark-only."""

import numpy as np

from spxa import BrownianMotion, GammaProcess, VarianceGamma
from spxa.analytics import european_call_price


def test_characteristic_function(benchmark):
    law = VarianceGamma(sigma=0.2, nu=0.4, theta=-0.1)
    frequencies = np.linspace(-50, 50, 2000)
    benchmark(law.char_func, frequencies)


def test_canonical_subordination(benchmark):
    parent, clock = BrownianMotion(mu=-0.1, sigma=0.2), GammaProcess(a=2, b=3)
    benchmark(lambda: (parent @ clock).cumulants(4))


def test_gamma_simulation(benchmark):
    law = GammaProcess(a=2, b=3)
    benchmark(law.simulate, n_steps=100, n_paths=100, rng=np.random.default_rng(42))


def test_fft_pricing(benchmark):
    benchmark(european_call_price, BrownianMotion(sigma=0.2), S0=100, K=100, t=1, r=0.05)
