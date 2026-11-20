"""GH increments match the characteristic function at short and long steps.

The general GH sampler used a fixed +-10 sd grid with ad hoc frequencies,
clipped the negative density and took the principal log of the Bessel
function. Its draws failed the variance test at dt = 1e-3 (p = 1e-35). It now
uses spxa.sim.cf_sampling with the continuous exponent.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxa.laws.stats import ecf_test, mean_test, variance_test
from spxa.zoo.levy import GeneralizedHyperbolic

LEVEL = 1e-5


@pytest.mark.parametrize("t", [1e-3, 0.1, 1.0, 3.0])
@pytest.mark.parametrize(
    "params", [(1.0, 3.0, 0.5, 1.0, 0.0), (-1.3, 2.0, -1.2, 4.0, 0.1), (2.5, 5.0, 4.0, 2.0, 0.0)]
)
def test_increments_match_the_cf(params, t: float) -> None:
    X = GeneralizedHyperbolic(*params)
    x = np.diff(X.simulate(2, 20_000, 2 * t, np.random.default_rng(0)), axis=1).ravel()
    k = X.cumulants(4)
    s = np.sqrt(k[2] * t)
    assert mean_test(x, k[1] * t, k[2] * t) > LEVEL
    assert variance_test(x, k[2] * t, k[4] * t) > LEVEL
    assert ecf_test(x, lambda u: X.char_func(u, t), [0.5 / s, 2.0 / s]) > LEVEL
