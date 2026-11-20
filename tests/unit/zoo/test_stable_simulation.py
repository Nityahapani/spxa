"""Chambers–Mallows–Stuck draws follow the S1 law that char_func describes.

At alpha = 1, S1 is not scale-equivariant. If X ~ S1(1, beta, 1, 0), then
cX ~ S1(1, beta, c, -(2/pi) beta c log c). The simulator scaled the standard
draw without adding the shift back, so for beta != 0 the empirical cf missed
by z = 20 to 150 at n = 100k. The checks below use empirical-cf z-scores with
|z| < 4.5 (a false-alarm rate of about 7e-6 each) and fixed seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxa.zoo.levy import AlphaStable


def _ecf_z(x: np.ndarray, X: AlphaStable, u: float, t: float) -> float:
    p, p2 = complex(X.char_func(u, t)), complex(X.char_func(2 * u, t))
    m = np.mean(np.exp(1j * u * x))
    n = len(x)
    z_re = (m.real - p.real) / np.sqrt(((1 + p2.real) / 2 - p.real**2) / n)
    z_im = (m.imag - p.imag) / np.sqrt(((1 - p2.real) / 2 - p.imag**2) / n)
    return float(max(abs(z_re), abs(z_im)))


@pytest.mark.parametrize(
    ("alpha", "beta"), [(1.0, 0.4), (1.0, -0.8), (1.0, 0.0), (0.7, -0.5), (1.5, 0.3)]
)
@pytest.mark.parametrize("dt", [1.0, 0.3])
def test_increments_match_the_cf(alpha: float, beta: float, dt: float) -> None:
    X = AlphaStable(alpha, beta, 0.5, 0.2)
    x = X.simulate(1, 100_000, dt, np.random.default_rng(11))[:, 1]
    for u in (0.3, 1.0, 3.0):
        assert _ecf_z(x, X, u, dt) < 4.5
