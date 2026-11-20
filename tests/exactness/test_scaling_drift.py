"""Drift correction of LevyTriplet.scale for processes with jumps (Sato 1999, Prop. 11.10).

For Z = cX with truncation x·1{|x| ≤ 1}:

    b_Z = c b_X + c ∫ x (1{|cx| ≤ 1} − 1{|x| ≤ 1}) ν_X(dx),

so E[Z_1] = c E[X_1] and ψ_Z(u) = ψ_X(cu). Before the fix, the correction was
integrated over the wrong range, and every jump process had E[cX] ≠ cE[X].
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spxa.zoo.levy import CGMY, NIG, CompoundPoisson, NormalJump, VarianceGamma

PROCESSES = {
    "VG": lambda: VarianceGamma(sigma=0.2, nu=0.35, theta=-0.15),
    "NIG": lambda: NIG(alpha=6.0, beta=-2.0, delta=0.3, mu=0.05),
    "CPN": lambda: CompoundPoisson(0.9, NormalJump(-0.1, 0.25)),
    "CGMY": lambda: CGMY(C=0.7, G=5.0, M=8.0, Y=0.6),
}
SCALES = [0.5, 2.0, -0.5, -3.0, 0.1, 7.0]


@pytest.mark.exactness
@pytest.mark.parametrize("name", sorted(PROCESSES))
@pytest.mark.parametrize("c", SCALES)
def test_mean_is_linear_under_scaling(name: str, c: float) -> None:
    x = PROCESSES[name]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = c * x
    k = x.cumulants(2)
    assert z.triplet.cumulant(1) == pytest.approx(c * float(k[1]), rel=1e-7, abs=1e-10)
    assert z.triplet.cumulant(2) == pytest.approx(c**2 * float(k[2]), rel=1e-9)


@pytest.mark.exactness
@pytest.mark.parametrize("name", ["VG", "NIG"])
@pytest.mark.parametrize("c", [0.5, -3.0])
def test_scaled_triplet_char_exponent(name: str, c: float) -> None:
    x = PROCESSES[name]()
    z_trip = x.triplet.scale(c)
    for u in (0.7, -2.0):
        expected = complex(x.char_func(c * u, t=1.0))
        assert abs(z_trip.char_exp(u, 1.0) - expected) < 1e-7


@pytest.mark.exactness
def test_first_moment_between_on_compound_and_scaled_measures() -> None:
    vg, nig = PROCESSES["VG"](), PROCESSES["NIG"]()
    total = (vg.triplet + nig.triplet).nu
    a, b = 0.2, 3.0
    assert total.first_moment_between(a, b) == pytest.approx(
        vg.triplet.nu.first_moment_between(a, b) + nig.triplet.nu.first_moment_between(a, b),
        rel=1e-12,
    )
    scaled = vg.triplet.scale(-2.0).nu
    assert scaled.first_moment_between(a, b) == pytest.approx(
        -2.0 * vg.triplet.nu.first_moment_between(a / 2, b / 2), rel=1e-12
    )
    with pytest.raises(ValueError):
        vg.triplet.nu.first_moment_between(1.0, 0.5)
    assert np.isfinite(vg.triplet.nu.first_moment_between(1.0, np.inf))
