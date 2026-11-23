"""Laplace exponents that overflow are +inf, not NaN (spxa-laws seed 5)."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.laws import Law
from spxa.laws.library.cumulants import cgf_is_convex
from spxa.laws.quantities import laplace_exponent
from spxa.laws.strategies import just
from spxa.zoo.levy import MertonJumpDiffusion, PoissonProcess

POISSON_CLOCKED = PoissonProcess(rate=1.0, jump_size=2.0) @ PoissonProcess(rate=1.0, jump_size=1.0)


def test_compound_clock_overflows_to_real_infinity() -> None:
    # K(z) = e^{e^{2z} - 1} - 1, and e^{e^7 - 1} overflows
    k = laplace_exponent(POISSON_CLOCKED, np.array([3.5, 1.2, -1.0]))
    assert k[0] == complex(np.inf, 0.0)
    np.testing.assert_allclose(
        k[1:].real, np.expm1(np.expm1(2 * np.array([1.2, -1.0]))), rtol=1e-12
    )
    assert np.all(k.imag == 0.0)


def test_normal_jumps_overflow_to_real_infinity() -> None:
    M = MertonJumpDiffusion(drift=0.0, sigma=0.2, rate=1.0, jump_mean=0.2, jump_std=1.0)
    k = laplace_exponent(M, np.array([40.0, 1e3]))
    assert np.all(k == complex(np.inf, 0.0))


@pytest.mark.parametrize("u", [0.7, -1.3 + 0.2j])
def test_off_axis_values_are_unchanged(u: complex) -> None:
    P = PoissonProcess(rate=2.0, jump_size=0.5)
    expected = 2.0 * np.expm1(0.5j * u)  # jump_size 0.5 <= 1, so -iu * 2 * 0.5
    expected -= 1j * u * 1.0
    assert P.triplet.nu.jump_exponent(complex(u)) == pytest.approx(expected, rel=1e-14)


def test_cgf_convexity_holds_past_the_overflow() -> None:
    strategies = {
        **cgf_is_convex.strategies,
        "X": just(POISSON_CLOCKED),
        "s": just(0.7),
        "v": just(0.0),
        "w": just(0.5),
    }
    assert Law("pinned", cgf_is_convex.fn, strategies).check(n_examples=1).status == "verified"
