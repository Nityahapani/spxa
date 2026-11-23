"""Compound Poisson measures with atoms: exact truncated means and jump exponents."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spxa.zoo.levy import CompoundPoisson, DiscreteJump


@pytest.mark.exactness
def test_discrete_jump_compound_poisson() -> None:
    jumps = DiscreteJump([-0.5, 1.5], [0.3, 0.7])
    cp = CompoundPoisson(1.2, jumps)
    mean = 1.2 * (0.3 * -0.5 + 0.7 * 1.5)
    assert cp.triplet.cumulant(1) == pytest.approx(mean, rel=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert (3.0 * cp).cumulants(1)[1] == pytest.approx(3.0 * mean, rel=1e-12)
        assert (0.5 * cp).cumulants(1)[1] == pytest.approx(0.5 * mean, rel=1e-12)
    nu = cp.triplet.nu
    assert nu.first_moment_between(1.0, np.inf) == pytest.approx(1.2 * 0.7 * 1.5)
    assert nu.first_moment_between(0.1, 1.0) == pytest.approx(1.2 * 0.3 * -0.5)
    with pytest.raises(ValueError):
        nu.first_moment_between(2.0, 1.0)


@pytest.mark.exactness
def test_compound_poisson_jump_exponent_is_closed_form() -> None:
    from spxa.zoo.levy import NormalJump

    for jumps in (DiscreteJump([-0.5, 1.5], [0.3, 0.7]), NormalJump(-0.1, 0.8)):
        cp = CompoundPoisson(1.2, jumps)
        for u in (0.4, -3.0):
            assert abs(cp.triplet.char_exp(u, 2.0) - complex(cp.char_func(u, t=2.0))) < 1e-12
