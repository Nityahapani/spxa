"""Cumulants from the characteristic function, and CGF convexity."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.laws import Law
from spxa.laws.library.cumulants import CUMULANTS, cauchy_cumulants, cumulants_from_cf
from spxa.laws.strategies import just
from spxa.zoo.levy import (
    CGMY,
    BrownianMotion,
    GammaProcess,
    PoissonProcess,
    VarianceGamma,
)


@pytest.mark.parametrize("law_", CUMULANTS, ids=lambda law_: law_.name)
def test_holds_on_the_zoo(law_: Law) -> None:
    result = law_.check(n_examples=12, seed=5)
    assert result.ok, str(result)


def test_cauchy_cumulants_of_closed_forms() -> None:
    G = GammaProcess(a=2.0, b=3.0)  # κ_n = a (n-1)! / b^n
    np.testing.assert_allclose(
        cauchy_cumulants(G, 4, radius=1.5),
        [2 / 3, 2 / 9, 4 / 27, 12 / 81],
        rtol=1e-7,
    )
    V = VarianceGamma(sigma=0.2, nu=0.3, theta=-0.1)
    np.testing.assert_allclose(
        cauchy_cumulants(V, 4, radius=1.0), list(V.cumulants(4).values()), rtol=1e-9
    )
    Z = BrownianMotion(mu=0.1, sigma=0.3) @ GammaProcess(a=2.0, b=3.0)
    np.testing.assert_allclose(
        cauchy_cumulants(Z, 4, radius=1.0), list(Z.cumulants(4).values()), rtol=1e-7
    )


def test_cauchy_cumulants_resolve_a_compound_clock() -> None:
    # A Poisson clock exponentiates K_X: |K_Z| ~ 1e5 near the branch point at
    # z = M = 2, so 32 trapezoid points left a 1.6e-6 error in kappa_1 (seed 2).
    Z = CGMY(C=1.0, G=2.5, M=2.0, Y=1.5) @ PoissonProcess(rate=1.0, jump_size=2.0)
    # kappa_1 = 2 C Gamma(-1/2) (sqrt(M) - sqrt(G)), by mpmath to 30 digits
    assert cauchy_cumulants(Z, 1, radius=1.0)[0] == pytest.approx(1.1834693342718554, rel=1e-10)
    np.testing.assert_allclose(
        cauchy_cumulants(Z, 4, radius=1.0), list(Z.cumulants(4).values()), rtol=1e-9
    )
    pinned = Law("pinned", cumulants_from_cf.fn, {**cumulants_from_cf.strategies, "X": just(Z)})
    assert pinned.check(n_examples=1).status == "verified"


class _OffGamma(GammaProcess):
    """A gamma process whose third cumulant is 0.1% too large."""

    def cumulants(self, order: int) -> dict[int, float]:
        k = super().cumulants(order)
        if 3 in k:
            k[3] *= 1.001
        return k


def test_a_small_cumulant_error_is_caught() -> None:
    law_ = Law("c", cumulants_from_cf.fn, {"X": just(_OffGamma(a=2.0, b=3.0))})
    r = law_.check(n_examples=1)
    assert r.status == "falsified" and "kappa_3" in r.message
