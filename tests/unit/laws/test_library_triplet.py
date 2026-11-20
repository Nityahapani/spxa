"""Triplet consistency laws: hold on the zoo, catch an inconsistent measure."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.triplet import LevyTriplet
from spxa.laws import Law
from spxa.laws.library.triplet import TRIPLET, measure_moments, measure_tail, triplet_cumulants
from spxa.laws.strategies import just
from spxa.zoo.levy import GammaProcess


@pytest.mark.parametrize("law_", TRIPLET, ids=lambda law_: law_.name)
def test_triplet_laws_hold_on_the_zoo(law_: Law) -> None:
    result = law_.check(n_examples=20, seed=11)
    assert result.ok, str(result)


class _SloppyGamma(GammaProcess):
    """Gamma whose measure reports kappa_2 = a/b instead of a/b^2 and a wrong tail."""

    def _triplet(self) -> LevyTriplet:
        tr = super()._triplet()
        a, b = self.a, self.b
        nu = DensityLevyMeasure(
            _density_fn=tr.nu.density,
            _total_mass=np.inf,
            _tail_fn=lambda x: a * np.exp(-b * x) / x,
            _moment_fns={2: lambda: a / b},
        )
        return LevyTriplet(b=tr.b, sigma_sq=0.0, nu=nu)


def _on(law_: Law, X: object) -> Law:
    return Law(law_.name, law_.fn, {**law_.strategies, "X": just(X)})


def test_inconsistent_measure_is_caught() -> None:
    X = _SloppyGamma(a=2.0, b=3.0)
    m = _on(measure_moments, X).check(10)
    assert m.status == "falsified" and m.counterexample["n"] == 2
    assert _on(measure_tail, X).check(10).status == "falsified"
    assert _on(triplet_cumulants, X).check(3).status == "falsified"


def test_atomic_measures_are_discarded() -> None:
    from spxa.zoo.levy import PoissonProcess

    r = _on(measure_tail, PoissonProcess(rate=1.0)).check(5)
    assert r.status == "unsatisfiable"
