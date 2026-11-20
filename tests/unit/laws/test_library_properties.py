"""Property-flag laws: hold on the zoo, catch wrong flags, shrink the drift fallacy."""

from __future__ import annotations

import dataclasses

import pytest

from spxa.core.properties import ProcessProperties
from spxa.laws import Law, implies, law
from spxa.laws.library.properties import (
    PROPERTIES,
    martingale,
    martingale_flag_is_complete,
    martingale_flag_is_sound,
    self_similarity_is_sound,
    subordinator_flag_is_sound,
)
from spxa.laws.processes import compositions, martingales
from spxa.laws.strategies import floats, just
from spxa.zoo.levy import BrownianMotion, Drift, GammaProcess, VarianceGamma

FAST = {"subordinator_flag_is_sound": 8}


@pytest.mark.parametrize("law_", PROPERTIES, ids=lambda law_: law_.name)
def test_holds_on_the_zoo(law_: Law) -> None:
    result = law_.check(n_examples=FAST.get(law_.name, 20), seed=7)
    assert result.ok, str(result)


def _on(law_: Law, X: object) -> Law:
    return Law(law_.name, law_.fn, {**law_.strategies, "X": just(X)})


class _Flagged(BrownianMotion):
    """Brownian motion with hand-edited property flags."""

    def __init__(self, mu: float, sigma: float, **flags: object) -> None:
        super().__init__(mu, sigma)
        self._flags = flags

    def _properties(self) -> ProcessProperties:
        return dataclasses.replace(super()._properties(), **self._flags)  # type: ignore[arg-type]


def test_unsound_martingale_flag_is_caught() -> None:
    result = _on(martingale_flag_is_sound, _Flagged(0.1, 0.2, is_martingale=True)).check(
        n_examples=3
    )
    assert result.status == "falsified"
    assert "kappa_1" in str(result)


def test_incomplete_martingale_flag_is_caught() -> None:
    result = _on(martingale_flag_is_complete, _Flagged(0.0, 0.2, is_martingale=False)).check(
        n_examples=3
    )
    assert result.status == "falsified"


def test_unsound_subordinator_flag_is_caught() -> None:
    result = _on(subordinator_flag_is_sound, _Flagged(0.1, 0.2, is_subordinator=True)).check(
        n_examples=3
    )
    assert result.status == "falsified"


def test_wrong_self_similarity_index_is_caught() -> None:
    result = _on(self_similarity_is_sound, _Flagged(0.0, 0.2, self_similarity_index=1.0)).check(
        n_examples=10
    )
    assert result.status == "falsified"


def test_drifted_brownian_motion_must_not_claim_self_similarity() -> None:
    """The bug fixed in the zoo: BM with drift claimed H = 1/2."""
    result = _on(self_similarity_is_sound, _Flagged(0.3, 0.2, self_similarity_index=0.5)).check(
        n_examples=10
    )
    assert result.status == "falsified"


def test_subordinator_law_discards_non_subordinators() -> None:
    result = _on(subordinator_flag_is_sound, VarianceGamma(0.2, 0.3, -0.1)).check(n_examples=3)
    assert result.status == "unsatisfiable"
    assert _on(subordinator_flag_is_sound, GammaProcess(2.0, 3.0)).check(n_examples=2).ok


def test_martingale_hypotheses_are_not_vacuous() -> None:
    """The closure laws see martingales: about half their draws are centred."""
    strategy = PROPERTIES[5].strategies["X"]
    hits = sum(martingale(strategy.example(seed=s)) for s in range(40))
    assert 10 <= hits <= 30


def test_drift_fallacy_shrinks_to_a_textbook_counterexample() -> None:
    """martingale(X) => martingale(X + Drift(c)) is false; the shrinker says why."""

    @law(X=martingales(compositions(max_depth=1)), c=floats(-1.0, 1.0, simplest=0.0))
    def drift_keeps_martingales(X: object, c: float) -> None:
        implies(martingale(X), lambda: martingale(X + Drift(c)))  # type: ignore[operator]

    result = drift_keeps_martingales.check(n_examples=30, seed=0)
    assert result.status == "falsified"
    cex = result.counterexample
    assert cex is not None
    assert cex["c"] != 0.0 and len(repr(cex["c"])) <= 4  # a short decimal such as 0.1
    assert martingale(cex["X"]) and not martingale(cex["X"] + Drift(cex["c"]))
