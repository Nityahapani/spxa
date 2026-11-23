"""The axiom laws hold on the zoo and catch processes that break them."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest

from spxa.core.process import Process
from spxa.laws import Law
from spxa.laws.library import AXIOMS
from spxa.laws.library.axioms import cf_is_bounded, cf_semigroup, even_cumulants_nonnegative
from spxa.laws.strategies import just
from spxa.zoo.levy import BrownianMotion


@pytest.mark.parametrize("law", AXIOMS, ids=lambda law: law.name)
def test_axioms_hold_on_the_zoo(law: Law) -> None:
    result = law.check(n_examples=40, seed=0)
    assert result.ok, str(result)


class _Broken(BrownianMotion):
    """A Brownian motion whose characteristic function has the wrong sign of t."""

    def char_func(self, u: Any, t: float = 1.0) -> np.ndarray:
        return np.exp(+0.5 * self.sigma**2 * np.asarray(u, dtype=float) ** 2 * t) + 0j

    def cumulants(self, order: int) -> dict[int, float]:
        k = super().cumulants(order)
        return {n: (-v if n == 2 else v) for n, v in k.items()}


def _with(law: Law, X: Process, **fixed: Optional[float]) -> Law:
    strategies = dict(law.strategies)
    strategies["X"] = just(X)
    for k, v in fixed.items():
        strategies[k] = just(v)
    return Law(law.name, law.fn, strategies)


def test_axioms_falsify_a_broken_process() -> None:
    bad = _Broken(mu=0.0, sigma=0.5)
    r = _with(cf_is_bounded, bad).check(20)
    assert r.status == "falsified" and "exceeds 1" in r.message
    assert _with(even_cumulants_nonnegative, bad).check(5).status == "falsified"
    # exp(+σ²u²t/2) does satisfy the semigroup law: the axioms are independent
    assert _with(cf_semigroup, bad).check(20).ok
