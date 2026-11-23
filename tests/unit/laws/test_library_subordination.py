"""Subordination laws hold on the zoo; a common-clock fallacy is falsified."""

from __future__ import annotations

import pytest

from spxa.core.process import Process
from spxa.laws import Law, expect_close, law
from spxa.laws.library.subordination import SUBORDINATION
from spxa.laws.processes import levy_processes, subordinators
from spxa.laws.quantities import char_func, variance


@pytest.mark.parametrize("law_", SUBORDINATION, ids=lambda law_: law_.name)
def test_subordination_holds_on_the_zoo(law_: Law) -> None:
    result = law_.check(n_examples=20, seed=3)
    assert result.ok, str(result)


def test_a_common_clock_does_not_distribute() -> None:
    # (X + Y) @ T runs X and Y on one clock; X @ T + Y @ T on two independent ones
    @law(
        X=levy_processes(["BrownianMotion"]),
        Y=levy_processes(["BrownianMotion"]),
        T=subordinators(["GammaProcess"]),
    )
    def clock_distributes(X: Process, Y: Process, T: Process) -> None:
        expect_close(variance((X + Y) @ T), variance(X @ T + Y @ T), rtol=1e-6)

    r = clock_distributes.check(n_examples=50, seed=0)
    assert r.status == "falsified"
    # Var differs by 2 Var(T) E[X] E[Y], which needs both drifts non-zero
    X, Y = r.counterexample["X"], r.counterexample["Y"]
    assert X.mu != 0 and Y.mu != 0


def test_time_change_is_not_scaling() -> None:
    from spxa.zoo.levy import Drift

    @law(X=levy_processes(["GammaProcess", "BrownianMotion"]))
    def clock_two_is_doubling(X: Process) -> None:
        expect_close(char_func(X @ Drift(2.0), 1.0), char_func(2.0 * X, 1.0), rtol=1e-6)

    assert clock_two_is_doubling.check(n_examples=20).status == "falsified"
