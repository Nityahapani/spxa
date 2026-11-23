"""The algebra laws hold on the zoo; plausible but wrong laws are falsified."""

from __future__ import annotations

import pytest

from spxa.core.process import Process
from spxa.laws import Law, expect_close, law
from spxa.laws.library.algebra import ALGEBRA
from spxa.laws.processes import levy_processes
from spxa.laws.quantities import char_func, cumulants, variance


@pytest.mark.parametrize("law_", ALGEBRA, ids=lambda law_: law_.name)
def test_algebra_holds_on_the_zoo(law_: Law) -> None:
    result = law_.check(n_examples=25, seed=7)
    assert result.ok, str(result)


def test_variance_of_a_difference_is_not_the_difference() -> None:
    @law(X=levy_processes(), Y=levy_processes())
    def variance_subtracts(X: Process, Y: Process) -> None:
        expect_close(variance(X - Y), variance(X) - variance(Y), rtol=1e-6, atol=1e-9)

    r = variance_subtracts.check(n_examples=50, seed=0)
    assert r.status == "falsified"


def test_scaling_by_c_does_not_scale_the_cf_value() -> None:
    @law(X=levy_processes(["BrownianMotion", "GammaProcess"]))
    def cf_of_double(X: Process) -> None:
        expect_close(char_func(2.0 * X, 1.0), 2.0 * char_func(X, 1.0), rtol=1e-6)

    r = cf_of_double.check(n_examples=20, seed=0)
    assert r.status == "falsified"
    # shrinks to the simplest Brownian motion
    assert repr(r.counterexample["X"]).startswith("BrownianMotion(")


def test_kurtosis_is_not_additive_for_scaled_sums() -> None:
    @law(X=levy_processes(["GammaProcess"]))
    def fourth_cumulant_doubles(X: Process) -> None:
        expect_close(cumulants(2.0 * X)[3], 2.0 * cumulants(X)[3], rtol=1e-6)

    assert fourth_cumulant_doubles.check(n_examples=10).status == "falsified"
