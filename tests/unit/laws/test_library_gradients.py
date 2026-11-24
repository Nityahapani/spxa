"""Gradient laws: they hold on the zoo and catch wrong derivatives."""

from __future__ import annotations

import pytest

from spxa.graph.autodiff.process import DiffProcess
from spxa.laws import Law
from spxa.laws.library.gradients import (
    GRADIENTS,
    cumulant_jacobian_matches_differences,
    hessian_is_symmetric_and_consistent,
    psi_gradient_matches_differences,
    scaling_chain_rule,
)
from spxa.laws.strategies import just
from spxa.zoo.levy import CGMY, PoissonProcess, VarianceGamma


@pytest.mark.parametrize("law_", GRADIENTS, ids=lambda law_: law_.name)
@pytest.mark.parametrize("seed", [1, 2])
def test_holds_on_the_zoo(law_: Law, seed: int) -> None:
    result = law_.check(n_examples=60, seed=seed)
    assert result.ok, str(result)
    assert result.n_discarded < result.n_examples


def test_holds_near_a_pole() -> None:
    # Gamma(-Y) has a pole at Y = 2; steps of 1e-3 left a 1.2e-6 truncation error here
    Z = CGMY(C=1.0, G=1.0, M=2.0, Y=1.9) @ PoissonProcess(rate=1.0, jump_size=1.2)
    pinned = Law(
        "pinned",
        psi_gradient_matches_differences.fn,
        {**psi_gradient_matches_differences.strategies, "X": just(Z), "u": just(3.0)},
    )
    assert pinned.check(n_examples=1).status == "verified"


def _scaled(method: str, factor: float):  # type: ignore[no-untyped-def]
    orig = getattr(DiffProcess, method)

    def wrong(self, *a, **k):  # type: ignore[no-untyped-def]
        out = orig(self, *a, **k)
        return (*out[:-1], out[-1] * factor)

    return wrong


@pytest.mark.parametrize(
    "law_, method",
    [
        (psi_gradient_matches_differences, "char_exponent_and_grad"),
        (cumulant_jacobian_matches_differences, "cumulants_and_jacobian"),
        (hessian_is_symmetric_and_consistent, "char_exponent_hessian"),
    ],
    ids=["psi", "cumulants", "hessian"],
)
def test_detects_a_derivative_off_by_1e_4(
    law_: Law, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DiffProcess, method, _scaled(method, 1.0001))
    pinned = Law("pinned", law_.fn, {**law_.strategies, "X": just(VarianceGamma(0.2, 0.3, -0.1))})
    assert pinned.check(n_examples=3).status == "falsified"


def test_scaling_rule_detects_a_wrong_scale_derivative(monkeypatch: pytest.MonkeyPatch) -> None:
    orig = DiffProcess.char_exponent_and_grad

    def wrong(self, u, theta=None):  # type: ignore[no-untyped-def]
        psi, g = orig(self, u, theta)
        if self.names[-1] == "scale":
            g = g.copy()
            g[..., -1] *= 1 + 1e-9
        return psi, g

    monkeypatch.setattr(DiffProcess, "char_exponent_and_grad", wrong)
    pinned = Law(
        "pinned",
        scaling_chain_rule.fn,
        {**scaling_chain_rule.strategies, "X": just(VarianceGamma(0.2, 0.3, -0.1))},
    )
    assert pinned.check(n_examples=3).status == "falsified"
