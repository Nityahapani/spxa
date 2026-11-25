"""Lowering whole process expressions to one IR exponent."""

from __future__ import annotations

import warnings
from typing import Any, List

import numpy as np
import pytest

from spxa.compile import ir as I
from spxa.compile.evaluate import cumulants_from_ir, evaluate
from spxa.compile.ir import CompileError
from spxa.compile.lower import LOpaque, LScale, LSub, LSum, is_analytic, lower
from spxa.zoo import levy as L
from spxa.zoo.levy import compound_poisson as cp

U_REAL = np.linspace(-30.0, 30.0, 61)
U_CPLX = np.linspace(-5.0, 5.0, 11) - 0.2j


def _zoo(proc: Any, u: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(proc._char_exponent(u), dtype=complex)


def _close(got: np.ndarray, ref: np.ndarray, rtol: float = 5e-14) -> None:
    got = np.broadcast_to(got, ref.shape)
    err = np.abs(got - ref) / np.maximum(1.0, np.abs(ref))
    assert np.max(err) <= rtol, f"max scaled error {np.max(err):.3g}"


def _compositions() -> List[Any]:
    bm = L.BrownianMotion(0.05, 0.2)
    vg = L.VarianceGamma(0.2, 0.3, -0.1)
    return [
        vg,
        vg + 0.5 * L.GammaProcess(2.0, 3.0),
        bm @ L.GammaProcess(5.0, 5.0),
        (bm + cp.CompoundPoisson(1.0, cp.ExponentialJump(5.0)))
        @ L.InverseGaussianProcess(1.0, 2.0),
        L.MertonJumpDiffusion(0.05, 0.2, 1.0, -0.1, 0.15) @ L.GammaProcess(10.0, 10.0),
        L.NIG(10.0, -2.0, 0.5) @ L.GammaProcess(5.0, 5.0) + 2.0 * L.Drift(0.01),
        (bm + vg) @ (L.GammaProcess(4.0, 4.0) @ L.InverseGaussianProcess(1.0, 3.0)),
        0.3 * (0.7 * (L.KouJumpDiffusion(0.0, 0.1, 2.0, 0.4, 10.0, 5.0) + bm)),
        L.CGMY(1.0, 5.0, 10.0, 0.5) + L.MeixnerProcess(0.3, -0.5, 1.0, 0.0) + vg + vg,
    ]


COMPOSITIONS = _compositions()


@pytest.mark.parametrize("k", range(len(COMPOSITIONS)))
def test_exponent_matches_the_zoo(k: int) -> None:
    proc = COMPOSITIONS[k]
    low = lower(proc)
    for uu in (U_REAL, U_CPLX):
        _close(evaluate(low.psi, uu, list(low.values)), _zoo(proc, uu))


@pytest.mark.parametrize("k", range(len(COMPOSITIONS)))
def test_cumulants_match_the_zoo(k: int) -> None:
    proc = COMPOSITIONS[k]
    low = lower(proc)
    got = cumulants_from_ir(low.psi, 4, list(low.values))
    ref = proc.cumulants(4)
    for n in range(1, 5):
        assert got[n - 1] == pytest.approx(ref[n], rel=1e-11, abs=1e-14)


@pytest.mark.parametrize("k", [0, 1, 2, 4, 5, 6, 7, 8])
def test_parameter_layout_matches_diff_process(k: int) -> None:
    from spxa.graph.autodiff.process import DiffProcess

    proc = COMPOSITIONS[k]
    low = lower(proc)
    dp = DiffProcess(proc)
    assert low.names == tuple(dp.names)
    np.testing.assert_array_equal(low.values, np.asarray(dp.values, dtype=float))


def test_names_for_repeated_families_and_scales() -> None:
    proc = 0.5 * L.GammaProcess(1.0, 2.0) + 0.25 * L.GammaProcess(3.0, 4.0)
    low = lower(proc)
    assert low.names == (
        "GammaProcess.a",
        "GammaProcess.b",
        "scale",
        "GammaProcess[1].a",
        "GammaProcess[1].b",
        "scale[1]",
    )
    np.testing.assert_array_equal(low.values, [1.0, 2.0, 0.5, 3.0, 4.0, 0.25])


def test_tree_structure_and_rebuild() -> None:
    proc = COMPOSITIONS[6]
    low = lower(proc)
    assert isinstance(low.tree, LSub) and isinstance(low.tree.x, LSum)
    assert isinstance(low.tree.clock, LSub)
    theta = low.values * 1.1
    rebuilt = low.build(theta)
    _close(_zoo(rebuilt, U_REAL), evaluate(low.psi, U_REAL, list(theta)))
    s = low.story()
    assert "@" in s and "BrownianMotion(mu=0.05" in s
    assert isinstance(lower(2.0 * L.Drift(0.1)).tree, LScale)
    assert low.build() is not proc


def test_shared_subexpressions_across_summands() -> None:
    vg = L.VarianceGamma(0.2, 0.3, -0.1)
    low_one = lower(vg)
    low_two = lower(vg + vg)
    # the two copies have separate parameters, so the DAG doubles minus shared constants
    assert I.count_nodes(low_two.psi) < 2 * I.count_nodes(low_one.psi) + 2


def test_opaque_leaves() -> None:
    class MyGamma(L.GammaProcess):
        pass

    g = MyGamma(2.0, 3.0)
    proc = L.BrownianMotion(0.0, 0.3) @ g + L.Drift(0.1)
    low = lower(proc)
    assert low.opaque == (g,)
    assert low.names == ("BrownianMotion.mu", "BrownianMotion.sigma", "Drift.mu")
    assert any(isinstance(n, LOpaque) for n in [low.tree.a.clock])  # type: ignore[attr-defined]
    _close(evaluate(low.psi, U_REAL, list(low.values)), _zoo(proc, U_REAL))
    got = cumulants_from_ir(low.psi, 3, list(low.values))
    ref = proc.cumulants(3)
    assert got[1] == pytest.approx(ref[2], rel=1e-12)
    assert low.build(low.values * 2.0)._right._char_exponent is not None
    with pytest.raises(CompileError, match="no lowering rule"):
        lower(proc, allow_opaque=False)


def test_non_levy_processes_are_rejected() -> None:
    from spxa.zoo.beyond.heston import Heston

    with pytest.raises(CompileError, match="not an exact"):
        lower(Heston(sigma_v=0.3))
    with pytest.raises(CompileError, match="Process"):
        lower(L.GammaProcess(1.0, 1.0)._char_exponent)  # type: ignore[arg-type]


def test_analyticity() -> None:
    assert lower(COMPOSITIONS[8]).analytic  # Meixner's sign guard only feeds a comparison
    stable = lower(L.AlphaStable(1.5, 0.3, 1.0, 0.0) + L.BrownianMotion(0.0, 0.1))
    assert not stable.analytic
    assert is_analytic([I.U * 2.0])
    assert not is_analytic(I.abs_(I.U))
    assert not is_analytic(I.abs_(I.U) + 1.0)
    assert is_analytic(I.where(I.gt(I.abs_(I.U), 1.0), I.U, -I.U))


def test_real_exponent_is_promoted_to_complex() -> None:
    low = lower(L.BrownianMotion(0.0, 0.3))
    assert low.psi.kind == I.COMPLEX
