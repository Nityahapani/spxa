"""Leaf lowering rules: IR exponents against the zoo and the differentiable families."""

from __future__ import annotations

import warnings
from typing import Any, List

import numpy as np
import pytest

from spxa.compile import ir as I
from spxa.compile.diff import derivative
from spxa.compile.evaluate import cumulants_from_ir, evaluate
from spxa.compile.ir import CompileError
from spxa.compile.rules import LEAF_RULES, spec_for, supported_families
from spxa.zoo import levy as L

U_REAL = np.concatenate([-np.logspace(2, -3, 12), [0.0], np.logspace(-3, 2, 12)])
U_CPLX = np.linspace(-6, 6, 13) - 0.3j  # inside every exponential-moment strip below


def _leaf_psi(proc: Any) -> tuple:
    spec = spec_for(proc)
    assert spec is not None
    params = [I.param(k, n) for k, n in enumerate(spec.names)]
    return spec, params, spec.psi(I.U, params)


def _zoo_psi(proc: Any, u: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(proc._char_exponent(u), dtype=complex)


def _close(got: np.ndarray, ref: np.ndarray, rtol: float) -> None:
    got = np.broadcast_to(got, ref.shape)
    scale = np.maximum(1.0, np.abs(ref))
    err = np.abs(got - ref) / scale
    assert np.max(err) <= rtol, f"max scaled error {np.max(err):.3g}"


FAMILY_CASES: List[Any] = [
    L.BrownianMotion(0.1, 0.3),
    L.BrownianMotion(-0.05, 1.7),
    L.Drift(0.2),
    L.GammaProcess(2.0, 3.0),
    L.GammaProcess(0.05, 40.0),
    L.InverseGaussianProcess(1.0, 2.0),
    L.InverseGaussianProcess(0.3, 0.01),
    L.TemperedStable(0.6, 1.0, 2.0),
    L.TemperedStable(0.2, 0.5, 7.0),
    L.NegativeBinomialProcess(2.0, 0.3),
    L.PoissonProcess(2.0, 0.5),
    L.VarianceGamma(0.2, 0.3, -0.1),
    L.VarianceGamma(0.05, 2.0, 0.3),
    L.NIG(10.0, -2.0, 0.5, 0.01),
    L.NIG(3.0, 2.5, 1.5, -0.2),
    L.NormalTemperedStableProcess(0.5, 2.0, 1.0, -0.2, 0.3),
    L.MeixnerProcess(0.3, -0.5, 1.0, 0.0),
    L.MeixnerProcess(2.0, 2.5, 0.2, 0.1),
    L.GeneralizedHyperbolicProcess(-0.5, 10.0, -2.0, 0.5, 0.0),
    L.GeneralizedHyperbolicProcess(1.7, 4.0, 1.0, 0.8, 0.05),
]


def _ids(cases: List[Any]) -> List[str]:
    return [f"{type(p).__name__}-{k}" for k, p in enumerate(cases)]


def _check_family(proc: Any, rtol: float = 2e-14, zoo_rtol: float = 2e-14) -> None:
    from spxa.graph.autodiff.families import RULES

    spec, params, psi = _leaf_psi(proc)
    fam = RULES[type(proc).__name__]
    assert spec.names == fam.params
    assert spec.values == tuple(float(v) for v in fam.getter(proc))
    for uu in (U_REAL, U_CPLX):
        got = evaluate(psi, uu, list(spec.values))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = np.asarray(fam.psi(uu, spec.values), dtype=complex)
        _close(got, ref, rtol)
    _close(evaluate(psi, U_REAL, list(spec.values)), _zoo_psi(proc, U_REAL), zoo_rtol)
    rebuilt = spec.build(spec.values)
    assert type(rebuilt) is type(proc)
    _close(_zoo_psi(rebuilt, U_REAL), _zoo_psi(proc, U_REAL), 0.0)


@pytest.mark.parametrize("proc", FAMILY_CASES, ids=_ids(FAMILY_CASES))
def test_family_exponent(proc: Any) -> None:
    _check_family(proc)


@pytest.mark.parametrize("proc", FAMILY_CASES, ids=_ids(FAMILY_CASES))
def test_family_parameter_gradient_matches_jets(proc: Any) -> None:
    from spxa.graph.autodiff.process import DiffProcess

    spec, params, psi = _leaf_psi(proc)
    _, ref = DiffProcess(proc).char_exponent_and_grad(U_CPLX)  # shape (n_u, n_params)
    assert ref.shape == U_CPLX.shape + (len(params),)
    for k, p in enumerate(params):
        got = evaluate(derivative(psi, p), U_CPLX, list(spec.values))
        _close(np.broadcast_to(got, U_CPLX.shape), ref[:, k], 1e-12)


@pytest.mark.parametrize("proc", FAMILY_CASES, ids=_ids(FAMILY_CASES))
def test_family_cumulants(proc: Any) -> None:
    spec, params, psi = _leaf_psi(proc)
    got = cumulants_from_ir(psi, 4, list(spec.values))
    ref = proc.cumulants(4)
    for n in range(1, 5):
        assert got[n - 1] == pytest.approx(ref[n], rel=1e-12, abs=1e-14)


def test_unknown_class_has_no_rule() -> None:
    class NotAFamily:
        pass

    assert spec_for(NotAFamily()) is None
    assert "GammaProcess" in supported_families()
    assert set(supported_families()) == set(LEAF_RULES)


def test_non_finite_parameters_are_rejected() -> None:
    from spxa.compile.rules import _f

    with pytest.raises(CompileError):
        _f(1.0, float("nan"))
