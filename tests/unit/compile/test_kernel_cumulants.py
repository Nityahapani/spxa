"""Generated cumulants agree with Taylor mode, and fall back to it where they must."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest

from spxa.compile.evaluate import cumulants_from_ir, cumulants_from_ir_many
from spxa.compile.kernel import _TAYLOR_ONLY, ExponentKernel
from spxa.compile.lower import lower
from spxa.zoo import levy as L
from spxa.zoo.levy import compound_poisson as cpm


def _taylor(k: ExponentKernel, order: int, theta: np.ndarray) -> Any:
    params = [theta[..., j] for j in range(k.n_free)]
    roots = [k.psi_expr] + k.grad_exprs
    series = cumulants_from_ir_many(roots, order, params)
    lead = theta.shape[:-1]
    kap = np.broadcast_to(series[0], lead + (order,))
    jac = np.stack([np.broadcast_to(s, lead + (order,)) for s in series[1:]], axis=-1)
    return kap, jac


CASES: List[Any] = [
    L.BrownianMotion(0.05, 0.2),
    L.GammaProcess(2.0, 3.0),
    L.InverseGaussianProcess(1.0, 2.0),
    L.TemperedStable(0.6, 1.0, 2.0),
    L.VarianceGamma(0.2, 0.3, -0.1),
    L.NIG(10.0, -2.0, 0.5),
    L.MeixnerProcess(0.3, -0.5, 1.0, 0.0),
    L.GeneralizedHyperbolicProcess(-0.5, 10.0, -2.0, 0.5, 0.0),
    L.CGMY(1.0, 5.0, 6.0, 0.5),
    L.CGMY(1.0, 5.0, 6.0, 1.5),
    L.KouJumpDiffusion(0.05, 0.2, 2.0, 0.4, 8.0, 5.0),
    L.MertonJumpDiffusion(0.05, 0.2, 1.0, -0.05, 0.1),
    (L.BrownianMotion(0.05, 0.2) + 0.5 * L.KouJumpDiffusion(0.05, 0.2, 2.0, 0.4, 8.0, 5.0))
    @ L.GammaProcess(4.0, 4.0)
    + L.CGMY(1.0, 5.0, 6.0, 0.5),
    (L.NIG(10.0, -2.0, 0.5) @ L.GammaProcess(5.0, 5.0)) @ L.InverseGaussianProcess(1.0, 2.0),
]


@pytest.mark.parametrize("proc", CASES, ids=lambda p: type(p).__name__)
def test_generated_cumulants_equal_taylor_mode(proc: Any) -> None:
    k = ExponentKernel(lower(proc))
    assert k._symbolic_cumulants_ok()
    rng = np.random.default_rng(3)
    theta = k.theta0 * (1.0 + 0.05 * rng.uniform(-1.0, 1.0, (5, k.n_free)))
    kap, jac = k.cumulants_and_jacobian(4, theta)
    kref, jref = _taylor(k, 4, theta)
    np.testing.assert_allclose(kap, kref, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(jac, jref, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(k.cumulants(4, theta), kref, rtol=1e-12, atol=1e-14)
    assert "cum_jac" in k._families and "cum" in k._families
    # single parameter vector and lower orders
    k1, j1 = k.cumulants_and_jacobian(2)
    np.testing.assert_allclose(k1, _taylor(k, 2, k.theta0)[0], rtol=1e-12, atol=1e-14)
    assert k1.shape == (2,) and j1.shape == (2, k.n_free)


def test_sinc_exponents_use_taylor_mode() -> None:
    # (e^{ibu} - e^{iau}) / (iu(b - a)) is written with sinc; its generated
    # derivatives at u = 0 are not exact, so Taylor mode must be used
    proc = cpm.CompoundPoisson(2.0, cpm.UniformJump(-0.2, 0.5))
    k = ExponentKernel(lower(proc))
    assert not k._symbolic_cumulants_ok()
    kap, jac = k.cumulants_and_jacobian(4)
    kref, jref = _taylor(k, 4, k.theta0)
    np.testing.assert_array_equal(kap, kref)
    np.testing.assert_array_equal(jac, jref)
    assert "cum_jac" not in k._families
    # uniform jumps: kappa_n = rate * E[J^n]
    lo, hi = -0.2, 0.5
    moments = [(hi ** (n + 1) - lo ** (n + 1)) / ((n + 1) * (hi - lo)) for n in range(1, 5)]
    np.testing.assert_allclose(kap, 2.0 * np.array(moments), rtol=1e-12)


def test_high_orders_use_taylor_mode() -> None:
    k = ExponentKernel(lower(L.NIG(10.0, -2.0, 0.5)))
    got = k.cumulants(6)
    np.testing.assert_array_equal(got, cumulants_from_ir(k.psi_expr, 6, list(k.theta0)))
    assert "cum" not in k._families


def test_non_finite_generated_values_fall_back() -> None:
    k = ExponentKernel(lower(L.GammaProcess(2.0, 3.0)))
    k.cumulants_and_jacobian(4)  # builds the generated family
    real = k._families["cum_jac"]

    class NanFamily:
        calls = 0

        def __call__(self, u: Any, params: Any) -> Any:
            NanFamily.calls += 1
            out = list(real(u, params))
            out[0] = np.complex128(np.nan)
            return tuple(out)

    k._families["cum_jac"] = NanFamily()  # type: ignore[assignment]
    kap, jac = k.cumulants_and_jacobian(4)
    assert NanFamily.calls == 1
    kref, jref = _taylor(k, 4, k.theta0)
    np.testing.assert_array_equal(kap, kref)
    np.testing.assert_array_equal(jac, jref)
    # Gamma(a, b): kappa_n = a (n - 1)! / b^n
    np.testing.assert_allclose(kap, [2 / 3, 2 / 9, 4 / 27, 12 / 81], rtol=1e-14)


def test_nan_guard_inside_the_fast_path() -> None:
    # a parameter set outside the domain gives nan; the call must still agree with Taylor
    k = ExponentKernel(lower(L.GammaProcess(2.0, 3.0)))
    theta = np.array([[2.0, 3.0], [2.0, 0.0]])
    kap = k.cumulants(4, theta)
    with np.errstate(all="ignore"):
        ref = cumulants_from_ir(k.psi_expr, 4, [theta[:, 0], theta[:, 1]])
    np.testing.assert_array_equal(np.isfinite(kap), np.isfinite(ref))
    np.testing.assert_allclose(kap[0], ref[0], rtol=1e-14)


def test_taylor_only_set_covers_the_taylor_special_series() -> None:
    assert {"sinc", "sinc_d1", "kve_dnu", "call"} <= _TAYLOR_ONLY
