"""calibrate_sabr's exact Jacobian (forward-mode jets, implicit ATM alpha)."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.graph.autodiff.jet import variable_vector, variables
from spxa.rates.sabr import (
    SABR,
    alpha_from_atm,
    calibrate_sabr,
    sabr_lognormal_vol,
    sabr_normal_vol,
)

F, T, BETA = 0.03, 2.0, 0.5
K = np.linspace(0.01, 0.07, 13)


def _rich(f, x, k, h=1e-4):  # type: ignore[no-untyped-def]
    def d(hh: float) -> np.ndarray:
        e = np.zeros_like(x)
        e[k] = hh
        return (np.asarray(f(x + e)) - np.asarray(f(x - e))) / (2 * hh)

    return (4 * d(h / 2) - d(h)) / 3


@pytest.mark.parametrize("method", ["hagan", "obloj"])
def test_lognormal_vol_parameter_gradient(method: str) -> None:
    th = np.array([0.02, -0.3, 0.45])
    a, r, n = variables(list(th))
    out = sabr_lognormal_vol(F, K, T, a, BETA, r, n, 0.001, method)
    f = lambda t: sabr_lognormal_vol(F, K, T, t[0], BETA, t[1], t[2], 0.001, method)  # noqa: E731
    np.testing.assert_allclose(out.value, f(th), rtol=1e-15)
    for k in range(3):
        np.testing.assert_allclose(out.grad[:, k], _rich(f, th, k), rtol=1e-8, atol=1e-12)


def test_normal_vol_gradient_including_the_atm_series_branch() -> None:
    th = np.array([0.02, -0.3, 0.45])
    Kq = np.array([0.02, F, F * (1 + 1e-7), 0.05])  # z ~ 0 hits the series
    a, r, n = variables(list(th))
    out = sabr_normal_vol(F, Kq, T, a, BETA, r, n)
    f = lambda t: sabr_normal_vol(F, Kq, T, t[0], BETA, t[1], t[2])  # noqa: E731
    for k in range(3):
        np.testing.assert_allclose(out.grad[:, k], _rich(f, th, k), rtol=1e-8, atol=1e-14)


def test_implicit_atm_alpha_derivative() -> None:
    r, n = variables([-0.3, 0.45])
    a = alpha_from_atm(0.2, F, T, BETA, r, n)
    assert float(a.value) == pytest.approx(alpha_from_atm(0.2, F, T, BETA, -0.3, 0.45), rel=1e-15)
    f = lambda t: alpha_from_atm(0.2, F, T, BETA, t[0], t[1])  # noqa: E731
    th = np.array([-0.3, 0.45])
    for k in range(2):
        assert a.grad[k] == pytest.approx(_rich(f, th, k), rel=1e-8)
    # the pinned alpha keeps the ATM vol fixed to first order
    vol = sabr_lognormal_vol(F, F, T, a, BETA, r, n)
    np.testing.assert_allclose(vol.grad, 0.0, atol=1e-12)
    with pytest.raises(NotImplementedError):
        r2, n2 = variables([-0.3, 0.45], order=2)
        alpha_from_atm(0.2, F, T, BETA, r2, n2)


@pytest.mark.parametrize(
    "vol_type, method, pin",
    [
        ("lognormal", "hagan", False),
        ("lognormal", "obloj", False),
        ("normal", "hagan", False),
        ("lognormal", "hagan", True),
    ],
)
def test_calibration_recovers_parameters_with_exact_jacobian(
    vol_type: str, method: str, pin: bool
) -> None:
    true = SABR(0.02, BETA, -0.3, 0.45)
    v = true.lognormal_vol(F, K, T, method) if vol_type == "lognormal" else true.normal_vol(F, K, T)
    atm = float(true.lognormal_vol(F, F, T)) if pin else None
    fit = calibrate_sabr(F, K, v, T, BETA, vol_type=vol_type, method=method, atm_vol=atm)
    assert fit.rmse < 1e-14
    np.testing.assert_allclose(
        [fit.model.alpha, fit.model.rho, fit.model.nu], [0.02, -0.3, 0.45], rtol=1e-10
    )
    with pytest.raises(ValueError):
        calibrate_sabr(F, K, v, T, BETA, jac="cs")


def test_jet_through_vector_of_unconstrained_coordinates() -> None:
    x = variable_vector([np.arctanh(-0.3), np.log(0.45), np.log(0.02)])
    out = sabr_lognormal_vol(F, K, T, np.exp(x[2]), BETA, np.tanh(x[0]), np.exp(x[1]))
    assert out.grad.shape == (K.size, 3)
