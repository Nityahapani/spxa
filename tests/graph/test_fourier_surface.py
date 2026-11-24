"""The multi-maturity COS pricer and the closed-form payoff derivatives behind it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from spxa.graph.autodiff import fourier
from spxa.graph.autodiff.fourier import cos_price, cos_price_surface
from spxa.graph.autodiff.jet import Jet, variable_vector, variables
from spxa.graph.autodiff.process import DiffProcess
from spxa.zoo.levy.brownian import BrownianMotion
from spxa.zoo.levy.cgmy import CGMY
from spxa.zoo.levy.drift import Drift
from spxa.zoo.levy.merton import MertonJumpDiffusion
from spxa.zoo.levy.nig import NIG
from spxa.zoo.levy.vg import VarianceGamma

S, R, Q = 100.0, 0.03, 0.01


def _payoff_cases() -> list[tuple[float, float, np.ndarray]]:
    rng = np.random.default_rng(11)
    cases = []
    for _ in range(60):
        a, b = rng.uniform(-4.0, -0.02), rng.uniform(0.02, 4.0)
        inside = S * np.exp(rng.uniform(1.4 * a, 1.4 * b, size=12))  # some clipped at each end
        ties = S * np.exp(np.array([a, b]))  # log(K/S) on the end points, up to rounding
        cases.append((a, b, np.concatenate([inside, ties, [S]])))
    return cases


@pytest.mark.parametrize("n_terms", [1, 2, 64, 1024])
def test_closed_form_payoff_derivatives_match_jets(n_terms: int) -> None:
    k_idx = np.arange(n_terms)
    for a, b, K in _payoff_cases():
        A, B = variables([a, b])
        ref = fourier._cos_payoff(K, S, A, B, k_idx)
        P, Pa, Pb = fourier._cos_payoff_ab(K, S, a, b, k_idx)
        scale = max(np.max(np.abs(ref.grad)), 1e-300)
        np.testing.assert_allclose(P, ref.value, rtol=0, atol=1e-14 * np.max(np.abs(ref.value)))
        np.testing.assert_allclose(Pa, ref.grad[..., 0], rtol=0, atol=2e-14 * scale)
        np.testing.assert_allclose(Pb, ref.grad[..., 1], rtol=0, atol=2e-14 * scale)


def test_closed_form_payoff_derivatives_match_differences() -> None:
    # independent of the jet rules: a Richardson difference of the payoff value
    k_idx = np.arange(48)
    K = np.array([40.0, 90.0, 100.0, 130.0, 900.0])  # 40 and 900 clipped at a and b
    a, b = -0.8, 1.7
    _, Pa, Pb = fourier._cos_payoff_ab(K, S, a, b, k_idx)
    h = 1e-4

    def val(aa: float, bb: float) -> np.ndarray:
        return fourier._cos_payoff_ab(K, S, aa, bb, k_idx)[0]

    for got, e in ((Pa, (1.0, 0.0)), (Pb, (0.0, 1.0))):

        def d(hh: float, e: tuple[float, float] = e) -> np.ndarray:
            up = val(a + hh * e[0], b + hh * e[1])
            dn = val(a - hh * e[0], b - hh * e[1])
            return (up - dn) / (2 * hh)

        ref = (4 * d(h / 2) - d(h)) / 3
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9 * np.max(np.abs(ref)))


PROCS = [
    BrownianMotion(0.02, 0.25),
    NIG(8.0, -2.0, 0.5, 0.01) + Drift(0.0),
    VarianceGamma(0.15, 0.3, -0.2),
    MertonJumpDiffusion(0.0, 0.2, 0.8, -0.1, 0.15),
    CGMY(1.0, 5.0, 8.0, 0.7),
]


@pytest.mark.parametrize("proc", PROCS, ids=lambda p: type(p).__name__)
@pytest.mark.parametrize("track", [True, False])
def test_surface_equals_per_maturity_prices(proc: Any, track: bool) -> None:
    dp = DiffProcess(proc)
    rng = np.random.default_rng(5)
    K = rng.uniform(60.0, 160.0, size=23)
    T = rng.choice([0.02, 0.25, 1.0, 4.0], size=23)  # shuffled maturity groups
    call = rng.random(23) < 0.5
    th = variable_vector(dp.values)
    surf = cos_price_surface(dp, S, K, T, R, Q, call, theta=th, track_interval=track)
    assert isinstance(surf, Jet) and surf.value.shape == (23,) and surf.grad.shape == (23, th.n)
    plain = cos_price_surface(dp, S, K, T, R, Q, call)
    assert isinstance(plain, np.ndarray) and plain.shape == (23,)
    for Ti in np.unique(T):
        i = T == Ti
        one = cos_price(dp, S, K[i], Ti, R, Q, call[i], theta=th, track_interval=track)
        np.testing.assert_allclose(surf.value[i], one.value, rtol=0, atol=1e-12 * S)
        np.testing.assert_allclose(plain[i], one.value, rtol=0, atol=1e-12 * S)
        gs = max(np.max(np.abs(one.grad)), 1.0)
        np.testing.assert_allclose(surf.grad[i], one.grad, rtol=0, atol=1e-11 * gs)


def test_surface_gradient_is_the_derivative_of_the_surface() -> None:
    dp = DiffProcess(VarianceGamma(0.15, 0.3, -0.2) + Drift(0.01))
    K = np.array([70.0, 95.0, 100.0, 110.0, 140.0, 80.0, 120.0])
    T = np.array([0.1, 0.1, 1.0, 1.0, 1.0, 3.0, 3.0])
    call = K >= 100.0
    th = dp.values
    J = cos_price_surface(dp, S, K, T, R, Q, call, theta=variable_vector(th)).grad
    for k in range(th.size):
        h = 1e-4 * max(abs(th[k]), 0.05)
        e = np.eye(th.size)[k]

        def d(hh: float, e: np.ndarray = e) -> np.ndarray:
            up = cos_price_surface(dp, S, K, T, R, Q, call, theta=th + hh * e)
            dn = cos_price_surface(dp, S, K, T, R, Q, call, theta=th - hh * e)
            return (up - dn) / (2 * hh)

        ref = (4 * d(h / 2) - d(h)) / 3
        np.testing.assert_allclose(J[:, k], ref, rtol=0, atol=1e-8 * max(1.0, np.max(np.abs(ref))))


def test_surface_second_order_and_validation() -> None:
    dp = DiffProcess(NIG(8.0, -2.0, 0.5, 0.0))
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([0.5, 0.5, 2.0])
    th2 = variable_vector(dp.values, order=2)
    surf = cos_price_surface(dp, S, K, T, R, Q, theta=th2)
    assert surf.hess is not None
    for Ti in (0.5, 2.0):
        i = T == Ti
        one = cos_price(dp, S, K[i], Ti, R, Q, theta=variable_vector(dp.values, order=2))
        np.testing.assert_allclose(surf.hess[i], one.hess, rtol=0, atol=1e-10 * S)
    scalar_T = cos_price_surface(dp, S, K, 0.5, R, Q)
    np.testing.assert_allclose(scalar_T, cos_price(dp, S, K, 0.5, R, Q), rtol=0, atol=1e-12 * S)
    with pytest.raises(ValueError):
        cos_price_surface(dp, S, K, np.array([0.5, -1.0, 1.0]))
    with pytest.raises(ValueError):
        cos_price_surface(dp, S, K, np.ones((3, 1)))
    with pytest.raises(ValueError):
        cos_price_surface(dp, S, K, np.ones(2))  # does not broadcast with three strikes


def test_ufunc_fast_path_agrees_with_general_path() -> None:
    x = variable_vector(np.array([0.3, -1.2, 2.0]), order=2)
    c = np.array([2.0, -0.5, 1.5])
    pairs = [
        (np.multiply(x, c), np.multiply(x, c, casting="same_kind")),
        (np.add(x, 2), np.add(x, 2, casting="same_kind")),
        (np.exp(x), np.exp(x, casting="same_kind")),
        (np.multiply(x, np.array([1, 2, 3])), np.multiply(x, np.array([1.0, 2.0, 3.0]))),
        (np.multiply(x, 2.0), np.multiply(x, np.float64(2.0), casting="same_kind")),
        (np.divide(True, x), np.divide(np.array(1.0), x, casting="same_kind")),
    ]
    for fast, slow in pairs:
        np.testing.assert_array_equal(fast.value, slow.value)
        np.testing.assert_array_equal(fast.grad, slow.grad)
        np.testing.assert_array_equal(fast.hess, slow.hess)
