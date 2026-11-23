"""
The α-stable Lévy triplet reproduces the closed-form characteristic exponent.

Regression for laws counterexample 3. Two bugs combined here. The Lévy
measure constant was half the value the exponent needs. The drift was set to
the location μ, ignoring the shift that the 1{|x|≤1} truncation brings in.
``cumulants`` also integrated divergent moments by quadrature, so it returned
numbers like 1.6e13 instead of inf/NaN.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import integrate  # type: ignore[import-untyped]

from spxa.zoo.levy import AlphaStable

PARAMS = [
    (0.4, -1.0),
    (0.7, 0.5),
    (1.0, 0.0),
    (1.0, 0.4),
    (1.5, 0.0),
    (1.5, 0.3),
    (1.8, -0.6),
    (1.95, 1.0),
]
U = np.array([-3.0, -1.1, -0.25, 0.3, 1.0, 2.7])


@pytest.mark.parametrize(("alpha", "beta"), PARAMS)
def test_triplet_exponent_matches_closed_form(alpha: float, beta: float) -> None:
    X = AlphaStable(alpha, beta, 0.6, 0.15)
    # LevyTriplet.char_exp returns the characteristic function exp(tψ)
    for t in (1.0, 0.37):
        np.testing.assert_allclose(
            X.triplet.char_exp(U, t), X.char_func(U, t), rtol=1e-12, atol=1e-14
        )


@pytest.mark.parametrize(("alpha", "beta"), PARAMS)
def test_triplet_mean_is_location_when_it_exists(alpha: float, beta: float) -> None:
    triplet = AlphaStable(alpha, beta, 0.6, 0.15).triplet
    if alpha > 1:
        assert triplet.cumulant(1) == pytest.approx(0.15, rel=1e-12)
    assert triplet.cumulant(2) == math.inf


def _jump_exponent_by_quadrature(X: AlphaStable, u: float) -> complex:
    nu = X.triplet.nu

    def integrand(x: float, part: int) -> float:
        c = np.exp(1j * u * x) - 1 - (1j * u * x if abs(x) <= 1 else 0)
        val = c.real if part == 0 else c.imag
        return float(val * nu.density(np.array([x]))[0])

    total = 0j
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        for part, unit in ((0, 1.0), (1, 1j)):
            for lo, hi in ((-np.inf, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, np.inf)):
                weight = "cos" if part == 0 else "sin"
                if np.isinf(lo) or np.isinf(hi):
                    # oscillatory Fourier tails: QAWF handles x^{-1-α} decay
                    sign = -1.0 if np.isinf(lo) else 1.0
                    dens = float(nu.density(np.array([sign]))[0])
                    tail = integrate.quad(
                        lambda y: dens * y ** (-1 - X.alpha), 1.0, np.inf, weight=weight, wvar=u
                    )[0]
                    if part == 0:
                        val = (
                            tail
                            - integrate.quad(lambda y: dens * y ** (-1 - X.alpha), 1.0, np.inf)[0]
                        )
                    else:
                        val = sign * tail
                else:
                    val = integrate.quad(integrand, lo, hi, args=(part,), limit=400)[0]
                total += unit * val
    return complex(total)


@pytest.mark.parametrize(("alpha", "beta"), [(0.7, 0.5), (1.0, 0.4), (1.5, 0.3), (1.8, -0.6)])
def test_jump_exponent_matches_its_density(alpha: float, beta: float) -> None:
    X = AlphaStable(alpha, beta, 0.6, 0.15)
    u = 1.3
    exact = X.triplet.nu.jump_exponent(u)
    assert _jump_exponent_by_quadrature(X, u) == pytest.approx(exact, rel=1e-6, abs=1e-8)


def test_measure_constant_matches_tail_asymptotics() -> None:
    """P(X_1 > x) ~ ν((x, ∞)) = σ^α (1+β) C_α x^{-α}, C_α = Γ(α) sin(πα/2)/π."""
    alpha, beta, sigma = 1.5, 0.3, 0.6
    nu = AlphaStable(alpha, beta, sigma).triplet.nu
    c_alpha = math.gamma(alpha) * math.sin(math.pi * alpha / 2) / math.pi
    assert nu.tail(10.0) == pytest.approx(sigma**alpha * (1 + beta) * c_alpha * 10.0**-alpha)


def test_symmetric_stable_has_no_drift_shift() -> None:
    for alpha in (0.5, 1.0, 1.5):
        assert AlphaStable(alpha, 0.0, 1.0, 0.25).triplet.b == pytest.approx(0.25)


def test_alpha_one_drift_uses_euler_gamma() -> None:
    X = AlphaStable(1.0, 0.5, 2.0, 0.0)
    c_diff = 2 * 0.5 * 2.0 / math.pi  # c₊ − c₋ = 2βσ/π
    assert X.triplet.b == pytest.approx(-c_diff * (1 - np.euler_gamma), rel=1e-14)


@pytest.mark.parametrize(("alpha", "beta"), PARAMS)
def test_cumulants_are_mean_then_divergent(alpha: float, beta: float) -> None:
    k = AlphaStable(alpha, beta, 0.5, 0.2).cumulants(4)
    if alpha > 1:
        assert k[1] == 0.2
    else:
        assert math.isnan(k[1])
    assert k[2] == math.inf and k[4] == math.inf
    assert math.isnan(k[3])


def test_gaussian_limit_cumulants() -> None:
    assert AlphaStable(2.0, 0.0, 0.5, 0.1).cumulants(4) == {1: 0.1, 2: 0.5, 3: 0.0, 4: 0.0}


def test_martingale_flag_follows_mean() -> None:
    assert AlphaStable(1.5, 0.3, 0.5, 0.0).properties.is_martingale
    assert not AlphaStable(1.5, 0.3, 0.5, 0.1).properties.is_martingale
    assert not AlphaStable(0.8, 0.0, 0.5, 0.0).properties.is_martingale  # no mean at all


def test_measure_rejects_complex_frequencies() -> None:
    with pytest.raises(ValueError, match="exponential moments"):
        AlphaStable(1.5).triplet.nu.jump_exponent(1 + 0.5j)
