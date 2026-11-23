"""Monte Carlo gradients (spxa.graph.autodiff.montecarlo): unbiasedness and calibrated errors."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.special as sps
from scipy.stats import invgauss, norm

from spxa.graph.autodiff.exactness import GradientExactness
from spxa.graph.autodiff.fourier import cos_price, price_and_gradient
from spxa.graph.autodiff.jet import variables
from spxa.graph.autodiff.montecarlo import (
    call,
    digital_call,
    gamma_shape_derivative,
    inverse_gaussian_reparam,
    likelihood_ratio_gradient,
    malliavin_greeks,
    pathwise_gradient,
    put,
)
from spxa.graph.autodiff.process import DiffProcess, UnsupportedProcessError
from spxa.zoo.levy.brownian import BrownianMotion
from spxa.zoo.levy.drift import Drift
from spxa.zoo.levy.gamma import GammaProcess
from spxa.zoo.levy.merton import MertonJumpDiffusion
from spxa.zoo.levy.nig import NIG
from spxa.zoo.levy.vg import VarianceGamma

S, K, T, R, Q = 100.0, 105.0, 0.5, 0.03, 0.01
SEEDS = range(6)
N = 40_000


def _pooled_z(est, dp, payoff, ref, **kw):  # type: ignore[no-untyped-def]
    """z-score of the mean over independent seeds (N(0,1) if unbiased with right SEs)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = np.array(
            [
                est(dp, payoff, S, T, R, Q, n_paths=N, rng=np.random.default_rng(s), **kw).z_scores(
                    ref
                )
                for s in SEEDS
            ]
        )
    return z.mean(axis=0) * np.sqrt(len(SEEDS)), z


def _cos_jac(dp: DiffProcess, k: float = K) -> np.ndarray:
    return price_and_gradient(dp, S, np.array([k]), T, method="cos", rate=R, dividend=Q).jacobian[0]


@pytest.mark.parametrize("a", [0.02, 0.3, 1.0, 3.3, 150.0, 1000.0])
def test_gamma_implicit_reparameterisation_against_mpmath(a: float) -> None:
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    U = np.array([1e-6, 0.01, 0.3, 0.5, 0.9, 0.999, 1 - 1e-7, 1 - 1e-12])
    G = sps.gammaincinv(a, U)
    d = gamma_shape_derivative(a, G)
    for g, dg in zip(G, d):
        gm = mp.mpf(g)
        dP = mp.diff(lambda aa: mp.gammainc(aa, 0, gm, regularized=True), mp.mpf(a))
        pdf = mp.exp((a - 1) * mp.log(gm) - gm - mp.loggamma(a))
        assert dg == pytest.approx(float(-dP / pdf), rel=5e-11)


def test_inverse_gaussian_reparameterisation_at_fixed_uniform() -> None:
    U = np.array([1e-4, 0.1, 0.5, 0.9, 0.9999])
    m0, s0 = 0.4, 0.9
    m, s = variables([m0, s0])

    def ppf(mm: float, ss: float) -> np.ndarray:
        return invgauss.ppf(U, mm / ss, scale=ss)

    x = ppf(m0, s0)
    jx = inverse_gaussian_reparam(m, s, x)
    h = 1e-6
    fd_m = (ppf(m0 + h, s0) - ppf(m0 - h, s0)) / (2 * h)
    fd_s = (ppf(m0, s0 + h) - ppf(m0, s0 - h)) / (2 * h)
    np.testing.assert_allclose(jx.grad[:, 0], fd_m, rtol=1e-6)
    np.testing.assert_allclose(jx.grad[:, 1], fd_s, rtol=1e-6)


PATHWISE = {
    "BM": BrownianMotion(0.0, 0.25),
    "VG+Drift": VarianceGamma(0.2, 0.3, -0.15) + Drift(0.02),
    "NIG": NIG(12.0, -3.0, 0.25, 0.0),
    "Merton": MertonJumpDiffusion(0.0, 0.15, 1.5, -0.08, 0.12),
    "BM@Gamma": BrownianMotion(-0.1, 0.2) @ GammaProcess(4.0, 4.0),
    "0.7*VG": 0.7 * VarianceGamma(0.2, 0.3, -0.15),
}


@pytest.mark.parametrize("key", list(PATHWISE))
def test_pathwise_gradient_is_unbiased(key: str) -> None:
    dp = DiffProcess(PATHWISE[key])
    ref = _cos_jac(dp)
    pooled, z = _pooled_z(pathwise_gradient, dp, call(K), ref)
    assert np.all(np.abs(pooled) < 4.0), pooled
    live = np.abs(ref) > 1e-10
    assert np.all(np.abs(z[:, live]) < 5.0)
    # standard errors are calibrated: per-seed z's have unit spread
    assert np.all(np.std(z[:, live], axis=0) < 2.5)


def test_pathwise_put_and_reproducibility() -> None:
    dp = DiffProcess(VarianceGamma(0.2, 0.3, -0.15))
    ref = price_and_gradient(
        dp, S, np.array([K]), T, method="cos", rate=R, dividend=Q, call=False
    ).jacobian[0]
    pooled, _ = _pooled_z(pathwise_gradient, dp, put(K), ref)
    assert np.all(np.abs(pooled) < 4.0)
    a = pathwise_gradient(dp, put(K), S, T, R, Q, n_paths=1000, rng=np.random.default_rng(9))
    b = pathwise_gradient(dp, put(K), S, T, R, Q, n_paths=1000, rng=np.random.default_rng(9))
    np.testing.assert_array_equal(a.grad, b.grad)
    assert a.exactness is GradientExactness.STOCHASTIC
    assert "STOCHASTIC" in a.__story__()


@pytest.mark.parametrize("key", ["VG", "NIG", "BM"])
def test_likelihood_ratio_digital_is_unbiased(key: str) -> None:
    proc = {
        "VG": VarianceGamma(0.2, 0.3, -0.15),
        "NIG": NIG(12.0, -3.0, 0.25, 0.0),
        "BM": BrownianMotion(0.0, 0.25),
    }[key]
    dp = DiffProcess(proc)
    h = 0.01  # digital = -dC/dK, so its gradient is -d/dK of the call gradient
    ref = -(_cos_jac(dp, K + h) - _cos_jac(dp, K - h)) / (2 * h)
    pooled, _ = _pooled_z(likelihood_ratio_gradient, dp, digital_call(K), ref)
    assert np.all(np.abs(pooled) < 4.0), pooled
    pooled_call, _ = _pooled_z(likelihood_ratio_gradient, dp, call(K), _cos_jac(dp))
    assert np.all(np.abs(pooled_call) < 4.0), pooled_call


def test_malliavin_matches_black_scholes_call_and_digital() -> None:
    sig = 0.25
    d1 = (np.log(S / K) + (R - Q + sig**2 / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    dp = DiffProcess(BrownianMotion(0.0, sig))
    call_ref = np.array(
        [np.exp(-Q * T) * norm.cdf(d1), S * np.exp(-Q * T) * norm.pdf(d1) * np.sqrt(T)]
    )
    dig_ref = np.array(
        [
            np.exp(-R * T) * norm.pdf(d2) / (S * sig * np.sqrt(T)),
            -np.exp(-R * T) * norm.pdf(d2) * d1 / sig,
        ]
    )
    for payoff, ref in [(call(K), call_ref), (digital_call(K), dig_ref)]:
        pooled, _ = _pooled_z(malliavin_greeks, dp, payoff, ref)
        assert np.all(np.abs(pooled) < 4.0), (payoff.label, pooled)


def test_malliavin_with_independent_jumps() -> None:
    dp = DiffProcess(
        VarianceGamma(0.2, 0.3, -0.15) + MertonJumpDiffusion(0.0, 0.15, 1.5, -0.08, 0.12)
    )
    price = lambda s: cos_price(dp, s, np.array([K]), T, R, Q)[0]  # noqa: E731
    delta = (price(S + 1e-3) - price(S - 1e-3)) / 2e-3
    vega = _cos_jac(dp)[dp.names.index("MertonJumpDiffusion.sigma")]
    pooled, _ = _pooled_z(malliavin_greeks, dp, call(K), np.array([delta, vega]))
    assert np.all(np.abs(pooled) < 4.0), pooled


def test_refusals() -> None:
    vg = DiffProcess(VarianceGamma(0.2, 0.3, -0.15))
    with pytest.raises(ValueError, match="Lipschitz"):
        pathwise_gradient(vg, digital_call(K), S, T)
    with pytest.raises(UnsupportedProcessError):
        likelihood_ratio_gradient(
            DiffProcess(VarianceGamma(0.2, 0.3, -0.15) + Drift(0.1)), call(K), S, T
        )
    with pytest.raises(UnsupportedProcessError):
        malliavin_greeks(vg, call(K), S, T)
