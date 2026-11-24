"""Closed-form gradients behind the SLSQP SVI and SSVI calibrators (``jac="exact"``)."""

from __future__ import annotations

import numpy as np
import pytest

import spxa
from spxa.vol import (
    SSVI,
    ATMVarianceCurve,
    HestonLikePhi,
    PowerLawPhi,
    SVIRaw,
    calibrate_ssvi,
    calibrate_svi_butterfly_free,
    calibrate_svi_surface,
    check_butterfly,
    check_calendar,
)
from spxa.vol.arbitrage import durrleman_g
from spxa.vol.calibration import _durrleman_g_and_gradient, _svi_and_gradients

WIDE = np.linspace(-10.0, 10.0, 20001)


def _svi_jet(x, kv):
    a, b, rho, m, sig = x[0], x[1], x[2], x[3], x[4]
    y = kv - m
    r = np.sqrt(y * y + sig * sig)
    return a + b * (rho * y + r), b * (rho + y / r), b * sig * sig / r**3


def _g_jet(x, kv, floor=1e-12):
    w, w1, w2 = _svi_jet(x, kv)
    w = np.maximum(w, floor)
    A = 1 - kv * w1 / (2 * w)
    return A * A - 0.25 * w1 * w1 * (1 / w + 0.25) + 0.5 * w2


def _random_slices(n: int):
    rng = np.random.default_rng(17)
    for _ in range(n):
        x = np.array(
            [
                rng.uniform(-0.05, 0.1),
                rng.uniform(0.01, 1.5),
                rng.uniform(-0.99, 0.99),
                rng.uniform(-0.5, 0.5),
                rng.uniform(1e-3, 1.0),
            ]
        )
        kv = np.concatenate([rng.uniform(-3, 3, 25), [x[3]]])  # includes k = m (y = 0)
        yield x, kv


def test_svi_value_and_gradients_match_jets() -> None:
    for x, kv in _random_slices(200):
        out = _svi_and_gradients(x, kv)
        for i in range(3):
            J = spxa.jacobian(lambda z, i=i: _svi_jet(z, kv)[i])(x)
            np.testing.assert_allclose(out[i], _svi_jet(x, kv)[i], rtol=1e-15, atol=1e-15)
            np.testing.assert_allclose(out[3 + i], J, rtol=0, atol=1e-14 * np.max(np.abs(J)))


def test_durrleman_gradient_matches_jets_and_the_library_g() -> None:
    for x, kv in _random_slices(200):
        g, dg = _durrleman_g_and_gradient(kv, x)
        w, w1, w2, *_ = _svi_and_gradients(x, kv)
        np.testing.assert_allclose(g, durrleman_g(kv, np.maximum(w, 1e-12), w1, w2), rtol=1e-14)
        J = spxa.jacobian(lambda z: _g_jet(z, kv))(x)
        np.testing.assert_allclose(dg, J, rtol=0, atol=1e-14 * np.max(np.abs(J)))
    # where the floor on w is active, w is a constant
    x = np.array([-0.5, 0.1, 0.0, 0.0, 0.1])
    kv = np.linspace(-0.2, 0.2, 5)
    J = spxa.jacobian(lambda z: _g_jet(z, kv))(x)
    np.testing.assert_allclose(_durrleman_g_and_gradient(kv, x)[1], J, rtol=1e-14)


def test_durrleman_gradient_matches_differences() -> None:
    # independent of the jet rules
    x = np.array([0.02, 0.4, -0.5, 0.1, 0.3])
    kv = np.linspace(-2, 2, 9)
    _, dg = _durrleman_g_and_gradient(kv, x)
    for j in range(5):
        h = 1e-4 * max(abs(x[j]), 0.05)
        e = np.eye(5)[j]

        def d(hh: float, e: np.ndarray = e) -> np.ndarray:
            up = _durrleman_g_and_gradient(kv, x + hh * e)[0]
            return (up - _durrleman_g_and_gradient(kv, x - hh * e)[0]) / (2 * hh)

        ref = (4 * d(h / 2) - d(h)) / 3
        np.testing.assert_allclose(dg[:, j], ref, rtol=0, atol=1e-8 * np.max(np.abs(ref)))


@pytest.mark.parametrize(
    "phi",
    [HestonLikePhi(2.0), HestonLikePhi(1e-3), HestonLikePhi(80.0), PowerLawPhi(1.1, 0.4)],
    ids=["heston", "heston-small-x", "heston-large-x", "power"],
)
def test_phi_parameter_gradient_against_mpmath(phi) -> None:
    mp = pytest.importorskip("mpmath")
    th = np.geomspace(1e-4, 5.0, 25)
    g = phi.parameter_gradient(th)
    heston = isinstance(phi, HestonLikePhi)
    assert g.shape == th.shape + ((1,) if heston else (2,))
    with mp.workdps(40):
        for i, t in enumerate(th):
            t = mp.mpf(t)
            if heston:
                f = lambda lam, t=t: (1 - (1 - mp.e ** (-lam * t)) / (lam * t)) / (lam * t)  # noqa: E731
                refs = [mp.diff(f, mp.mpf(phi.lam))]
            else:
                f = lambda e, c, t=t: e * t ** (-c) * (1 + t) ** (c - 1)  # noqa: E731
                e0, c0 = mp.mpf(phi.eta), mp.mpf(phi.gamma)
                refs = [mp.diff(lambda e: f(e, c0), e0), mp.diff(lambda c: f(e0, c), c0)]
            for j, ref in enumerate(refs):
                assert g[i, j] == pytest.approx(float(ref), rel=1e-14, abs=1e-300)


T = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
TH = np.array([0.004, 0.009, 0.017, 0.033, 0.064])
KS = [np.linspace(-1.2 * np.sqrt(t), 0.8 * np.sqrt(t), 15) for t in T]


@pytest.mark.parametrize(
    "true, kind",
    [
        (SSVI(-0.6, HestonLikePhi(2.0), ATMVarianceCurve(T, TH)), "heston"),
        (SSVI(-0.3, HestonLikePhi(100.0), ATMVarianceCurve(T, TH)), "heston"),
        (SSVI(-0.7, PowerLawPhi(1.1, 0.4), ATMVarianceCurve(T, TH)), "power_law"),
        (SSVI(0.5, PowerLawPhi(0.3, 0.2), ATMVarianceCurve(T, TH)), "power_law"),
    ],
)
def test_ssvi_exact_recovery(true: SSVI, kind: str) -> None:
    data = [np.asarray(true.total_variance(k, t)) for k, t in zip(KS, T)]
    fit = calibrate_ssvi(T, KS, data, phi=kind)
    assert fit.success
    # stage 1 recovers theta even where the slice's SVI m = -rho/phi and sigma lie
    # far outside the strikes (phi small)
    np.testing.assert_allclose(fit.thetas, TH, rtol=1e-11)
    assert fit.rmse < 1e-9
    assert fit.surface.rho == pytest.approx(true.rho, abs=1e-6)
    params = ("lam",) if kind == "heston" else ("eta", "gamma")
    for p in params:
        assert getattr(fit.surface.phi, p) == pytest.approx(getattr(true.phi, p), rel=1e-6)


def test_ssvi_exact_is_at_least_as_good_as_differences() -> None:
    true = SSVI(-0.6, HestonLikePhi(2.0), ATMVarianceCurve(T, TH))
    data = [np.asarray(true.total_variance(k, t)) for k, t in zip(KS, T)]
    ex = calibrate_ssvi(T, KS, data, phi="heston", jac="exact")
    fd = calibrate_ssvi(T, KS, data, phi="heston", jac="2-point")
    assert ex.rmse <= fd.rmse
    with pytest.raises(ValueError, match="jac"):
        calibrate_ssvi(T, KS, data, jac="3-point")


def test_butterfly_free_exact_and_differences_agree() -> None:
    vogt = SVIRaw(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)  # has arbitrage
    k = np.linspace(-1.5, 1.5, 41)
    ex = calibrate_svi_butterfly_free(k, vogt(k), jac="exact")
    fd = calibrate_svi_butterfly_free(k, vogt(k), jac="2-point")
    assert ex.success and fd.success
    assert check_butterfly(ex.params, WIDE).arbitrage_free
    assert ex.sse <= fd.sse * (1 + 1e-8)
    np.testing.assert_allclose(ex.params.to_array(), fd.params.to_array(), rtol=0, atol=1e-5)
    assert ex.nfev < fd.nfev
    with pytest.raises(ValueError, match="jac"):
        calibrate_svi_butterfly_free(k, vogt(k), jac="cs")


def test_surface_exact_is_calendar_and_butterfly_free() -> None:
    K = np.linspace(-0.6, 0.4, 21)
    short = SVIRaw(0.01, 0.1, -0.5, 0.0, 0.15)
    long = SVIRaw(0.03, 0.06, -0.3, 0.05, 0.2)  # crosses short: calendar arbitrage in quotes
    ex = calibrate_svi_surface([0.5, 1.0], [K, K], [short(K), long(K)], jac="exact")
    fd = calibrate_svi_surface([0.5, 1.0], [K, K], [short(K), long(K)], jac="2-point")
    assert ex.success and check_calendar(ex.slices, WIDE).arbitrage_free
    assert all(check_butterfly(s, WIDE).arbitrage_free for s in ex.slices)
    for a, b in zip(ex.fits, fd.fits):
        assert a.sse <= b.sse * (1 + 1e-7) + 1e-18
