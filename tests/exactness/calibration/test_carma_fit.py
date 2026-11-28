"""Kalman likelihood and Gaussian QML fitting of CARMA processes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg, optimize, stats

from spxa.calibration.filter.carma import (
    _coefficients,
    _factor_hurwitz,
    _hurwitz_poly,
    carma_kalman_filter,
    carma_log_likelihood,
    fit_carma,
    select_carma_order,
)
from spxa.zoo.beyond.carma import CARMAProcess
from spxa.zoo.levy.brownian import BrownianMotion
from spxa.zoo.levy.gamma import GammaProcess

DT = 0.1
CARMA31 = CARMAProcess.from_roots([-0.5, -1 + 2j, -1 - 2j], [-2.0], driver=BrownianMotion(0.1, 0.3))
CARMA21 = CARMAProcess([1.2, 1.0], [0.8, 1.0], driver=BrownianMotion(0.2, 0.5))


def _exact_gaussian_loglik(model: CARMAProcess, y: np.ndarray, times: np.ndarray) -> float:
    cov = model.autocovariance(np.abs(times[:, None] - times[None, :]))
    chol = linalg.cho_factor(0.5 * (cov + cov.T))
    r = y - model.stationary_mean
    return float(
        -0.5
        * (
            y.size * np.log(2 * np.pi)
            + 2 * np.sum(np.log(np.diag(chol[0])))
            + r @ linalg.cho_solve(chol, r)
        )
    )


@pytest.fixture(scope="module")
def series21() -> np.ndarray:
    return CARMA21.simulate(3000, 1, T=300.0, rng=np.random.default_rng(1))[0]


# ---------------------------------------------------------------------------
# the likelihood
# ---------------------------------------------------------------------------


def test_kalman_likelihood_is_the_exact_gaussian_likelihood() -> None:
    y = CARMA31.simulate(400, 1, T=40.0, rng=np.random.default_rng(1))[0]
    kal = carma_kalman_filter(CARMA31, y, DT)
    assert 0 < kal.steady_from < y.size  # the steady-state path was used
    chol = linalg.cho_factor(linalg.toeplitz(CARMA31.sampled_autocovariance(DT, y.size - 1)))
    r = y - CARMA31.stationary_mean
    exact = -0.5 * (
        y.size * np.log(2 * np.pi)
        + 2 * np.sum(np.log(np.diag(chol[0])))
        + r @ linalg.cho_solve(chol, r)
    )
    assert kal.log_likelihood == pytest.approx(exact, abs=1e-8)
    assert carma_log_likelihood(CARMA31, y, DT) == kal.log_likelihood


def test_irregular_times() -> None:
    rng = np.random.default_rng(5)
    t = np.cumsum(0.05 + rng.exponential(0.13, 300))
    cov = CARMA31.autocovariance(np.abs(t[:, None] - t[None, :]))
    y = np.random.default_rng(6).multivariate_normal(np.full(t.size, CARMA31.stationary_mean), cov)
    ll = carma_log_likelihood(CARMA31, y, times=t)
    assert ll == pytest.approx(_exact_gaussian_loglik(CARMA31, y, t), abs=1e-8)
    # an even grid given as times takes the full recursion and agrees with the steady-state path
    z = CARMA31.simulate(3000, 1, T=300.0, rng=np.random.default_rng(7))[0]
    fast = carma_kalman_filter(CARMA31, z, DT)
    full = carma_kalman_filter(CARMA31, z, times=DT * np.arange(z.size))
    assert full.steady_from == z.size
    np.testing.assert_allclose(fast.innovations, full.innovations, atol=1e-9)
    np.testing.assert_allclose(fast.variances, full.variances, rtol=1e-9)
    assert fast.log_likelihood == pytest.approx(full.log_likelihood, abs=1e-7)


def test_car1_likelihood_is_the_exact_ar1_likelihood() -> None:
    """CAR(1) on a grid is a Gaussian AR(1) with phi = exp(-a dt): an independent closed form."""
    a, sigma, drift = 1.7, 0.4, 0.5
    model = CARMAProcess([a], driver=BrownianMotion(drift, sigma))
    y = model.simulate(500, 1, T=50.0, rng=np.random.default_rng(8))[0]
    mu, phi, v = drift / a, np.exp(-a * DT), sigma**2 / (2 * a)
    ll = stats.norm.logpdf(y[0], mu, np.sqrt(v)) + np.sum(
        stats.norm.logpdf(y[1:], mu + phi * (y[:-1] - mu), np.sqrt(v * (1 - phi**2)))
    )
    assert carma_log_likelihood(model, y, DT) == pytest.approx(ll, abs=1e-9)


def test_standardized_innovations_of_the_true_model_are_white() -> None:
    y = CARMA31.simulate(5000, 1, T=500.0, rng=np.random.default_rng(9))[0]
    z = carma_kalman_filter(CARMA31, y, DT).standardized
    n = z.size
    assert abs(z.mean()) < 4 / np.sqrt(n)
    assert z.var() == pytest.approx(1.0, abs=4 * np.sqrt(2 / n))
    for lag in (1, 2, 5):
        assert abs(np.corrcoef(z[:-lag], z[lag:])[0, 1]) < 4 / np.sqrt(n)


def test_hurwitz_parametrisation() -> None:
    rng = np.random.default_rng(10)
    for p in (1, 2, 3, 4, 5):
        for _ in range(20):
            poly = _hurwitz_poly(rng.normal(0.0, 2.0, p))
            assert poly.size == p + 1 and np.all(np.roots(poly).real < 0)
    roots = np.array([-0.5, -1 + 2j, -1 - 2j, -3.0, -0.2])
    np.testing.assert_allclose(
        _hurwitz_poly(_factor_hurwitz(roots)), np.real(np.poly(roots)), rtol=1e-12
    )
    ar, ma = _coefficients(np.log([2.0, 5.0, 0.5, 3.0]), 3, 1)
    # a = (z^2 + 2z + 5)(z + 0.5); b = z + 3, with b_q = 1
    np.testing.assert_allclose(ar, [2.5, 6.0, 2.5])
    np.testing.assert_allclose(ma, [3.0, 1.0])


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fit21(series21: np.ndarray):  # type: ignore[no-untyped-def]
    return fit_carma(series21, 2, 1, DT, n_starts=2)


def test_fit_maximises_the_likelihood(series21: np.ndarray, fit21) -> None:  # type: ignore[no-untyped-def]
    # the profiled optimum is the likelihood of the reported process
    assert fit21.log_likelihood == pytest.approx(
        carma_log_likelihood(fit21.process, series21, DT), abs=1e-8
    )
    assert fit21.log_likelihood >= carma_log_likelihood(CARMA21, series21, DT)
    assert fit21.converged

    # no direction in the natural coordinates (a1, a2, b0, sigma, drift) improves it
    def negll(v: np.ndarray) -> float:
        try:
            m = CARMAProcess(v[:2], [v[2], 1.0], driver=BrownianMotion(v[4], v[3]))
            return -carma_log_likelihood(m, series21, DT)
        except ValueError:
            return np.inf

    v0 = np.array([*fit21.ar, fit21.ma[0], fit21.sigma, fit21.process.driver.mu])
    polish = optimize.minimize(
        negll, v0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-10}
    )
    assert -polish.fun - fit21.log_likelihood < 1e-5


def test_fit_recovers_a_gaussian_carma(fit21) -> None:  # type: ignore[no-untyped-def]
    np.testing.assert_allclose(fit21.ar, CARMA21.ar, rtol=0.15)
    np.testing.assert_allclose(fit21.ma, CARMA21.ma, rtol=0.25)
    assert fit21.sigma == pytest.approx(0.5, rel=0.05)
    assert fit21.mean == pytest.approx(CARMA21.stationary_mean, abs=0.05)
    assert fit21.process.stationary_mean == pytest.approx(fit21.mean, rel=1e-12)
    assert (fit21.p, fit21.q, fit21.n_params, fit21.n_obs) == (2, 1, 5, 3001)
    assert fit21.aic == pytest.approx(-2 * fit21.log_likelihood + 10)
    assert fit21.bic == pytest.approx(-2 * fit21.log_likelihood + 5 * np.log(3001))
    assert np.all(fit21.ma_roots.real < 0) and np.all(fit21.ar_roots.real < 0)
    z = fit21.residuals
    assert z.var() == pytest.approx(1.0, abs=0.01)  # sigma^2 is profiled
    assert "CARMA(2,1)" in fit21.summary()


def test_quasi_likelihood_recovers_a_levy_driven_carma() -> None:
    """Gamma driver: the Gaussian QMLE still recovers the zeros, sigma^2 = kappa_2 and the mean."""
    truth = CARMAProcess([1.2, 1.0], [0.8, 1.0], driver=GammaProcess(a=3.0, b=2.0))
    y = truth.simulate(3000, 1, T=300.0, rng=np.random.default_rng(2))[0]
    fit = fit_carma(y, 2, 1, DT, n_starts=2)
    np.testing.assert_allclose(fit.ar, truth.ar, rtol=0.1)
    np.testing.assert_allclose(fit.ma, truth.ma, rtol=0.15)
    assert fit.sigma == pytest.approx(np.sqrt(3.0 / 4.0), rel=0.05)
    assert fit.mean == pytest.approx(truth.stationary_mean, rel=0.03)
    # innovations are uncorrelated but not Gaussian: the driver's skew survives
    z = fit.residuals
    assert abs(np.corrcoef(z[:-1], z[1:])[0, 1]) < 4 / np.sqrt(z.size)
    assert stats.skew(z) > 0.3


def test_fit_on_irregular_times() -> None:
    rng = np.random.default_rng(3)
    t = np.round(np.cumsum(0.02 + rng.exponential(0.08, 800)), 2)
    t = np.unique(t)
    n_fine = int(round(t[-1] / 0.01)) + 1
    fine = CARMA21.simulate(n_fine, 1, T=n_fine * 0.01, rng=np.random.default_rng(4))[0]
    y = fine[np.round(t / 0.01).astype(int)]
    fit = fit_carma(y, 2, 1, times=t, n_starts=2)
    assert fit.log_likelihood >= carma_log_likelihood(CARMA21, y, times=t)
    assert fit.log_likelihood == pytest.approx(
        carma_log_likelihood(fit.process, y, times=t), abs=1e-8
    )
    np.testing.assert_allclose(fit.ar, CARMA21.ar, rtol=0.35)
    assert fit.sigma == pytest.approx(0.5, rel=0.1)


def test_order_selection_finds_the_true_order(series21: np.ndarray) -> None:
    fits = select_carma_order(series21, DT, max_p=2, n_starts=2)
    assert [(f.p, f.q) for f in fits][0] == (2, 1)
    assert len(fits) == 3
    bics = [f.bic for f in fits]
    assert bics == sorted(bics)
    by_aic = select_carma_order(series21[:600], DT, max_p=1, criterion="aic")
    assert [(f.p, f.q) for f in by_aic] == [(1, 0)]


def test_validation() -> None:
    y = CARMA21.simulate(100, 1, T=10.0, rng=np.random.default_rng(11))[0]
    with pytest.raises(ValueError, match="exactly one"):
        carma_log_likelihood(CARMA21, y)
    with pytest.raises(ValueError, match="exactly one"):
        carma_log_likelihood(CARMA21, y, DT, times=np.arange(y.size))
    with pytest.raises(ValueError, match="increasing"):
        carma_log_likelihood(CARMA21, y, times=np.zeros(y.size))
    with pytest.raises(ValueError, match="one entry"):
        carma_log_likelihood(CARMA21, y, times=np.arange(3.0))
    with pytest.raises(ValueError, match="positive"):
        carma_log_likelihood(CARMA21, y, 0.0)
    with pytest.raises(ValueError, match="finite"):
        carma_log_likelihood(CARMA21, np.array([np.nan, 1.0]), DT)
    with pytest.raises(ValueError, match="q < p"):
        fit_carma(y, 2, 2, DT)
    with pytest.raises(ValueError, match="observations"):
        fit_carma(y[:6], 2, 1, DT)
    with pytest.raises(ValueError, match="n_starts"):
        fit_carma(y, 1, 0, DT, n_starts=0)
    with pytest.raises(ValueError, match="criterion"):
        select_carma_order(y, DT, criterion="hqc")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_p"):
        select_carma_order(y, DT, max_p=0)
