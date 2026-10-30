"""Tests for Newey-West HAC covariance estimator."""

import numpy as np

from spxa.calibration.mle.hac import newey_west_hac


class TestMLEHAC:
    def test_hac_positive_definite(self):
        rng = np.random.default_rng(42)
        n = 500
        # AR(1) autocorrelated series: x_t = 0.6 * x_{t-1} + eps
        eps = rng.normal(size=n)
        x = np.empty(n)
        x[0] = eps[0]
        for t in range(1, n):
            x[t] = 0.6 * x[t - 1] + eps[t]

        # Moments matrix with 2 dimensions
        m = np.column_stack([x, x**2 - 1.0])
        omega = newey_west_hac(m, lags=5)

        assert omega.shape == (2, 2)
        # Check symmetry
        np.testing.assert_allclose(omega, omega.T, atol=1e-10)
        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(omega)
        assert np.all(eigvals > 0.0)

    def test_hac_automatic_lag_length(self):
        # 1000 observations -> lag = floor(4 * (10)^(2/9)) = floor(4 * 1.668) = 6
        m = np.random.normal(size=(1000, 3))
        omega = newey_west_hac(m, lags=None)
        assert omega.shape == (3, 3)


def test_quadratic_spectral_kernel_differs_from_bartlett_and_matches_formula():
    import math

    import numpy as np
    import pytest

    from spxa.calibration.mle.hac import newey_west_hac

    rng = np.random.default_rng(3)
    e = rng.normal(size=401)
    x = (e[1:] + 0.6 * e[:-1])[:, None]  # MA(1): long-run variance (1 + 0.6)^2 = 2.56
    lags = 6
    qs = newey_west_hac(x, lags=lags, kernel="quadratic_spectral")
    bt = newey_west_hac(x, lags=lags, kernel="bartlett")
    assert not np.isclose(qs[0, 0], bt[0, 0])

    xd = x[:, 0] - x[:, 0].mean()
    n = len(xd)
    ref = xd @ xd / n
    for j in range(1, n):
        z = 6 * math.pi * (j / (lags + 1)) / 5
        ref += (
            2
            * (25 / (12 * math.pi**2 * (j / (lags + 1)) ** 2))
            * (math.sin(z) / z - math.cos(z))
            * (xd[j:] @ xd[:-j] / n)
        )
    assert qs[0, 0] == pytest.approx(max(ref, 1e-12), rel=1e-10)
    assert abs(qs[0, 0] - 2.56) < 0.8

    with pytest.raises(ValueError):
        newey_west_hac(x, kernel="parzen")
