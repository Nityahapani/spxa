"""
Tests for multivariate Lévy processes.
"""

import numpy as np
import pytest

from spxa.zoo.multivariate import (
    CorrelatedLevy,
    MultivariateBrownianMotion,
    MultivariateLevyTriplet,
    correlated_brownian_motion,
    independent_levy_vector,
)
from spxa.zoo.levy import BrownianMotion, GammaProcess, VarianceGamma


class TestMultivariateLevyTriplet:
    def test_construction(self) -> None:
        from spxa.zoo.levy.brownian import _ZeroLevyMeasure
        b = np.array([1.0, 2.0])
        sigma = np.eye(2)
        t = MultivariateLevyTriplet(b=b, sigma=sigma, nu=_ZeroLevyMeasure(), dim=2)
        assert t.dim == 2

    def test_non_symmetric_sigma_raises(self) -> None:
        from spxa.zoo.levy.brownian import _ZeroLevyMeasure
        with pytest.raises(ValueError, match="symmetric"):
            MultivariateLevyTriplet(
                b=np.zeros(2), sigma=np.array([[1.0, 0.5], [0.0, 1.0]]),
                nu=_ZeroLevyMeasure(), dim=2
            )

    def test_non_psd_sigma_raises(self) -> None:
        from spxa.zoo.levy.brownian import _ZeroLevyMeasure
        with pytest.raises(ValueError, match="positive semidefinite"):
            MultivariateLevyTriplet(
                b=np.zeros(2), sigma=np.array([[-1.0, 0.0], [0.0, 1.0]]),
                nu=_ZeroLevyMeasure(), dim=2
            )

    def test_addition(self) -> None:
        from spxa.zoo.levy.brownian import _ZeroLevyMeasure
        nu = _ZeroLevyMeasure()
        t1 = MultivariateLevyTriplet(b=np.array([1.0, 0.0]), sigma=np.eye(2), nu=nu, dim=2)
        t2 = MultivariateLevyTriplet(b=np.array([0.0, 2.0]), sigma=np.eye(2), nu=nu, dim=2)
        t3 = t1 + t2
        np.testing.assert_array_equal(t3.b, [1.0, 2.0])
        np.testing.assert_array_equal(t3.sigma, 2 * np.eye(2))


class TestMultivariateBrownianMotion:
    def test_simulate_shape(self) -> None:
        mbm = MultivariateBrownianMotion(dim=3)
        paths = mbm.simulate(n_steps=50, n_paths=10, T=1.0)
        assert paths.shape == (10, 51, 3)

    def test_simulate_starts_at_zero(self) -> None:
        mbm = MultivariateBrownianMotion(dim=2)
        paths = mbm.simulate(n_steps=50, n_paths=5, T=1.0)
        np.testing.assert_array_equal(paths[:, 0, :], 0.0)

    def test_covariance_matrix_recovered(self) -> None:
        """Empirical covariance at T=1 should match Σ."""
        rng = np.random.default_rng(0)
        Sigma = np.array([[1.0, 0.7], [0.7, 2.0]])
        mbm = correlated_brownian_motion(Sigma)
        paths = mbm.simulate(n_steps=100, n_paths=10000, T=1.0, rng=rng)
        empirical_cov = np.cov(paths[:, -1, :].T)
        np.testing.assert_allclose(empirical_cov, Sigma, atol=0.06)

    def test_correlation_matrix(self) -> None:
        Sigma = np.array([[4.0, 2.0], [2.0, 4.0]])
        mbm = correlated_brownian_motion(Sigma)
        corr = mbm.correlation()
        expected = np.array([[1.0, 0.5], [0.5, 1.0]])
        np.testing.assert_allclose(corr, expected, atol=1e-10)

    def test_covariance_scales_with_t(self) -> None:
        """Cov(X_t) = t·Σ."""
        mbm = correlated_brownian_motion(np.eye(2))
        cov_t2 = mbm.covariance(t=2.0)
        np.testing.assert_allclose(cov_t2, 2 * np.eye(2))

    def test_martingale_when_zero_drift(self) -> None:
        mbm = MultivariateBrownianMotion(mu=np.zeros(2), sigma=np.eye(2))
        assert mbm.properties.is_martingale is True

    def test_not_martingale_with_drift(self) -> None:
        mbm = MultivariateBrownianMotion(mu=np.array([1.0, 0.0]), sigma=np.eye(2))
        assert mbm.properties.is_martingale is False

    def test_multivariate_triplet(self) -> None:
        Sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
        mbm = correlated_brownian_motion(Sigma)
        t = mbm.multivariate_triplet()
        assert t.dim == 2
        np.testing.assert_allclose(t.sigma, Sigma)


class TestCorrelatedLevy:
    def test_shape(self) -> None:
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        C = CorrelatedLevy([bm1, bm2], A=A)
        paths = C.simulate(n_steps=50, n_paths=10, T=1.0)
        assert paths.shape == (10, 51, 2)

    def test_sigma_from_triplet(self) -> None:
        """Σ_Z = A·diag(σ²_X)·Aᵀ."""
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=2.0)
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        C = CorrelatedLevy([bm1, bm2], A=A)
        t = C.multivariate_triplet()
        expected = np.diag([1.0, 4.0])
        np.testing.assert_allclose(t.sigma, expected, atol=1e-10)

    def test_dimension_mismatch_raises(self) -> None:
        bm = BrownianMotion(sigma=1.0)
        A = np.eye(3)
        with pytest.raises(ValueError, match="columns"):
            CorrelatedLevy([bm], A=A)

    def test_independent_levy_vector_shape(self) -> None:
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)
        g = GammaProcess(a=1.0, b=2.0)
        Z = independent_levy_vector(vg, g)
        assert Z.dim_out == 2
        assert Z.dim_in == 2
        paths = Z.simulate(n_steps=50, n_paths=5, T=1.0)
        assert paths.shape == (5, 51, 2)

    def test_mixing_empirical_cov(self) -> None:
        """Cov(Z_T) ≈ A·diag(σ²)·Aᵀ for BM components."""
        rng = np.random.default_rng(1)
        bm1 = BrownianMotion(sigma=1.0)
        bm2 = BrownianMotion(sigma=1.0)
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        C = CorrelatedLevy([bm1, bm2], A=A)
        paths = C.simulate(n_steps=100, n_paths=10000, T=1.0, rng=rng)
        empirical_cov = np.cov(paths[:, -1, :].T)
        expected_cov = A @ A.T
        np.testing.assert_allclose(empirical_cov, expected_cov, atol=0.06)
