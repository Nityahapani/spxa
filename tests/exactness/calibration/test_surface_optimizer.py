"""Tests for surface calibration optimizers and parameter transformations."""

import numpy as np
import pytest

from spxa.calibration.surface.optimizer import (
    DifferentialEvolutionOptimizer,
    HybridOptimizer,
    LeastSquaresOptimizer,
    ParameterTransform,
)


class TestSurfaceOptimizer:
    def test_parameter_transform_roundtrip(self):
        bounds = [(0.01, 10.0), (-0.99, 0.99), (0.001, 1.0)]
        transform = ParameterTransform(bounds)

        test_params = np.array([2.5, -0.65, 0.04])
        uncon = transform.to_unconstrained(test_params)
        reconstructed = transform.to_constrained(uncon)

        np.testing.assert_allclose(reconstructed, test_params, rtol=1e-7, atol=1e-7)

    def test_least_squares_optimizer_rosenbrock(self):
        # Fit 2 parameters (a, b) to non-linear residuals: [10*(y - x^2), 1 - x]
        def residual_fn(p):
            x, y = p[0], p[1]
            return np.array([10.0 * (y - x**2), 1.0 - x])

        opt = LeastSquaresOptimizer()
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        x0 = np.array([0.0, 0.0])

        res = opt.minimize(residual_fn, x0, bounds)
        assert res.success is True
        np.testing.assert_allclose(res.params, [1.0, 1.0], atol=1e-5)
        assert res.cost < 1e-10

    def test_least_squares_optimizer_exact_jacobian(self):
        from spxa.graph.autodiff.jet import variable_vector

        def residual_fn(p):
            x, y = p[0], p[1]
            return 10.0 * (y - x**2), 1.0 - x

        def resid(p):
            return np.array(residual_fn(p))

        def jac(p):
            v = variable_vector(p)
            return np.array([np.asarray(r.grad) for r in residual_fn(v)])

        opt = LeastSquaresOptimizer(ftol=1e-15, xtol=1e-15, gtol=1e-15)
        res = opt.minimize(resid, np.array([-1.2, 1.0]), [(-2.0, 2.0), (-2.0, 2.0)], jac=jac)
        np.testing.assert_allclose(res.params, [1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(res.jacobian, [[-20.0, 10.0], [-1.0, 0.0]], atol=1e-10)

    def test_differential_evolution_optimizer(self):
        # Multimodal Rastrigin-like residual
        def residual_fn(p):
            x = p[0]
            return np.array([x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0])

        opt = DifferentialEvolutionOptimizer(maxiter=30, seed=42)
        bounds = [(-5.0, 5.0)]
        res = opt.minimize(residual_fn, np.array([3.0]), bounds)

        assert res.success is True
        # Global minimum is at 0.0
        assert res.params[0] == pytest.approx(0.0, abs=1e-3)

    def test_hybrid_optimizer_accuracy(self):
        # Non-convex valley: (y - x^2)^2 + (1 - x)^2
        def residual_fn(p):
            x, y = p[0], p[1]
            return np.array([y - x**2, 1.0 - x])

        hybrid = HybridOptimizer(global_maxiter=20, local_max_nfev=100, seed=42)
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]
        res = hybrid.minimize(residual_fn, np.array([2.0, -1.0]), bounds)

        assert res.success is True
        np.testing.assert_allclose(res.params, [1.0, 1.0], atol=1e-5)
