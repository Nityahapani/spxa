"""Tests for spxa.pde.operators: stencils and operator assembly."""

import numpy as np
import pytest

from spxa.pde.grid import Grid1D
from spxa.pde.operators import (
    central_first_weights,
    central_second_weights,
    derivative_matrices,
    one_sided_first_weights,
)


def _random_grid(n=15, seed=1):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.uniform(0.05, 0.3, n))


def _apply3(weights, u):
    lo, di, up = weights
    return lo * u[:-2] + di * u[1:-1] + up * u[2:]


class TestStencils:
    def test_first_derivative_exact_for_quadratics(self):
        x = _random_grid()
        u = 3.0 - 2.0 * x + 0.7 * x**2
        np.testing.assert_allclose(
            _apply3(central_first_weights(x), u), -2.0 + 1.4 * x[1:-1], atol=1e-10
        )

    def test_second_derivative_exact_for_quadratics(self):
        x = _random_grid()
        u = 3.0 - 2.0 * x + 0.7 * x**2
        np.testing.assert_allclose(_apply3(central_second_weights(x), u), 1.4, atol=1e-9)

    def test_weights_annihilate_constants(self):
        x = _random_grid()
        for w in (central_first_weights(x), central_second_weights(x)):
            np.testing.assert_allclose(sum(w), 0.0, atol=1e-10)

    @pytest.mark.parametrize("deriv", [1, 2])
    def test_second_order_on_sinh_grid(self, deriv):
        errs = []
        for n in (81, 161, 321):
            x = Grid1D.sinh(0.0, 3.0, n, center=1.0, alpha=0.3).nodes
            w = central_first_weights(x) if deriv == 1 else central_second_weights(x)
            exact = np.cos(x[1:-1]) if deriv == 1 else -np.sin(x[1:-1])
            errs.append(np.max(np.abs(_apply3(w, np.sin(x)) - exact)))
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.85)

    @pytest.mark.parametrize("pts", [(0.0, 0.3, 0.7), (1.0, 0.8, 0.5)])
    def test_one_sided_exact_for_quadratics(self, pts):
        x0, x1, x2 = pts
        w = one_sided_first_weights(x0, x1, x2)
        f = lambda z: 1.0 + 2.0 * z - 5.0 * z**2  # noqa: E731
        assert w[0] * f(x0) + w[1] * f(x1) + w[2] * f(x2) == pytest.approx(2.0 - 10.0 * x0)

    def test_derivative_matrices(self):
        x = _random_grid(12)
        D1, D2 = derivative_matrices(x)
        u = 1.0 + x - 0.5 * x**2
        np.testing.assert_allclose(D1 @ u, 1.0 - x, atol=1e-9)
        np.testing.assert_allclose(D2 @ u, -1.0, atol=1e-8)
        with pytest.raises(ValueError):
            derivative_matrices(np.array([0.0, 1.0]))
