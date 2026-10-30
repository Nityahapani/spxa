"""Tests for payoffs and cell-average projection."""

import numpy as np
import pytest

from spxa.pde.grid import Grid1D
from spxa.pde.payoffs import Payoff, asset_or_nothing, digital, project_payoff, vanilla


class TestPayoffs:
    def test_vanilla(self):
        s = np.array([80.0, 100.0, 120.0])
        np.testing.assert_allclose(vanilla(100, "call")(s), [0, 0, 20])
        np.testing.assert_allclose(vanilla(100, "put")(s), [20, 0, 0])
        assert vanilla(100).kinks == (100.0,)

    def test_digitals(self):
        s = np.array([90.0, 100.0, 110.0])
        np.testing.assert_allclose(digital(100, "call", 2.0)(s), [0, 1.0, 2.0])
        np.testing.assert_allclose(digital(100, "put")(s), [1, 0.5, 0])
        np.testing.assert_allclose(asset_or_nothing(100, "call")(s), [0, 50, 110])
        np.testing.assert_allclose(asset_or_nothing(100, "put")(s), [90, 50, 0])

    def test_algebra(self):
        straddle = vanilla(100, "call") + vanilla(100, "put")
        np.testing.assert_allclose(straddle(np.array([90.0, 110.0])), [10, 10])
        spread = vanilla(90) - vanilla(110)
        np.testing.assert_allclose(spread(np.array([80.0, 100.0, 130.0])), [0, 10, 20])
        assert spread.kinks == (90.0, 110.0)
        np.testing.assert_allclose((2 * vanilla(100))(np.array([105.0])), [10.0])
        with pytest.raises(TypeError):
            vanilla(1) + 3

    def test_validation(self):
        with pytest.raises(ValueError):
            vanilla(-1.0)
        with pytest.raises(ValueError):
            vanilla(1.0, "forward")


class TestProjection:
    def test_pointwise(self):
        g = Grid1D.uniform(0, 200, 11)
        np.testing.assert_allclose(
            project_payoff(vanilla(100), g, "none"), np.maximum(g.nodes - 100, 0)
        )

    def test_average_is_exact_cell_mean(self):
        g = Grid1D.uniform(0.0, 4.0, 5)  # faces 0, .5, 1.5, 2.5, 3.5, 4
        out = project_payoff(vanilla(1.7), g)
        # node 2 cell [1.5, 2.5]: (1/1) * int_{1.7}^{2.5} (s - 1.7) ds = 0.32
        assert out[2] == pytest.approx(0.32)
        assert out[0] == 0.0
        assert out[4] == pytest.approx(4.0 - 0.25 - 1.7)

    def test_average_preserves_integral(self):
        g = Grid1D.sinh(0.0, 3.0, 57, center=1.1, alpha=0.2)
        p = digital(1.234)
        out = project_payoff(p, g)
        assert np.dot(out, g.cell_widths) == pytest.approx(3.0 - 1.234, abs=1e-12)

    def test_smooth_payoff_unchanged_to_second_order(self):
        g = Grid1D.uniform(0.0, 1.0, 101)
        p = Payoff(np.sin)
        out = project_payoff(p, g)
        assert np.max(np.abs(out[1:-1] - np.sin(g.nodes[1:-1]))) < 1e-5

    def test_log_space_transform(self):
        g = Grid1D.uniform(np.log(50), np.log(200), 41)
        out = project_payoff(vanilla(100), g, transform=np.exp, inverse=np.log)
        # integral over the whole log-domain of (e^x - K)^+
        exact = (200 - 100) - 100 * (np.log(200) - np.log(100))
        assert np.dot(out, g.cell_widths) == pytest.approx(exact, rel=1e-10)
        with pytest.raises(ValueError):
            project_payoff(vanilla(100), g, transform=np.exp)

    def test_kink_override_and_method_validation(self):
        g = Grid1D.uniform(0.0, 2.0, 11)
        fn = lambda s: np.abs(s - 0.95)  # noqa: E731
        out = project_payoff(fn, g, kinks=(0.95,))
        assert np.dot(out, g.cell_widths) == pytest.approx(0.95**2 / 2 + 1.05**2 / 2)
        with pytest.raises(ValueError):
            project_payoff(fn, g, method="spline")
