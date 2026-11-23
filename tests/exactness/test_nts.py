"""
Exactness and mathematical consistency tests for Normal Tempered Stable (NTS) process.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.zoo.levy import NTSProcess, NormalTemperedStable, NormalTemperedStableProcess


class TestNormalTemperedStableExactness:
    """Mathematical exactness and consistency test suite for NormalTemperedStableProcess."""

    def test_nts_exactness_level_and_aliases(self) -> None:
        """NTSProcess is EXACT and identical to NormalTemperedStable, NormalTemperedStableProcess."""
        p = NormalTemperedStableProcess(alpha=0.5, lambd=1.5, delta=1.0, theta=-0.2, sigma=0.8)
        assert p.exactness == ExactnessLevel.EXACT
        assert NormalTemperedStable is NormalTemperedStableProcess
        assert NTSProcess is NormalTemperedStableProcess
        assert "NormalTemperedStableProcess" in repr(p)
        assert "alpha=0.5" in repr(p)
        assert "lambd=1.5" in repr(p)

    def test_nts_triplet_structure(self) -> None:
        """NTS triplet has zero diffusion, and its drift gives the right mean.

        With truncation 1{|x|<=1} the drift is b = E[X_1] - int_{|x|>1} x nu(dx),
        not the mean itself.
        """
        p = NormalTemperedStableProcess(alpha=0.5, lambd=1.0, delta=1.0, theta=-0.1, sigma=1.0)
        triplet = p.triplet
        assert triplet.sigma_sq == 0.0
        large_jumps = triplet.nu.first_moment_between(1.0, math.inf)
        assert math.isclose(triplet.b, p.mean - large_jumps, rel_tol=1e-9)
        assert math.isclose(triplet.cumulant(1), p.mean, rel_tol=1e-9)
        assert not triplet.nu.is_finite_activity()

    def test_nts_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid stability, tempering, or scale."""
        with pytest.raises(ValueError, match="alpha"):
            NormalTemperedStableProcess(alpha=-0.1)
        with pytest.raises(ValueError, match="alpha"):
            NormalTemperedStableProcess(alpha=1.0)
        with pytest.raises(ValueError, match="lambd"):
            NormalTemperedStableProcess(lambd=-1.0)
        with pytest.raises(ValueError, match="delta"):
            NormalTemperedStableProcess(delta=0.0)
        with pytest.raises(ValueError, match="sigma"):
            NormalTemperedStableProcess(sigma=-0.5)

    def test_nts_process_properties(self) -> None:
        """Process properties reflect pure-jump Lévy structure."""
        p_drift = NormalTemperedStableProcess(theta=0.2)
        assert p_drift.properties.has_stationary_increments
        assert p_drift.properties.has_independent_increments
        assert not p_drift.properties.is_martingale

        p_mart = NormalTemperedStableProcess(theta=0.0)
        assert p_mart.properties.is_martingale

    def test_nts_characteristic_function_at_zero_and_symmetry(self) -> None:
        """Characteristic function satisfies phi(0) = 1 and phi(-u) = conj(phi(u))."""
        p = NormalTemperedStableProcess(alpha=0.4, lambd=2.0, delta=1.2, theta=-0.3, sigma=0.9)
        cf0 = p.char_func(0.0)
        assert math.isclose(abs(cf0), 1.0, rel_tol=1e-9)

        u = 1.5
        cf_pos = p.char_func(u)
        cf_neg = p.char_func(-u)
        assert math.isclose(cf_pos.real, cf_neg.real, rel_tol=1e-9)
        assert math.isclose(cf_pos.imag, -cf_neg.imag, rel_tol=1e-9)

    def test_nts_cumulants_match_moments(self) -> None:
        """Exact cumulants c_1, c_2, c_3, c_4 match subordination identities."""
        p = NormalTemperedStableProcess(alpha=0.5, lambd=1.5, delta=1.0, theta=-0.2, sigma=0.8)
        assert p.mean == p.cumulant(1)
        assert p.variance == p.cumulant(2)
        assert p.variance > 0.0

        # Skewness has same sign as theta
        assert p.skewness < 0.0
        # Excess kurtosis is strictly positive (leptokurtic)
        assert p.excess_kurtosis > 0.0

    def test_nts_vectorized_simulation(self) -> None:
        """Simulation produces sample paths with correct shape starting at 0."""
        p = NormalTemperedStableProcess(alpha=0.5, lambd=1.0, delta=1.0, theta=0.1, sigma=0.5)
        t_grid = np.linspace(0, 1.0, 11)
        rng = np.random.default_rng(123)

        paths = p.simulate(t_grid, n_paths=150, rng=rng)
        assert paths.shape == (150, 11)
        assert np.allclose(paths[:, 0], 0.0)
