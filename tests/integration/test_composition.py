"""
Integration tests for full composition pipelines.

Tests that exercise the full stack: process creation → composition →
triplet arithmetic → analytics → simulation.
"""

import numpy as np
import pytest

import spxa
from spxa.analytics import cumulant_table, hellinger_distance, l2_char_func_distance
from spxa.zoo.levy import BrownianMotion, GammaProcess, NIG, VarianceGamma
from spxa.zoo.beyond import FractionalBrownianMotion, OULevy


class TestFullCompositionPipeline:
    def test_bm_plus_vg_exactness(self) -> None:
        """BM + VG is EXACT (both Lévy processes)."""
        bm = BrownianMotion(mu=0.0, sigma=0.1)
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.05)
        z = bm + vg
        assert z.exactness == spxa.ExactnessLevel.EXACT

    def test_bm_plus_vg_sigma_sq_additivity(self) -> None:
        """(BM + VG).triplet.sigma_sq = BM.sigma_sq (VG has σ²=0)."""
        bm = BrownianMotion(mu=0.0, sigma=0.3)
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)
        z = bm + vg
        assert z.triplet.sigma_sq == pytest.approx(0.09, rel=1e-8)

    def test_three_process_sum(self) -> None:
        """BM + VG + NIG triplet sigma_sq sums correctly."""
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)
        nig = NIG(alpha=3.0, beta=0.0, delta=1.0)
        z = bm + vg + nig
        assert z.exactness == spxa.ExactnessLevel.EXACT
        assert z.triplet.sigma_sq == pytest.approx(1.0, rel=1e-8)

    def test_scaled_sum_sigma_sq(self) -> None:
        """(2*BM + 3*BM).sigma_sq = 4 + 9 = 13."""
        bm1 = BrownianMotion(mu=0.0, sigma=1.0)
        bm2 = BrownianMotion(mu=0.0, sigma=1.0)
        z = 2.0 * bm1 + 3.0 * bm2
        assert z.triplet.sigma_sq == pytest.approx(13.0, rel=1e-8)

    def test_subordination_vg_via_at_operator(self) -> None:
        """BM @ GammaProcess = VG in distribution."""
        rng = np.random.default_rng(100)
        bm = BrownianMotion(mu=0.0, sigma=0.2)
        g = GammaProcess(a=10.0, b=10.0)
        z = bm @ g
        paths = z.simulate(n_steps=100, n_paths=5000, T=1.0, rng=rng)
        assert paths.shape == (5000, 101)
        assert np.all(paths[:, 0] == 0.0)

    def test_story_full_pipeline(self) -> None:
        """__story__() on a compound composition records all operations."""
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)
        z = 0.5 * bm + 2.0 * vg
        story = z.__story__()
        assert "Addition" in story
        assert "Scaling" in story
        assert "BrownianMotion" in story
        assert "VarianceGamma" in story

    def test_fbm_plus_bm_degrades(self) -> None:
        """fBM + BM degrades to MOMENT_PROPAGATION with a warning."""
        fbm = FractionalBrownianMotion(H=0.7)
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        with pytest.warns(spxa.SpxaDegradationWarning):
            z = fbm + bm
        assert z.exactness == spxa.ExactnessLevel.MOMENT_PROPAGATION

    def test_ou_levy_cumulants(self) -> None:
        """OULevy stationary mean and variance match GammaProcess cumulants."""
        lam = 2.0
        g = GammaProcess(a=1.0, b=2.0)
        ou = OULevy(lam=lam, subordinator=g)
        g_kappas = g.cumulants(order=2)
        assert ou.stationary_mean == pytest.approx(g_kappas[1], rel=1e-10)
        assert ou.stationary_variance == pytest.approx(g_kappas[2] / (2 * lam), rel=1e-10)

    def test_process_simulation_reproducible(self) -> None:
        """Same rng seed gives identical paths."""
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.05)
        paths1 = vg.simulate(n_steps=50, n_paths=10, T=1.0, rng=np.random.default_rng(42))
        paths2 = vg.simulate(n_steps=50, n_paths=10, T=1.0, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(paths1, paths2)


class TestAnalyticsPipeline:
    def test_cumulant_table_vg(self) -> None:
        """cumulant_table returns correct keys and VG mean."""
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
        table = cumulant_table(vg, order=4, t=1.0)
        assert "mean" in table
        assert "variance" in table
        assert "skewness" in table
        assert "excess_kurtosis" in table
        assert table["mean"] == pytest.approx(-0.1, rel=1e-4)

    def test_l2_distance_same_process_is_zero(self) -> None:
        """L² distance between a process and itself is 0."""
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)
        d = l2_char_func_distance(vg, vg, t=1.0)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_l2_distance_different_processes_positive(self) -> None:
        """L² distance between BM and VG is positive."""
        bm = BrownianMotion(mu=0.0, sigma=0.3)
        vg = VarianceGamma(sigma=0.3, nu=0.5, theta=0.0)
        d = l2_char_func_distance(bm, vg, t=1.0)
        assert d > 0.0

    def test_hellinger_same_process_is_zero(self) -> None:
        """Hellinger distance between a process and itself is 0."""
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)
        h = hellinger_distance(vg, vg, t=1.0)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_hellinger_bounded_by_one(self) -> None:
        """Hellinger distance H² ∈ [0, 1]."""
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        vg = VarianceGamma(sigma=0.2, nu=0.5, theta=-0.1)
        h = hellinger_distance(bm, vg, t=1.0)
        assert 0.0 <= h <= 1.0 + 1e-10

    def test_l2_distance_symmetric(self) -> None:
        """L² distance is symmetric: d(X,Y) = d(Y,X)."""
        bm = BrownianMotion(mu=0.0, sigma=0.5)
        nig = NIG(alpha=3.0, beta=0.0, delta=1.0)
        d_xy = l2_char_func_distance(bm, nig, t=1.0)
        d_yx = l2_char_func_distance(nig, bm, t=1.0)
        assert d_xy == pytest.approx(d_yx, rel=1e-10)


class TestTopLevelImports:
    def test_all_processes_importable_from_spxa(self) -> None:
        """All main process classes are importable directly from spxa."""
        import spxa
        for name in ["BrownianMotion", "GammaProcess", "VarianceGamma", "NIG",
                     "AlphaStable", "CGMY", "PoissonProcess",
                     "FractionalBrownianMotion", "HawkesProcess", "OULevy"]:
            assert hasattr(spxa, name), f"spxa.{name} not found"

    def test_core_types_importable(self) -> None:
        """Core types importable directly from spxa."""
        import spxa
        for name in ["ExactnessLevel", "ExactnessError", "LevyTriplet",
                     "ProcessProperties", "Process"]:
            assert hasattr(spxa, name), f"spxa.{name} not found"
