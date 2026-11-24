"""
Unit tests and numerical exactness tests for spxa.copula.

Tests:
1. Base copulas & Fréchet-Hoeffding bounds (Independence, Comonotonicity, Countermonotonicity).
2. Elliptical copulas (Gaussian, Student-t) exact rank correlations and tail dependence.
3. Archimedean copulas (Clayton, Gumbel, Frank, Joe, AMH) generators, exact tau, and tail dependence.
4. Extreme Value copulas (Galambos, Hüsler-Reiss) Pickands functions and upper tail dependence.
5. CoupledProcess joint multi-asset path simulations and basket / worst-of / spread option pricing.
6. Calibration & Model Selection (CML, Kendall's tau inversion, AIC/BIC).
7. Visualization functions (3D CDF/PDF, scatter sampling, tail dependence curves).
"""

import os
import tempfile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from spxa import (
    AliMikhailHaqCopula,
    BrownianMotion,
    ClaytonCopula,
    ComonotonicityCopula,
    CountermonotonicityCopula,
    CoupledProcess,
    FrankCopula,
    GalambosCopula,
    GaussianCopula,
    GumbelCopula,
    HuslerReissCopula,
    IndependenceCopula,
    JoeCopula,
    MertonJumpDiffusion,
    StudentTCopula,
    fit_copula,
    plot_copula_cdf,
    plot_copula_pdf,
    plot_copula_samples,
    plot_tail_dependence_curves,
    pseudo_observations,
    select_best_copula,
)


class TestBaseCopulas:
    """Test basic copula properties and Fréchet-Hoeffding bounds."""

    def test_independence_copula_properties(self):
        indep = IndependenceCopula(dim=2)
        assert indep.dim == 2
        assert indep.parameters == {}

        # C(u, v) = u * v
        assert np.isclose(indep.cdf([0.4, 0.7]), 0.28)
        assert np.isclose(indep.pdf([0.4, 0.7]), 1.0)
        assert indep.kendall_tau() == 0.0
        assert indep.spearman_rho() == 0.0
        assert indep.tail_dependence() == (0.0, 0.0)

        # Sampling
        samples = indep.sample(100)
        assert samples.shape == (100, 2)
        assert np.all(samples >= 0.0) and np.all(samples <= 1.0)

    def test_comonotonicity_copula_bounds(self):
        como = ComonotonicityCopula(dim=2)
        # C(u, v) = min(u, v)
        assert np.isclose(como.cdf([0.3, 0.8]), 0.3)
        assert como.kendall_tau() == 1.0
        assert como.spearman_rho() == 1.0
        assert como.tail_dependence() == (1.0, 1.0)

        samples = como.sample(50)
        assert np.allclose(samples[:, 0], samples[:, 1])

    def test_countermonotonicity_copula_bounds(self):
        counter = CountermonotonicityCopula()
        assert np.isclose(counter.cdf([0.6, 0.7]), 0.3)
        assert np.isclose(counter.cdf([0.3, 0.4]), 0.0)
        assert counter.kendall_tau() == -1.0
        assert counter.spearman_rho() == -1.0
        assert counter.tail_dependence() == (0.0, 0.0)

        samples = counter.sample(50)
        assert np.allclose(samples[:, 0] + samples[:, 1], 1.0)

    def test_frechet_hoeffding_inequality(self):
        cop = GaussianCopula(corr=0.4)
        rng = np.random.default_rng(42)
        pts = rng.uniform(0.01, 0.99, size=(50, 2))

        c_vals = np.asarray(cop.cdf(pts), dtype=float)
        w_vals, m_vals = cop.frechet_hoeffding_bounds(pts)

        # W(u) <= C(u) <= M(u)
        assert np.all(c_vals >= w_vals - 1e-12)
        assert np.all(c_vals <= m_vals + 1e-12)

    def test_rosenblatt_transform(self):
        cop = GaussianCopula(corr=0.6)
        u = np.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]])
        w = cop.rosenblatt_transform(u)
        assert w.shape == (3, 2)
        assert np.all(w >= 0.0) and np.all(w <= 1.0)

        # Inverse Rosenblatt
        u_rec = cop.inverse_rosenblatt_transform(w)
        assert np.allclose(u_rec, u, atol=1e-5)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            GaussianCopula(corr=1.5)
        with pytest.raises(ValueError):
            StudentTCopula(df=-1.0)
        with pytest.raises(ValueError):
            ClaytonCopula(theta=-0.5)
        with pytest.raises(ValueError):
            GumbelCopula(theta=0.5)
        with pytest.raises(ValueError):
            FrankCopula(theta=0.0)
        with pytest.raises(ValueError):
            JoeCopula(theta=0.5)
        with pytest.raises(ValueError):
            AliMikhailHaqCopula(theta=1.5)
        with pytest.raises(ValueError):
            GalambosCopula(theta=-1.0)
        with pytest.raises(ValueError):
            HuslerReissCopula(theta=-1.0)


class TestEllipticalCopulas:
    """Test Gaussian and Student-t copulas."""

    def test_gaussian_copula_exact_tau_and_rho(self):
        rho = 0.6
        cop = GaussianCopula(corr=rho)
        assert np.isclose(cop.rho, rho)

        # Exact Kendall's tau: 2/pi * arcsin(rho)
        expected_tau = 2.0 / np.pi * np.arcsin(rho)
        assert np.isclose(cop.kendall_tau(), expected_tau)

        # Exact Spearman's rho: 6/pi * arcsin(rho/2)
        expected_rho_s = 6.0 / np.pi * np.arcsin(rho / 2.0)
        assert np.isclose(cop.spearman_rho(), expected_rho_s)

        # Asymptotic tail independence
        assert cop.tail_dependence() == (0.0, 0.0)

    def test_gaussian_conditional_cdf_inversion(self):
        cop = GaussianCopula(corr=0.5)
        u1 = 0.35
        u2 = 0.65

        # Forward conditional CDF C_{2|1}(u2 | u1)
        p = float(cop.conditional_cdf(u2, u1))
        assert 0.0 < p < 1.0

        # Inverse conditional CDF
        u2_recovered = float(cop.inverse_conditional_cdf(p, u1))
        assert np.isclose(u2_recovered, u2, atol=1e-6)

    def test_gaussian_multivariate(self):
        r_mat = np.array(
            [
                [1.0, 0.5, 0.2],
                [0.5, 1.0, 0.4],
                [0.2, 0.4, 1.0],
            ]
        )
        cop = GaussianCopula(corr=r_mat, dim=3)
        assert cop.dim == 3

        samples = cop.sample(2000, rng=np.random.default_rng(42))
        assert samples.shape == (2000, 3)
        assert np.all(samples >= 0.0) and np.all(samples <= 1.0)

        # Empirical normal scores correlation matches R
        z = stats.norm.ppf(samples)
        emp_r = np.corrcoef(z, rowvar=False)
        assert np.allclose(emp_r, r_mat, atol=0.08)

    def test_student_t_copula_tail_dependence(self):
        nu = 4.0
        rho = 0.5
        cop = StudentTCopula(df=nu, corr=rho)

        # Exact Kendall's tau matches Gaussian
        expected_tau = 2.0 / np.pi * np.arcsin(rho)
        assert np.isclose(cop.kendall_tau(), expected_tau)

        # Tail dependence is strictly positive!
        arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
        expected_lambda = float(2.0 * stats.t.cdf(arg, df=nu + 1.0))
        lambda_l, lambda_u = cop.tail_dependence()

        assert np.isclose(lambda_l, expected_lambda)
        assert np.isclose(lambda_u, expected_lambda)
        assert lambda_l > 0.2  # Heavy tail co-movement

    def test_student_t_conditional_inversion(self):
        cop = StudentTCopula(df=5.0, corr=0.4)
        u1 = 0.3
        u2 = 0.7

        p = float(cop.conditional_cdf(u2, u1))
        u2_rec = float(cop.inverse_conditional_cdf(p, u1))
        assert np.isclose(u2_rec, u2, atol=1e-5)


class TestArchimedeanCopulas:
    """Test Archimedean families: Clayton, Gumbel, Frank, Joe, AMH."""

    def test_clayton_exact_properties(self):
        theta = 2.0
        clayton = ClaytonCopula(theta=theta)

        # Exact Kendall's tau: theta / (theta + 2)
        assert np.isclose(clayton.kendall_tau(), 0.5)

        # Lower tail dependence: 2^(-1/theta), upper = 0
        lambda_l, lambda_u = clayton.tail_dependence()
        assert np.isclose(lambda_l, 2.0 ** (-0.5))
        assert lambda_u == 0.0

        # Exact conditional inversion
        u1 = 0.4
        u2 = 0.6
        p = float(clayton.conditional_cdf(u2, u1))
        u2_inv = float(clayton.inverse_conditional_cdf(p, u1))
        assert np.isclose(u2_inv, u2, atol=1e-6)

        # Sampling
        samples = clayton.sample(1000, rng=np.random.default_rng(42))
        assert samples.shape == (1000, 2)
        # Check empirical tau near theoretical 0.5
        tau_emp = stats.kendalltau(samples[:, 0], samples[:, 1])[0]
        assert np.isclose(tau_emp, 0.5, atol=0.08)

    def test_gumbel_exact_properties(self):
        theta = 2.5
        gumbel = GumbelCopula(theta=theta)

        # Exact Kendall's tau: 1 - 1/theta
        assert np.isclose(gumbel.kendall_tau(), 1.0 - 1.0 / 2.5)

        # Upper tail dependence: 2 - 2^(1/theta), lower = 0
        lambda_l, lambda_u = gumbel.tail_dependence()
        assert lambda_l == 0.0
        assert np.isclose(lambda_u, 2.0 - 2.0 ** (1.0 / 2.5))

        # Sampling
        samples = gumbel.sample(1000, rng=np.random.default_rng(42))
        assert samples.shape == (1000, 2)
        tau_emp = stats.kendalltau(samples[:, 0], samples[:, 1])[0]
        assert np.isclose(tau_emp, 0.6, atol=0.08)

    def test_frank_exact_properties(self):
        theta = 3.5
        frank = FrankCopula(theta=theta)

        # Zero tail dependence
        assert frank.tail_dependence() == (0.0, 0.0)

        # Exact conditional inversion
        u1 = 0.45
        u2 = 0.75
        p = float(frank.conditional_cdf(u2, u1))
        u2_inv = float(frank.inverse_conditional_cdf(p, u1))
        assert np.isclose(u2_inv, u2, atol=1e-5)

        # Sampling
        samples = frank.sample(300)
        assert samples.shape == (300, 2)

    def test_joe_copula_properties(self):
        theta = 2.0
        joe = JoeCopula(theta=theta)
        lambda_l, lambda_u = joe.tail_dependence()
        assert lambda_l == 0.0
        assert np.isclose(lambda_u, 2.0 - 2.0 ** (0.5))

        assert 0.0 < joe.kendall_tau() < 1.0

        # Conditional cdf and inverse
        u1, u2 = 0.4, 0.7
        p = float(joe.conditional_cdf(u2, u1))
        u2_inv = float(joe.inverse_conditional_cdf(p, u1))
        assert np.isclose(u2_inv, u2, atol=1e-4)

        # Sampling
        samples = joe.sample(100)
        assert samples.shape == (100, 2)

    def test_amh_copula_properties(self):
        theta = 0.5
        amh = AliMikhailHaqCopula(theta=theta)
        assert amh.tail_dependence() == (0.0, 0.0)
        assert 0.0 < amh.kendall_tau() < 0.5

        u1, u2 = 0.35, 0.65
        p = float(amh.conditional_cdf(u2, u1))
        u2_inv = float(amh.inverse_conditional_cdf(p, u1))
        assert np.isclose(u2_inv, u2, atol=1e-4)

        samples = amh.sample(100)
        assert samples.shape == (100, 2)


class TestExtremeValueCopulas:
    """Test Pickands representation, Galambos, and Hüsler-Reiss."""

    def test_galambos_pickands_and_tail_dependence(self):
        theta = 1.5
        gal = GalambosCopula(theta=theta)

        # Pickands A(0) = A(1) = 1
        assert np.isclose(gal.pickands_a(0.0), 1.0)
        assert np.isclose(gal.pickands_a(1.0), 1.0)
        assert gal.pickands_a(0.5) < 1.0
        assert gal.pickands_a(0.5) >= 0.5

        # Upper tail dependence: 2^(-1/theta)
        lambda_l, lambda_u = gal.tail_dependence()
        assert lambda_l == 0.0
        assert np.isclose(lambda_u, 2.0 ** (-1.0 / 1.5))

        # Density and sampling
        pdf_val = float(gal.pdf([0.4, 0.6]))
        assert pdf_val > 0.0
        samples = gal.sample(100)
        assert samples.shape == (100, 2)
        assert np.all(samples >= 0.0) and np.all(samples <= 1.0)

        # Rank correlations
        tau = gal.kendall_tau()
        assert 0.0 < tau < 1.0
        rho = gal.spearman_rho()
        assert 0.0 < rho < 1.0

    def test_husler_reiss_properties(self):
        theta = 2.0
        hr = HuslerReissCopula(theta=theta)
        assert np.isclose(hr.pickands_a(0.0), 1.0)
        assert np.isclose(hr.pickands_a(1.0), 1.0)

        lambda_l, lambda_u = hr.tail_dependence()
        assert lambda_l == 0.0
        assert np.isclose(lambda_u, 2.0 * stats.norm.cdf(-1.0 / 2.0))

        pdf_val = float(hr.pdf([0.3, 0.7]))
        assert pdf_val > 0.0
        samples = hr.sample(100)
        assert samples.shape == (100, 2)


class TestCoupledProcess:
    """Test CoupledProcess multi-asset simulation and option pricing."""

    def test_coupled_brownian_paths(self):
        bm1 = BrownianMotion(mu=0.05, sigma=0.2)
        bm2 = BrownianMotion(mu=0.08, sigma=0.25)
        copula = GaussianCopula(corr=0.7)

        coupled = CoupledProcess([bm1, bm2], copula=copula)
        assert coupled.dim == 2
        assert "CoupledProcess" in repr(coupled)

        n = 2_000
        paths = coupled.simulate(n_steps=40, n_paths=n, T=1.0, rng=np.random.default_rng(7))
        assert paths.shape == (n, 41, 2)

        # The Gaussian copula with normal marginals gives jointly normal steps with
        # correlation 0.7, so the increments over [0, T] have correlation 0.7. The sample
        # correlation has standard error (1 - rho^2)/sqrt(n); allow five of them. The run
        # is seeded: unseeded, 200 paths and atol=0.1 failed about 1% of CI runs.
        increments_1 = paths[:, -1, 0] - paths[:, 0, 0]
        increments_2 = paths[:, -1, 1] - paths[:, 0, 1]
        emp_corr = np.corrcoef(increments_1, increments_2)[0, 1]
        assert np.isclose(emp_corr, 0.7, atol=5 * (1 - 0.7**2) / np.sqrt(n))

    def test_coupled_jump_diffusion_and_options(self):
        merton1 = MertonJumpDiffusion(rate=1.0, jump_mean=-0.05, jump_std=0.1)
        merton2 = MertonJumpDiffusion(rate=1.0, jump_mean=-0.05, jump_std=0.1)
        copula = ClaytonCopula(theta=2.0)

        coupled = CoupledProcess([merton1, merton2], copula=copula)
        paths = coupled.simulate(n_steps=20, n_paths=100, T=0.5, rng=np.random.default_rng(1))
        assert paths.shape == (100, 21, 2)

        # worst-of <= basket <= best-of holds path by path; with common random numbers
        # (one seed for all three) it holds exactly for the estimators too
        def crn():
            return np.random.default_rng(2)

        b_call = coupled.basket_call_price(
            weights=[0.5, 0.5], strike=1.0, T=0.5, n_paths=5000, rng=crn()
        )
        w_call = coupled.worst_of_call_price(strike=1.0, T=0.5, n_paths=5000, rng=crn())
        best_call = coupled.best_of_call_price(strike=1.0, T=0.5, n_paths=5000, rng=crn())
        assert w_call <= b_call <= best_call

        # Spread option pricing
        spread_call = coupled.spread_option_price(
            strike=0.0, T=0.5, n_paths=3000, rng=np.random.default_rng(3)
        )
        assert spread_call >= 0.0


class TestCopulaFitting:
    """Test calibration, pseudo-observations, and model selection."""

    def test_pseudo_observations(self):
        data = np.array(
            [
                [10.0, 50.0],
                [12.0, 45.0],
                [15.0, 60.0],
            ]
        )
        u = pseudo_observations(data)
        assert u.shape == (3, 2)
        assert np.all(u > 0.0) and np.all(u < 1.0)
        # Check rank ordering
        assert u[0, 0] < u[1, 0] < u[2, 0]

    def test_fit_clayton_itau_and_cml(self):
        true_theta = 2.0
        clay = ClaytonCopula(theta=true_theta)
        samples = clay.sample(1500, rng=np.random.default_rng(123))

        # Inversion of tau
        fitted_itau = fit_copula(samples, ClaytonCopula, method="itau")
        assert np.isclose(fitted_itau.theta, true_theta, atol=0.25)

        # CML
        fitted_cml = fit_copula(samples, ClaytonCopula, method="cml")
        assert np.isclose(fitted_cml.theta, true_theta, atol=0.25)

    def test_fit_gaussian_copula(self):
        true_rho = 0.65
        gauss = GaussianCopula(corr=true_rho)
        samples = gauss.sample(1500, rng=np.random.default_rng(123))

        fitted = fit_copula(samples, GaussianCopula, method="itau")
        assert np.isclose(fitted.rho, true_rho, atol=0.1)

    def test_select_best_copula(self):
        # Generate Clayton data (lower tail dependence)
        clay = ClaytonCopula(theta=3.0)
        samples = clay.sample(1000, rng=np.random.default_rng(99))

        best, report = select_best_copula(
            samples, candidate_families=[GaussianCopula, ClaytonCopula, FrankCopula]
        )
        assert isinstance(best, ClaytonCopula)
        assert report["best_model"] == "ClaytonCopula"


class TestCopulaVisualization:
    """Test 3D surface, scatter, and tail dependence plots."""

    def test_plot_copula_cdf_save(self):
        cop = ClaytonCopula(theta=2.0)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig, ax = plot_copula_cdf(cop, n_points=20, save_path=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_plot_copula_pdf_save(self):
        cop = GaussianCopula(corr=0.5)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig, ax = plot_copula_pdf(cop, n_points=20, save_path=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_plot_copula_samples_save(self):
        cop = StudentTCopula(df=4.0, corr=0.6)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig, axes = plot_copula_samples(cop, n_samples=200, save_path=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            assert len(axes) == 2
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_plot_tail_dependence_curves_save(self):
        copulas = {
            "Clayton": ClaytonCopula(theta=2.0),
            "Gaussian": GaussianCopula(corr=0.5),
        }
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig, axes = plot_tail_dependence_curves(copulas, n_points=25, save_path=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            assert len(axes) == 2
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
