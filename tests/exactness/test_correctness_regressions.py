"""Independent regressions for the October 2026 consistency review.

References
----------
Sato (1999), Theorems 8.1, 25.3, 30.1 and linear-image triplets.
Tweedie (1957), inverse-Gaussian cumulant generating function.
Carr et al. (2002), CGMY characteristic exponent.
Schoutens & Teugels (1998), Meixner characteristic function.
Nolan (2020), S1 stable parameterization; Chambers et al. (1976).
Applebaum (2009), Chapter 4, Ito isometry.
Barndorff-Nielsen & Shephard (2001), OU time change.
Glasserman (2003), Section 3.2, conditional Gaussian/Gamma bridges.
Carr & Madan (1999); Black & Scholes (1973).

Reference paths deliberately include SciPy distribution implementations,
independent differentiation, quadrature, law identities and seeded simulation.
"""

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import invgauss, norm, norminvgauss

from spxa import (
    CGMY,
    NIG,
    AlphaStable,
    BrownianMotion,
    FractionalBrownianMotion,
    GammaProcess,
    HawkesProcess,
    InverseGaussianProcess,
    MarkovModulatedPoissonProcess,
    MeixnerProcess,
    NegativeBinomialProcess,
    OULevy,
    PoissonProcess,
    TemperedStable,
    independent_levy_vector,
)
from spxa.analytics import (
    cumulant_table,
    european_call_price,
    european_put_price,
    hellinger_distance,
    implied_volatility,
    kl_divergence_mc,
    wasserstein2_distance,
)
from spxa.core import ExactnessError, ExactnessLevel, SpxaDegradationWarning
from spxa.ops import StochasticIntegral, multiply
from spxa.ops.subordination import char_exp_subordinated
from spxa.sim import brownian_bridge_interpolate, convergence_order_estimate, gamma_bridge
from spxa.sim.exact import sample_alpha_stable


@pytest.mark.exactness
@pytest.mark.parametrize(
    "process",
    [
        GammaProcess(2, 3),
        InverseGaussianProcess(1, 2),
        NIG(3, 0.7, 1, 0.1),
        MeixnerProcess(0.8, -0.7, 0.6, 0.1),
        CGMY(1, 2, 4, 0.5),
        CGMY(1, 2, 4, 1),
        CGMY(1, 2, 4, 1.5),
        PoissonProcess(2),
        NegativeBinomialProcess(2, 0.4),
    ],
)
def test_triplet_matches_named_law(process):
    """Sato 8.1/25.3: a triplet and named law must describe the same distribution."""
    np.testing.assert_allclose(
        process.triplet.char_exp(np.array([-0.7, 0, 0.4, 1.0])),
        process.char_func(np.array([-0.7, 0, 0.4, 1.0])),
        atol=2e-7,
        rtol=2e-7,
    )
    for n in (1, 2):
        assert process.triplet.cumulant(n) == pytest.approx(
            process.cumulants(2)[n], rel=2e-7, abs=1e-8
        )


@pytest.mark.exactness
@pytest.mark.parametrize(
    "process",
    [
        GammaProcess(2, 3),
        InverseGaussianProcess(1, 2),
        NIG(3, 0.5, 1),
        PoissonProcess(2, 0.5),
        PoissonProcess(2, 1),
        PoissonProcess(2, 2),
        NegativeBinomialProcess(2, 0.4),
        TemperedStable(0.5, 1, 2),
    ],
)
@pytest.mark.parametrize("c", [-2.0, -1.0, -0.5, 0.5, 2.0])
def test_scalar_image_cumulants(process, c):
    """Sato 11.10: kappa_n(cX)=c^n*kappa_n(X), including boundary atoms."""
    for n in (1, 2, 3):
        assert (c * process).triplet.cumulant(n) == pytest.approx(
            c**n * process.cumulants(3)[n], rel=3e-7, abs=1e-8
        )


@pytest.mark.exactness
@pytest.mark.parametrize(
    "process", [GammaProcess(2, 3), PoissonProcess(2), NegativeBinomialProcess(2, 0.4)]
)
def test_reflected_positive_jumps_have_no_right_tail(process):
    assert (-process).triplet.nu.tail(0.5) == 0


@pytest.mark.exactness
def test_ig_against_scipy_reference():
    """SciPy Wald parameter mapping: invgauss(mu/lambda,scale=lambda)."""
    p = InverseGaussianProcess(1.4, 2.3)
    reference = invgauss(mu=1.4 / 2.3, scale=2.3)
    assert p.cumulants(2)[2] == pytest.approx(reference.var())
    t = 0.6
    x = p.simulate(1, 60000, t, np.random.default_rng(61))[:, -1]
    assert x.mean() == pytest.approx(reference.mean() * t, rel=0.015)
    assert x.var() == pytest.approx(reference.var() * t, rel=0.035)
    for s in (0.1, 1, 4):
        assert p.bernstein_function(s) == pytest.approx(-np.log(complex(p.char_func(1j * s)).real))


@pytest.mark.exactness
def test_nig_density_second_moment_and_scipy():
    p = NIG(3, 0.5, 1.2, 0.2)
    reference = norminvgauss(a=3 * 1.2, b=0.5 * 1.2, loc=0.2, scale=1.2)
    assert p.cumulants(2)[1] == pytest.approx(reference.mean())
    assert p.cumulants(2)[2] == pytest.approx(reference.var())
    value = sum(
        quad(lambda x: x * x * p.triplet.nu.density(np.array([x]))[0], lo, hi)[0]
        for lo, hi in [(-np.inf, 0), (0, np.inf)]
    )
    assert value == pytest.approx(reference.var(), rel=1e-7)


@pytest.mark.exactness
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (1.0, 0.5), (1.5, 0.4)])
def test_stable_triplet_normalization(alpha, beta):
    p = AlphaStable(alpha, beta, 0.8, 0.2)
    np.testing.assert_allclose(p.triplet.char_exp(0.7), p.char_func(0.7), atol=3e-5)
    if alpha == 1 and beta == 0:
        assert p.triplet.nu.density(np.array([1.0]))[0] == pytest.approx(0.8 / np.pi)


@pytest.mark.exactness
def test_s1_alpha_one_time_scaling_in_both_samplers():
    """S1 scaling includes the logarithmic location correction (Nolan 2020)."""
    p = AlphaStable(1, 0.7, 0.8, 0.2)
    values = p.simulate(1, 80000, 0.4, np.random.default_rng(62))[:, -1]
    assert np.exp(1j * values).mean() == pytest.approx(complex(p.char_func(1, 0.4)), abs=0.012)
    values = sample_alpha_stable(1, 0.7, 0.8, 0.4, 1, 80000, np.random.default_rng(62))[:, 0]
    assert np.exp(1j * values).mean() == pytest.approx(
        complex(AlphaStable(1, 0.7, 0.8).char_func(1, 0.4)), abs=0.012
    )


@pytest.mark.exactness
@pytest.mark.parametrize("y", [0.0, 1.0, 1.5, 1 - 1e-7, 1 + 1e-7])
def test_cgmy_removable_poles_and_finite_tempered_mean(y):
    p = CGMY(1, 2, 4, y)
    assert complex(p.char_func(0)) == pytest.approx(1)
    assert np.isfinite(p.cumulants(4)[1])
    assert p.triplet.cumulant(1) == pytest.approx(p.cumulants(1)[1], abs=1e-7)
    np.testing.assert_allclose(
        p.char_func(1, 0.4) * p.char_func(1, 0.6), p.char_func(1), atol=1e-10
    )


@pytest.mark.exactness
def test_meixner_cumulants_by_independent_high_precision_differentiation():
    p = MeixnerProcess(0.7, -0.8, 0.4, 0.1)
    with mp.workdps(55):
        a, b, d, m = map(mp.mpf, (0.7, -0.8, 0.4, 0.1))

        def cgf(s):
            return m * s + 2 * d * (mp.log(mp.cos(b / 2)) - mp.log(mp.cos((b + a * s) / 2)))

        for n in range(1, 9):
            assert p.cumulants(n)[n] == pytest.approx(float(mp.diff(cgf, 0, n)), rel=1e-10)
    assert complex(p.char_func(2, t=0)) == 1
    np.testing.assert_allclose(
        p.char_func(2, 0.3) * p.char_func(2, 0.7), p.char_func(2), atol=1e-12
    )


@pytest.mark.exactness
def test_subordination_branch_and_known_triplet():
    """Sato 30.1: compose unwrapped exponents, not principal logs of CFs."""
    b = BrownianMotion(20, 0.7)
    g = GammaProcess(2, 3)
    z = b @ g
    u = 2.0
    t = 0.7
    expected = (3 / (3 - (20j * u - 0.5 * 0.7**2 * u * u))) ** (2 * t)
    assert complex(z.char_func(u, t)) == pytest.approx(expected)
    assert complex(char_exp_subordinated(b, g, u, t)) == pytest.approx(expected)
    assert z.exactness == ExactnessLevel.EXACT
    assert complex(z.triplet.char_exp(0.2)) == pytest.approx(complex(z.char_func(0.2)), rel=1e-7)


@pytest.mark.exactness
def test_generic_subordination_has_honest_triplet_limit_and_moments():
    x, t = GammaProcess(2, 3), GammaProcess(4, 5)
    with pytest.warns(SpxaDegradationWarning, match="explicit triplet"):
        z = x @ t
    assert z.exactness == ExactnessLevel.MOMENT_PROPAGATION
    with pytest.raises(ExactnessError):
        _ = z.triplet
    assert z.cumulants(2)[1] == pytest.approx(2 / 3 * 4 / 5)
    assert z.cumulants(2)[2] == pytest.approx(4 / 5 * 2 / 9 + (2 / 3) ** 2 * 4 / 25)
    values = z.simulate(1, 15000, rng=np.random.default_rng(63))[:, -1]
    assert values.mean() == pytest.approx(z.cumulants(1)[1], rel=0.035)


@pytest.mark.exactness
@pytest.mark.parametrize(
    "clock", [HawkesProcess(1, 0.5, 2), OULevy(2), FractionalBrownianMotion(0.7)]
)
def test_non_levy_clocks_are_rejected(clock):
    with pytest.raises(ValueError, match="subordinator"):
        _ = BrownianMotion() @ clock


@pytest.mark.exactness
def test_time_aware_moment_propagation():
    f = FractionalBrownianMotion(0.7)
    with pytest.warns(SpxaDegradationWarning):
        z = 2 * f + BrownianMotion()
    assert cumulant_table(z, t=2)["variance"] == pytest.approx(4 * 2**1.4 + 2)
    with pytest.warns(SpxaDegradationWarning):
        product = multiply(BrownianMotion(), BrownianMotion())
    assert cumulant_table(product, t=3)["variance"] == pytest.approx(9)
    p = MarkovModulatedPoissonProcess([[-1, 1], [2, -2]], [1, 7])
    for t in (0.3, 2):
        mean, var = p.count_moments(t)
        table = cumulant_table(p, t=t)
        assert table["mean"] == pytest.approx(mean)
        assert table["variance"] == pytest.approx(var)


@pytest.mark.exactness
def test_hawkes_finite_horizon_mean_and_poisson_limit():
    """Mean from the independent linear ODE for expected intensity, empty history."""
    p = HawkesProcess(1, 0.5, 2)
    for t in (0.1, 1, 3):
        decay = 2 - 0.5
        expected = 2 / decay * t - 0.5 / decay**2 * (1 - np.exp(-decay * t))
        assert p.cumulants_at(2, t)[1] == pytest.approx(expected)
    assert p.fano_factor == pytest.approx(1 / (1 - 0.5 / 2) ** 2)
    poisson = HawkesProcess(2, 0, 2)
    assert list(poisson.cumulants_at(4, 1.5).values()) == pytest.approx([3] * 4, abs=1e-10)


@pytest.mark.exactness
def test_immutable_parameters_and_defensive_properties():
    p = BrownianMotion(sigma=1)
    z = 2 * p
    with pytest.raises(AttributeError):
        p.sigma = 3
    assert z.triplet.sigma_sq == 4
    properties = z.properties
    properties.has_finite_variance = False
    assert z.properties.has_finite_variance
    with pytest.raises(AttributeError):
        z._node.name = "different"
    a = np.eye(2)
    from spxa import MultivariateBrownianMotion

    vector = MultivariateBrownianMotion(np.zeros(2), a)
    a[0, 0] = 9
    assert vector.sigma_mat[0, 0] == 1
    with pytest.raises(ValueError):
        vector.sigma_mat.setflags(write=True)


@pytest.mark.exactness
@pytest.mark.parametrize(
    "f,expected",
    [(lambda x: 0.0, 0.0), (lambda x: 2.0, 4.0), (lambda x: x, 0.5), (lambda x: x * x, 1.0)],
)
def test_integral_uses_integrand_and_time(f, expected):
    b = BrownianMotion()
    with pytest.warns(SpxaDegradationWarning):
        z = StochasticIntegral(b, b, f)
    assert z.cumulants(2)[2] == pytest.approx(expected, abs=1e-8)


@pytest.mark.exactness
def test_same_object_integral_shares_brownian_path():
    """Discrete Ito identity sum W_k*dW_k = (W_T^2-sum dW_k^2)/2."""
    b = BrownianMotion()
    with pytest.warns(SpxaDegradationWarning):
        z = StochasticIntegral(b, b, lambda x: x)
    path = b.simulate(32, 20, rng=np.random.default_rng(64))
    values = z.simulate(32, 20, rng=np.random.default_rng(64))[:, -1]
    expected = (path[:, -1] ** 2 - (np.diff(path) ** 2).sum(axis=1)) / 2
    np.testing.assert_allclose(values, expected, atol=1e-14)


@pytest.mark.exactness
def test_ou_transient_and_grid_refinement():
    """Integrate exp(-lambda*(t-s)) against dL_(lambda*s), BNS (2001)."""
    p = OULevy(2, GammaProcess(1, 2), v0=0.3)
    expected = 0.3 * np.exp(-2) + 0.5 * (1 - np.exp(-2))
    assert p.cumulants_at(2, 1)[1] == pytest.approx(expected)
    assert p.stationary_variance == pytest.approx(0.25 / 2)
    for n in (10, 100):
        values = p.simulate(n, 30000, rng=np.random.default_rng(65))[:, -1]
        assert values.mean() == pytest.approx(expected, rel=0.025)
    assert p.cumulants_at(2, 0) == {1: 0.3, 2: 0.0}
    assert not p.properties.is_subordinator


@pytest.mark.exactness
def test_directional_jump_projection_and_linear_transforms():
    p = independent_levy_vector(GammaProcess(1, 2), GammaProcess(2, 2))
    triplet = p.multivariate_triplet()
    assert triplet.cumulant(2, np.array([1.0, 0.0])) == pytest.approx(0.25)
    direction = np.array([2.0, -3.0])
    assert triplet.cumulant(1, direction) == pytest.approx(2 * 0.5 - 3)
    assert triplet.cumulant(2, direction) == pytest.approx(4 * 0.25 + 9 * 0.5)
    A = np.array([[2.0, -3.0], [1.0, 0.5]])
    B = np.array([[0.5, -1.0]])
    assert triplet.transform(A).transform(B).cumulant(2, np.ones(1)) == pytest.approx(
        triplet.cumulant(2, (B @ A)[0])
    )
    assert triplet.transform(A).cumulant(1, np.array([1.0, 0.0])) == pytest.approx(-2.0)


@pytest.mark.exactness
def test_bridge_joint_covariances():
    """Glasserman 3.2: conditional Gaussian covariance and Dirichlet Gamma increments."""
    values = brownian_bridge_interpolate(
        np.zeros((60000, 2)),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.25, 0.75, 1.0]),
        rng=np.random.default_rng(66),
    )
    assert np.cov(values[:, 1], values[:, 2])[0, 1] == pytest.approx(0.0625, abs=0.003)
    _, values = gamma_bridge(
        2, 4, x_end=1, n_points=3, n_paths=60000, rng=np.random.default_rng(67)
    )
    assert values[:, 2].var() == pytest.approx(1 / 12, rel=0.025)
    assert np.cov(values[:, 1], values[:, 3])[0, 1] == pytest.approx(1 / 48, rel=0.04)
    assert np.all(np.diff(values, axis=1) >= 0)


@pytest.mark.exactness
def test_strong_errors_use_coupled_driver():
    result = convergence_order_estimate(
        BrownianMotion(),
        lambda z, t: 0.0,
        lambda z, t: 1.0,
        n_paths_ref=40,
        step_sizes=(10, 20, 40),
    )
    assert max(result["errors"]) < 1e-13


@pytest.mark.exactness
@pytest.mark.parametrize("sigma", [0.2, 1.0, 4.0])
def test_hellinger_normal_reference_and_self(sigma):
    p = BrownianMotion(sigma=sigma)
    assert hellinger_distance(p, BrownianMotion(sigma=sigma)) == pytest.approx(0.0, abs=1e-12)
    assert hellinger_distance(p, BrownianMotion(mu=2 * sigma, sigma=sigma)) == pytest.approx(
        1 - np.exp(-0.5)
    )


@pytest.mark.exactness
def test_discrete_and_singular_hellinger():
    assert hellinger_distance(PoissonProcess(1), PoissonProcess(4)) == pytest.approx(
        1 - np.exp(-0.5), abs=1e-10
    )
    assert hellinger_distance(PoissonProcess(1), BrownianMotion()) == 1
    assert hellinger_distance(BrownianMotion(1, 0), BrownianMotion(2, 0)) == 1


@pytest.mark.exactness
def test_wasserstein_translation_scale_and_infinite_moment_guard():
    assert wasserstein2_distance(BrownianMotion(100, 2), BrownianMotion(-100, 3)) == pytest.approx(
        np.sqrt(40001)
    )
    with pytest.raises(ValueError, match="second moments"):
        wasserstein2_distance(AlphaStable(1), BrownianMotion())


@pytest.mark.exactness
def test_kl_known_normal_and_singularity():
    p, q = BrownianMotion(), BrownianMotion(1, 2)
    expected = np.log(2) + (1 + 1) / 8 - 0.5
    assert kl_divergence_mc(p, q, n_samples=25000, rng=np.random.default_rng(68)) == pytest.approx(
        expected, abs=0.02
    )
    assert np.isinf(kl_divergence_mc(PoissonProcess(1), q))


@pytest.mark.exactness
@pytest.mark.parametrize("t", [0.3, 1.0, 2.0])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_fourier_prices_against_black_scholes(t, strike):
    sigma, r, spot = 0.2, 0.05, 100.0
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call = spot * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)
    computed = european_call_price(BrownianMotion(sigma=sigma), spot, strike, r, t)
    assert computed == pytest.approx(call, abs=0.006)
    put = european_put_price(BrownianMotion(sigma=sigma), spot, strike, r, t)
    assert put == pytest.approx(call - spot + strike * np.exp(-r * t), abs=0.006)
    assert implied_volatility(call, spot, strike, r, t) == pytest.approx(sigma, abs=2e-5)


@pytest.mark.exactness
def test_tempered_stable_laplace_reference_and_sampler_moments():
    """Exponential tilting: normalized Laplace exponent a*((lambda+s)^alpha-lambda^alpha)."""
    p = TemperedStable(0.6, 0.8, 2)
    values = p.simulate(1, 70000, 0.4, np.random.default_rng(69))[:, -1]
    assert values.mean() == pytest.approx(0.4 * p.cumulants(2)[1], rel=0.02)
    assert values.var() == pytest.approx(0.4 * p.cumulants(2)[2], rel=0.04)
    assert np.exp(-values).mean() == pytest.approx(complex(p.char_func(1j, 0.4)).real, abs=0.004)
