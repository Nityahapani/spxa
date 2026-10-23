"""Independent regression oracles for the six families added after 78cf452.

References: Sato (1999), Chs. 8/11/25/30; Cox--Ingersoll--Ross (1985);
Uhlenbeck--Ornstein (1930); Heston (1993), Albrecher et al. (2007);
Revuz--Yor (1999), Ch. XI; Schroder (1989); Merton (1976).

These tests exercise laws, not just mutually agreeing library methods. In
particular they use high-precision affine derivatives, independent Bessel/CDF
identities, a Fourier pricing oracle, spectral normalization and sampled ECDFs.
"""

from __future__ import annotations

import pickle
from contextlib import nullcontext
from copy import deepcopy

import mpmath as mp
import numpy as np
import pytest
from scipy import special, stats
from scipy.integrate import quad

from spxa import (
    AlphaStable,
    BrownianMotion,
    CEVProcess,
    CIRProcess,
    CompoundPoisson,
    DiscreteJump,
    DoubleExponentialJump,
    ExponentialJump,
    GammaProcess,
    HestonProcess,
    MertonJumpDiffusion,
    NormalJump,
    OrnsteinUhlenbeckProcess,
    PoissonProcess,
    UniformJump,
)
from spxa.analytics._marginals import known_marginal
from spxa.analytics.divergence import hellinger_distance
from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import SpxaDegradationWarning
from spxa.sim.exact import sample_compound_poisson

pytestmark = pytest.mark.exactness

NEW_PROCESSES = [
    CompoundPoisson(1.3, DiscreteJump([0, 0.5, 2], [0.2, 0.3, 0.5])),
    MertonJumpDiffusion(drift=0.1, sigma=0.2, rate=0.7),
    CIRProcess(1.7, 0.06, 0.2, r0=0.03),
    OrnsteinUhlenbeckProcess(1.4, 0.3, 0.2, x0=-0.1),
    HestonProcess(v0=0.03, kappa=1.8, theta=0.04, sigma_v=0.3, rho=-0.6),
    CEVProcess(mu=0.03, sigma=0.4, gamma=0.5, s0=1),
]


@pytest.mark.parametrize("process", NEW_PROCESSES)
@pytest.mark.parametrize("t", [-0.1, np.nan, np.inf])
def test_new_time_contract_rejects_invalid_horizons(process, t):
    with pytest.raises(ValueError):
        process.cumulants_at(2, t)
    if not isinstance(process, CEVProcess):
        with pytest.raises(ValueError):
            process.char_func(0.2, t)


@pytest.mark.parametrize("process", NEW_PROCESSES)
@pytest.mark.parametrize("order", [0, -1, True, 1.5])
def test_new_cumulant_orders_are_integral_and_positive(process, order):
    with pytest.raises(ValueError):
        process.cumulants_at(order, 0.7)


@pytest.mark.parametrize("process", NEW_PROCESSES)
def test_new_processes_compose_with_keyword_simulation_and_initial_states(process):
    initial = process.cumulants_at(1, 0)[1]
    scaled = -2 * process
    path = scaled.simulate(8, n_paths=3, T=0.6, rng=np.random.default_rng(182))
    assert path.shape == (3, 9)
    assert np.all(path[:, 0] == -2 * initial)
    assert np.all(np.isfinite(path))
    bm = BrownianMotion(mu=0.2, sigma=0.3)
    warning = (
        pytest.warns(SpxaDegradationWarning)
        if process.exactness != ExactnessLevel.EXACT
        else nullcontext()
    )
    with warning:
        total = process + bm
    for t in (0, 0.4, 2):
        a, b, result = process.cumulants_at(2, t), bm.cumulants_at(2, t), total.cumulants_at(2, t)
        assert result == pytest.approx({n: a[n] + b[n] for n in (1, 2)})
    paths = total.simulate(4, n_paths=2, T=0.4, rng=np.random.default_rng(183))
    assert paths.shape == (2, 5)
    assert np.all(paths[:, 0] == initial)


@pytest.mark.parametrize("process", NEW_PROCESSES[:-1])
def test_new_characteristic_functions_accept_vectors_through_composition(process):
    bm = BrownianMotion(mu=-0.1, sigma=0.2)
    warning = (
        pytest.warns(SpxaDegradationWarning)
        if process.exactness != ExactnessLevel.EXACT
        else nullcontext()
    )
    with warning:
        total = process + bm
    u = np.array([0, 0.2, 1.3, -0.5j])
    assert np.allclose(total.char_func(u, 0.7), process.char_func(u, 0.7) * bm.char_func(u, 0.7))


@pytest.mark.parametrize("scale", [-3, -0.3, 0.25, 4])
@pytest.mark.parametrize(
    "jump",
    [
        DiscreteJump([-2, 0, 0.5, 1], [0.1, 0.2, 0.3, 0.4]),
        NormalJump(2, 1e-6),
        ExponentialJump(2),
        DoubleExponentialJump(0.3, 1.4, 2.5),
        UniformJump(-0.2, 1.5),
    ],
)
def test_new_jump_laws_remain_consistent_under_triplet_images(jump, scale):
    process = CompoundPoisson(1.4, jump)
    image = scale * process
    u = np.array([-1.2, 0, 0.3, 1.7])
    reference = process.char_func(scale * u, 0.4)
    assert np.allclose(image.triplet.char_exp(u, 0.4), reference, atol=3e-10, rtol=3e-10)
    assert image.triplet.cumulant(1) == pytest.approx(scale * process.cumulants(1)[1], abs=2e-9)
    assert image.triplet.cumulant(2) == pytest.approx(scale**2 * process.cumulants(2)[2])


@pytest.mark.parametrize(
    ("rate", "jump", "kind"),
    [
        (1e16, NormalJump(0, 1e-8), "normal"),
        (1e16, DoubleExponentialJump(0.5, 1e8, 1e8), "double"),
        (3e16, UniformJump(-1e-8, 1e-8), "uniform"),
        (1e16, DiscreteJump([-1e-8, 1e-8]), "discrete"),
    ],
)
def test_small_jump_high_rate_exponents_do_not_round_to_zero(rate, jump, kind):
    process = CompoundPoisson(rate, jump)
    with mp.workdps(65):
        u, h = mp.mpf("0.7"), mp.mpf("1e-8")
        if kind == "normal":
            increment = mp.expm1(-u * u * h * h / 2)
        elif kind == "double":
            increment = -((u * h) ** 2) / (1 + (u * h) ** 2)
        elif kind == "uniform":
            increment = mp.sin(u * h) / (u * h) - 1
        else:
            increment = mp.cos(u * h) - 1
        expected = complex(mp.exp(mp.mpf(str(rate)) * increment))
    assert process.char_func(0.7) == pytest.approx(expected, rel=1e-13)
    assert process.triplet.char_exp(0.7) == pytest.approx(expected, rel=1e-13)


def test_zero_size_events_are_not_levy_atoms_and_do_not_forbid_a_clock():
    process = CompoundPoisson(5, DiscreteJump([0, 0.5, 0.5], [0.4, 0.2, 0.4]))
    assert process.triplet.nu.total_mass() == pytest.approx(3)
    assert process.triplet.nu.integrate(lambda _x: 1, 0, 0) == 0
    assert process.triplet.nu.tail(-0.1) == pytest.approx(3)
    zero = PoissonProcess(4, jump_size=0)
    assert zero.triplet.nu.total_mass() == 0
    assert zero.properties.is_subordinator and zero.properties.is_martingale
    assert CompoundPoisson(2, DiscreteJump([-3, 0, 1], [0, 0.2, 0.8])).properties.is_subordinator
    with pytest.warns(SpxaDegradationWarning):
        subordinated = BrownianMotion(mu=0.1, sigma=0.3) @ process
    u, t = 0.7, 0.8
    psi = 0.1j * u - 0.5 * 0.3**2 * u**2
    expected = np.exp(3 * t * (np.exp(0.5 * psi) - 1))
    assert subordinated.char_func(u, t) == pytest.approx(expected)


def test_discrete_jump_inputs_are_copied_and_nested_parameters_are_immutable():
    values, probabilities = np.array([-1.0, 2]), np.array([0.4, 0.6])
    jump = DiscreteJump(values, probabilities)
    process = CompoundPoisson(2, jump)
    before = process.char_func(0.7)
    values[:] = 99
    probabilities[:] = 0.5
    assert process.char_func(0.7) == before
    for array in (jump.values, jump.probs):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(AttributeError):
        jump.values = np.array([9])
    merton = MertonJumpDiffusion()
    with pytest.raises(AttributeError):
        merton.jump_distribution.std = 3


@pytest.mark.parametrize("jump", [ExponentialJump(0.7), DoubleExponentialJump(0.3, 0.7, 2)])
def test_new_complex_jump_cf_respects_true_mgf_domain(jump):
    process = CompoundPoisson(2, jump)
    with pytest.raises(ValueError, match="MGF"):
        process.char_func(-1j)
    assert process.char_func(-0.5j).real > 0
    assert process.char_func(-100j, 0) == 1


def test_invalid_jump_alias_does_not_silently_use_default_rate():
    with pytest.raises(ValueError):
        CompoundPoisson(2, "exponential", jump_rate=0)
    with pytest.raises(TypeError):
        CompoundPoisson(2, "exponential", jump_rate=1, beta=2)
    for values in ([[1, 2]], [np.nan]):
        with pytest.raises(ValueError):
            DiscreteJump(values)
    with pytest.raises(ValueError):
        DiscreteJump([1, 2], [np.nan, 0])


def test_compounding_does_not_swallow_and_retry_sampler_errors():
    calls = []

    def broken(_rng, size):
        calls.append(size)
        raise RuntimeError("deliberate sampler failure")

    with pytest.raises(RuntimeError, match="deliberate"):
        sample_compound_poisson(50, broken, 1, 2, 2, np.random.default_rng(31))
    assert len(calls) == 1
    with pytest.raises(ValueError, match="requested number"):
        sample_compound_poisson(50, lambda _g, _n: np.ones(1), 1, 2, 2, np.random.default_rng(31))


def test_uniform_near_zero_frequency_preserves_a_large_location():
    jump = UniformJump(1e8, 1e8 + 1)
    assert jump.moment(1) == 1e8 + 0.5
    expected = np.exp(1j * 1e-9 * (1e8 + 0.5)) * np.sinc(1e-9 / (2 * np.pi))
    assert jump.char_func(1e-9) == pytest.approx(expected, abs=1e-15)
    assert jump.truncation_mean(2e8) == pytest.approx(1e8 + 0.5)


@pytest.mark.parametrize("t", [0, 0.2, 1.7])
def test_cir_conditional_cumulants_match_independent_noncentral_chisquare(t):
    process = CIRProcess(1.3, 0.07, 0.25, r0=0.02)
    if t == 0:
        assert process.cumulants_at(4, t) == {1: 0.02, 2: 0, 3: 0, 4: 0}
        return
    scale = 0.25**2 * (-np.expm1(-1.3 * t)) / (4 * 1.3)
    distribution = stats.ncx2(
        4 * 1.3 * 0.07 / 0.25**2, 0.02 * np.exp(-1.3 * t) / scale, scale=scale
    )
    mean, variance, skew, kurtosis = distribution.stats(moments="mvsk")
    expected = {1: mean, 2: variance, 3: skew * variance**1.5, 4: kurtosis * variance**2}
    assert process.cumulants_at(4, t) == pytest.approx(expected, rel=2e-13)
    assert known_marginal(process, t).distribution.mean() == pytest.approx(mean)


def test_small_time_variances_and_covariances_do_not_cancel_to_zero():
    t = 1e-20
    cir = CIRProcess(2, 0.1, 0.3, r0=0.02)
    ou = OrnsteinUhlenbeckProcess(2, 1, 0.3, x0=0)
    heston = HestonProcess(v0=0.02, sigma_v=0.3)
    assert cir.conditional_variance(t) == pytest.approx(0.02 * 0.3**2 * t, rel=1e-13, abs=0)
    assert ou.two_time_covariance(t, t) == pytest.approx(0.3**2 * t, rel=1e-13, abs=0)
    assert heston.variance_var(t) == pytest.approx(0.02 * 0.3**2 * t, rel=1e-13, abs=0)
    assert heston.expected_integrated_variance(t) == pytest.approx(0.02 * t, rel=1e-13, abs=0)
    with pytest.raises(ValueError, match="MGF"):
        cir.char_func(-100j, 2)


def test_two_sided_ou_spectrum_integrates_to_stationary_covariance():
    process = OrnsteinUhlenbeckProcess(1.7, 0.1, 0.4)
    mass = 2 * quad(lambda w: process.spectral_density(w), 0, np.inf)[0]
    assert mass == pytest.approx(process.stationary_variance, rel=1e-12)
    lag = 0.8
    covariance = (
        2 * quad(lambda w: process.spectral_density(w), 0, np.inf, weight="cos", wvar=lag)[0]
    )
    assert covariance == pytest.approx(process.stationary_variance * np.exp(-1.7 * lag), abs=2e-10)


def test_new_nonzero_initial_atoms_and_gaussian_ou_hellinger():
    a = OrnsteinUhlenbeckProcess(1.2, 0.3, 0.4, x0=-0.2)
    b = OrnsteinUhlenbeckProcess(0.7, -0.1, 0.3, x0=0.4)
    assert hellinger_distance(a, b, t=0) == 1
    t = 0.6
    m1, m2 = a.conditional_mean(t), b.conditional_mean(t)
    v1, v2 = a.conditional_variance(t), b.conditional_variance(t)
    expected = 1 - np.sqrt(2 * np.sqrt(v1 * v2) / (v1 + v2)) * np.exp(
        -((m1 - m2) ** 2) / (4 * (v1 + v2))
    )
    assert hellinger_distance(a, b, t=t) == pytest.approx(expected, abs=2e-10)
    mixed = CompoundPoisson(2, NormalJump())
    with pytest.raises(ValueError, match="mixture"):
        hellinger_distance(mixed, BrownianMotion(), t=0.7, n_points=128)


@pytest.mark.parametrize("t", [0.4, 1.7])
def test_heston_ode_cumulants_match_high_precision_affine_derivatives(t):
    process = HestonProcess(v0=0.02, kappa=1.3, theta=0.07, sigma_v=0.4, rho=-0.45, mu=0.05)
    with mp.workdps(70):
        v0, k, theta, sigma, rho, mu, time = map(
            mp.mpf, map(str, [0.02, 1.3, 0.07, 0.4, -0.45, 0.05, t])
        )

        def log_mgf(z):
            a = k - rho * sigma * z
            d = mp.sqrt(a * a + sigma * sigma * (z - z * z))
            g = (a - d) / (a + d)
            decay = mp.exp(-d * time)
            D = (a - d) / (sigma * sigma) * (1 - decay) / (1 - g * decay)
            C = (
                k
                * theta
                / (sigma * sigma)
                * ((a - d) * time - 2 * mp.log((1 - g * decay) / (1 - g)))
            )
            return C + D * v0 + mu * z * time

        expected = {n: float(mp.diff(log_mgf, 0, n)) for n in range(1, 7)}
    assert process.cumulants_at(6, t) == pytest.approx(expected, rel=2e-8, abs=2e-12)
    with pytest.raises(NotImplementedError, match="budget"):
        process.cumulants_at(13, t)


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_heston_first_moment_remains_valid_when_kappa_minus_rho_sigma_is_nonpositive(rho):
    process = HestonProcess(kappa=0.5, sigma_v=1, rho=rho, mu=0.07)
    for t in (0.3, 2, 20):
        assert process.char_func(-1j, t) == pytest.approx(np.exp(0.07 * t))
        assert np.all(np.isfinite(process.char_func(np.array([0, -1j, 1]), t)))
    with pytest.raises(ValueError):
        process.cumulants_at(2, -1)


def test_heston_moment_explosions_are_not_analytic_continuations():
    process = HestonProcess(kappa=0.5, sigma_v=2, rho=0.9, mu=0.02)
    with pytest.raises(ValueError, match="moment"):
        process.char_func(-2j, 5)
    value = process.char_func(-0.5j, 5)
    assert 0 < value.real <= np.exp(0.02 * 5 / 2)


@pytest.mark.parametrize("physical_mu", [-0.2, 0, 0.3])
def test_heston_rate_dividend_override_matches_black_scholes_limit(physical_mu):
    process = HestonProcess(
        v0=0.04, kappa=2, theta=0.04, sigma_v=1e-6, rho=0, mu=physical_mu, s0=100
    )
    t, rate, dividend = 0.6, 0.03, 0.02
    d1 = (rate - dividend + 0.5 * 0.2**2) * np.sqrt(t) / 0.2
    expected = 100 * np.exp(-dividend * t) * stats.norm.cdf(d1) - 100 * np.exp(
        -rate * t
    ) * stats.norm.cdf(d1 - 0.2 * np.sqrt(t))
    assert process.call_price(100, T=t, r=rate, q=dividend) == pytest.approx(expected, abs=2e-8)


@pytest.mark.parametrize("gamma", [0, 0.2, 0.5, 0.75, 0.9, 1, 1.2, 1.7, 2])
@pytest.mark.parametrize("mu", [-0.15, 0.15])
def test_exact_cev_markov_transitions_match_bessel_cdfs_at_coarse_steps(gamma, mu):
    # Four coarse steps must still have the correct terminal law. A Milstein
    # fallback at these parameters does not pass the same deterministic ECDF check.
    process = CEVProcess(mu=mu, sigma=1, gamma=gamma, s0=1)
    values = process.simulate(4, n_paths=50_000, T=1.2, rng=np.random.default_rng(719))[:, -1]
    for threshold in (0, 0.05, 0.2, 0.7, 1, 1.5, 3):
        expected = process.transition_cdf(threshold, 1.2)
        assert abs(np.mean(values <= threshold) - expected) < 0.012
    assert np.all(np.isfinite(values)) and np.all(values >= 0)


def test_cev_inverse_bessel_mean_tail_criterion_and_collateralized_call():
    process = CEVProcess(mu=0, sigma=1, gamma=2, s0=1)
    expected_mean = special.erf(1 / np.sqrt(2))  # reciprocal BES(3), not s0
    assert process.conditional_mean(1) == pytest.approx(expected_mean, rel=1e-14)
    # E[S^2] from a negative moment of a noncentral chi-square (independent formula).
    second = (
        0.5 * np.exp(special.gammaln(0.5) - special.gammaln(1.5)) * special.hyp1f1(1, 1.5, -0.5)
    )
    assert process.conditional_variance(1) == pytest.approx(second - expected_mean**2, abs=2e-9)
    expected_payoff = quad(lambda z: stats.ncx2.cdf(1 / z**2, 3, 1), 1, np.inf)[0]
    defect = 1 - expected_mean
    assert process.call_price(1, 1) == pytest.approx(expected_payoff + defect, abs=2e-9)
    for gamma in (1.2, 1.5):
        law = CEVProcess(mu=0, sigma=1, gamma=gamma, s0=1)
        assert not law.properties.has_finite_variance
        assert np.isinf(law.conditional_variance(1))
        with pytest.raises(ValueError, match="second moment"):
            law.cumulants_at(2, 1)


def test_cev_initial_state_override_and_immutable_greeks_are_not_mutations():
    process = CEVProcess(mu=0.03, sigma=0.8, gamma=0.7, s0=1)
    other = CEVProcess(mu=0.03, sigma=0.8, gamma=0.7, s0=2)
    assert process.conditional_variance(0.8, s0=2) == pytest.approx(
        other.conditional_variance(0.8), rel=2e-10
    )
    story = process.__story__()
    values = process.delta(1, 1), process.gamma_greek(1, 1), process.vega(1, 1)
    assert np.all(np.isfinite(values))
    assert process.s0 == 1 and process.sigma == 0.8 and process.__story__() == story
    with pytest.raises(ValueError):
        process.vega(-1, 1)
    assert process.__story__() == story
    with pytest.raises(NotImplementedError, match="order two"):
        process.cumulants_at(3, 1)
    with pytest.raises(ValueError, match="agree"):
        process.simulate(2, T=1, time_horizon=2)
    with pytest.raises(ValueError, match="equal T"):
        process.simulate(2, dt=0.2, T=1)


def test_merton_high_intensity_series_has_no_underflow_or_silent_tail_truncation():
    process = MertonJumpDiffusion(rate=1000, jump_mean=0, jump_std=0.001, sigma=0.2)
    with pytest.raises(ValueError, match="max_jumps"):
        process.call_price(100, T=1, r=0.03, q=0.01)
    result = process.call_price(100, T=1, r=0.03, q=0.01, max_jumps=1500)
    # Independent Gil--Pelaez inversion of the risk-neutral characteristic exponent.
    mu = 0.03 - 0.01 - 0.5 * 0.2**2 - 1000 * np.expm1(0.5 * 0.001**2)

    def phi(u):
        return np.exp(1j * mu * u - 0.5 * 0.2**2 * u**2 + 1000 * np.expm1(-0.5 * 0.001**2 * u**2))

    p1 = 0.5 + quad(lambda u: (phi(u - 1j) * np.exp(-0.02) / (1j * u)).real, 0, np.inf)[0] / np.pi
    p2 = 0.5 + quad(lambda u: (phi(u) / (1j * u)).real, 0, np.inf)[0] / np.pi
    expected = 100 * np.exp(-0.01) * p1 - 100 * np.exp(-0.03) * p2
    assert result == pytest.approx(expected, abs=2e-8)


def test_degenerate_time_zero_law_exists_even_when_future_levy_moments_do_not():
    stable = AlphaStable(alpha=0.7, beta=0.2)
    assert stable.cumulants_at(4, 0) == {1: 0, 2: 0, 3: 0, 4: 0}
    assert hellinger_distance(stable, BrownianMotion(), t=0) == 0
    with pytest.warns(SpxaDegradationWarning):
        subordinated = stable @ GammaProcess(a=1, b=2)
    assert subordinated.cumulants_at(4, 0) == {1: 0, 2: 0, 3: 0, 4: 0}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "constructor",
    [
        lambda x: CompoundPoisson(x),
        lambda x: MertonJumpDiffusion(drift=x),
        lambda x: CIRProcess(x, 0.1, 0.2),
        lambda x: OrnsteinUhlenbeckProcess(1, x, 0.2),
        lambda x: HestonProcess(mu=x, sigma_v=0.3),
        lambda x: CEVProcess(mu=x),
    ],
)
def test_new_constructors_reject_nonfinite_parameters(constructor, bad):
    with pytest.raises(ValueError, match="finite|positive"):
        constructor(bad)


@pytest.mark.parametrize("process", NEW_PROCESSES)
@pytest.mark.parametrize(
    "bad", [{"n_steps": 1.5}, {"n_paths": 0}, {"n_paths": True}, {"T": np.nan}, {"T": np.inf}]
)
def test_new_simulation_grid_guards_are_shared_through_composition(process, bad):
    kwargs = {"n_steps": 2, "n_paths": 2, "T": 0.5} | bad
    with pytest.raises(ValueError):
        (2 * process).simulate(**kwargs)


def test_cev_existing_positional_dt_api_and_portable_t_keyword_both_work():
    process = CEVProcess(mu=0.01, sigma=0.6, gamma=0.7, s0=1)
    old_style = process.simulate(3, 0.2, 0.6, 4, np.random.default_rng(129), "exact")
    portable = process.simulate(3, n_paths=4, T=0.6, rng=np.random.default_rng(129))
    assert np.allclose(old_style, portable, atol=1e-13)


def test_nested_image_integrals_preserve_singletons_and_do_not_double_count_atoms():
    values = np.array([-2.0, -1, 0, 0.5, 1, 2])
    probabilities = np.array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2])
    process = CompoundPoisson(3, DiscreteJump(values, probabilities))
    image = -2 * (0.5 * process)
    measure = image.triplet.nu
    for lower, upper in [(-np.inf, np.inf), (-1, 1), (1, 1), (-0.5, 0.5)]:
        mask = (values != 0) & (-values >= lower) & (-values <= upper)
        expected = 3 * np.sum(probabilities[mask] * (1 + values[mask] ** 2))
        assert measure.integrate(lambda x: 1 + x**2, lower, upper) == pytest.approx(expected)


@pytest.mark.parametrize("process", NEW_PROCESSES)
def test_deepcopy_preserves_new_process_laws_and_distinct_driver_identity(process):
    copied = deepcopy(process)
    assert copied is not process  # same-object integral drivers deliberately share paths
    assert copied.cumulants_at(2, 0.4) == pytest.approx(process.cumulants_at(2, 0.4))
    if not isinstance(process, CEVProcess):
        assert copied.char_func(0.7, 0.4) == pytest.approx(process.char_func(0.7, 0.4))
    with pytest.raises(AttributeError):
        copied._sealed = False


def test_discrete_parameter_metadata_and_copy_restore_cannot_change_the_law():
    law = CompoundPoisson(2, DiscreteJump([0.5, 2], [0.3, 0.7]))
    expected = law.char_func(0.8, 0.4)
    for candidate in (law, deepcopy(law), pickle.loads(pickle.dumps(law))):
        exposed = candidate.jump_dist.values
        exposed.shape = (2, 1)
        exposed.dtype = np.int64
        assert candidate.jump_dist.values.shape == (2,)
        assert candidate.jump_dist.values.dtype == np.dtype(float)
        with pytest.raises(ValueError):
            candidate.jump_dist.values.setflags(write=True)
        assert candidate.char_func(0.8, 0.4) == pytest.approx(expected)


def test_multivariate_triplet_array_metadata_and_copy_are_immutable():
    from spxa.core.levy_measure import CompoundLevyMeasure
    from spxa.zoo.multivariate import MultivariateLevyTriplet

    triplet = MultivariateLevyTriplet(np.array([1.0, 2.0]), np.eye(2), CompoundLevyMeasure([]), 2)
    for candidate in (triplet, deepcopy(triplet), pickle.loads(pickle.dumps(triplet))):
        candidate.b.shape = (2, 1)
        candidate.sigma.dtype = np.int64
        assert candidate.b.shape == (2,)
        assert np.array_equal(candidate.sigma, np.eye(2))
        with pytest.raises(ValueError):
            candidate.sigma.setflags(write=True)


def test_nested_array_parameters_expose_metadata_only_on_disposable_views():
    from spxa.core._immutable import Immutable, ImmutableMeta

    class ParameterSet(Immutable, metaclass=ImmutableMeta):
        def __init__(self):
            self.parts = [(np.array([1.0, 2.0]),)]

    parameters = ParameterSet()
    parameters.parts[0][0].shape = (2, 1)
    assert parameters.parts[0][0].shape == (2,)
    copied = deepcopy(parameters)
    with pytest.raises(ValueError):
        copied.parts[0][0].setflags(write=True)


@pytest.mark.parametrize("method", ["call_price", "put_price"])
@pytest.mark.parametrize("T", [0.0, 1.0])
@pytest.mark.parametrize("rates", [{"r": np.nan}, {"q": np.nan}, {"r": np.inf}, {"q": -np.inf}])
def test_cev_price_rate_overrides_must_be_finite(method, T, rates):
    process = CEVProcess(mu=0, sigma=0.4, gamma=0.5, s0=1)
    with pytest.raises(ValueError, match="finite"):
        getattr(process, method)(K=1, T=T, **rates)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_cev_option_pricing_uses_continuous_zero_drift_limit(gamma):
    process = CEVProcess(mu=0, sigma=0.4, gamma=gamma, s0=1)
    for method in (process.call_price, process.put_price):
        reference = method(0.9, 1.0, r=0, q=0)
        for drift in (-1e-14, 1e-14):
            assert method(0.9, 1.0, r=drift, q=0) == pytest.approx(reference, rel=1e-12, abs=1e-13)
