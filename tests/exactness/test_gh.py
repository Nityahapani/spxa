"""
Exactness and mathematical consistency tests for Generalized Hyperbolic (GH) Lévy process.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.zoo.levy import (
    GH,
    GHProcess,
    GeneralizedHyperbolic,
    GeneralizedHyperbolicProcess,
    NIG,
)


class TestGeneralizedHyperbolicExactness:
    """Mathematical exactness and consistency test suite for GeneralizedHyperbolicProcess."""

    def test_gh_exactness_level_and_aliases(self) -> None:
        """GH process is EXACT and aliased to GH, GHProcess, GeneralizedHyperbolic."""
        gh = GeneralizedHyperbolicProcess(lambda_param=1.0, alpha=2.0, beta=0.5, delta=1.0, mu=0.1)
        assert gh.exactness == ExactnessLevel.EXACT
        assert GH is GeneralizedHyperbolicProcess
        assert GHProcess is GeneralizedHyperbolicProcess
        assert GeneralizedHyperbolic is GeneralizedHyperbolicProcess
        assert "GeneralizedHyperbolicProcess" in repr(gh)
        assert "lambda_param=1.0" in repr(gh)
        assert "alpha=2.0" in repr(gh)

    def test_gh_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on out-of-domain inputs."""
        with pytest.raises(ValueError, match="alpha"):
            GeneralizedHyperbolicProcess(alpha=-0.5)
        with pytest.raises(ValueError, match="alpha"):
            GeneralizedHyperbolicProcess(alpha=0.0)
        with pytest.raises(ValueError, match="delta"):
            GeneralizedHyperbolicProcess(delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            GeneralizedHyperbolicProcess(delta=-1.0)
        with pytest.raises(ValueError, match=r"\|beta\|"):
            GeneralizedHyperbolicProcess(alpha=1.5, beta=1.5)
        with pytest.raises(ValueError, match=r"\|beta\|"):
            GeneralizedHyperbolicProcess(alpha=1.5, beta=-2.0)

    def test_gh_subfamily_identification(self) -> None:
        """GH correctly classifies sub-families: NIG (λ=-0.5), Hyperbolic (λ=1.0)."""
        nig_gh = GeneralizedHyperbolicProcess(lambda_param=-0.5, alpha=2.0, beta=0.2, delta=1.0)
        assert nig_gh.is_nig
        assert not nig_gh.is_hyperbolic
        assert nig_gh.subfamily == "NIG"

        hyp_gh = GeneralizedHyperbolicProcess(lambda_param=1.0, alpha=2.0, beta=0.2, delta=1.0)
        assert not hyp_gh.is_nig
        assert hyp_gh.is_hyperbolic
        assert hyp_gh.subfamily == "Hyperbolic"

    def test_gh_properties(self) -> None:
        """Lévy process structural properties."""
        gh = GeneralizedHyperbolicProcess(lambda_param=1.0, alpha=2.0, beta=0.0, delta=1.0, mu=0.0)
        props = gh.properties
        assert props.has_stationary_increments
        assert props.has_independent_increments
        assert props.has_finite_mean
        assert props.has_finite_variance
        assert not props.is_subordinator
        assert props.is_martingale  # Symmetric with zero drift -> E[X_1] = 0

    def test_gh_characteristic_function_properties(self) -> None:
        """Characteristic function satisfies phi(0)=1, |phi|<=1, and time-additivity."""
        gh = GeneralizedHyperbolicProcess(lambda_param=1.2, alpha=2.5, beta=-0.3, delta=1.5, mu=0.2)
        # phi(0; t) = 1
        assert gh.char_func(0.0, t=1.0) == pytest.approx(1.0 + 0.0j, abs=1e-12)
        assert gh.char_func(0.0, t=2.5) == pytest.approx(1.0 + 0.0j, abs=1e-12)

        # Modulus <= 1 for real u
        u_vals = np.array([-5.0, -1.0, 0.5, 2.0, 10.0])
        cf_vals = gh.char_func(u_vals, t=1.0)
        assert np.all(np.abs(cf_vals) <= 1.0 + 1e-12)

        # Time-additivity: phi(u; t1 + t2) = phi(u; t1) * phi(u; t2)
        cf_t1 = gh.char_func(u_vals, t=0.7)
        cf_t2 = gh.char_func(u_vals, t=1.3)
        cf_sum = gh.char_func(u_vals, t=2.0)
        np.testing.assert_allclose(cf_t1 * cf_t2, cf_sum, rtol=1e-12)

    def test_gh_reduction_to_nig(self) -> None:
        """When lambda = -0.5, GH characteristic function agrees with NIG."""
        alpha = 2.2
        beta = -0.4
        delta = 1.3
        mu = 0.15

        gh_nig = GeneralizedHyperbolicProcess(
            lambda_param=-0.5, alpha=alpha, beta=beta, delta=delta, mu=mu
        )
        pure_nig = NIG(alpha=alpha, beta=beta, delta=delta, mu=mu)

        u_grid = np.linspace(-4.0, 4.0, 25)
        for t in [0.5, 1.0, 2.0]:
            cf_gh = gh_nig.char_func(u_grid, t=t)
            cf_nig = pure_nig.char_func(u_grid, t=t)
            np.testing.assert_allclose(cf_gh, cf_nig, rtol=1e-11)

    def test_gh_cumulants_exact_and_time_scaling(self) -> None:
        """Exact cumulants match numerical differentiation and scale linearly with t."""
        gh = GeneralizedHyperbolicProcess(lambda_param=1.5, alpha=2.5, beta=-0.4, delta=1.8, mu=0.1)

        k1_1 = gh.cumulant_exact(1, t=1.0)
        k2_1 = gh.cumulant_exact(2, t=1.0)
        k3_1 = gh.cumulant_exact(3, t=1.0)
        k4_1 = gh.cumulant_exact(4, t=1.0)

        assert k2_1 > 0.0

        for t in [0.5, 2.0, 3.5]:
            assert gh.cumulant_exact(1, t=t) == pytest.approx(t * k1_1, rel=1e-12)
            assert gh.cumulant_exact(2, t=t) == pytest.approx(t * k2_1, rel=1e-12)
            assert gh.cumulant_exact(3, t=t) == pytest.approx(t * k3_1, rel=1e-12)
            assert gh.cumulant_exact(4, t=t) == pytest.approx(t * k4_1, rel=1e-12)

        cums = gh.cumulants(order=4, t=1.0)
        assert cums[1] == pytest.approx(k1_1)
        assert cums[2] == pytest.approx(k2_1)
        assert cums[3] == pytest.approx(k3_1)
        assert cums[4] == pytest.approx(k4_1)

    def test_gh_levy_triplet(self) -> None:
        """Triplet has sigma_sq = 0.0 and positive infinite activity jump measure."""
        gh = GeneralizedHyperbolicProcess(lambda_param=1.0, alpha=2.0, beta=0.2, delta=1.0, mu=0.0)
        triplet = gh.triplet
        assert triplet.sigma_sq == 0.0
        assert math.isinf(triplet.nu.total_mass())

    def test_gh_option_pricing_put_call_parity(self) -> None:
        """European call and put option prices satisfy exact Put-Call parity."""
        gh = GeneralizedHyperbolicProcess(lambda_param=0.8, alpha=3.0, beta=-0.8, delta=1.2, mu=0.0)
        s0 = 100.0
        r = 0.04
        q = 0.01
        t = 1.0
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

        df_r = math.exp(-r * t)
        df_q = math.exp(-q * t)

        for k in strikes:
            call = gh.call_price(K=k, T=t, r=r, q=q, s0=s0)
            put = gh.put_price(K=k, T=t, r=r, q=q, s0=s0)
            expected_parity = s0 * df_q - k * df_r
            actual_parity = call - put
            assert actual_parity == pytest.approx(expected_parity, rel=1e-7, abs=1e-8)

    def test_gh_vectorized_option_pricing(self) -> None:
        """call_price and put_price accept array-like strikes."""
        gh = GeneralizedHyperbolicProcess(lambda_param=1.0, alpha=3.0, beta=-0.5, delta=1.0)
        strikes = np.array([85.0, 100.0, 115.0])
        calls = gh.call_price(strikes, T=1.0, r=0.03, s0=100.0)
        puts = gh.put_price(strikes, T=1.0, r=0.03, s0=100.0)
        assert calls.shape == (3,)
        assert puts.shape == (3,)
        assert calls[0] > calls[1] > calls[2]
        assert puts[0] < puts[1] < puts[2]

    def test_gh_simulation_paths_and_mean(self) -> None:
        """Simulated sample paths start at 0 and empirical mean matches theoretical mean."""
        rng = np.random.default_rng(2026)
        gh = GeneralizedHyperbolicProcess(
            lambda_param=-0.5, alpha=2.5, beta=-0.3, delta=1.2, mu=0.2
        )
        paths = gh.simulate(n_steps=20, time_horizon=1.0, n_paths=25_000, rng=rng)
        assert paths.shape == (25_000, 21)
        assert np.all(paths[:, 0] == 0.0)

        terminal_mean = float(np.mean(paths[:, -1]))
        theo_mean = gh.cumulant_exact(1, t=1.0)
        assert terminal_mean == pytest.approx(theo_mean, rel=0.03)


# ── cumulants of every order and the triplet drift ──────────────────────────

_GH_CASES = [
    (1.2, 3.0, -0.5, 0.8, 0.03),
    (-0.5, 2.0, 0.4, 1.5, 0.0),  # NIG
    (2.5, 1.5, 1.0, 0.3, 0.1),
    (-2.0, 5.0, -3.0, 0.2, -0.4),
    (0.0, 1.1, 0.9, 4.0, 0.0),
]


def _mp_gh_cgf(mp, z, lam, alpha, beta, delta, mu):  # type: ignore[no-untyped-def]
    """log E[e^{z X_1}] of GH, at mpmath precision."""
    g = mp.sqrt(alpha**2 - beta**2)
    s = mp.sqrt(alpha**2 - (beta + z) ** 2)
    return (
        mu * z
        + lam * mp.log(g / s)
        + mp.log(mp.besselk(lam, delta * s) / mp.besselk(lam, delta * g))
    )


@pytest.mark.parametrize("p", _GH_CASES)
def test_gh_cumulants_of_every_order_match_mpmath(p: tuple) -> None:
    mp = pytest.importorskip("mpmath")
    gh = GeneralizedHyperbolicProcess(*p)
    with mp.workdps(40):
        mpp = [mp.mpf(v) for v in p]
        co = mp.taylor(lambda z: _mp_gh_cgf(mp, z, *mpp), 0, 10)
        ref = [float(mp.factorial(n) * co[n]) for n in range(1, 11)]
    got = [gh.cumulant_exact(n) for n in range(1, 11)]
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=0)
    assert gh.cumulant_exact(7, t=2.5) == pytest.approx(2.5 * ref[6], rel=1e-14)
    # orders >= 5 used to return a difference quotient of psi'(0), i.e. kappa_1
    assert abs(gh.cumulant_exact(5) - gh.cumulant_exact(1)) > 1e-3 * abs(ref[4])


@pytest.mark.parametrize("p", _GH_CASES)
def test_gh_triplet_drift_reproduces_the_closed_form_mean(p: tuple) -> None:
    gh = GeneralizedHyperbolicProcess(*p)
    trip = gh._triplet()
    # was 2.5e-9 off for the first case: the integral skipped |x| < 1e-8
    assert trip.cumulant(1) == pytest.approx(gh.cumulant_exact(1), rel=1e-14, abs=1e-15)
    assert gh._exact_cumulants(4) == gh.cumulants(4)


def test_nig_levy_density_is_finite_in_the_tails() -> None:
    mp = pytest.importorskip("mpmath")
    a, b, d = 2.0, 0.4, 1.5
    nu = GeneralizedHyperbolicProcess(-0.5, a, b, d, 0.0)._triplet().nu
    xs = np.array([-400.0, -40.0, -3.0, -0.01, 1e-6, 0.5, 7.0, 60.0, 400.0])
    with mp.workdps(30):
        ref = [
            float(a * d / mp.pi * mp.exp(b * x) * mp.besselk(1, a * abs(x)) / abs(x)) for x in xs
        ]
    got = nu.density(xs)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=0)
