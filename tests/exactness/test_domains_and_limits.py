"""Domain/limit regressions: Sato (1999), Thm 25.3; CGMY (2002); Rosinski (2007).

Exponential moments are expectations, not unrestricted analytic continuations.
"""

import numpy as np
import pytest

from spxa import BrownianMotion, GammaProcess
from spxa.analytics import european_call_price
from spxa.sim import geometric_levy
from spxa.zoo.levy import (
    CGMY,
    NIG,
    AlphaStable,
    InverseGaussianProcess,
    MeixnerProcess,
    TemperedStable,
    VarianceGamma,
)

pytestmark = pytest.mark.exactness


@pytest.mark.parametrize(
    "law",
    [
        GammaProcess(a=1, b=0.5),
        InverseGaussianProcess(mu=1, lam=1),
        NIG(alpha=1, beta=0.5, delta=1),
        VarianceGamma(sigma=2, nu=1, theta=1),
        TemperedStable(alpha=0.5, C=1, lam=0.5),
        MeixnerProcess(alpha=4, beta=0.2, delta=1),
        CGMY(C=1, G=2, M=0.5, Y=0.5),
    ],
)
def test_absent_asset_exponential_moment_is_rejected(law):
    """Carr–Madan (1999): martingale normalization requires a finite MGF at one."""
    with pytest.raises(ValueError):
        european_call_price(law, S0=100, K=100, r=0.05, t=1)


@pytest.mark.parametrize("Y", [0.0, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("frequency", [-3.1j, 2.1j])
def test_cgmy_complex_frequency_strip(Y, frequency):
    """CGMY (2002): positive M / negative G exponential tail convergence."""
    law = CGMY(C=0.5, G=2, M=3, Y=Y)
    with pytest.raises(ValueError, match="convergence strip"):
        law.char_func(frequency)


@pytest.mark.parametrize("alpha", [0.6, 1.0, 1.5])
def test_stable_triplet_cannot_invent_a_finite_variance(alpha):
    """Sato (1999), Thm 25.3: tail moment order must be strictly below alpha."""
    with pytest.raises(ValueError, match="does not exist"):
        AlphaStable(alpha=alpha, beta=0.2).triplet.cumulant(2)


@pytest.mark.parametrize(
    "law",
    [
        AlphaStable(alpha=1.5, beta=0.2),
        NIG(alpha=3, beta=0.2, delta=1),
        MeixnerProcess(alpha=1, beta=0.2, delta=1),
        CGMY(C=0.2, G=2, M=3, Y=1.5),
    ],
)
def test_infinite_variation_measure_first_moment_not_raw_cumulant(law):
    """Sato (1999), Thm 21.9: infinite small-jump absolute first moment."""
    with pytest.raises(ValueError, match="does not exist"):
        law.triplet.nu.moment(1)
    assert np.isfinite(law.cumulants(1)[1])


@pytest.mark.parametrize("alpha,C,lam", [(0.2, 0.1, 0.4), (0.5, 0.4, 2.0), (0.85, 0.1, 1.5)])
def test_tempered_stable_empirical_characteristic_function(alpha, C, lam):
    """Rosinski (2007): exponential tilting must agree with the Laplace/CF law."""
    law = TemperedStable(alpha=alpha, C=C, lam=lam)
    samples = law.simulate(n_steps=1, n_paths=50000, T=1, rng=np.random.default_rng(537))[:, -1]
    u = np.array([0.3, 2.0])
    empirical = np.mean(np.exp(1j * samples[:, None] * u), axis=0)
    np.testing.assert_allclose(empirical, law.char_func(u), atol=0.02, rtol=0)
    assert np.all(samples >= 0)


def test_tempered_stable_extreme_budget_has_no_truncation_fallback():
    """Rosinski (2007) law: reject a resource-exhausting exact scheme, do not change the law."""
    with pytest.raises(ValueError):
        TemperedStable(alpha=0.5, C=100000, lam=1).simulate(n_steps=1, n_paths=1, T=1)


def test_geometric_levy_does_not_use_gaussian_surrogate_for_an_absent_mgf():
    """Exponential asset normalization requires its actual finite first moment."""
    with pytest.raises(ValueError):
        geometric_levy(GammaProcess(a=1, b=0.5), mu=0.05, sigma=1, n_steps=1, n_paths=1)


def test_deterministic_geometric_asset_limit():
    """Deterministic X cancels exactly under the martingale normalization."""
    values = geometric_levy(BrownianMotion(mu=0.3, sigma=0), mu=0.05, sigma=2, s0=100, n_steps=4)
    np.testing.assert_allclose(values[0], 100 * np.exp(0.05 * np.linspace(0, 1, 5)), rtol=1e-14)


@pytest.mark.parametrize(
    "mu,sigma,index", [(0.0, 1.0, 0.5), (1.0, 1.0, None), (1.0, 0.0, 1.0), (0.0, 0.0, None)]
)
def test_brownian_similarity_flag_requires_a_single_scaling_law(mu, sigma, index):
    """Brownian scaling (Sato 1999, Example 8.3): drift and noise have different indices."""
    assert BrownianMotion(mu=mu, sigma=sigma).properties.self_similarity_index == index


def test_deterministic_brownian_clock_is_an_exact_time_rescaling():
    """Bochner subordination (Sato 1999, Thm 30.1), deterministic clock T_t=3t."""
    parent = BrownianMotion(mu=0.2, sigma=0.7)
    clock = BrownianMotion(mu=3, sigma=0)
    law = parent @ clock
    assert clock.properties.is_subordinator
    u = np.array([0.0, 1.0, 5.0])
    np.testing.assert_allclose(
        law.char_func(u, t=0.4), parent.char_func(u, t=1.2), rtol=1e-14, atol=1e-14
    )
    assert law.triplet.sigma_sq == pytest.approx(3 * 0.7**2)


@pytest.mark.parametrize("t", [-1.0, np.nan, np.inf])
def test_brownian_characteristic_function_rejects_invalid_time(t):
    """A probability CF requires a nonnegative finite marginal time."""
    with pytest.raises(ValueError):
        BrownianMotion().char_func(1, t=t)
