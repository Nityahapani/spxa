"""
Exactness tests for BrownianMotion.

Each test verifies a closed-form result against a published formula.
These tests must never be skipped or marked xfail.
"""

import numpy as np
import pytest

from spxa.zoo.levy import BrownianMotion


@pytest.mark.exactness
def test_bm_triplet_drift_zero() -> None:
    """
    BM with mu=0 has drift b=0.
    Sato (1999), Example 8.3: BM triplet is (0, σ², 0).
    """
    bm = BrownianMotion(mu=0.0, sigma=1.0)
    assert bm.triplet.b == 0.0


@pytest.mark.exactness
def test_bm_triplet_sigma_sq() -> None:
    """
    BM with sigma=2 has sigma_sq=4.
    Sato (1999), Example 8.3: triplet is (mu, σ², 0).
    """
    bm = BrownianMotion(mu=0.0, sigma=2.0)
    assert bm.triplet.sigma_sq == pytest.approx(4.0)


@pytest.mark.exactness
def test_bm_triplet_with_drift() -> None:
    """
    BM with mu=1.5 has b=1.5.
    Sato (1999), Example 8.3.
    """
    bm = BrownianMotion(mu=1.5, sigma=1.0)
    assert bm.triplet.b == pytest.approx(1.5)


@pytest.mark.exactness
def test_bm_levy_measure_zero() -> None:
    """
    BM Lévy measure is zero: ν(B) = 0 for all B.
    BM is a continuous process — no jumps.
    Sato (1999), Example 8.3.
    """
    bm = BrownianMotion(mu=0.0, sigma=1.0)
    assert bm.triplet.nu.total_mass() == pytest.approx(0.0)


@pytest.mark.exactness
def test_bm_cumulant_1_equals_drift() -> None:
    """
    κ_1(X_1) = μ for BM with drift μ.
    Sato (1999), Theorem 25.3, n=1: κ_1 = b + ∫_{|x|>1} x ν(dx) = b (ν=0).
    """
    bm = BrownianMotion(mu=0.75, sigma=1.0)
    kappas = bm.cumulants(order=2)
    assert kappas[1] == pytest.approx(0.75, rel=1e-6)


@pytest.mark.exactness
def test_bm_cumulant_2_equals_variance() -> None:
    """
    κ_2(X_1) = σ² for BM.
    Sato (1999), Theorem 25.3, n=2: κ_2 = σ² + ∫ x² ν(dx) = σ² (ν=0).
    """
    bm = BrownianMotion(mu=0.0, sigma=1.5)
    kappas = bm.cumulants(order=2)
    assert kappas[2] == pytest.approx(2.25, rel=1e-6)


@pytest.mark.exactness
def test_bm_higher_cumulants_zero() -> None:
    """
    κ_n(X_1) = 0 for n ≥ 3, BM.
    Sato (1999), Theorem 25.3: κ_n = ∫ xⁿ ν(dx) = 0 for n ≥ 3 when ν=0.
    """
    bm = BrownianMotion(mu=0.0, sigma=1.0)
    kappas = bm.cumulants(order=4)
    assert kappas[3] == pytest.approx(0.0, abs=1e-10)
    assert kappas[4] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.exactness
def test_bm_addition_triplet() -> None:
    """
    (BM(μ₁,σ₁) + BM(μ₂,σ₂)).triplet == BM(μ₁+μ₂, sqrt(σ₁²+σ₂²)).triplet
    Sato (1999), Proposition 11.10: triplets add component-wise.
    """
    bm1 = BrownianMotion(mu=1.0, sigma=1.0)
    bm2 = BrownianMotion(mu=2.0, sigma=2.0)
    z = bm1 + bm2
    t = z.triplet
    assert t.b == pytest.approx(3.0, rel=1e-10)
    assert t.sigma_sq == pytest.approx(5.0, rel=1e-10)


@pytest.mark.exactness
def test_bm_scaling_triplet() -> None:
    """
    (c * BM(0, σ)).triplet has sigma_sq = c²σ².
    Sato (1999), Prop. 11.10: scaling by c gives (cb, c²σ², ν(·/c)).
    """
    bm = BrownianMotion(mu=0.0, sigma=1.0)
    z = 3.0 * bm
    assert z.triplet.sigma_sq == pytest.approx(9.0, rel=1e-10)


@pytest.mark.exactness
def test_bm_is_martingale_iff_zero_drift() -> None:
    """
    BM(μ=0) is a martingale; BM(μ≠0) is not.
    A Lévy process is a martingale iff b=0 and ∫_{|x|>1} |x| ν(dx) < ∞ with zero mean jump.
    Applebaum (2009), Theorem 2.4.5.
    """
    assert BrownianMotion(mu=0.0, sigma=1.0).properties.is_martingale is True
    assert BrownianMotion(mu=0.5, sigma=1.0).properties.is_martingale is False


@pytest.mark.exactness
def test_bm_self_similarity_index() -> None:
    """
    BM has self-similarity index H = 0.5.
    B_{ct} ∼ c^{1/2} B_t for standard BM.
    Sato (1999), Example 13.2.
    """
    bm = BrownianMotion(mu=0.0, sigma=1.0)
    assert bm.properties.self_similarity_index == pytest.approx(0.5)


@pytest.mark.exactness
def test_bm_simulate_mean() -> None:
    """
    Empirical mean of BM(μ, σ)_1 converges to μ.
    E[X_1] = μ·1 = μ. Law of large numbers.
    """
    rng = np.random.default_rng(42)
    bm = BrownianMotion(mu=2.0, sigma=0.5)
    paths = bm.simulate(n_steps=100, n_paths=10_000, T=1.0, rng=rng)
    empirical_mean = paths[:, -1].mean()
    assert empirical_mean == pytest.approx(2.0, abs=0.05)


@pytest.mark.exactness
def test_bm_simulate_variance() -> None:
    """
    Empirical variance of BM(0, σ)_1 converges to σ².
    Var(X_1) = σ². Law of large numbers.
    """
    rng = np.random.default_rng(42)
    bm = BrownianMotion(mu=0.0, sigma=2.0)
    paths = bm.simulate(n_steps=100, n_paths=10_000, T=1.0, rng=rng)
    empirical_var = paths[:, -1].var()
    assert empirical_var == pytest.approx(4.0, rel=0.05)
