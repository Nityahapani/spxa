"""
Exactness tests for AlphaStable process.

Each test verifies a closed-form result against a published formula.
These tests must never be skipped or marked xfail.
"""

import numpy as np
import pytest

from spxa.zoo.levy.stable import AlphaStable


@pytest.mark.exactness
def test_stable_invalid_alpha_raises() -> None:
    """
    Alpha must be in (0, 2]. Samorodnitsky & Taqqu (1994), Definition 1.1.1.
    """
    with pytest.raises(ValueError):
        AlphaStable(alpha=0.0)
    with pytest.raises(ValueError):
        AlphaStable(alpha=2.5)


@pytest.mark.exactness
def test_stable_invalid_beta_raises() -> None:
    """
    Beta must be in [-1, 1]. Samorodnitsky & Taqqu (1994), Definition 1.1.1.
    """
    with pytest.raises(ValueError):
        AlphaStable(alpha=1.5, beta=1.5)
    with pytest.raises(ValueError):
        AlphaStable(alpha=1.5, beta=-2.0)


@pytest.mark.exactness
def test_stable_alpha2_is_gaussian() -> None:
    """
    α=2 stable is Brownian motion: σ²_triplet = 2σ², ν = 0.
    Samorodnitsky & Taqqu (1994), Property 1.2.3.
    """
    s = AlphaStable(alpha=2.0, beta=0.0, sigma=1.0)
    assert s.triplet.sigma_sq == pytest.approx(2.0)
    assert s.triplet.nu.total_mass() == pytest.approx(0.0)


@pytest.mark.exactness
def test_stable_char_func_at_zero() -> None:
    """
    φ(0; t) = 1. Definition of characteristic function.
    """
    s = AlphaStable(alpha=1.5, beta=0.0, sigma=1.0)
    cf = s.char_func_exact(u=0.0, t=1.0)
    assert abs(complex(cf) - 1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.exactness
def test_stable_char_func_modulus_leq_1() -> None:
    """
    |φ(u)| ≤ 1. Characteristic functions are bounded by 1.
    """
    s = AlphaStable(alpha=1.5, beta=0.3, sigma=1.0)
    u_vals = np.linspace(-5.0, 5.0, 50)
    cf_vals = s.char_func_exact(u=u_vals, t=1.0)
    assert np.all(np.abs(cf_vals) <= 1.0 + 1e-12)


@pytest.mark.exactness
def test_stable_char_func_time_additivity() -> None:
    """
    φ(u; s+t) = φ(u; s)·φ(u; t). Lévy process property.
    Sato (1999), Definition 1.6.
    """
    s = AlphaStable(alpha=1.5, beta=0.2, sigma=1.0)
    u, t1, t2 = 1.0, 0.3, 0.7
    cf1 = complex(s.char_func_exact(u=u, t=t1))
    cf2 = complex(s.char_func_exact(u=u, t=t2))
    cf12 = complex(s.char_func_exact(u=u, t=t1 + t2))
    assert abs(cf1 * cf2 - cf12) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.exactness
def test_stable_symmetric_when_beta_zero() -> None:
    """
    Symmetric stable (β=0) has real characteristic function for real u.
    φ(u) = exp(-σ^α|u|^α) is real. Samorodnitsky & Taqqu (1994), Property 1.2.9.
    """
    s = AlphaStable(alpha=1.5, beta=0.0, sigma=1.0)
    u_vals = np.array([0.5, 1.0, 2.0])
    cf_vals = s.char_func_exact(u=u_vals, t=1.0)
    assert np.all(np.abs(cf_vals.imag) < 1e-12)


@pytest.mark.exactness
def test_stable_self_similarity_index() -> None:
    """
    α-stable process has self-similarity index H = 1/α.
    X_{ct} ∼ c^{1/α} X_t. Samorodnitsky & Taqqu (1994), Property 1.2.1.
    """
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        s = AlphaStable(alpha=alpha)
        assert s.properties.self_similarity_index == pytest.approx(1.0 / alpha)


@pytest.mark.exactness
def test_stable_finite_mean_iff_alpha_gt_1() -> None:
    """
    E[|X_1|] < ∞ iff α > 1.
    Samorodnitsky & Taqqu (1994), Property 1.2.16.
    """
    assert AlphaStable(alpha=1.5).properties.has_finite_mean is True
    assert AlphaStable(alpha=0.5).properties.has_finite_mean is False
    assert AlphaStable(alpha=1.0).properties.has_finite_mean is False


@pytest.mark.exactness
def test_stable_finite_variance_only_at_alpha2() -> None:
    """
    Var(X_1) < ∞ iff α = 2.
    Samorodnitsky & Taqqu (1994), Property 1.2.16.
    """
    assert AlphaStable(alpha=2.0).properties.has_finite_variance is True
    assert AlphaStable(alpha=1.9).properties.has_finite_variance is False


@pytest.mark.exactness
def test_stable_simulate_shape() -> None:
    """Simulation produces correct array shape."""
    s = AlphaStable(alpha=1.5, beta=0.0, sigma=1.0)
    paths = s.simulate(n_steps=50, n_paths=10, T=1.0)
    assert paths.shape == (10, 51)
    assert np.all(paths[:, 0] == 0.0)


@pytest.mark.exactness
def test_stable_cauchy_simulate_symmetry() -> None:
    """
    Cauchy process (α=1, β=0) is symmetric: median ≈ 0 for large samples.
    The Cauchy distribution has undefined mean but median = 0 when symmetric.
    Samorodnitsky & Taqqu (1994), Property 1.2.9.
    """
    rng = np.random.default_rng(30)
    s = AlphaStable(alpha=1.0, beta=0.0, sigma=1.0)
    paths = s.simulate(n_steps=10, n_paths=50_000, T=1.0, rng=rng)
    empirical_median = np.median(paths[:, -1])
    assert empirical_median == pytest.approx(0.0, abs=0.05)
