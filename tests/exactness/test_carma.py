"""Exactness tests for Lévy-driven CARMA(p, q) processes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, linalg, stats

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.carma import CARMA, CARMAProcess
from spxa.zoo.beyond.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess
from spxa.zoo.levy.brownian import BrownianMotion
from spxa.zoo.levy.gamma import GammaProcess


def _carma31() -> CARMAProcess:
    """CARMA(3, 1) with AR zeros -0.5, -1 +- 2i and MA zero -2."""
    return CARMAProcess.from_roots(
        [-0.5, -1 + 2j, -1 - 2j], [-2.0], driver=BrownianMotion(0.1, 0.3)
    )


def _gamma_carma() -> CARMAProcess:
    return CARMAProcess([1.5, 0.8], [0.5, 1.0], driver=GammaProcess(a=2.0, b=4.0))


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


@pytest.mark.exactness
def test_from_roots_builds_the_polynomials() -> None:
    m = _carma31()
    # a(z) = (z + 0.5)(z^2 + 2z + 5), b(z) = z + 2
    np.testing.assert_allclose(m.ar, [2.5, 6.0, 2.5], atol=1e-12)
    np.testing.assert_allclose(m.ma, [2.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        np.sort_complex(m.ar_roots), np.sort_complex([-1 - 2j, -1 + 2j, -0.5])
    )
    np.testing.assert_allclose(m.ma_roots, [-2.0])
    assert (m.p, m.q) == (3, 1)
    assert CARMA is CARMAProcess
    assert m.exactness == ExactnessLevel.MOMENT_PROPAGATION
    A = m.state_matrix
    np.testing.assert_allclose(A[-1], [-2.5, -6.0, -2.5])
    np.testing.assert_allclose(np.diag(A, 1), 1.0)
    np.testing.assert_allclose(m.output_vector, [2.0, 1.0, 0.0])


@pytest.mark.exactness
def test_validation() -> None:
    with pytest.raises(ValueError, match="negative real parts"):
        CARMAProcess([-1.0])
    with pytest.raises(ValueError, match="negative real parts"):
        CARMAProcess([0.0, 1.0])  # zeros +-i on the axis
    with pytest.raises(ValueError, match="q < p"):
        CARMAProcess([1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="vanish"):
        CARMAProcess([1.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="length"):
        CARMAProcess([1.0, 1.0], x0=[0.0])
    with pytest.raises(ValueError, match="substeps"):
        CARMAProcess([1.0], substeps=0)
    with pytest.raises(ValueError, match="conjugate"):
        CARMAProcess.from_roots([-1 + 1j, -2.0])
    with pytest.raises(ExactnessError):
        _ = _carma31().triplet


# ---------------------------------------------------------------------------
# second-order structure
# ---------------------------------------------------------------------------


@pytest.mark.exactness
def test_carma_1_0_is_the_ornstein_uhlenbeck_process() -> None:
    m = CARMAProcess([2.0], [1.0], driver=BrownianMotion(0.6, 0.5))
    ou = OrnsteinUhlenbeckProcess(theta=2.0, mu=0.3, sigma=0.5)
    assert m.stationary_mean == pytest.approx(ou.stationary_mean, rel=1e-14)
    assert m.stationary_variance == pytest.approx(ou.stationary_variance, rel=1e-14)
    for h in (0.0, 0.3, 1.7):
        assert m.autocovariance(h) == pytest.approx(ou.autocovariance(h), rel=1e-12)
    np.testing.assert_allclose(m.kernel(np.array([0.0, 1.0])), np.exp([0.0, -2.0]), rtol=1e-14)


@pytest.mark.exactness
def test_lyapunov_equation_and_stationary_state() -> None:
    m = _carma31()
    A, S = m.state_matrix, m.stationary_state_covariance / 0.09
    e = np.zeros(3)
    e[-1] = 1.0
    np.testing.assert_allclose(A @ S + S @ A.T + np.outer(e, e), 0.0, atol=1e-13)
    assert m.stationary_variance == pytest.approx(0.09 * m.output_vector @ S @ m.output_vector)
    assert m.stationary_mean == pytest.approx(m.output_vector @ m.stationary_state_mean, rel=1e-12)
    assert m.stationary_mean == pytest.approx(0.1 * 2.0 / 2.5, rel=1e-14)


@pytest.mark.exactness
def test_spectral_density_is_the_fourier_transform_of_the_autocovariance() -> None:
    m = _carma31()
    w = np.linspace(-400.0, 400.0, 400_001)
    f = m.spectral_density(w)
    assert np.trapezoid(f, w) == pytest.approx(m.stationary_variance, rel=1e-7)
    for h in (0.4, 0.8, 2.5):
        assert np.trapezoid(f * np.cos(w * h), w) == pytest.approx(m.autocovariance(h), rel=1e-7)
    assert m.spectral_density(0.0) == pytest.approx(0.09 / (2 * np.pi) * (2.0 / 2.5) ** 2)


@pytest.mark.exactness
def test_sampled_autocovariance_and_autocorrelation() -> None:
    m = _carma31()
    gam = m.sampled_autocovariance(0.25, 12)
    np.testing.assert_allclose(gam, m.autocovariance(0.25 * np.arange(13)), rtol=1e-10, atol=1e-16)
    assert m.autocorrelation(0.0) == pytest.approx(1.0)
    assert m.autocovariance(-0.6) == pytest.approx(m.autocovariance(0.6))
    with pytest.raises(ValueError):
        m.sampled_autocovariance(0.0, 3)


# ---------------------------------------------------------------------------
# kernel, cumulants, characteristic function
# ---------------------------------------------------------------------------


@pytest.mark.exactness
@pytest.mark.parametrize("repeated", [False, True], ids=["distinct-zeros", "repeated-zero"])
def test_kernel_and_its_power_integrals(repeated: bool) -> None:
    m = (
        CARMAProcess.from_roots([-1.0, -1.0, -3.0], [-0.5], driver=BrownianMotion(0.0, 1.0))
        if repeated
        else _carma31()
    )
    t = np.array([0.0, 0.2, 1.0, 3.0])
    by_expm = [m.output_vector @ linalg.expm(m.state_matrix * s) @ np.eye(m.p)[-1] for s in t]
    np.testing.assert_allclose(m.kernel(t), by_expm, rtol=1e-10, atol=1e-14)
    assert m.kernel(-1.0) == 0.0
    for n in (1, 2, 3, 4):
        quad = integrate.quad(lambda s: float(m.kernel(s)) ** n, 0.0, 80.0, limit=400)[0]
        assert m.kernel_power_integral(n) == pytest.approx(quad, rel=1e-8, abs=1e-14)


@pytest.mark.exactness
def test_stationary_cumulants_scale_the_driver_cumulants() -> None:
    m = _gamma_carma()
    driver = GammaProcess(a=2.0, b=4.0).cumulants(5)
    cum = m.cumulants(5)
    for n in range(1, 6):
        assert cum[n] == pytest.approx(driver[n] * m.kernel_power_integral(n), rel=1e-12)
        assert m.stationary_cumulant(n) == pytest.approx(cum[n], rel=1e-12)
    assert cum[1] == pytest.approx(m.stationary_mean)
    assert cum[2] == pytest.approx(m.stationary_variance)
    # a Brownian driver has no cumulants beyond the second
    assert _carma31().stationary_cumulant(3) == 0.0


@pytest.mark.exactness
def test_stationary_characteristic_function() -> None:
    m = _carma31()
    u = np.array([0.3, 1.0, 2.0, 5.0])
    gauss = np.exp(1j * u * m.stationary_mean - 0.5 * u**2 * m.stationary_variance)
    np.testing.assert_allclose(m.stationary_char_func(u), gauss, rtol=1e-9, atol=1e-12)
    # the gamma-driven law: small-u expansion reproduces the exact cumulants
    g = _gamma_carma()
    k = g.cumulants(4)
    u0 = 0.05
    series = 1j * u0 * k[1] - u0**2 * k[2] / 2 - 1j * u0**3 * k[3] / 6 + u0**4 * k[4] / 24
    assert np.log(g.stationary_char_func(u0)) == pytest.approx(series, abs=1e-8)
    with pytest.raises(ExactnessError):
        CARMAProcess([1.0], driver=_carma31()).stationary_char_func(1.0)


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


@pytest.mark.exactness
def test_gaussian_transition_is_exact() -> None:
    m = CARMAProcess.from_roots(
        [-0.5, -1 + 2j, -1 - 2j], [-2.0], driver=BrownianMotion(0.1, 0.3), x0=[0.2, -0.1, 0.4]
    )
    dt, n = 0.7, 200_000
    y = m.simulate(1, n, T=dt, rng=np.random.default_rng(1))
    A, b = m.state_matrix, m.output_vector
    F = linalg.expm(A * dt)
    e = np.eye(3)[-1]
    mean = b @ (F @ m.x0 + 0.1 * np.linalg.solve(A, (F - np.eye(3)) @ e))
    S = m.stationary_state_covariance
    var = b @ (S - F @ S @ F.T) @ b
    assert y[0, 0] == pytest.approx(b @ m.x0)
    assert abs(y[:, 1].mean() - mean) < 4 * np.sqrt(var / n)
    assert y[:, 1].var() == pytest.approx(var, rel=4 * np.sqrt(2 / n))


@pytest.mark.exactness
def test_gaussian_paths_are_stationary_with_the_right_autocovariance() -> None:
    m = _carma31()
    y = m.simulate(400, 2000, T=40.0, rng=np.random.default_rng(2))
    for col in (0, 200, 400):
        assert abs(y[:, col].mean() - m.stationary_mean) < 4 * np.sqrt(m.stationary_variance / 2000)
        assert y[:, col].var() == pytest.approx(m.stationary_variance, rel=0.15)
    lag = 8  # 0.8 time units
    cov = np.mean((y[:, :-lag] - m.stationary_mean) * (y[:, lag:] - m.stationary_mean))
    assert cov == pytest.approx(m.autocovariance(0.8), rel=0.05)


@pytest.mark.exactness
def test_gamma_driven_paths_match_the_stationary_cumulants() -> None:
    m = _gamma_carma()
    y = m.simulate(400, 400, T=400.0, rng=np.random.default_rng(3))
    z = y[:, ::10].ravel()  # 41 well-separated points per path
    k1, k2, k3 = m.cumulants(3).values()
    n = z.size
    assert abs(z.mean() - k1) < 4 * np.sqrt(k2 / n) * 3  # autocorrelation inflates the error
    assert z.var() == pytest.approx(k2, rel=0.08)
    assert stats.skew(z) * z.var() ** 1.5 == pytest.approx(k3, rel=0.25)
    cf = m.stationary_char_func(np.array([0.5, 1.0]))
    emp = np.mean(np.exp(1j * np.outer([0.5, 1.0], z)), axis=1)
    np.testing.assert_allclose(emp, cf, atol=0.01)


@pytest.mark.exactness
def test_substeps_converge_for_a_non_gaussian_driver() -> None:
    """Averaged-kernel substeps keep the mean exact; the variance error falls with the substep."""

    def one_step_variance(substeps: int) -> float:
        m = CARMAProcess(
            [1.5, 0.8],
            [0.5, 1.0],
            driver=GammaProcess(a=2.0, b=4.0),
            x0=[0.0, 0.0],
            substeps=substeps,
        )
        y = m.simulate(1, 200_000, T=1.0, rng=np.random.default_rng(4))
        return float(y[:, 1].var()), float(y[:, 1].mean())

    m = _gamma_carma()
    A, b = m.state_matrix, m.output_vector
    e = np.eye(2)[-1]
    # exact one-step moments from X_0 = 0: mean k1 b'A^{-1}(F-I)e, variance k2 int_0^1 g^2
    F = linalg.expm(A)
    k1, k2 = 0.5, 0.125
    exact_mean = k1 * b @ np.linalg.solve(A, (F - np.eye(2)) @ e)
    exact_var = k2 * integrate.quad(lambda s: float(m.kernel(s)) ** 2, 0.0, 1.0)[0]
    var1, mean1 = one_step_variance(1)
    var32, mean32 = one_step_variance(32)
    se_mean = np.sqrt(exact_var / 200_000)
    assert abs(mean1 - exact_mean) < 4 * se_mean and abs(mean32 - exact_mean) < 4 * se_mean
    assert abs(var32 - exact_var) < 0.02 * exact_var
    assert abs(var1 - exact_var) > abs(var32 - exact_var)


@pytest.mark.exactness
def test_properties_and_repr() -> None:
    m = _gamma_carma()
    props = m.properties
    assert not props.has_independent_increments and props.has_finite_variance
    assert "CARMAProcess(ar=[1.5, 0.8]" in repr(m)
    import spxa

    assert spxa.CARMAProcess is CARMAProcess and spxa.CARMA is CARMAProcess
