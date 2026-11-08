"""
Exactness tests for MG1WorkloadProcess (M/G/1 workload, reflected compound Poisson).

Every expected value is computed here from a published closed form, written out
in the test docstring, and never from the class's own formulas.
"""

import math

import numpy as np
import pytest

from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.mg1_workload import JobSizeDistribution, MG1WorkloadProcess
from spxa.zoo.levy.compound_poisson import ExponentialJump


def _deterministic(d: float) -> JobSizeDistribution:
    return JobSizeDistribution(lambda rng, n: np.full(n, d), d, d**2, d**3)


@pytest.mark.exactness
@pytest.mark.parametrize("lam, mu", [(0.5, 1.0), (0.9, 1.0), (1.2, 2.0)])
def test_mm1_stationary_mean_and_second_moment(lam: float, mu: float) -> None:
    """
    M/M/1 waiting-time law (Kleinrock, 1975, Queueing Systems Vol. I, M/M/1
    waiting-time distribution: P(W > x) = rho * exp(-mu (1 - rho) x)). By PASTA
    this is also the stationary workload.

    With Exp(mu) sizes, W_inf has an atom 1 - rho at 0 and is Exp(mu(1 - rho))
    otherwise, so E[W] = rho / (mu (1 - rho)) and E[W^2] = 2 rho / (mu^2 (1 - rho)^2).
    """
    rho = lam / mu
    q = MG1WorkloadProcess(lam=lam, job_size=ExponentialJump(rate=mu))
    expected_mean = rho / (mu * (1.0 - rho))
    expected_m2 = 2.0 * rho / (mu**2 * (1.0 - rho) ** 2)
    assert q.rho == pytest.approx(rho, rel=1e-14)
    assert q.stationary_mean == pytest.approx(expected_mean, rel=1e-12)
    assert q.stationary_second_moment == pytest.approx(expected_m2, rel=1e-12)
    assert q.stationary_variance == pytest.approx(expected_m2 - expected_mean**2, rel=1e-12)
    assert q.idle_probability == pytest.approx(1.0 - rho, rel=1e-14)


@pytest.mark.exactness
@pytest.mark.parametrize("lam, d", [(1.0, 0.5), (2.0, 0.3), (0.1, 5.0)])
def test_md1_stationary_mean_and_variance(lam: float, d: float) -> None:
    """
    M/D/1 workload moments (Takacs, 1962, moment recurrence
    E[W^k] = lam/(1 - rho) sum_{i=1}^k C(k, i) E[S^{i+1}]/(i + 1) E[W^{k-i}]).

    With constant size d this gives E[W] = rho d / (2 (1 - rho)) and
    Var(W) = rho d^2 (4 - rho) / (12 (1 - rho)^2).
    """
    rho = lam * d
    q = MG1WorkloadProcess(lam=lam, job_size=_deterministic(d))
    assert q.stationary_mean == pytest.approx(rho * d / (2.0 * (1.0 - rho)), rel=1e-12)
    assert q.stationary_variance == pytest.approx(
        rho * d**2 * (4.0 - rho) / (12.0 * (1.0 - rho) ** 2), rel=1e-12
    )


@pytest.mark.exactness
def test_cumulants_are_pk_mean_variance_then_nan() -> None:
    """
    Pollaczek (1930) and Khinchine (1932) mean, E[W] = lam E[S^2] / (2 (1 - rho)),
    and second moment from Takacs (1962), E[W^2] = 2 E[W]^2 + lam E[S^3] / (3 (1 - rho)).

    Uniform(0, 2) sizes: E[S] = 1, E[S^2] = 4/3, E[S^3] = 2. With lam = 0.4,
    kappa_1 = 4/9 and kappa_2 = 2 (4/9)^2 + 4/9 - (4/9)^2 = 52/81. Orders 3 and
    up have no closed form here and must be nan.
    """
    jobs = JobSizeDistribution(lambda rng, n: rng.uniform(0.0, 2.0, n), 1.0, 4.0 / 3.0, 2.0)
    kappa = MG1WorkloadProcess(lam=0.4, job_size=jobs).cumulants(5)
    assert kappa[1] == pytest.approx(4.0 / 9.0, rel=1e-12)
    assert kappa[2] == pytest.approx(52.0 / 81.0, rel=1e-12)
    assert all(math.isnan(kappa[n]) for n in (3, 4, 5))


@pytest.mark.exactness
@pytest.mark.parametrize("lam", [1.0, 1.5])
def test_unstable_queue_raises(lam: float) -> None:
    """
    Pollaczek (1930), Khinchine (1932): a stationary law exists iff
    rho = lam E[S] < 1 (Lindley, 1952, Theorem 1). For rho >= 1 the
    stationary quantities must raise ValueError.
    """
    q = MG1WorkloadProcess(lam=lam, job_size=ExponentialJump(rate=1.0))
    assert not q.has_stationary_distribution
    for attr in ("stationary_mean", "stationary_variance", "idle_probability"):
        with pytest.raises(ValueError, match="rho"):
            getattr(q, attr)
    with pytest.raises(ValueError, match="rho"):
        q.cumulants(2)


@pytest.mark.exactness
def test_triplet_raises_exactness_error() -> None:
    """
    Lindley (1952), recursion w_{r+1} = max(w_r + u_r, 0) with u_r = S_r - T_r:
    the next workload
    depends on the current one, so increments are not independent and no
    Lévy-Khintchine triplet exists.
    """
    q = MG1WorkloadProcess(lam=0.5, job_size=ExponentialJump(rate=1.0))
    with pytest.raises(ExactnessError):
        _ = q.triplet
    with pytest.raises(ExactnessError, match="independent"):
        q._triplet()
    assert not q.properties.has_independent_increments


@pytest.mark.exactness
def test_simulate_shape_start_and_support() -> None:
    """
    Lindley (1952) recursion W_{n+1} = max(W_n + S_n - T_n, 0): paths start at
    W_0 and are non-negative.
    """
    q = MG1WorkloadProcess(lam=2.0, job_size=ExponentialJump(rate=3.0), w0=0.7)
    paths = q.simulate(n_steps=50, n_paths=40, T=5.0, rng=np.random.default_rng(1))
    assert paths.shape == (40, 51)
    assert np.all(paths[:, 0] == 0.7)
    assert np.all(paths >= 0.0)
    # Between grid points the workload drains at rate at most 1.
    assert np.all(np.diff(paths, axis=1) >= -0.1 - 1e-12)


@pytest.mark.exactness
def test_simulate_drain_without_arrivals() -> None:
    """
    Lindley (1952) / Skorokhod reflection W_t = w0 + X_t + sup_{s<=t}(-(w0 + X_s))^+:
    with no arrivals X_t = -t, so W_t = max(w0 - t, 0) exactly.
    """
    q = MG1WorkloadProcess(lam=1e-12, job_size=ExponentialJump(rate=1.0), w0=1.5)
    t = np.linspace(0.0, 3.0, 31)
    paths = q.simulate(n_steps=30, n_paths=5, T=3.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(paths, np.tile(np.maximum(1.5 - t, 0.0), (5, 1)), atol=1e-14)


@pytest.mark.exactness
def test_simulate_single_job_matches_reflection_formula() -> None:
    """
    Lindley (1952) recursion with one arrival at time a of size s from w0 = 0:
    W_t = 0 for t < a and W_t = max(s - (t - a), 0) for t >= a.

    The arrival time is recovered from the path, then every grid value is
    checked against this formula.
    """
    s = 0.8
    q = MG1WorkloadProcess(lam=0.3, job_size=_deterministic(s))
    rng = np.random.default_rng(7)
    paths = q.simulate(n_steps=400, n_paths=200, T=2.0, rng=rng)
    t = np.linspace(0.0, 2.0, 401)
    checked = 0
    for path in paths:
        jumps = np.flatnonzero(np.diff(path) > 1e-12)
        if len(jumps) != 1:
            continue
        i = jumps[0]
        a = t[i + 1] + path[i + 1] - s  # path[i + 1] = s - (t[i+1] - a)
        expected = np.where(t >= a, np.maximum(s - (t - a), 0.0), 0.0)
        np.testing.assert_allclose(path, expected, atol=1e-12)
        checked += 1
    assert checked >= 30


@pytest.mark.exactness
def test_simulate_long_run_matches_mm1_stationary_law() -> None:
    """
    M/M/1 stationary law (Kleinrock, 1975, P(W > x) = rho * exp(-mu (1 - rho) x)):
    P(W = 0) = 1 - rho and
    E[W] = rho / (mu (1 - rho)). By ergodicity the time averages of simulated
    paths converge to these values.

    lam = 0.5, mu = 1 gives idle fraction 0.5, P(W > 1) = 0.5 exp(-0.5) and
    mean 1. The tolerances are about five standard errors for 20 paths of
    length 2e4.
    """
    q = MG1WorkloadProcess(lam=0.5, job_size=ExponentialJump(rate=1.0))
    paths = q.simulate(n_steps=20_000, n_paths=20, T=20_000.0, rng=np.random.default_rng(2026))
    body = paths[:, 200:]
    assert np.mean(body == 0.0) == pytest.approx(0.5, abs=0.02)
    assert np.mean(body > 1.0) == pytest.approx(0.5 * math.exp(-0.5), abs=0.02)
    assert np.mean(body) == pytest.approx(1.0, abs=0.08)
