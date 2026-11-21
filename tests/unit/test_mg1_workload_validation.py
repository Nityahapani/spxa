"""Input validation for MG1WorkloadProcess and JobSizeDistribution."""

import numpy as np
import pytest

from spxa.zoo.beyond.mg1_workload import JobSizeDistribution, MG1WorkloadProcess
from spxa.zoo.levy.compound_poisson import DiscreteJump, ExponentialJump, UniformJump


def _const(d: float) -> JobSizeDistribution:
    return JobSizeDistribution(lambda rng, n: np.full(n, d), d, d**2, d**3)


@pytest.mark.parametrize("d", [0.1, 0.3, 1.0 / 3.0, 7.0])
def test_point_mass_moments_accepted(d: float) -> None:
    # Point masses attain equality in both Cauchy-Schwarz bounds.
    MG1WorkloadProcess(lam=0.01, job_size=_const(d))


@pytest.mark.parametrize(
    "law", [ExponentialJump(rate=2.0), UniformJump(low=0.0, high=1.0), DiscreteJump([1.0, 2.0])]
)
def test_library_job_laws_accepted(law: object) -> None:
    MG1WorkloadProcess(lam=0.1, job_size=law)  # type: ignore[arg-type]


def test_variance_passed_as_second_moment_rejected() -> None:
    # Mean 2 and variance 1 give E[S^2] = 5; passing the variance 1 instead
    # violates E[S^2] >= E[S]^2 = 4.
    jobs = JobSizeDistribution(lambda rng, n: np.full(n, 2.0), 2.0, 1.0, 8.0)
    with pytest.raises(ValueError, match="variance"):
        MG1WorkloadProcess(lam=0.1, job_size=jobs)


def test_third_moment_below_cauchy_schwarz_rejected() -> None:
    # E[S] = 1, E[S^2] = 2 needs E[S^3] >= 4.
    jobs = JobSizeDistribution(lambda rng, n: np.ones(n), 1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="Inconsistent"):
        MG1WorkloadProcess(lam=0.1, job_size=jobs)


def test_finite_third_with_infinite_second_rejected() -> None:
    jobs = JobSizeDistribution(lambda rng, n: np.ones(n), 1.0, float("inf"), 5.0)
    with pytest.raises(ValueError, match="infinite"):
        MG1WorkloadProcess(lam=0.1, job_size=jobs)


def test_heavy_tailed_moments_accepted() -> None:
    jobs = JobSizeDistribution(lambda rng, n: np.ones(n), 1.0, 3.0, float("inf"))
    q = MG1WorkloadProcess(lam=0.5, job_size=jobs)
    assert q.stationary_mean == pytest.approx(1.5)
    assert q.stationary_variance == float("inf")
    assert not q.properties.has_finite_variance


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.5])
def test_non_finite_or_negative_samples_rejected(bad: float) -> None:
    jobs = JobSizeDistribution(lambda rng, n: np.full(n, bad), 1.0, 1.0, 1.0)
    q = MG1WorkloadProcess(lam=50.0, job_size=jobs)
    with pytest.raises(ValueError, match="finite and non-negative"):
        q.simulate(n_steps=10, n_paths=3, rng=np.random.default_rng(0))


def test_wrong_sample_count_rejected() -> None:
    jobs = JobSizeDistribution(lambda rng, n: np.ones(n + 1), 1.0, 1.0, 1.0)
    q = MG1WorkloadProcess(lam=50.0, job_size=jobs)
    with pytest.raises(ValueError, match="expected"):
        q.simulate(n_steps=10, n_paths=3, rng=np.random.default_rng(0))


@pytest.mark.parametrize(
    "kwargs", [{"lam": 0.0}, {"lam": -1.0}, {"lam": float("nan")}, {"lam": 1.0, "w0": -0.1}]
)
def test_bad_parameters_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        MG1WorkloadProcess(job_size=ExponentialJump(rate=1.0), **kwargs)  # type: ignore[arg-type]


def test_simulate_argument_checks() -> None:
    q = MG1WorkloadProcess(lam=1.0, job_size=ExponentialJump(rate=2.0))
    with pytest.raises(ValueError):
        q.simulate(n_steps=0)
    with pytest.raises(ValueError):
        q.simulate(n_steps=5, n_paths=0)
    with pytest.raises(ValueError):
        q.simulate(n_steps=5, T=0.0)


def test_job_size_distribution_moment_range() -> None:
    with pytest.raises(ValueError):
        _const(1.0).moment(4)
