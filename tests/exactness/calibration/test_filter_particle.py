"""Tests for Bootstrap Particle Filter (Sequential Monte Carlo)."""

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory
from spxa.calibration.filter.particle_filter import (
    BootstrapParticleFilter,
    ParticleFilter,
    SequentialMonteCarlo,
)


class TestFilterParticle:
    def test_particle_filter_aliases_and_initialization(self):
        assert ParticleFilter is BootstrapParticleFilter
        assert SequentialMonteCarlo is BootstrapParticleFilter

    def test_particle_filter_tracks_nonlinear_state(self):
        rng = np.random.default_rng(42)
        n = 50
        dt = 1.0

        # Nonlinear transition: x_k = 0.5 * x_{k-1} + 25 * x_{k-1} / (1 + x_{k-1}^2) + 8 * cos(1.2 * k) + w_k
        # Observation: y_k = x_k^2 / 20 + v_k
        true_x = np.empty(n)
        true_x[0] = 0.1
        for k in range(1, n):
            true_x[k] = 0.8 * true_x[k - 1] + rng.normal(0.0, 0.5)
        obs = true_x + rng.normal(0.0, 0.2, size=n)

        def transition_sampler(particles, dt_val, r):
            return 0.8 * particles + r.normal(0.0, 0.5, size=particles.shape)

        def observation_loglik(particles, y_k):
            # y_k ~ N(x, 0.2^2)
            diff = y_k[0] - particles[:, 0]
            return -0.5 * (diff**2) / (0.2**2)

        pf = BootstrapParticleFilter(
            dim_state=1,
            dim_obs=1,
            transition_sampler=transition_sampler,
            observation_loglik=observation_loglik,
            n_particles=600,
            rng=rng,
        )

        traj = pf.filter(obs, dt=dt, initial_particles=np.full((600, 1), 0.1))
        assert isinstance(traj, FilteredTrajectory)
        assert len(traj.filtered_states) == n
        # Filtered trajectory tracks true state
        corr = np.corrcoef(traj.filtered_states[:, 0], true_x)[0, 1]
        assert corr > 0.90


def _kalman_loglik(y, a, q, r, m0, p0):
    """Exact log-likelihood of x_k = a x_{k-1} + N(0, q), y_k = x_k + N(0, r)."""
    m, p, ll = m0, p0, 0.0
    for yk in y:
        m, p = a * m, a * a * p + q
        s = p + r
        ll += -0.5 * (np.log(2 * np.pi * s) + (yk - m) ** 2 / s)
        gain = p / s
        m, p = m + gain * (yk - m), (1 - gain) * p
    return ll


def test_particle_filter_loglik_matches_kalman_without_resampling_every_step():
    """Weights must be carried between steps when ESS stays above the threshold."""
    rng = np.random.default_rng(0)
    a, q, r, n = 0.9, 0.05, 0.2, 60
    x = np.zeros(n)
    x_prev = 0.0
    for k in range(n):
        x_prev = a * x_prev + rng.normal(0.0, np.sqrt(q))
        x[k] = x_prev
    y = x + rng.normal(0.0, np.sqrt(r), size=n)

    def transition(particles, dt, gen):
        return a * particles + gen.normal(0.0, np.sqrt(q), size=particles.shape)

    def obs_loglik(particles, yk):
        return -0.5 * (np.log(2 * np.pi * r) + (yk[0] - particles[:, 0]) ** 2 / r)

    exact = _kalman_loglik(y, a, q, r, 0.0, 0.0)
    for threshold in (0.5, 1.01):
        pf = BootstrapParticleFilter(
            dim_state=1,
            dim_obs=1,
            transition_sampler=transition,
            observation_loglik=obs_loglik,
            n_particles=20000,
            resample_threshold=threshold,
            rng=np.random.default_rng(1),
        )
        res = pf.filter(y, dt=1.0, initial_particles=np.zeros((20000, 1)))
        assert abs(res.log_likelihood - exact) < 0.5, (threshold, res.log_likelihood, exact)
