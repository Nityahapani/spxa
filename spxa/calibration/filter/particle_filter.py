r"""Bootstrap Particle Filter (Sequential Monte Carlo) for non-Gaussian state-space models.

Implements Sequential Importance Resampling (SIR) with systematic/stratified resampling:
    x_k^{(i)} \sim p(x_k | x_{k-1}^{(i)})
    w_k^{(i)} \propto w_{k-1}^{(i)} * p(y_k | x_k^{(i)})
    \log \hat{p}(y_{1:n}) = \sum_k \log \sum_i w_{k-1}^{(i)} p(y_k | x_k^{(i)})

Systematic resampling is triggered when ESS < resample_threshold * N; weights are reset to
1/N after resampling and carried forward otherwise. The likelihood estimator is unbiased
(Del Moral 2004).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory
from spxa.calibration.filter.resampling import effective_sample_size, systematic_resample


class BootstrapParticleFilter:
    r"""Sequential Importance Resampling (SIR) Particle Filter."""

    def __init__(
        self,
        dim_state: int,
        dim_obs: int,
        transition_sampler: Callable[[np.ndarray, float, np.random.Generator], np.ndarray],
        observation_loglik: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_particles: int = 500,
        resample_threshold: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.dim_state = int(dim_state)
        self.dim_obs = int(dim_obs)
        self.transition_sampler = transition_sampler
        self.observation_loglik = observation_loglik
        self.n_particles = int(n_particles)
        self.resample_threshold = float(resample_threshold)
        self.rng = rng if rng is not None else np.random.default_rng()

    def filter(
        self,
        observations: Sequence[float] | np.ndarray,
        dt: float,
        initial_particles: Optional[np.ndarray] = None,
    ) -> FilteredTrajectory:
        r"""Execute particle filtering across sequential observations."""
        y_obs = np.asarray(observations, dtype=float)
        if y_obs.ndim == 1 and self.dim_obs == 1:
            y_obs = y_obs[:, None]

        n_steps, _ = y_obs.shape
        n_p = self.n_particles

        # Initialize particles
        if initial_particles is not None:
            particles = np.asarray(initial_particles, dtype=float)
        else:
            particles = self.rng.standard_normal((n_p, self.dim_state))

        weights = np.full(n_p, 1.0 / n_p, dtype=float)

        filtered_states = np.empty((n_steps, self.dim_state), dtype=float)
        filtered_covs = np.empty((n_steps, self.dim_state, self.dim_state), dtype=float)
        innovations = np.zeros((n_steps, self.dim_obs), dtype=float)
        total_loglik = 0.0

        times = np.arange(n_steps) * dt

        for k in range(n_steps):
            y_k = y_obs[k]

            # 1. State propagation: sample x_k from transition
            particles = self.transition_sampler(particles, dt, self.rng)

            # 2. Measurement update: compute log likelihood of y_k given each particle
            # observation_loglik takes particles of shape (n_p, dim_state) and y_k, returns (n_p,)
            log_obs = self.observation_loglik(particles, y_k)
            # Sequential importance update w_k ∝ w_{k-1} p(y_k | x_k), carried in log-space.
            # Incremental marginal likelihood: p(y_k | y_{1:k-1}) ≈ sum_i w_{k-1}^i p(y_k | x_k^i).
            log_weights = np.log(np.maximum(weights, 1e-300)) + log_obs
            max_logw = np.max(log_weights)
            unnorm_w = np.exp(log_weights - max_logw)
            marginal_ll_k = max_logw + math.log(max(1e-300, float(np.sum(unnorm_w))))
            total_loglik += marginal_ll_k

            weights = unnorm_w / np.sum(unnorm_w)

            # 3. Estimate posterior mean and covariance
            mean_k = np.sum(weights[:, None] * particles, axis=0)
            diff = particles - mean_k[None, :]
            cov_k = diff.T @ (weights[:, None] * diff)

            filtered_states[k] = mean_k
            filtered_covs[k] = cov_k

            # 4. Resample if ESS < threshold * N
            ess = effective_sample_size(weights)
            if ess < self.resample_threshold * n_p:
                idx = systematic_resample(weights, rng=self.rng)
                particles = particles[idx]
                weights.fill(1.0 / n_p)

        return FilteredTrajectory(
            times=times,
            filtered_states=filtered_states,
            filtered_covariances=filtered_covs,
            innovations=innovations,
            log_likelihood=total_loglik,
        )


ParticleFilter = BootstrapParticleFilter
SequentialMonteCarlo = BootstrapParticleFilter
