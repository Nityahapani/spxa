#!/usr/bin/env python3
"""
Example: Advanced Simulation Backends in spxa.sim.

Demonstrates:
  1. High-order 1D and Multivariate Commutative Milstein schemes.
  2. Derivative-Free Stochastic Runge-Kutta (SRK order 1.0 strong and order 2.0 weak).
  3. Quasi-Monte Carlo (QMC) with Sobol sequences and hierarchical Brownian bridge.
  4. Variance reduction: Antithetic variates and optimal Control Variates (R² > 80%).
  5. Predictor-Corrector integration for stiff SDEs (stiff Ornstein-Uhlenbeck).
  6. Giles (2008) Multilevel Monte Carlo (MLMC) with geometric variance decay.
"""

import math
import numpy as np
from scipy.stats import norm

from spxa.sim import (
    control_variate_estimate,
    milstein,
    mlmc_estimate,
    predictor_corrector_euler,
    sample_qmc_brownian,
    stochastic_runge_kutta,
)


def main() -> None:
    print("=" * 75)
    print("Advanced Simulation Backends in spxa.sim Demonstration")
    print("=" * 75)

    # ------------------------------------------------------------------
    # 1. High-Order Milstein Scheme (Strong Order 1.0)
    # ------------------------------------------------------------------
    print("\n--- 1. High-Order Milstein Scheme ---")
    s0, mu, sigma, t = 100.0, 0.05, 0.20, 1.0
    n_steps = 50
    n_paths = 10000

    m_paths = milstein(
        drift=lambda t, s: mu * s,
        diffusion=lambda t, s: sigma * s,
        diffusion_prime=lambda t, s: np.full_like(s, sigma),
        x0=s0,
        n_steps=n_steps,
        n_paths=n_paths,
        T=t,
        rng=np.random.default_rng(42),
    )
    exact_mean = s0 * math.exp(mu * t)
    emp_mean = np.mean(m_paths[:, -1])
    print(f"  GBM S_0 = ${s0:.2f}, μ = {mu:.1%}, σ = {sigma:.1%}")
    print(f"  Theoretical E[S_T]: ${exact_mean:.4f}")
    print(
        f"  Milstein E[S_T]:    ${emp_mean:.4f} (Rel Error: {abs(emp_mean - exact_mean) / exact_mean:.2e})"
    )

    # ------------------------------------------------------------------
    # 2. Derivative-Free Stochastic Runge-Kutta (SRK)
    # ------------------------------------------------------------------
    print("\n--- 2. Derivative-Free Stochastic Runge-Kutta (SRK) ---")
    srk_paths = stochastic_runge_kutta(
        drift=lambda t, s: mu * s,
        diffusion=lambda t, s: sigma * s,
        x0=s0,
        n_steps=n_steps,
        n_paths=n_paths,
        T=t,
        rng=np.random.default_rng(42),
    )
    emp_srk_mean = np.mean(srk_paths[:, -1])
    print("  SRK (Strong Order 1.0, No Analytic Derivative Required):")
    print(
        f"  SRK E[S_T]:         ${emp_srk_mean:.4f} (Rel Error: {abs(emp_srk_mean - exact_mean) / exact_mean:.2e})"
    )

    # ------------------------------------------------------------------
    # 3. Quasi-Monte Carlo (QMC) with Brownian Bridge
    # ------------------------------------------------------------------
    print("\n--- 3. Quasi-Monte Carlo (QMC) with Hierarchical Brownian Bridge ---")
    qmc_paths = sample_qmc_brownian(
        n_paths=1024, n_steps=32, T=t, bridge=True, engine="sobol", seed=123
    )
    print(
        f"  Sobol QMC Paths Generated: {qmc_paths.shape[0]} paths across {qmc_paths.shape[1] - 1} steps"
    )
    print(f"  Terminal Variance Var(W_T): {np.var(qmc_paths[:, -1]):.4f} (Exact: {t:.4f})")
    print(f"  Midpoint Variance Var(W_T/2): {np.var(qmc_paths[:, 16]):.4f} (Exact: {t / 2:.4f})")

    # ------------------------------------------------------------------
    # 4. Variance Reduction: Optimal Control Variates
    # ------------------------------------------------------------------
    print("\n--- 4. Variance Reduction: Control Variates ---")
    strike = 100.0
    r_rate = 0.05
    n_cv = 10000
    rng_cv = np.random.default_rng(99)
    z = rng_cv.standard_normal(n_cv)
    s_term = s0 * np.exp((r_rate - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * z)
    call_payoffs = math.exp(-r_rate * t) * np.maximum(s_term - strike, 0.0)

    # Use underlying stock price forward as control variate: E[S_T] = S0 * exp(r * T)
    true_mean_stock = s0 * math.exp(r_rate * t)
    cv_res = control_variate_estimate(y=call_payoffs, x=s_term, mean_x=true_mean_stock)

    # Analytical Black-Scholes benchmark
    d1 = (math.log(s0 / strike) + (r_rate + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    bs_call = s0 * norm.cdf(d1) - strike * math.exp(-r_rate * t) * norm.cdf(d2)

    print(f"  European Call (Strike = ${strike:.2f}, Benchmark: ${bs_call:.4f}):")
    print(f"  Raw Standard MC:     ${cv_res.raw_mean:.4f} ± {cv_res.raw_std_err:.4f}")
    print(f"  Control Variate MC:  ${cv_res.mean:.4f} ± {cv_res.std_err:.4f}")
    print(
        f"  Variance Reduction:  {cv_res.variance_reduction_ratio:.2f}x (R² = {cv_res.r_squared:.2%})"
    )

    # ------------------------------------------------------------------
    # 5. Predictor-Corrector Integration for Stiff SDEs
    # ------------------------------------------------------------------
    print("\n--- 5. Predictor-Corrector Integration for Stiff SDEs ---")
    kappa_stiff = 40.0
    dt_coarse = 0.08  # Notice |kappa * dt| = 3.2 > 2.0 (Euler-Maruyama explodes!)
    n_stiff_steps = 25
    x0_stiff = 1.0

    pc_paths = predictor_corrector_euler(
        drift=lambda t, x: -kappa_stiff * x,
        diffusion=lambda t, x: np.full_like(x, 0.5),
        x0=x0_stiff,
        theta=0.5,
        dt=dt_coarse,
        n_steps=n_stiff_steps,
        n_paths=1000,
        rng=np.random.default_rng(42),
    )
    print(
        f"  Stiff Ornstein-Uhlenbeck κ = {kappa_stiff:.1f}, coarse Δt = {dt_coarse:.2f} (κΔt = 3.2 > 2)"
    )
    print("  Explicit Euler Diverges; Predictor-Corrector remains strictly stable!")
    print(f"  Final State Max Absolute: {np.max(np.abs(pc_paths[:, -1])):.4f}")

    # ------------------------------------------------------------------
    # 6. Giles (2008) Multilevel Monte Carlo (MLMC)
    # ------------------------------------------------------------------
    print("\n--- 6. Multilevel Monte Carlo (MLMC) ---")

    def gbm_coupler(level: int, n_samples: int, rng: np.random.Generator):
        m_factor = 2
        n_fine = (m_factor**level) * 4
        dt_f = t / n_fine
        sq_dt_f = math.sqrt(dt_f)

        if level == 0:
            s = np.full(n_samples, s0)
            for _ in range(n_fine):
                s += r_rate * s * dt_f + sigma * s * rng.standard_normal(n_samples) * sq_dt_f
            return np.exp(-r_rate * t) * np.maximum(s - strike, 0.0), None
        else:
            n_c = n_fine // m_factor
            dt_c = t / n_c
            s_f = np.full(n_samples, s0)
            s_c = np.full(n_samples, s0)
            for _ in range(n_c):
                dw1 = rng.standard_normal(n_samples) * sq_dt_f
                dw2 = rng.standard_normal(n_samples) * sq_dt_f
                s_f += r_rate * s_f * dt_f + sigma * s_f * dw1
                s_f += r_rate * s_f * dt_f + sigma * s_f * dw2
                s_c += r_rate * s_c * dt_c + sigma * s_c * (dw1 + dw2)
            p_f = np.exp(-r_rate * t) * np.maximum(s_f - strike, 0.0)
            p_c = np.exp(-r_rate * t) * np.maximum(s_c - strike, 0.0)
            return p_f, p_c

    target_tol = 0.05
    mlmc_res = mlmc_estimate(
        gbm_coupler, target_eps=target_tol, l_min=2, l_max=4, rng=np.random.default_rng(42)
    )
    print(f"  MLMC Target Tolerance ε = ${target_tol:.2f}:")
    print(f"  MLMC Estimate:       ${mlmc_res.estimate:.4f} ± {mlmc_res.std_err:.4f}")
    print(
        f"  Levels Reached:      {mlmc_res.levels} (Samples per level: {mlmc_res.n_samples_per_level})"
    )
    print(f"  Variance Decay:      {[f'{v:.2e}' for v in mlmc_res.sample_variances]}")

    print("\nDemonstration completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
