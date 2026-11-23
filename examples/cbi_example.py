#!/usr/bin/env python3
"""
Example: Continuous-State Branching Process with Immigration (CBI) / Dawson-Watanabe.

Demonstrates:
  1. Initialization of subcritical Dawson-Watanabe CBI process with jump immigration.
  2. Exact closed-form Riccati solution v(t, λ) and Laplace transform.
  3. Exact conditional expectations and variance with jump-diffusion immigration.
  4. Extinction probability for pure CSBP (no immigration).
  5. High-order Milstein path simulation with non-negative reflection.
"""

import numpy as np

from spxa import CBISubordinator


def main() -> None:
    print("=" * 72)
    print("Continuous-State Branching with Immigration (CBI) Demonstration")
    print("=" * 72)

    # 1. Model Configuration
    # Continuous immigration a = 0.05, branching decay b = 0.8, diffusion c = 0.1
    # Compound Poisson jump immigration: rate λ_J = 0.4, mean jump 1/η = 1/20 = 0.05
    cbi = CBISubordinator(
        a=0.05,
        b=0.8,
        c=0.1,
        lambda_j=0.4,
        eta=20.0,
        x0=0.1,
    )

    print(f"\nModel Configuration: {cbi}")
    print(
        f"  Branching Drift (b):                   {cbi.b:.2f} (Subcritical: {cbi.is_subcritical})"
    )
    print(f"  Branching Diffusion (c):               {cbi.c:.2f}")
    print(f"  Continuous Immigration Rate (a):       {cbi.a:.4f}")
    print(
        f"  Jump Immigration Rate (λ_J):           {cbi.lambda_j:.2f} (Mean Jump: {1.0 / cbi.eta:.4f})"
    )
    print(f"  Stationary Mean:                       {cbi.stationary_mean:.4f}")
    print(f"  Stationary Variance:                   {cbi.stationary_variance:.6f}")

    # 2. Exact Laplace Transform at horizon T = 1.0 yr
    tau = 1.0
    lam = 1.5
    laplace_val = cbi.laplace_transform(lam=lam, t=tau)
    v_val = cbi.riccati_v(lam=lam, t=tau)

    print(f"\nExact Affine Structure at horizon T = {tau:.1f} yr:")
    print(f"  Riccati Solution v(T, λ = {lam}):       {v_val:.6f}")
    print(f"  Laplace Transform E[e^(-λ X_T)]:       {laplace_val:.6f}")

    # 3. Exact Conditional Moments
    mean_val = cbi.conditional_mean(t=tau)
    var_val = cbi.conditional_variance(t=tau)
    std_val = cbi.conditional_std(t=tau)

    print(f"\nConditional Moments at horizon T = {tau:.1f} yr:")
    print(f"  Expected State E[X_T]:                 {mean_val:.4f}")
    print(f"  Conditional Variance Var(X_T):         {var_val:.6f} (Std: {std_val:.4f})")

    # 4. Extinction Probability (Pure CSBP)
    pure_csbp = CBISubordinator(a=0.0, b=0.5, c=0.2, lambda_j=0.0, x0=0.05)
    p_ext = pure_csbp.extinction_probability(t=2.0)
    print(
        f"\nPure CSBP Extinction (X0 = {pure_csbp.x0}, a = 0, b = {pure_csbp.b}, c = {pure_csbp.c}):"
    )
    print(f"  Extinction Probability P(X_2 = 0):     {p_ext:.4%}")

    # 5. Simulation
    n_paths = 40000
    n_steps = 100
    print(f"\nSimulating {n_paths:,} sample paths ({n_steps} steps over T = {tau:.1f} yr)...")
    paths = cbi.simulate(
        n_steps=n_steps, time_horizon=tau, n_paths=n_paths, rng=np.random.default_rng(42)
    )

    term = paths[:, -1]
    emp_mean = np.mean(term)
    emp_var = np.var(term)
    emp_laplace = np.mean(np.exp(-lam * term))

    print(f"  Empirical Mean:                        {emp_mean:.4f} (Theoretical: {mean_val:.4f})")
    print(f"  Empirical Variance:                    {emp_var:.6f} (Theoretical: {var_val:.6f})")
    print(
        f"  Empirical Laplace E[e^(-λ X_T)]:       {emp_laplace:.6f} (Theoretical: {laplace_val:.6f})"
    )
    print(f"  Paths strictly non-negative:           {np.all(paths >= 0.0)}")

    print("\nDemonstration completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
