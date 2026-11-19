#!/usr/bin/env python3
"""
Example: Chen Three-Factor Short Rate Model (Chen 1996).

Demonstrates:
  1. Initialization of Chen 3-factor short rate model with stochastic mean and volatility.
  2. Verification of Feller conditions for central tendency and volatility processes.
  3. Exact conditional expectations of short rate, central tendency, and variance.
  4. Exact matrix exponential 10-dimensional second moments and variance system.
  5. Zero-coupon bond pricing and term structure yield curve.
  6. Coupled 3-factor path simulation.
"""

import numpy as np

from spxa import ChenModel


def main() -> None:
    print("=" * 72)
    print("Chen Three-Factor Short Rate Model Demonstration (Chen 1996)")
    print("=" * 72)

    # 1. Model Configuration
    # Short rate r0 = 4.0%, speed k = 1.5
    # Central tendency: target theta_bar = 5.0%, kappa = 0.8, alpha = 0.04
    # Variance factor: target zeta_bar = 0.02, lambda = 1.0, beta = 0.05
    model = ChenModel(
        k=1.5,
        theta_bar=0.05,
        kappa=0.8,
        alpha=0.04,
        zeta_bar=0.02,
        lambda_param=1.0,
        beta=0.05,
        r0=0.04,
        theta0=0.05,
        zeta0=0.02,
    )

    print(f"\nModel Configuration: {model}")
    print(f"  Feller Central Tendency (2κθ_bar ≥ α²): {model.feller_theta}")
    print(f"  Feller Variance Factor (2λζ_bar ≥ β²): {model.feller_zeta}")
    print(f"  Stationary Mean Short Rate:            {model.stationary_mean_short_rate:.2%}")
    print(f"  Stationary Mean Variance Factor:       {model.stationary_mean_volatility:.4f}")

    # 2. Exact Conditional Expectations
    tau = 1.0
    mean_r = model.conditional_mean(t=tau)
    mean_th = model.conditional_mean_central_tendency(t=tau)
    mean_z = model.conditional_mean_volatility(t=tau)

    print(f"\nConditional Means at horizon T = {tau:.1f} yr:")
    print(f"  Expected Short Rate E[r_T]:            {mean_r:.4%}")
    print(f"  Expected Central Tendency E[θ_T]:      {mean_th:.4%}")
    print(f"  Expected Variance Factor E[ζ_T]:       {mean_z:.4f}")

    # 3. Exact 10-Dimensional Matrix Exponential Moment System
    moments = model.conditional_moments_system(t=tau)
    var_r = model.conditional_variance(t=tau)
    std_r = model.conditional_std(t=tau)

    print(f"\nExact Matrix Exponential Moments at horizon T = {tau:.1f} yr:")
    print(f"  Second Moment E[r_T²]:                 {moments['r2']:.6f}")
    print(f"  Covariance Term E[r_T * θ_T]:          {moments['r_theta']:.6f}")
    print(f"  Covariance Term E[r_T * ζ_T]:          {moments['r_zeta']:.6f}")
    print(f"  Conditional Variance Var(r_T):         {var_r:.6e} (Std: {std_r:.4%})")

    # 4. Term Structure Yield Curve
    maturities = [0.25, 0.5, 1.0, 2.0, 5.0]
    yields = model.yield_curve(maturities, n_steps=50, n_paths=8000, rng=np.random.default_rng(42))

    print("\nTerm Structure Zero-Coupon Yield Curve:")
    print("  Maturity (yr)     Bond Price P(0, T)     Zero Yield y(T)")
    print("  --------------------------------------------------------")
    for mat, yld in zip(maturities, yields):
        bp = np.exp(-yld * mat)
        print(f"  {mat:<16.2f}  {bp:<21.4f}  {yld:.4%}")

    # 5. Coupled Simulation
    n_paths = 30000
    n_steps = 100
    print(
        f"\nSimulating {n_paths:,} coupled 3-factor paths ({n_steps} steps over T = {tau:.1f} yr)..."
    )
    r_paths, th_paths, z_paths = model.simulate_factors(
        n_steps=n_steps, time_horizon=tau, n_paths=n_paths, rng=np.random.default_rng(123)
    )

    emp_mean_r = np.mean(r_paths[:, -1])
    emp_var_r = np.var(r_paths[:, -1])
    emp_mean_th = np.mean(th_paths[:, -1])
    emp_mean_z = np.mean(z_paths[:, -1])

    print(f"  Empirical Mean Short Rate:             {emp_mean_r:.4%} (Theoretical: {mean_r:.4%})")
    print(f"  Empirical Variance Short Rate:         {emp_var_r:.6e} (Theoretical: {var_r:.6e})")
    print(
        f"  Empirical Mean Central Tendency:       {emp_mean_th:.4%} (Theoretical: {mean_th:.4%})"
    )
    print(f"  Empirical Mean Variance Factor:        {emp_mean_z:.4f} (Theoretical: {mean_z:.4f})")

    print("\nDemonstration completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
