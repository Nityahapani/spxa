"""
Example demonstrating the Jacobi bounded diffusion process.

This script illustrates:
1. Model initialization on general bounded intervals [y_min, y_max].
2. Boundary classifications at both endpoints (entrance vs regular reflecting).
3. Exact polynomial moments (mean, second moment, variance).
4. Beta stationary distribution density and cumulative distribution function.
5. High-order Milstein path simulation with strict boundary compliance.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spxa import JacobiProcess


def main() -> None:
    print("=" * 72)
    print("Jacobi Bounded Diffusion Process Demonstration")
    print("=" * 72)

    # 1. Model Configuration
    kappa = 1.50
    theta = 0.40
    sigma = 0.50
    y_min = 0.00
    y_max = 1.00
    y0 = 0.30

    jp = JacobiProcess(
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        y_min=y_min,
        y_max=y_max,
        y0=y0,
    )

    print(f"\nInitialized Process: {jp}")
    print(f"  Domain Interval:              [{jp.y_min:.2f}, {jp.y_max:.2f}]")
    print(f"  Initial State (Y_0):           {jp.y0:.2f}")
    print(f"  Mean-Reversion Speed (κ):      {jp.kappa:.2f}")
    print(f"  Long-Term Target (θ):          {jp.theta:.2f}")
    print(f"  Diffusion Parameter (σ):       {jp.sigma:.2f}")
    print(f"  Stationary Beta Shapes:        α = {jp.alpha:.2f}, β = {jp.beta:.2f}")
    print(f"  Lower Boundary Classification: {jp.boundary_classification_lower}")
    print(f"  Upper Boundary Classification: {jp.boundary_classification_upper}")

    # 2. Moments & Variance
    print("\nStationary Moments:")
    print(f"  Stationary Mean:               {jp.stationary_mean:.4f}")
    print(
        f"  Stationary Variance:           {jp.stationary_variance:.6f} (Std: {jp.stationary_std:.4f})"
    )

    t = 1.0
    cond_mean = jp.conditional_mean(t=t)
    cond_m2 = jp.conditional_second_moment(t=t)
    cond_var = jp.conditional_variance(t=t)
    cond_std = jp.conditional_std(t=t)

    print(f"\nConditional Moments at horizon T = {t:.1f} yr:")
    print(f"  Expected State E[Y_T]:         {cond_mean:.4f}")
    print(f"  Second Moment E[Y_T²]:         {cond_m2:.4f}")
    print(f"  Conditional Variance:          {cond_var:.6f} (Std: {cond_std:.4f})")

    # 3. Stationary Distribution Function Values
    grid = np.linspace(0.1, 0.9, 9)
    print("\nStationary Beta Distribution Profile:")
    print(f"  {'State y':<15} {'Density p(y)':<20} {'CDF P(Y <= y)':<20}")
    print("  " + "-" * 55)
    for y_val in grid:
        p_val = jp.stationary_density(y_val)
        cdf_val = jp.stationary_cdf(y_val)
        print(f"  {y_val:<15.2f} {p_val:<20.6f} {cdf_val:<20.6%}")

    # 4. High-Order Milstein Simulation
    n_paths = 40_000
    n_steps = 30
    rng = np.random.default_rng(2026)

    print(f"\nSimulating {n_paths:,} sample paths over T = {t:.1f} ({n_steps} steps)...")
    paths = jp.simulate(n_steps=n_steps, time_horizon=t, n_paths=n_paths, rng=rng)
    mc_mean = float(np.mean(paths[:, -1]))
    mc_var = float(np.var(paths[:, -1]))
    min_val = float(np.min(paths))
    max_val = float(np.max(paths))

    print(
        f"  Empirical Mean Y_T:            {mc_mean:.4f} (Theoretical: {cond_mean:.4f}, error: {abs(mc_mean - cond_mean):.4e})"
    )
    print(
        f"  Empirical Variance Var(Y_T):   {mc_var:.6f} (Theoretical: {cond_var:.6f}, error: {abs(mc_var - cond_var):.4e})"
    )
    print(
        f"  Path Range [min, max]:         [{min_val:.4f}, {max_val:.4f}] (Inside bounds: {bool(min_val >= y_min and max_val <= y_max)})"
    )

    print("\nDemonstration completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
