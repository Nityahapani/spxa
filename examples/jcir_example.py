"""
Example demonstrating Jump-Extended Cox-Ingersoll-Ross (JCIR) process.

This script illustrates:
1. Model initialization and Feller boundary condition.
2. Stationary distribution moments and conditional expectation dynamics.
3. Closed-form Zero-Coupon Bond pricing, term structure yield curve, and forward rates.
4. Characteristic function and Fourier inversion properties.
5. Exact non-central chi-squared path simulation with compound Poisson jump shocks.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spxa import JCIRProcess


def main() -> None:
    print("=" * 72)
    print("Jump-Extended Cox-Ingersoll-Ross (JCIR) Process Demonstration")
    print("=" * 72)

    # 1. Model Configuration
    kappa = 1.2
    theta = 0.05
    sigma = 0.15
    lambda_j = 0.8
    mu_j = 0.03
    r0 = 0.04

    jcir = JCIRProcess(
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        lambda_j=lambda_j,
        mu_j=mu_j,
        r0=r0,
    )
    print(f"\nInitialized Process: {jcir}")
    print(f"  Mean Reversion (κ):          {jcir.kappa:.2f}")
    print(f"  Diffusion Mean (θ):          {jcir.theta:.4f}")
    print(f"  Volatility (σ):              {jcir.sigma:.4f}")
    print(f"  Jump Intensity (λ_J):        {jcir.lambda_j:.2f} jumps/year")
    print(f"  Mean Jump Size (μ_J):        {jcir.mu_j:.4f} ({jcir.mu_j * 10000:.0f} bps)")
    print(f"  Initial Short Rate (r_0):    {jcir.r0:.4f}")
    print(
        f"  Diffusion Feller Ratio:      {jcir.feller_ratio:.2f} (Satisfied: {jcir.feller_satisfied})"
    )

    # 2. Stationary and Conditional Moments
    print("\nMoment Dynamics:")
    print(
        f"  Stationary Mean θ_∞:         {jcir.stationary_mean:.4f} (baseline θ: {jcir.theta:.4f})"
    )
    print(f"  Stationary Variance:         {jcir.stationary_variance:.6f}")
    print(f"  Stationary Std Dev:          {jcir.stationary_std:.4f}")

    t = 1.0
    cond_mean = jcir.conditional_mean(t=t)
    cond_var = jcir.conditional_variance(t=t)
    cond_std = jcir.conditional_std(t=t)
    print(f"  Conditional Mean at t={t:.1f}:     {cond_mean:.4f}")
    print(f"  Conditional Variance at t={t:.1f}: {cond_var:.6f} (Std: {cond_std:.4f})")

    # 3. Term Structure & Bond Pricing
    maturities = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    print("\nTerm Structure of Zero-Coupon Bonds & Yields:")
    print(
        f"  {'Maturity (yr)':<15} {'Bond Price P(0, T)':<20} {'Yield y(0, T)':<18} {'Forward Rate f(0, T)':<20}"
    )
    print("  " + "-" * 73)
    for mat in maturities:
        price = jcir.bond_price(mat)
        yld = jcir.yield_curve(mat)
        fwd = jcir.forward_rate(mat)
        print(f"  {mat:<15.2f} {price:<20.6f} {yld:<18.4%} {fwd:<20.4%}")

    # 4. Characteristic Function
    u = 1.0
    cf = jcir.char_func(u=u, t=t)
    print(f"\nCharacteristic Function at t = {t:.1f}:")
    print(f"  φ({u}; {t:.1f}) = {cf.real:.6f} + {cf.imag:.6f}i (modulus: {abs(cf):.6f})")

    # 5. Exact Path Simulation
    n_paths = 40_000
    n_steps = 25
    rng = np.random.default_rng(2026)

    print(f"\nSimulating {n_paths:,} sample paths over T = {t:.1f} ({n_steps} steps)...")
    paths = jcir.simulate(n_steps=n_steps, time_horizon=t, n_paths=n_paths, rng=rng)
    mc_mean = float(np.mean(paths[:, -1]))
    mc_var = float(np.var(paths[:, -1]))

    print(
        f"  Empirical Mean:              {mc_mean:.4f} (Theoretical: {cond_mean:.4f}, error: {abs(mc_mean - cond_mean):.4e})"
    )
    print(
        f"  Empirical Variance:          {mc_var:.6f} (Theoretical: {cond_var:.6f}, error: {abs(mc_var - cond_var):.4e})"
    )
    print(f"  All simulated rates non-negative: {bool(np.all(paths >= 0.0))}")

    print("\nDemonstration completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
