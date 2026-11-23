"""
Example demonstrating Squared Bessel (BESQ) and Bessel (BES) processes.

This script illustrates:
1. Analytical properties, dimension parameter, and Feller boundary classification.
2. Exact cumulants, moments, Laplace transform, and characteristic function.
3. Transition probability density and cumulative distribution function.
4. Bessel process moments and relation R_t = sqrt(X_t).
5. Exact non-central chi-squared path simulation with zero discretization error.
"""

import math
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spxa import SquaredBesselProcess, BesselProcess


def main() -> None:
    print("=" * 72)
    print("Squared Bessel (BESQ) & Bessel (BES) Processes Demonstration")
    print("=" * 72)

    # 1. Model Configuration
    dim = 3.0
    x0 = 2.0
    besq = SquaredBesselProcess(dim=dim, x0=x0)
    print(f"\nInitialized Process: {besq}")
    print(f"  Dimension (δ):               {besq.dim:.2f}")
    print(f"  Index (ν = δ/2 - 1):         {besq.index:.2f}")
    print(f"  Initial Value (X_0):         {besq.x0:.2f}")
    print(f"  Feller Boundary at 0:        {besq.boundary_classification}")

    # 2. Exact Cumulants & Moments
    t = 1.0
    cums = besq.conditional_cumulants(t=t, order=4)
    print(f"\nExact Cumulants at t = {t:.1f}:")
    print(f"  κ_1 (Mean E[X_t]):           {cums[1]:.4f} (formula: x0 + δt = {x0 + dim * t:.4f})")
    print(
        f"  κ_2 (Variance Var(X_t)):     {cums[2]:.4f} (formula: 4 x0 t + 2 δ t² = {4 * x0 * t + 2 * dim * t**2:.4f})"
    )
    print(f"  κ_3 (Third cumulant):        {cums[3]:.4f}")
    print(f"  κ_4 (Fourth cumulant):       {cums[4]:.4f}")

    # 3. Exact Laplace Transform and Characteristic Function
    lmbda = 0.5
    lt = besq.laplace_transform(lmbda=lmbda, t=t)
    u = 1.0
    cf = besq.char_func(u=u, t=t)
    print(f"\nTransform Analysis at t = {t:.1f}:")
    print(f"  Laplace Transform E[e^(-{lmbda} X_t)]: {lt:.6f}")
    print(
        f"  Char Function E[e^(i*{u}*X_t)]:      {cf.real:.6f} + {cf.imag:.6f}i (modulus: {abs(cf):.6f})"
    )

    # 4. Bessel Process Analysis (R_t = sqrt(X_t))
    r0 = math.sqrt(x0)
    bes = BesselProcess(dim=dim, r0=r0)
    bes_mean = bes.conditional_mean(t=t)
    bes_sec_mom = bes.conditional_second_moment(t=t)
    bes_var = bes.conditional_variance(t=t)
    print(f"\nBessel Process BES^{dim:.1f} (R_0 = {r0:.4f}):")
    print(f"  E[R_t] (via 1F1 hypergeometric):     {bes_mean:.6f}")
    print(f"  E[R_t²] = E[X_t]:                    {bes_sec_mom:.6f}")
    print(f"  Var(R_t):                            {bes_var:.6f}")

    # 5. Exact Path Simulation
    n_paths = 50_000
    n_steps = 25
    rng = np.random.default_rng(2026)

    print(f"\nSimulating {n_paths:,} exact paths over T = {t:.1f} ({n_steps} steps)...")
    besq_paths = besq.simulate(n_steps=n_steps, time_horizon=t, n_paths=n_paths, rng=rng)
    mc_mean = float(np.mean(besq_paths[:, -1]))
    mc_var = float(np.var(besq_paths[:, -1]))

    print(
        f"  Empirical Mean:                      {mc_mean:.4f} (Theoretical: {cums[1]:.4f}, error: {abs(mc_mean - cums[1]):.4e})"
    )
    print(
        f"  Empirical Variance:                  {mc_var:.4f} (Theoretical: {cums[2]:.4f}, error: {abs(mc_var - cums[2]):.4e})"
    )
    print(f"  All simulated values non-negative:   {bool(np.all(besq_paths >= 0.0))}")

    print("\nDemonstration completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
