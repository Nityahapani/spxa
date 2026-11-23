"""
Example demonstrating the Schwartz Two-Factor commodity pricing model.

This script illustrates:
1. Model initialization with spot and convenience yield parameters.
2. Convenience yield mean-reversion and log-spot analytical moments.
3. Term structure of commodity futures prices (contango vs backwardation).
4. European Call and Put option pricing with exact Put-Call parity verification.
5. Joint Monte Carlo simulation of commodity spot and convenience yield paths.
"""

import math
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spxa import SchwartzTwoFactor


def main() -> None:
    print("=" * 72)
    print("Schwartz Two-Factor Commodity Model Demonstration")
    print("=" * 72)

    # 1. Model Configuration
    s0 = 20.0
    delta0 = 0.05
    kappa = 1.20
    alpha = 0.08
    sigma1 = 0.30
    sigma2 = 0.20
    rho = 0.60
    rate = 0.05

    s2f = SchwartzTwoFactor(
        s0=s0,
        delta0=delta0,
        kappa=kappa,
        alpha=alpha,
        sigma1=sigma1,
        sigma2=sigma2,
        rho=rho,
        rate=rate,
    )
    print(f"\nInitialized Model: {s2f}")
    print(f"  Spot Price (S_0):             ${s2f.s0:.2f}")
    print(f"  Convenience Yield (δ_0):       {s2f.delta0:.2%}")
    print(f"  Yield Mean Reversion (κ):      {s2f.kappa:.2f}")
    print(f"  Long-Term Target (α_hat):      {s2f.alpha_hat:.2%}")
    print(f"  Spot Volatility (σ_1):         {s2f.sigma1:.2%}")
    print(f"  Yield Volatility (σ_2):        {s2f.sigma2:.2%}")
    print(f"  Spot-Yield Correlation (ρ):    {s2f.rho:.2f}")
    print(f"  Risk-Free Rate (r):            {s2f.rate:.2%}")

    # 2. Moments & Variance
    t = 1.0
    y_mean = s2f.yield_mean(t=t)
    y_std = s2f.yield_std(t=t)
    log_mean = s2f.log_spot_mean(t=t)
    log_std = s2f.log_spot_std(t=t)

    print(f"\nMoment Analytics at horizon T = {t:.1f} yr:")
    print(f"  Convenience Yield E[δ_T]:      {y_mean:.4%} (Std: {y_std:.4%})")
    print(f"  Log-Spot Mean E[ln S_T]:       {log_mean:.4f} (Std: {log_std:.4f})")
    print(f"  Spot Price E[S_T]:             ${math.exp(log_mean + 0.5 * log_std**2):.2f}")

    # 3. Commodity Futures Term Structure
    maturities = [0.25, 0.50, 1.0, 2.0, 3.0, 5.0]
    print("\nFutures Term Structure F(S_0, δ_0, T):")
    print(
        f"  {'Maturity (yr)':<15} {'Futures Price':<18} {'Basis (F - S0)':<16} {'Market Regime':<15}"
    )
    print("  " + "-" * 64)
    for mat in maturities:
        fwd = s2f.futures_price(tau=mat)
        basis = fwd - s0
        regime = "Contango" if basis > 0 else "Backwardation"
        print(f"  {mat:<15.2f} ${fwd:<17.4f} ${basis:<15.4f} {regime:<15}")

    # 4. European Options on Commodity Spot
    strike = 20.0
    call = s2f.call_price(strike=strike, time_to_maturity=t)
    put = s2f.put_price(strike=strike, time_to_maturity=t)
    fwd_1y = s2f.futures_price(tau=t)
    parity_lhs = call - put
    parity_rhs = math.exp(-rate * t) * (fwd_1y - strike)

    print(f"\nEuropean Options on Spot at Strike K = ${strike:.2f} (T = {t:.1f} yr):")
    print(f"  Call Price:                    ${call:.4f}")
    print(f"  Put Price:                     ${put:.4f}")
    print(
        f"  Put-Call Parity: (C - P) = ${parity_lhs:.6f}, e^(-rT)(F - K) = ${parity_rhs:.6f} (Diff: {abs(parity_lhs - parity_rhs):.2e})"
    )

    # 5. Joint Path Simulation
    n_paths = 30_000
    n_steps = 25
    rng = np.random.default_rng(2026)

    print(f"\nSimulating {n_paths:,} joint sample paths over T = {t:.1f} ({n_steps} steps)...")
    s_paths, d_paths = s2f.simulate_joint(n_steps=n_steps, time_horizon=t, n_paths=n_paths, rng=rng)
    mc_spot = float(np.mean(s_paths[:, -1]))
    mc_yield = float(np.mean(d_paths[:, -1]))

    print(
        f"  Empirical Mean Spot S_T:       ${mc_spot:.4f} (Theoretical F: ${fwd_1y:.4f}, error: {abs(mc_spot - fwd_1y):.4e})"
    )
    print(
        f"  Empirical Mean Yield δ_T:      {mc_yield:.4%} (Theoretical: {y_mean:.4%}, error: {abs(mc_yield - y_mean):.4e})"
    )
    print(f"  All simulated spot prices > 0: {bool(np.all(s_paths > 0.0))}")

    print("\nDemonstration completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
