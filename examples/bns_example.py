"""
Example demonstrating the Barndorff-Nielsen & Shephard (BNS) stochastic volatility model.

This script illustrates:
1. Model configuration and stationary Gamma-OU variance parameters.
2. Exact variance moments (conditional, stationary, and integrated).
3. Characteristic function and risk-neutral martingale drift.
4. European Call and Put option pricing via Gil-Pelaez Fourier inversion.
5. Black-Scholes implied volatility smile across strikes.
6. Vectorized coupled simulation of asset price and stochastic variance paths.
"""

import math
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spxa import BNSModel


def main() -> None:
    print("=" * 72)
    print("Barndorff-Nielsen & Shephard (BNS) Stochastic Volatility Demonstration")
    print("=" * 72)

    # 1. Model Configuration
    lam = 1.50
    alpha = 2.00
    beta_bdlp = 25.0
    rho = -0.50
    beta = -0.50
    sigma0_sq = 0.04
    s0 = 100.0
    rate = 0.05
    dividend = 0.01

    bns = BNSModel(
        lambda_param=lam,
        alpha=alpha,
        beta_bdlp=beta_bdlp,
        rho=rho,
        beta=beta,
        sigma0_sq=sigma0_sq,
        s0=s0,
        rate=rate,
        dividend=dividend,
    )

    print(f"\nInitialized Model: {bns}")
    print(f"  Spot Asset Price (S_0):       ${bns.s0:.2f}")
    print(
        f"  Initial Variance (σ_0²):      {bns.sigma0_sq:.4f} (Vol: {math.sqrt(bns.sigma0_sq):.2%})"
    )
    print(f"  Variance Mean Reversion (λ):  {bns.lambda_param:.2f}")
    print(f"  BDLP Gamma Shape (α):         {bns.alpha:.2f}")
    print(f"  BDLP Gamma Rate (β_bdlp):     {bns.beta_bdlp:.2f}")
    print(f"  Leverage Parameter (ρ):       {bns.rho:.2f}")
    print(f"  Risk-Free Rate (r):           {bns.rate:.2%}")
    print(f"  Dividend Yield (q):           {bns.dividend:.2%}")
    print(f"  Risk-Neutral Drift (μ):       {bns.mu:.4f}")

    # 2. Variance Analytics
    t = 1.0
    stat_var = bns.stationary_variance_mean
    stat_std = bns.stationary_variance_std
    cond_var = bns.variance_mean(t=t)
    int_var = bns.integrated_variance_mean(t=t)

    print(f"\nVariance Dynamics (horizon T = {t:.1f} yr):")
    print(
        f"  Stationary Variance E[σ_∞²]:  {stat_var:.4f} (Long-term Vol: {math.sqrt(stat_var):.2%})"
    )
    print(f"  Stationary Variance Std:      {stat_std:.4f}")
    print(f"  Expected Variance E[σ_T²]:    {cond_var:.4f}")
    print(f"  Expected Integrated Variance: {int_var:.4f}")

    # 3. Characteristic Function
    u = 1.0
    cf = bns.char_func(u=u, t=t)
    print(f"\nCharacteristic Function at t = {t:.1f}:")
    print(f"  φ({u}; {t:.1f}) = {cf.real:.6f} + {cf.imag:.6f}i (modulus: {abs(cf):.6f})")

    # 4. European Options and Implied Volatility Smile
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    print(f"\nEuropean Options & Implied Volatility Surface (T = {t:.1f} yr):")
    print(
        f"  {'Strike K':<12} {'Call Price':<14} {'Put Price':<14} {'Implied Vol':<14} {'Parity Error':<14}"
    )
    print("  " + "-" * 68)

    disc_r = math.exp(-rate * t)
    disc_q = math.exp(-dividend * t)

    for k in strikes:
        call = bns.call_price(strike=k, time_to_maturity=t)
        put = bns.put_price(strike=k, time_to_maturity=t)
        iv = bns.implied_volatility(strike=k, time_to_maturity=t)
        err = abs((call - put) - (s0 * disc_q - k * disc_r))
        print(f"  ${k:<11.2f} ${call:<13.4f} ${put:<13.4f} {iv:<14.2%} {err:<14.2e}")

    # 5. Joint Path Simulation
    n_paths = 30_000
    n_steps = 25
    rng = np.random.default_rng(2026)

    print(f"\nSimulating {n_paths:,} joint sample paths over T = {t:.1f} ({n_steps} steps)...")
    s_paths, v_paths = bns.simulate_joint(n_steps=n_steps, time_horizon=t, n_paths=n_paths, rng=rng)
    mc_spot = float(np.mean(s_paths[:, -1]))
    mc_var = float(np.mean(v_paths[:, -1]))
    expected_spot = s0 * math.exp((rate - dividend) * t)

    print(
        f"  Empirical Mean Spot S_T:      ${mc_spot:.4f} (Theoretical E[S_T]: ${expected_spot:.4f}, error: {abs(mc_spot - expected_spot):.4e})"
    )
    print(
        f"  Empirical Mean Variance σ_T²: {mc_var:.4f} (Theoretical: {cond_var:.4f}, error: {abs(mc_var - cond_var):.4e})"
    )
    print(f"  All simulated variances non-negative: {bool(np.all(v_paths >= 0.0))}")

    print("\nDemonstration completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
