"""
Kou Double Exponential Jump-Diffusion Process demonstration in spxa.

Run with: python examples/kou_example.py
"""

import math
import numpy as np

from spxa import BrownianMotion, KouJumpDiffusion

print("=" * 72)
print("spxa — Kou (2002) Double Exponential Jump-Diffusion Process Demo")
print("=" * 72)

# 1. Model Definition
drift = 0.05
sigma = 0.15
rate = 1.5
p = 0.4
eta1 = 10.0
eta2 = 5.0

kou = KouJumpDiffusion(
    drift=drift,
    sigma=sigma,
    rate=rate,
    p=p,
    eta1=eta1,
    eta2=eta2,
)

print("\n── 1. Model Definition & Exact Triplet ──")
print(f"  Model               : {kou}")
print(f"  Exactness Level     : {kou.exactness.name}")
print(f"  Continuous Drift    : {kou.drift:.4f}")
print(f"  Diffusion Vol σ     : {kou.sigma * 100:.1f}%")
print(f"  Jump Rate λ         : {kou.rate:.2f} jumps/yr")
print(f"  Prob Up Jump p      : {kou.p * 100:.1f}%")
print(f"  Up Decay Rate η_1   : {kou.eta1:.1f} (mean up jump: +{1 / kou.eta1 * 100:.1f}%)")
print(f"  Down Decay Rate η_2 : {kou.eta2:.1f} (mean down jump: -{1 / kou.eta2 * 100:.1f}%)")
print(f"  Overall Jump Mean   : {kou.jump_mean * 100:.2f}% (negative skew)")
print(f"  Expected Rel Jump   : {kou.expected_relative_jump * 100:.2f}% (E[exp(Y) - 1])")
print(f"  Triplet Drift b     : {kou.triplet.b:.6f} (truncation cutoff=1.0)")
print(f"  Triplet σ²          : {kou.triplet.sigma_sq:.6f}")
print(f"  Total Jump Mass     : {kou.triplet.nu.total_mass():.4f} (finite activity)")

# 2. Exact Cumulants of X_1
k = kou.cumulants(order=5, t=1.0)
print("\n── 2. Exact Analytical Cumulants of X_1 ──")
print(f"  κ_1 (Mean)          : {k[1]:.6f}")
print(f"  κ_2 (Variance)      : {k[2]:.6f}")
print(f"  Std Dev             : {np.sqrt(k[2]):.6f}")
print(f"  κ_3 (Skewness)      : {k[3]:.6f} (asymmetric downward jump bias)")
print(f"  κ_4 (Ex Kurtosis)   : {k[4]:.6f} (leptokurtic jump tails)")
print(f"  κ_5                 : {k[5]:.6f}")

# 3. Algebraic Composition: Z = Kou + 0.5 * BM
bm = BrownianMotion(mu=0.0, sigma=0.10)
z = kou + 0.5 * bm
print("\n── 3. Algebraic Composition: Z = Kou + 0.5 * BM ──")
print(f"  Composed Process    : {z}")
print(f"  Exactness Level     : {z.exactness.name}")
print(
    f"  Combined σ²         : {z.triplet.sigma_sq:.6f} (theory: {kou.sigma**2 + 0.25 * bm.sigma**2:.6f})"
)

# 4. European Option Pricing & Implied Volatility
S0 = 100.0
r = 0.05
q = 0.01
T = 0.5
strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

call_prices = kou.call_price(strikes, T=T, S0=S0, r=r, q=q)
put_prices = kou.put_price(strikes, T=T, S0=S0, r=r, q=q)
implied_vols = kou.implied_vol_smile(strikes, T=T, S0=S0, r=r, q=q)

print(f"\n── 4. Kou Option Pricing & Implied Volatility (T = {T:.1f} yr) ──")
print(
    f"  {'Strike K':>8} | {'Call':>9} | {'Put':>9} | {'C - P Parity Diff':>18} | {'Implied Vol':>12}"
)
print("  " + "-" * 66)
for K, c, p, iv in zip(strikes, call_prices, put_prices, implied_vols):
    parity_theor = S0 * math.exp(-q * T) - K * math.exp(-r * T)
    parity_diff = abs((c - p) - parity_theor)
    print(f"  {K:>8.1f} | {c:>9.4f} | {p:>9.4f} | {parity_diff:>18.2e} | {iv * 100:>11.2f}%")

# 5. Exact Monte Carlo Simulation
print("\n── 5. Exact Monte Carlo Path Simulation ──")
n_paths = 10_000
n_steps = 50
rng = np.random.default_rng(42)

paths = kou.simulate(n_steps=n_steps, time_horizon=T, n_paths=n_paths, rng=rng)
final_vals = paths[:, -1]

mc_mean = np.mean(final_vals)
th_mean = k[1] * T
mc_var = np.var(final_vals)
th_var = k[2] * T

print(
    f"  Simulated Paths     : {n_paths:,} paths across {n_steps} steps (zero discretization error)"
)
print(
    f"  Sample Mean X_T     : {mc_mean:.4f} (analytical: {th_mean:.4f}, diff: {abs(mc_mean - th_mean):.4f})"
)
print(
    f"  Sample Var X_T      : {mc_var:.4f} (analytical: {th_var:.4f}, rel diff: {abs(mc_var - th_var) / th_var * 100:.2f}%)"
)

print("\n" + "=" * 72)
print("Demo finished successfully.")
print("=" * 72)
