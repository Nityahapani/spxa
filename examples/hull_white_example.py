"""
Hull-White (1990) One-Factor Short-Rate Process demonstration in spxa.

Run with: PYTHONPATH=. python examples/hull_white_example.py
"""

import math
import numpy as np

from spxa import HullWhiteProcess

print("=" * 72)
print("spxa — Hull-White (1990) One-Factor Short-Rate Process Demo")
print("=" * 72)

# 1. Model Definition
a = 0.10  # Mean reversion speed (10% per year)
sigma = 0.015  # 1.5% short rate volatility
b = 0.035  # 3.5% long term baseline rate
r0 = 0.025  # 2.5% current short rate

hw = HullWhiteProcess(
    a=a,
    sigma=sigma,
    b=b,
    r0=r0,
)

print("\n── 1. Model Definition & Structural Properties ──")
print(f"  Model               : {hw}")
print(f"  Exactness Level     : {hw.exactness.name}")
print(f"  Current Rate r0     : {hw.r0 * 100:.2f}%")
print(f"  Long-Term Target b  : {hw.b * 100:.2f}%")
print(f"  Mean Reversion a    : {hw.a:.2f} (half-life: {math.log(2) / hw.a:.1f} yr)")
print(f"  Rate Volatility σ   : {hw.sigma * 100:.2f}%")
print(f"  Stationary Mean     : {hw.stationary_mean * 100:.2f}%")
print(f"  Stationary Std Dev  : {math.sqrt(hw.stationary_variance) * 100:.2f}%")

# 2. Distributional Analytics & Moments
T = 2.0
m = hw.conditional_mean(t=T)
v = hw.conditional_variance(t=T)
s = hw.conditional_std(t=T)
int_m = hw.integrated_mean(t=T)
int_v = hw.integrated_variance(t=T)

print(f"\n── 2. Moments & Cumulants at T = {T:.1f} yr ──")
print(f"  E[r_T]              : {m * 100:.3f}% (pull toward {hw.b * 100:.2f}%)")
print(f"  Var(r_T)            : {v:.6e} (Std Dev: {s * 100:.3f}%)")
print(f"  E[∫ r_s ds]         : {int_m:.4f}")
print(f"  Var(∫ r_s ds)       : {int_v:.6e} (Std Dev: {math.sqrt(int_v):.4f})")

# 3. Zero-Coupon Bond Prices, Yield Curve & Forward Rates
maturities = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
bond_prices = hw.bond_price(maturities)
yields = hw.yield_to_maturity(maturities)
forwards = hw.instantaneous_forward_rate(maturities)

print("\n── 3. Term Structure of Interest Rates (at t = 0) ──")
print(f"  {'Maturity':>8} | {'Bond Price':>10} | {'Spot Yield':>10} | {'Instant Fwd':>12}")
print("  " + "-" * 50)
for mat, p, y, f in zip(maturities, bond_prices, yields, forwards):
    print(f"  {mat:>8.1f} | {p:>10.4f} | {y * 100:>9.3f}% | {f * 100:>11.3f}%")

# 4. European Bond Options & Jamshidian Pricing
opt_expiry = 1.0
bond_mat = 5.0
strikes = [0.80, 0.85, 0.90, 0.95]

p_T = float(hw.bond_price(opt_expiry))
p_S = float(hw.bond_price(bond_mat))

print(
    f"\n── 4. Jamshidian European Bond Options (Option T = {opt_expiry:.1f} yr, Bond S = {bond_mat:.1f} yr) ──"
)
print(f"  Underlying Bond P(0, 5): {p_S:.4f}, Discount P(0, 1): {p_T:.4f}")
print(f"  {'Strike K':>8} | {'Call Price':>10} | {'Put Price':>10} | {'Parity Diff':>16}")
print("  " + "-" * 52)
for K in strikes:
    c = hw.bond_call_price(T=opt_expiry, S=bond_mat, K=K)
    p = hw.bond_put_price(T=opt_expiry, S=bond_mat, K=K)
    parity_diff = abs((c - p) - (p_S - K * p_T))
    print(f"  {K:>8.2f} | {c:>10.4f} | {p:>10.4f} | {parity_diff:>16.2e}")

# 5. Caplet & Floorlet Pricing
strike_rate = 0.03
tau = 0.5
caplet = hw.caplet_price(T=opt_expiry, strike_rate=strike_rate, tau=tau)
floorlet = hw.floorlet_price(T=opt_expiry, strike_rate=strike_rate, tau=tau)
swaplet = p_T - (1.0 + tau * strike_rate) * hw.bond_price(opt_expiry + tau)
print("\n── 5. Interest Rate Derivatives Pricing ──")
print(f"  Caplet Price (R_K={strike_rate * 100:.1f}%, τ={tau}y)   : {caplet:.6f}")
print(f"  Floorlet Price (R_K={strike_rate * 100:.1f}%, τ={tau}y) : {floorlet:.6f}")
print(f"  Caplet - Floorlet Parity Diff    : {abs((caplet - floorlet) - swaplet):.2e}")

# 6. Exact Path Simulation
n_paths = 10_000
n_steps = 50
rng = np.random.default_rng(42)

paths = hw.simulate(n_steps=n_steps, time_horizon=T, n_paths=n_paths, rng=rng, method="exact")
final_r = paths[:, -1]

mc_mean = float(np.mean(final_r))
mc_var = float(np.var(final_r))

print("\n── 6. Exact Short-Rate Path Simulation ──")
print(
    f"  Simulated Paths     : {n_paths:,} paths across {n_steps} steps (zero discretization error)"
)
print(
    f"  Sample Mean r_T     : {mc_mean * 100:.3f}% (analytical: {m * 100:.3f}%, diff: {abs(mc_mean - m) * 100:.4f}%)"
)
print(
    f"  Sample Var r_T      : {mc_var:.6e} (analytical: {v:.6e}, rel diff: {abs(mc_var - v) / v * 100:.2f}%)"
)

print("\n" + "=" * 72)
print("Demo finished successfully.")
print("=" * 72)
