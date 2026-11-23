"""
Black-Karasinski (1991) Lognormal Short-Rate Process demonstration in spxa.

Run with: PYTHONPATH=. python examples/black_karasinski_example.py
"""

import math
import numpy as np

from spxa import BlackKarasinskiProcess

print("=" * 72)
print("spxa — Black-Karasinski (1991) Lognormal Short-Rate Process Demo")
print("=" * 72)

# 1. Model Definition
a = 0.15  # Mean-reversion speed (15% per year)
sigma = 0.20  # Volatility of log-rate (20%)
r0 = 0.03  # Current short rate (3.0%)
b = math.log(0.035)  # Reversion target ln(3.5%)

bk = BlackKarasinskiProcess(
    a=a,
    sigma=sigma,
    b=b,
    r0=r0,
)

print("\n── 1. Model Definition & Structural Properties ──")
print(f"  Model                   : {bk}")
print(f"  Exactness Level         : {bk.exactness.name}")
print(f"  Initial Short Rate r0   : {bk.r0 * 100:.2f}% (x0 = {bk.x0:.4f})")
print(f"  Mean Reversion Speed a  : {bk.a:.2f} (half-life: {math.log(2) / bk.a:.1f} yr)")
print(f"  Log-Rate Volatility σ   : {bk.sigma * 100:.2f}%")
print(f"  Long-Term Target Rate   : {math.exp(bk.b) * 100:.2f}% (b = {bk.b:.4f})")
print(f"  Stationary Median Rate  : {bk.stationary_median * 100:.2f}%")
print(f"  Stationary Mean Rate    : {bk.stationary_mean * 100:.2f}%")
print(f"  Stationary Std Dev      : {math.sqrt(bk.stationary_variance) * 100:.2f}%")

# 2. Distributional Analytics & Moments
T = 2.0
mu_x = bk.log_rate_mean(t=T)
var_x = bk.log_rate_variance(t=T)
m_r = bk.conditional_mean(t=T)
v_r = bk.conditional_variance(t=T)
s_r = bk.conditional_std(t=T)

print(f"\n── 2. Moments & Cumulants at T = {T:.1f} yr ──")
print(f"  Log-Rate E[ln r_T]      : {mu_x:.4f} (Var: {var_x:.6f}, Std: {math.sqrt(var_x):.4f})")
print(f"  Short Rate E[r_T]       : {m_r * 100:.3f}%")
print(f"  Short Rate Var(r_T)     : {v_r:.6e} (Std Dev: {s_r * 100:.3f}%)")

# 3. Zero-Coupon Bond Pricing & Yield Curve
maturities = [0.5, 1.0, 2.0, 5.0, 10.0]
rng = np.random.default_rng(2026)
bond_prices = bk.bond_price(maturities, n_paths=20_000, rng=rng)
yields = -np.log(bond_prices) / np.array(maturities)

print("\n── 3. Term Structure of Interest Rates (Monte Carlo) ──")
print(f"  {'Maturity':>8} | {'Bond Price':>10} | {'Spot Yield':>10}")
print("  " + "-" * 36)
for mat, p, y in zip(maturities, bond_prices, yields):
    print(f"  {mat:>8.1f} | {p:>10.4f} | {y * 100:>9.3f}%")

# 4. Exact Path Simulation
n_paths = 25_000
n_steps = 50
rng = np.random.default_rng(42)

paths = bk.simulate(n_steps=n_steps, time_horizon=T, n_paths=n_paths, rng=rng)
final_r = paths[:, -1]

mc_mean = float(np.mean(final_r))
mc_var = float(np.var(final_r))

print("\n── 4. Exact Log-Rate Discretization Simulation ──")
print(f"  Simulated Paths         : {n_paths:,} paths across {n_steps} steps")
print(
    f"  Strict Positivity Check : min(r_t) = {np.min(paths) * 100:.4f}% > 0 (rates strictly positive)"
)
print(
    f"  Sample Mean r_T         : {mc_mean * 100:.3f}% (analytical: {m_r * 100:.3f}%, diff: {abs(mc_mean - m_r) * 100:.4f}%)"
)
print(f"  Sample Var r_T          : {mc_var:.6e} (analytical: {v_r:.6e})")

print("\n" + "=" * 72)
print("Demo finished successfully.")
print("=" * 72)
