"""
Bates (1996) Stochastic Volatility with Jumps Process demonstration in spxa.

Run with: PYTHONPATH=. python examples/bates_example.py
"""

import math
import numpy as np

from spxa import BatesProcess, HestonProcess

print("=" * 72)
print("spxa — Bates (1996) Stochastic Volatility with Jumps Process Demo")
print("=" * 72)

# 1. Model Definition
v0 = 0.04  # 20% initial instantaneous volatility
kappa = 2.0  # Mean reversion speed
theta = 0.04  # Long term variance
sigma_v = 0.30  # Volatility of variance
rho = -0.60  # Spot-vol correlation (leverage effect)
rate = 1.2  # 1.2 jumps per year
jump_mean = -0.06  # -6% average jump size
jump_std = 0.12  # 12% jump volatility
mu = 0.05  # Drift (r - q)
s0 = 100.0  # Spot price

bates = BatesProcess(
    v0=v0,
    kappa=kappa,
    theta=theta,
    sigma_v=sigma_v,
    rho=rho,
    rate=rate,
    jump_mean=jump_mean,
    jump_std=jump_std,
    mu=mu,
    s0=s0,
)

print("\n── 1. Model Definition & Structural Properties ──")
print(f"  Model               : {bates}")
print(f"  Exactness Level     : {bates.exactness.name}")
print(f"  Initial Vol √v0     : {math.sqrt(bates.v0) * 100:.1f}%")
print(f"  Long-term Vol √θ    : {math.sqrt(bates.theta) * 100:.1f}%")
print(f"  Mean Reversion κ    : {bates.kappa:.2f}")
print(f"  Vol of Vol σ_v      : {bates.sigma_v * 100:.1f}%")
print(f"  Spot-Vol Corr ρ     : {bates.rho:.2f} (leverage effect)")
print(f"  Feller Ratio        : {bates.feller_ratio:.3f} (satisfied: {bates.is_feller_satisfied})")
print(f"  Jump Rate λ         : {bates.rate:.2f} jumps/yr")
print(f"  Mean Log Jump μ_J   : {bates.jump_mean * 100:.1f}%")
print(f"  Jump Vol σ_J        : {bates.jump_std * 100:.1f}%")
print(f"  Expected Rel Jump k̄ : {bates.expected_relative_jump * 100:.2f}% (E[exp(Y) - 1])")
print(f"  Drift Compensator   : {bates.drift_compensator:.4f} (-λ k̄)")

# 2. Moments & Cumulants of Log-Return X_t
T = 0.5
k = bates.conditional_cumulants(t=T, order=4)
mean_theor = bates.conditional_mean(t=T)
var_theor = bates.conditional_variance(t=T)

print(f"\n── 2. Moments & Cumulants of Log-Return X_T (T = {T:.1f} yr) ──")
print(f"  E[X_T] (κ_1)        : {k[1]:.6f} (theory: {mean_theor:.6f})")
print(f"  Var(X_T) (κ_2)      : {k[2]:.6f} (theory: {var_theor:.6f})")
print(f"  Std Dev             : {math.sqrt(k[2]):.6f}")
print(f"  Skewness (κ_3)      : {k[3]:.6f} (negative asymmetry from ρ < 0 and μ_J < 0)")
print(f"  Ex Kurtosis (κ_4)   : {k[4]:.6f} (heavy leptokurtic tails from jumps & vol-of-vol)")

# 3. Pure Heston Limit Comparison
heston = HestonProcess(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, mu=mu, s0=s0)
c_bates_atm = float(bates.call_price(s0, T=T, r=0.05, q=0.0))
c_heston_atm = float(heston.call_price(s0, T=T, r=0.05, q=0.0))
print("\n── 3. Pure Heston Comparison (Jump Impact on ATM Option) ──")
print(f"  ATM Heston Call     : {c_heston_atm:.4f}")
print(
    f"  ATM Bates Call      : {c_bates_atm:.4f} (jump risk premium: +{c_bates_atm - c_heston_atm:.4f})"
)

# 4. European Option Pricing & Implied Volatility Smile
strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
r = 0.05
q = 0.01

call_prices = bates.call_price(strikes, T=T, r=r, q=q)
put_prices = bates.put_price(strikes, T=T, r=r, q=q)
implied_vols = bates.implied_vol_smile(strikes, T=T, r=r, q=q)

print(f"\n── 4. Bates Option Pricing & Implied Volatility (T = {T:.1f} yr) ──")
print(
    f"  {'Strike K':>8} | {'Call':>9} | {'Put':>9} | {'C - P Parity Diff':>18} | {'Implied Vol':>12}"
)
print("  " + "-" * 66)
for K, c, p, iv in zip(strikes, call_prices, put_prices, implied_vols):
    parity_theor = s0 * math.exp(-q * T) - K * math.exp(-r * T)
    parity_diff = abs((c - p) - parity_theor)
    print(f"  {K:>8.1f} | {c:>9.4f} | {p:>9.4f} | {parity_diff:>18.2e} | {iv * 100:>11.2f}%")

# 5. Joint Simulation (Paths of X_t and V_t)
n_paths = 10_000
n_steps = 50
rng = np.random.default_rng(42)

px, pv = bates.simulate_joint(n_steps=n_steps, time_horizon=T, n_paths=n_paths, rng=rng)
final_x = px[:, -1]
final_v = pv[:, -1]

mc_mean_x = float(np.mean(final_x))
mc_var_x = float(np.var(final_x))
mc_mean_v = float(np.mean(final_v))
theor_mean_v = bates.variance_mean(t=T)

print("\n── 5. Monte Carlo Joint Path Simulation (X_t, V_t) ──")
print(f"  Simulated Paths     : {n_paths:,} paths across {n_steps} steps (Lord full truncation)")
print(
    f"  Sample Mean X_T     : {mc_mean_x:.4f} (analytical: {mean_theor:.4f}, diff: {abs(mc_mean_x - mean_theor):.4f})"
)
print(
    f"  Sample Var X_T      : {mc_var_x:.4f} (analytical: {var_theor:.4f}, diff: {abs(mc_var_x - var_theor):.4f})"
)
print(
    f"  Sample Mean V_T     : {mc_mean_v:.4f} (analytical: {theor_mean_v:.4f}, diff: {abs(mc_mean_v - theor_mean_v):.4f})"
)

print("\n" + "=" * 72)
print("Demo finished successfully.")
print("=" * 72)
