"""
SABR Stochastic Volatility Model demonstration in spxa.

Run with: PYTHONPATH=. python examples/sabr_example.py
"""

import math
import numpy as np

from spxa import SABRProcess

print("=" * 72)
print("spxa — SABR Stochastic Volatility Model Demo (Hagan et al. 2002)")
print("=" * 72)

# 1. Model Definition
f0 = 100.0  # Initial forward price
alpha = 0.25  # Initial volatility parameter
beta = 0.50  # CEV elasticity exponent (square-root backbone)
nu = 0.40  # Volatility of volatility
rho = -0.30  # Forward-volatility correlation (negative skew)

sabr = SABRProcess(
    alpha=alpha,
    beta=beta,
    nu=nu,
    rho=rho,
    f0=f0,
)

print("\n── 1. Model Definition & Structural Properties ──")
print(f"  Model               : {sabr}")
print(f"  Exactness Level     : {sabr.exactness.name}")
print(f"  Initial Forward F0  : {sabr.f0:.2f}")
print(f"  Initial Vol Alpha   : {sabr.alpha:.3f}")
print(f"  Elasticity Beta     : {sabr.beta:.2f} (backbone: {sabr.backbone})")
print(f"  Vol-of-Vol Nu       : {sabr.nu:.2f}")
print(f"  Correlation Rho     : {sabr.rho:+.2f}")
print(f"  Is Martingale       : {sabr.properties.is_martingale}")

# 2. Implied Volatility Smile & ATM Volatility (Hagan 2002 Asymptotic Formula)
T = 1.0
strikes = np.array([75.0, 85.0, 95.0, 100.0, 105.0, 115.0, 125.0])
vols = sabr.implied_volatility(strikes, T=T)
atm_vol = sabr.atm_volatility(T=T)

print(f"\n── 2. Black Implied Volatility Smile at T = {T:.1f} yr ──")
print(f"  At-the-Money Vol (analytical formula) : {atm_vol * 100:.2f}%")
print(f"  {'Strike K':>8} | {'Implied Vol':>12} | {'Moneyness (K/F0)':>18}")
print("  " + "-" * 44)
for k, v in zip(strikes, vols):
    print(f"  {k:>8.1f} | {v * 100:>11.2f}% | {k / f0:>18.2f}")

# 3. European Option Pricing & Black-76 Put-Call Parity
rate = 0.05
df = math.exp(-rate * T)
calls = sabr.call_price(strikes, T=T, r=rate)
puts = sabr.put_price(strikes, T=T, r=rate)

print(f"\n── 3. Black-76 European Options & Put-Call Parity (r = {rate * 100:.1f}%) ──")
print(
    f"  {'Strike K':>8} | {'Call Price':>10} | {'Put Price':>10} | {'Parity C - P':>12} | {'Expected':>10}"
)
print("  " + "-" * 60)
for k, c, p in zip(strikes, calls, puts):
    parity_diff = c - p
    expected = df * (f0 - k)
    print(f"  {k:>8.1f} | {c:>10.4f} | {p:>10.4f} | {parity_diff:>12.4f} | {expected:>10.4f}")

# 4. Volatility Moments
print(f"\n── 4. Volatility Analytics at T = {T:.1f} yr ──")
vol_mean = sabr.volatility_mean(T)
vol_var = sabr.volatility_variance(T)
log_vol_mean = sabr.log_volatility_mean(T)
log_vol_std = math.sqrt(sabr.log_volatility_variance(T))

print(f"  Expected Volatility E[α_T]       : {vol_mean:.4f}")
print(f"  Volatility Variance Var(α_T)     : {vol_var:.6f} (Std Dev: {math.sqrt(vol_var):.4f})")
print(f"  Expected Log-Vol E[ln α_T]       : {log_vol_mean:.4f}")
print(f"  Log-Vol Std Dev                  : {log_vol_std:.4f}")

# 5. Coupled Path Simulation & Forward Martingale Validation
n_paths = 25_000
n_steps = 50
rng = np.random.default_rng(2026)

f_paths, a_paths = sabr.simulate_joint(
    n_steps=n_steps, time_horizon=T, n_paths=n_paths, rng=rng, method="log_euler"
)
terminal_f = f_paths[:, -1]
terminal_a = a_paths[:, -1]

mc_f_mean = float(np.mean(terminal_f))
mc_a_mean = float(np.mean(terminal_a))
mc_a_var = float(np.var(terminal_a))

print("\n── 5. Vectorized Path Simulation (Log-Euler Method) ──")
print(f"  Simulated Paths                  : {n_paths:,} paths across {n_steps} steps")
print(
    f"  Terminal Forward Mean E[F_T]     : {mc_f_mean:.3f} (theoretical: {f0:.3f}, diff: {abs(mc_f_mean - f0):.4f})"
)
print(f"  Terminal Volatility Mean E[α_T]  : {mc_a_mean:.4f} (theoretical: {vol_mean:.4f})")
print(f"  Terminal Volatility Var Var(α_T) : {mc_a_var:.6f} (theoretical: {vol_var:.6f})")

print("\n" + "=" * 72)
print("Demo finished successfully.")
print("=" * 72)
