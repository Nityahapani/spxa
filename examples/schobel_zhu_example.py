"""
Schöbel-Zhu (1999) Stochastic Volatility Model demonstration in spxa.

Run with: PYTHONPATH=. python examples/schobel_zhu_example.py
"""

import math
import numpy as np

from spxa import SchobelZhuProcess

print("=" * 72)
print("spxa — Schöbel-Zhu (1999) Stochastic Volatility Model Demo")
print("=" * 72)

# 1. Model Definition
s0 = 100.0  # Spot price
sigma0 = 0.20  # Initial volatility (20%)
kappa = 1.50  # Mean-reversion speed
theta = 0.20  # Long-term volatility target (20%)
tau = 0.30  # Vol-of-vol
rho = -0.50  # Spot-volatility leverage correlation (equity skew)
r = 0.04  # Risk-free rate (4.0%)
q = 0.01  # Dividend yield (1.0%)

sz = SchobelZhuProcess(
    sigma0=sigma0,
    kappa=kappa,
    theta=theta,
    tau=tau,
    rho=rho,
    s0=s0,
    rate=r,
    dividend=q,
)

print("\n── 1. Model Definition & Structural Properties ──")
print(f"  Model                   : {sz}")
print(f"  Exactness Level         : {sz.exactness.name}")
print(f"  Initial Spot S0         : ${sz.s0:.2f}")
print(f"  Initial Volatility σ0   : {sz.sigma0 * 100:.2f}%")
print(f"  Mean Reversion κ        : {sz.kappa:.2f}")
print(f"  Long-Term Volatility θ  : {sz.theta * 100:.2f}%")
print(f"  Vol-of-Vol τ            : {sz.tau:.2f}")
print(f"  Correlation ρ           : {sz.rho:+.2f}")
print(f"  Interest Rate r / Div q : {sz.rate * 100:.1f}% / {sz.dividend * 100:.1f}%")

# 2. Volatility Analytics & Stationary Law
T = 1.0
vol_m = sz.volatility_mean(T)
vol_v = sz.volatility_variance(T)
var_m = sz.variance_mean(T)
var_v = sz.variance_variance(T)

print(f"\n── 2. Volatility Moments at T = {T:.1f} yr ──")
print(f"  Expected Volatility E[σ_T]      : {vol_m * 100:.2f}%")
print(f"  Volatility Variance Var(σ_T)    : {vol_v:.6f} (Std Dev: {math.sqrt(vol_v) * 100:.2f}%)")
print(f"  Expected Variance E[σ_T²]       : {var_m:.6f} (RMS Vol: {math.sqrt(var_m) * 100:.2f}%)")
print(f"  Stationary Variance E[σ_∞²]     : {sz.stationary_variance_mean:.6f}")

# 3. European Option Pricing & Put-Call Parity
strikes = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
calls = sz.call_price(strikes, T=T)
puts = sz.put_price(strikes, T=T)
ivs = sz.implied_volatility(strikes, T=T)
df_r = math.exp(-r * T)
df_q = math.exp(-q * T)

print("\n── 3. European Options & Put-Call Parity ──")
print(
    f"  {'Strike K':>8} | {'Call Price':>10} | {'Put Price':>10} | {'Imp Vol':>10} | {'Parity Diff':>12}"
)
print("  " + "-" * 60)
for k, c, p, iv in zip(strikes, calls, puts, ivs):
    parity_diff = abs((c - p) - (s0 * df_q - k * df_r))
    iv_str = f"{iv * 100:.2f}%" if math.isfinite(iv) else "N/A"
    print(f"  {k:>8.1f} | {c:>10.4f} | {p:>10.4f} | {iv_str:>10} | {parity_diff:>12.2e}")

# 4. Vectorized Simulation
n_paths = 20_000
n_steps = 30
rng = np.random.default_rng(2026)

s_paths, sig_paths = sz.simulate_joint(n_steps=n_steps, time_horizon=T, n_paths=n_paths, rng=rng)
terminal_s = s_paths[:, -1]
terminal_sig = sig_paths[:, -1]

mc_s_mean = float(np.mean(terminal_s))
theo_s_mean = s0 * math.exp((r - q) * T)
mc_sig_mean = float(np.mean(terminal_sig))

print("\n── 4. Coupled Path Simulation ──")
print(f"  Simulated Paths         : {n_paths:,} paths across {n_steps} steps")
print(
    f"  Terminal Price Mean     : ${mc_s_mean:.2f} (theoretical: ${theo_s_mean:.2f}, diff: ${abs(mc_s_mean - theo_s_mean):.3f})"
)
print(f"  Terminal Vol Mean       : {mc_sig_mean * 100:.2f}% (theoretical: {vol_m * 100:.2f}%)")

print("\n" + "=" * 72)
print("Demo finished successfully.")
print("=" * 72)
