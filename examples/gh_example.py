"""
Generalized Hyperbolic (GH) Lévy Process demonstration in spxa.

Run with: PYTHONPATH=. python examples/gh_example.py
"""

import math
import numpy as np

from spxa import GeneralizedHyperbolicProcess

print("=" * 72)
print("spxa — Generalized Hyperbolic (GH) Lévy Process Demo (Barndorff-Nielsen 1977)")
print("=" * 72)

# 1. Model Definition
lambda_param = 1.0  # Hyperbolic process
alpha = 2.5  # Tail decay / heaviness
beta = -0.5  # Asymmetry (negative skewness)
delta = 1.2  # Scale parameter
mu = 0.05  # Drift / location parameter

gh = GeneralizedHyperbolicProcess(
    lambda_param=lambda_param,
    alpha=alpha,
    beta=beta,
    delta=delta,
    mu=mu,
)

print("\n── 1. Model Definition & Structural Properties ──")
print(f"  Model               : {gh}")
print(f"  Subfamily           : {gh.subfamily}")
print(f"  Exactness Level     : {gh.exactness.name}")
print(f"  Lambda (shape)      : {gh.lambda_param:.2f}")
print(f"  Alpha (tail decay)  : {gh.alpha:.2f}")
print(f"  Beta (skewness)     : {gh.beta:+.2f}")
print(f"  Delta (scale)       : {gh.delta:.2f}")
print(f"  Mu (location)       : {gh.mu:+.2f}")
print(f"  Gamma root          : {gh.gamma:.4f}")

# 2. Cumulants & Distribution Analytics
T = 1.0
cums = gh.cumulants(order=4, t=T)
print(f"\n── 2. Cumulants & Shape at T = {T:.1f} yr ──")
print(f"  Mean κ1(T)          : {cums[1]:.4f}")
print(f"  Variance κ2(T)      : {cums[2]:.4f} (Std Dev: {math.sqrt(cums[2]):.4f})")
print(f"  Skewness κ3/κ2^(1.5): {cums[3] / (cums[2] ** 1.5):.4f}")
print(f"  Excess Kurt κ4/κ2^2 : {cums[4] / (cums[2] ** 2.0):.4f}")

# 3. European Option Pricing & Put-Call Parity
s0 = 100.0
r = 0.04
q = 0.01
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

calls = gh.call_price(strikes, T=T, r=r, q=q, s0=s0)
puts = gh.put_price(strikes, T=T, r=r, q=q, s0=s0)
ivs = gh.implied_volatility(strikes, T=T, r=r, q=q, s0=s0)
df_r = math.exp(-r * T)
df_q = math.exp(-q * T)

print(
    f"\n── 3. European Options & Put-Call Parity (S0={s0}, r={r * 100:.1f}%, q={q * 100:.1f}%) ──"
)
print(
    f"  {'Strike K':>8} | {'Call Price':>10} | {'Put Price':>10} | {'Imp Vol':>10} | {'Parity Diff':>12}"
)
print("  " + "-" * 60)
for k, c, p, iv in zip(strikes, calls, puts, ivs):
    parity_diff = abs((c - p) - (s0 * df_q - k * df_r))
    iv_str = f"{iv * 100:.2f}%" if math.isfinite(iv) else "N/A"
    print(f"  {k:>8.1f} | {c:>10.4f} | {p:>10.4f} | {iv_str:>10} | {parity_diff:>12.2e}")

# 4. Simulation
n_paths = 20_000
n_steps = 25
rng = np.random.default_rng(2026)
paths = gh.simulate(n_steps=n_steps, time_horizon=T, n_paths=n_paths, rng=rng)
terminal_x = paths[:, -1]

mc_mean = float(np.mean(terminal_x))
mc_var = float(np.var(terminal_x))

print("\n── 4. Vectorized Path Simulation ──")
print(f"  Simulated Paths     : {n_paths:,} paths across {n_steps} steps")
print(
    f"  Empirical Mean      : {mc_mean:.4f} (theoretical: {cums[1]:.4f}, diff: {abs(mc_mean - cums[1]):.4f})"
)
print(f"  Empirical Variance  : {mc_var:.4f} (theoretical: {cums[2]:.4f})")

print("\n" + "=" * 72)
print("Demo finished successfully.")
print("=" * 72)
