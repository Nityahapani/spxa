"""
Constant Elasticity of Variance (CEV) Process demonstration in spxa.

Run with: python examples/cev_example.py
"""

import math

import numpy as np

from spxa import CEVProcess

print("=" * 72)
print("spxa — Constant Elasticity of Variance (CEV) Process Demo")
print("=" * 72)

# 1. Process Definition
s0 = 100.0
r = 0.05
q = 0.01
mu = r - q
sigma = 2.0  # ATM local volatility = 20% when gamma = 0.5 and s0 = 100
gamma = 0.5  # Square-root / equity leverage regime

cev = CEVProcess(
    mu=mu,
    sigma=sigma,
    gamma=gamma,
    s0=s0,
    r=r,
    q=q,
)

print("\n── 1. Model Definition & Structural Properties ──")
print(f"  Model               : {cev}")
print(f"  Exactness Level     : {cev.exactness.name}")
print(f"  Initial Spot S_0    : {cev.s0:.2f}")
print(f"  Drift Rate μ        : {cev.mu * 100:.2f}% (r={cev.r * 100:.2f}%, q={cev.q * 100:.2f}%)")
print(f"  Vol Scale Parameter : {cev.sigma:.4f}")
print(f"  Elasticity Exponent : γ = {cev.gamma:.2f}")
print(f"  Variance Elasticity : 2(γ - 1) = {cev.elasticity:.2f}")
print(f"  Vol Elasticity      : γ - 1 = {cev.vol_elasticity:.2f}")
print(f"  Regime              : {cev.regime}")
print(f"  Absorbed at Zero    : {cev.is_absorbed_at_zero}")

# 2. Local Volatility Function
print("\n── 2. State-Dependent Local Volatility σ_loc(S) = σ S^{γ - 1} ──")
spot_levels = [25.0, 50.0, 75.0, 100.0, 150.0, 200.0]
print(f"  {'Spot S':>8} | {'Local Vol σ(S)':>15} | {'Effect':>20}")
print("  " + "-" * 48)
for S in spot_levels:
    vol = cev.local_volatility(S)
    comment = (
        "Elevated (distress)" if s0 > S else ("Baseline (ATM)" if s0 == S else "Subdued (rally)")
    )
    print(f"  {S:>8.1f} | {vol * 100:>14.2f}% | {comment:>20}")

# 3. Schroder (1989) European Option Pricing & Greeks
T = 1.0  # 1 year
strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

print(f"\n── 3. Closed-Form Option Representation & Finite-Difference Greeks (T = {T:.1f} yr) ──")
print(
    "  "
    f"{'Strike K':>8}"
    " | "
    f"{'Call':>9}"
    " | "
    f"{'Put':>9}"
    " | "
    f"{'C - P Parity Diff':>18}"
    " | "
    f"{'Call Delta':>10}"
    " | "
    f"{'Gamma':>8}"
)
print("  " + "-" * 72)

for K in strikes:
    c = cev.call_price(K, T)
    p = cev.put_price(K, T)
    parity_theoretical = s0 * math.exp(-q * T) - K * math.exp(-r * T)
    parity_diff = abs((c - p) - parity_theoretical)
    delta = cev.delta(K, T, option_type="call")
    gamma_g = cev.gamma_greek(K, T)
    print(
        "  "
        f"{K:>8.1f}"
        " | "
        f"{c:>9.4f}"
        " | "
        f"{p:>9.4f}"
        " | "
        f"{parity_diff:>18.2e}"
        " | "
        f"{delta:>10.4f}"
        " | "
        f"{gamma_g:>8.4f}"
    )

# 4. Analytical Absorption / Bankruptcy Probability
horizons = [0.5, 1.0, 2.0, 5.0, 10.0]
print("\n── 4. Analytical Absorption (Default / Bankruptcy) Probability P(S_T = 0) ──")
print(f"  {'Horizon T':>10} | {'P(S_T = 0)':>14} | {'Expected S_T':>14}")
print("  " + "-" * 44)
for t in horizons:
    p_abs = cev.absorption_probability(t)
    exp_s = cev.conditional_mean(t)
    print(f"  {t:>8.1f} yr | {p_abs * 100:>13.4f}% | {exp_s:>14.2f}")

# 5. Vectorized Path Simulation with Absorbing Zero Boundary
print("\n── 5. Vectorized Monte Carlo Path Simulation ──")
n_paths = 10_000
n_steps = 100
dt = T / n_steps
rng = np.random.default_rng(42)

paths = cev.simulate(n_steps=n_steps, time_horizon=T, n_paths=n_paths, rng=rng, method="exact")
print("  Simulation Method   : exact absorbing gamma-mixture transitions")
final_spots = paths[:, -1]

mc_mean = np.mean(final_spots)
th_mean = cev.conditional_mean(T)
mc_var = np.var(final_spots)
th_var = cev.conditional_variance(T)

absorbed_count = np.sum(final_spots == 0.0)
mc_abs_prob = absorbed_count / n_paths
th_abs_prob = cev.absorption_probability(T)

print(f"  Simulated Paths     : {n_paths:,} paths across {n_steps} steps (dt = {dt:.4f})")
print(
    "  Empirical Mean S_T  : "
    f"{mc_mean:.4f}"
    " (analytical: "
    f"{th_mean:.4f}"
    ", diff: "
    f"{abs(mc_mean - th_mean):.4f}"
    ")"
)
print(
    "  Empirical Var S_T   : "
    f"{mc_var:.4f}"
    " (analytical: "
    f"{th_var:.4f}"
    ", rel diff: "
    f"{abs(mc_var - th_var) / th_var * 100:.2f}"
    "%)"
)
print(
    "  Absorbed Paths at 0 : "
    f"{absorbed_count}"
    " / "
    f"{n_paths}"
    " ("
    f"{mc_abs_prob * 100:.2f}"
    "% vs theoretical: "
    f"{th_abs_prob * 100:.2f}"
    "%)"
)
print(f"  Minimum Path Value  : {np.min(paths):.6f} (strictly non-negative)")

print("\n" + "=" * 72)
print("Demo finished successfully.")
print("=" * 72)
