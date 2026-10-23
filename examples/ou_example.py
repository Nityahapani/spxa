"""
Gaussian Ornstein–Uhlenbeck (Vasicek) process demonstration in spxa.

Run with: python examples/ou_example.py
"""

import numpy as np

from spxa import OrnsteinUhlenbeck

print("=" * 65)
print("spxa — Gaussian Ornstein–Uhlenbeck (Vasicek) Process Demo")
print("=" * 65)

# 1. Process Definition & Timescales
ou = OrnsteinUhlenbeck(theta=1.5, mu=0.05, sigma=0.03, x0=0.02)
print("\n── 1. Process Definition & Timescales ──")
print(f"  Model            : {ou}")
print(f"  Exactness        : {ou.exactness.name}")
print(f"  Relaxation time  : {ou.relaxation_time:.4f} yrs (τ = 1/θ)")
print(f"  Half-life        : {ou.half_life:.4f} yrs (t_1/2 = ln(2)/θ)")
print(f"  Stationary Mean  : {ou.stationary_mean:.6f}")
print(f"  Stationary Var   : {ou.stationary_variance:.8f} (σ² / (2θ))")
print(f"  Stationary Std   : {ou.stationary_std:.6f}")
print(f"  Stationary Skew  : {ou.stationary_skewness:.4f} (Gaussian)")
print(f"  Stationary Kurt  : {ou.stationary_kurtosis:.4f} (excess kurtosis)")

# 2. Analytical Moments at Horizon t=1 vs Stationarity
t_eval = 1.0
cond_kappas = ou.conditional_cumulants(t=t_eval, order=4)
print(f"\n── 2. Analytical Moments at t={t_eval} vs Stationarity ──")
print(
    "  Conditional Mean (t="
    f"{t_eval}"
    ") : "
    f"{cond_kappas[1]:.6f}"
    " (theory: "
    f"{ou.conditional_mean(t_eval):.6f}"
    ")"
)
print(
    "  Conditional Var  (t="
    f"{t_eval}"
    ") : "
    f"{cond_kappas[2]:.8f}"
    " (theory: "
    f"{ou.conditional_variance(t_eval):.8f}"
    ")"
)
print(f"  Conditional Std  (t={t_eval}) : {ou.conditional_std(t_eval):.6f}")
print(
    f"  Cumulants κ_3, κ_4       : {cond_kappas[3]:.1f}, {cond_kappas[4]:.1f} (vanish for Gaussian)"
)

# 3. Temporal Dependence & Power Spectral Density
print("\n── 3. Temporal Dependence & Power Spectral Density ──")
for h in [0.0, ou.half_life, 1.0, 2.0, 5.0]:
    acov = ou.autocovariance(h)
    corr = ou.autocorrelation(h)
    print(f"  Lag h={h:5.3f} | Autocov: {acov:.8f} | Autocorr: {corr:.4f}")

# Spectral density at key frequencies
s_zero = ou.spectral_density(0.0)
s_cutoff = ou.spectral_density(ou.theta)
print(f"  Spectral Density S(0)    : {s_zero:.8f}")
print(f"  Spectral Density S(θ)    : {s_cutoff:.8f} (half-power frequency ω=θ)")

# 4. Vasicek Zero-Coupon Bond Pricing & Term Structure Yield Curve
print("\n── 4. Vasicek Term Structure & Bond Pricing ──")
maturities = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
bond_prices = ou.bond_price(maturities)
bond_yields = ou.bond_yield(maturities)
print(f"  Spot Rate r_0    : {ou.x0:.4f}")
print(f"  Asymptotic Yield : {ou.asymptotic_yield:.6f} (τ → ∞)")
print("    Maturity (yrs)     Bond Price P(τ)     Zero Yield y(τ)")
for tau, p, y in zip(maturities, bond_prices, bond_yields, strict=True):
    print(f"      {tau:5.2f}               {p:.6f}              {y:.6f}")

# 5. Exact Simulation Check vs Theory
print("\n── 5. Monte Carlo Simulation Check ──")
n_paths = 50_000
n_steps = 100
T = 1.0
rng = np.random.default_rng(12345)

paths_exact = ou.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng, method="exact")
terminal_exact = paths_exact[:, -1]

mc_mean = float(np.mean(terminal_exact))
mc_var = float(np.var(terminal_exact))
th_mean = ou.conditional_mean(T)
th_var = ou.conditional_variance(T)

print(f"  Simulation Method: exact Gaussian increments (N={n_paths:,}, steps={n_steps})")
print(
    "  Terminal E[X_T]  : "
    f"{mc_mean:.6f}"
    "  (theory: "
    f"{th_mean:.6f}"
    ", diff: "
    f"{abs(mc_mean - th_mean):.6f}"
    ")"
)
print(
    f"  Terminal Var[X_T]: {mc_var:.8f}  (theory: {th_var:.8f}, diff: {abs(mc_var - th_var):.8f})"
)

# Euler scheme comparison
paths_euler = ou.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng, method="euler")
euler_mean = float(np.mean(paths_euler[:, -1]))
euler_var = float(np.var(paths_euler[:, -1]))
print(f"  Euler Scheme E[X]: {euler_mean:.6f}  (var: {euler_var:.8f})")

# 6. Derivation Story Trace
print("\n── 6. Derivation Story Trace ──")
print(ou.__story__())
print("=" * 65)
