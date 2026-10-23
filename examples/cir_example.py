"""
Cox–Ingersoll–Ross (CIR) square-root diffusion process demonstration in spxa.

Run with: python examples/cir_example.py
"""

import numpy as np

from spxa import CIRProcess

print("=" * 65)
print("spxa — Cox–Ingersoll–Ross (CIR) Square-Root Diffusion Demo")
print("=" * 65)

# 1. Process Definition & Feller Regime Analysis
cir = CIRProcess(kappa=1.5, theta=0.05, sigma=0.15, r0=0.03)
print("\n── 1. Process Definition & Feller Condition ──")
print(f"  Model          : {cir}")
print(f"  Exactness      : {cir.exactness.name}")
print(f"  Feller ratio   : {cir.feller_ratio:.4f} (2κθ / σ²)")
feller_status = (
    "Satisfied (strictly positive paths)" if cir.feller_condition else "Violated (reflecting at 0)"
)
print(f"  Feller status  : {feller_status}")
print(f"  Deg of freedom : {cir.degrees_of_freedom:.4f} (d = 4κθ / σ²)")

# Feller violated example
cir_reflecting = CIRProcess(kappa=0.6, theta=0.02, sigma=0.25, r0=0.01)
print(f"  Reflecting CIR : {cir_reflecting}")
print(
    "  Feller ratio   : "
    f"{cir_reflecting.feller_ratio:.4f}"
    " (condition: "
    f"{cir_reflecting.feller_condition}"
    ")"
)

# 2. Analytical Moments: Conditional vs Stationary
t_eval = 1.0
cond_kappas = cir.conditional_cumulants(t=t_eval, order=4)
print(f"\n── 2. Analytical Moments at t={t_eval} vs Stationarity ──")
print(
    "  Conditional Mean (t="
    f"{t_eval}"
    ") : "
    f"{cond_kappas[1]:.6f}"
    " (theory: "
    f"{cir.conditional_mean(t_eval):.6f}"
    ")"
)
print(
    "  Conditional Var  (t="
    f"{t_eval}"
    ") : "
    f"{cond_kappas[2]:.8f}"
    " (theory: "
    f"{cir.conditional_variance(t_eval):.8f}"
    ")"
)
print(f"  Conditional Std  (t={t_eval}) : {cir.conditional_std(t_eval):.6f}")
print(f"  Stationary Mean (t→∞)    : {cir.stationary_mean:.6f}")
print(f"  Stationary Var  (t→∞)    : {cir.stationary_variance:.8f}")
print(f"  Stationary Skewness      : {cir.stationary_skewness:.4f}")
print(f"  Stationary Excess Kurt   : {cir.stationary_kurtosis:.4f}")
print(
    "  Stationary Gamma law     : Gamma(shape="
    f"{cir.stationary_shape:.4f}"
    ", scale="
    f"{cir.stationary_scale:.6f}"
    ")"
)

# 3. Autocovariance & Autocorrelation
print("\n── 3. Temporal Dependence Structure ──")
for h in [0.0, 0.5, 1.0, 2.0, 5.0]:
    acov = cir.autocovariance(h)
    corr = cir.autocorrelation(h)
    print(f"  Lag h={h:3.1f} | Autocov: {acov:.8f} | Autocorr: {corr:.4f}")

# 4. Zero-Coupon Bond Pricing & Term Structure of Interest Rates
print("\n── 4. Affine Bond Pricing & Term Structure Yield Curve ──")
maturities = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
bond_prices = cir.bond_price(maturities)
bond_yields = cir.bond_yield(maturities)
print(f"  Spot Rate r_0    : {cir.r0:.4f}")
print(f"  Asymptotic Yield : {cir.asymptotic_yield:.6f} (τ → ∞)")
print("    Maturity (yrs)     Bond Price P(τ)     Zero Yield y(τ)")
for tau, p, y in zip(maturities, bond_prices, bond_yields, strict=True):
    print(f"      {tau:5.2f}               {p:.6f}              {y:.6f}")

# 5. Exact Non-Central Chi-Squared Simulation vs Theory
print("\n── 5. Monte Carlo Simulation Check ──")
n_paths = 50_000
n_steps = 100
T = 1.0
rng = np.random.default_rng(12345)

paths_exact = cir.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng, method="exact")
terminal_exact = paths_exact[:, -1]

mc_mean = float(np.mean(terminal_exact))
mc_var = float(np.var(terminal_exact))
th_mean = cir.conditional_mean(T)
th_var = cir.conditional_variance(T)

print(f"  Simulation Method: exact non-central chi-squared (N={n_paths:,}, steps={n_steps})")
print(
    "  Terminal E[r_T]  : "
    f"{mc_mean:.6f}"
    "  (theory: "
    f"{th_mean:.6f}"
    ", diff: "
    f"{abs(mc_mean - th_mean):.6f}"
    ")"
)
print(
    f"  Terminal Var[r_T]: {mc_var:.8f}  (theory: {th_var:.8f}, diff: {abs(mc_var - th_var):.8f})"
)
print(f"  Minimum Path Val : {np.min(paths_exact):.6f}  (non-negativity strictly preserved)")

# Truncated Euler comparison
paths_euler = cir.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng, method="truncated_euler")
euler_mean = float(np.mean(paths_euler[:, -1]))
euler_var = float(np.var(paths_euler[:, -1]))
print(f"  Euler Scheme E[r]: {euler_mean:.6f}  (var: {euler_var:.8f})")

# 6. Composition Story Trace
print("\n── 6. Derivation Story Trace ──")
print(cir.__story__())
print("=" * 65)
