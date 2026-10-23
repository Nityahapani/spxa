"""
Heston Stochastic Volatility Model demonstration in spxa.

Run with: python examples/heston_example.py
"""

import numpy as np

from spxa import Heston

print("=" * 70)
print("spxa — Heston Stochastic Volatility Model Demo")
print("=" * 70)

# 1. Process Definition & Parameters
v0 = 0.04  # Initial variance (20% vol)
kappa = 2.0  # Mean-reversion rate of variance
theta = 0.04  # Long-term mean variance (20% vol)
sigma_v = 0.30  # Volatility of variance (vol-of-vol)
rho = -0.70  # Spot-variance correlation (leverage effect)
mu = 0.05  # Risk-free rate / drift (r - q)
s0 = 100.0  # Spot asset price

heston = Heston(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, mu=mu, s0=s0)

print("\n── 1. Model Definition & Feller Condition ──")
print(f"  Model            : {heston}")
print(f"  Exactness        : {heston.exactness.name}")
print(f"  Spot Price S_0   : {heston.s0:.2f}")
print(f"  Initial Vol √v0  : {np.sqrt(heston.v0) * 100:.1f}%")
print(f"  Long-term Vol √θ : {np.sqrt(heston.theta) * 100:.1f}%")
print(f"  Mean-Reversion κ : {heston.kappa:.2f}")
print(f"  Vol-of-Vol σ_v   : {heston.sigma_v:.2f}")
print(f"  Correlation ρ    : {heston.rho:.2f} (negative leverage)")
print(f"  Feller Ratio     : {heston.feller_ratio:.4f} (2κθ / σ_v²)")
feller_status = (
    "Strictly Positive (V_t > 0 a.s.)"
    if heston.is_feller_satisfied
    else "Accessible (Reflecting at 0)"
)
print(f"  Feller Boundary  : {feller_status}")

# 2. Variance Process Analytics & Integrated Variance
T = 1.0
print(f"\n── 2. Variance Process Analytics at Horizon T={T} yrs ──")
print(
    "  E[V_T]           : "
    f"{heston.variance_mean(T):.6f}"
    " (stationary: "
    f"{heston.stationary_variance_mean:.6f}"
    ")"
)
print(
    "  Var(V_T)         : "
    f"{heston.variance_var(T):.8f}"
    " (stationary: "
    f"{heston.stationary_variance_var:.8f}"
    ")"
)
print(f"  E[∫_0^T V_s ds]  : {heston.expected_integrated_variance(T):.6f}")

# 3. Log-Return Moments & Cumulants
kappas = heston.conditional_cumulants(t=T, order=4)
print(f"\n── 3. Affine CF & Numerical Log-Return Cumulants (T={T}) ──")
print(f"  Expected Return E[X_T]   : {kappas[1]:.6f} (theory: {heston.conditional_mean(T):.6f})")
print(f"  Return Variance Var(X_T) : {kappas[2]:.6f}")
print(f"  Return Volatility        : {np.sqrt(kappas[2]) * 100:.2f}%")
print(f"  Skewness κ_3             : {kappas[3]:.8f} (negative skew due to ρ={heston.rho})")
print(f"  Excess Kurtosis κ_4      : {kappas[4]:.8f} (heavy tails from vol clustering)")

# 4. European Option Pricing via Gil-Pelaez Fourier Inversion
print("\n── 4. Option Pricing & Put-Call Parity (T=1.0 yr, r=0.05) ──")
strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
call_prices = heston.call_price(strikes, T=T, r=mu)
put_prices = heston.put_price(strikes, T=T, r=mu)
implied_vols = heston.implied_vol_smile(strikes, T=T, r=mu)

print("   Strike K     Call Price     Put Price     Implied Vol     Parity Error")
for k, c, p, iv in zip(strikes, call_prices, put_prices, implied_vols, strict=True):
    parity_err = abs((c - p) - (s0 - k * np.exp(-mu * T)))
    print(
        "    "
        f"{k:5.1f}"
        "        "
        f"{c:8.4f}"
        "       "
        f"{p:8.4f}"
        "        "
        f"{iv * 100:5.2f}"
        "%         "
        f"{parity_err:.2e}"
    )

# 5. Monte Carlo Simulation Check
print("\n── 5. Monte Carlo Simulation Check (Full Truncation Scheme) ──")
n_paths = 50_000
n_steps = 100
rng = np.random.default_rng(12345)

s_paths, v_paths = heston.simulate_joint(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
terminal_x = np.log(s_paths[:, -1] / s0)
terminal_v = v_paths[:, -1]

mc_mean_x = float(np.mean(terminal_x))
mc_var_x = float(np.var(terminal_x))
mc_mean_v = float(np.mean(terminal_v))
mc_martingale = float(np.mean(s_paths[:, -1] / s0))

print(f"  Paths Simulated  : {n_paths:,} trajectories, {n_steps} steps")
print(
    "  Terminal E[X_T]  : "
    f"{mc_mean_x:.6f}"
    "  (theory: "
    f"{heston.conditional_mean(T):.6f}"
    ", diff: "
    f"{abs(mc_mean_x - heston.conditional_mean(T)):.6f}"
    ")"
)
print(
    "  Terminal Var(X_T): "
    f"{mc_var_x:.6f}"
    "  (theory: "
    f"{kappas[2]:.6f}"
    ", diff: "
    f"{abs(mc_var_x - kappas[2]):.6f}"
    ")"
)
print(f"  Terminal E[V_T]  : {mc_mean_v:.6f}  (theory: {heston.variance_mean(T):.6f})")
print(f"  Martingale E[S_T]: {mc_martingale:.6f}  (theory: {np.exp(mu * T):.6f})")
print(f"  Min Variance Path: {np.min(v_paths):.8f} (guaranteed non-negative)")

# 6. Derivation Story Trace
print("\n── 6. Derivation Story Trace ──")
print(heston.__story__())
print("=" * 70)
