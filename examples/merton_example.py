"""
Merton Jump-Diffusion Process demonstration in spxa.

Run with: python examples/merton_example.py
"""

import numpy as np

from spxa import BrownianMotion, MertonJumpDiffusion

print("=" * 70)
print("spxa — Merton (1976) Jump-Diffusion Process Demo")
print("=" * 70)

# 1. Process Definition
drift = 0.05
sigma = 0.15
rate = 1.5
jump_mean = -0.08
jump_std = 0.12

merton = MertonJumpDiffusion(
    drift=drift,
    sigma=sigma,
    rate=rate,
    jump_mean=jump_mean,
    jump_std=jump_std,
)

print("\n── 1. Model Definition & Triplet ──")
print(f"  Model            : {merton}")
print(f"  Exactness        : {merton.exactness.name}")
print(f"  Continuous Drift : {merton.drift:.4f}")
print(f"  Diffusion Vol σ  : {merton.sigma * 100:.1f}%")
print(f"  Jump Rate λ      : {merton.rate:.2f} jumps/yr")
print(f"  Jump Mean μ_J    : {merton.jump_mean * 100:.1f}%")
print(f"  Jump Std σ_J     : {merton.jump_std * 100:.1f}%")
print(f"  Expected Rel Jump: {merton.expected_relative_jump * 100:.2f}% (E[exp(Y) - 1])")
print(f"  Triplet Drift b  : {merton.triplet.b:.6f} (truncation cutoff=1.0)")
print(f"  Triplet σ²       : {merton.triplet.sigma_sq:.6f}")
print(f"  Total Jump Mass  : {merton.triplet.nu.total_mass():.4f} (finite activity)")

# 2. Analytical Cumulants of X_1
k = merton.cumulants(order=5)
print("\n── 2. Exact Cumulants of X_1 ──")
print(f"  κ_1 (Mean)       : {k[1]:.6f}")
print(f"  κ_2 (Variance)   : {k[2]:.6f}")
print(f"  Std Dev          : {np.sqrt(k[2]):.6f}")
print(f"  κ_3 (Skewness)   : {k[3]:.6f} (negative skew from μ_J < 0)")
print(f"  κ_4 (Ex Kurtosis): {k[4]:.6f} (heavy tails from jump risk)")
print(f"  κ_5              : {k[5]:.6f}")

# 3. Algebraic Composition: Z = Merton + 0.5 * BM
bm = BrownianMotion(mu=0.0, sigma=0.10)
z = merton + 0.5 * bm
print("\n── 3. Algebraic Composition: Z = Merton + 0.5 * BM ──")
print(f"  Composed Process : {z}")
print(f"  Exactness Level  : {z.exactness.name}")
print(
    "  Combined σ²      : "
    f"{z.triplet.sigma_sq:.6f}"
    " (theory: "
    f"{merton.sigma**2 + 0.25 * bm.sigma**2:.6f}"
    ")"
)

# 4. Merton (1976) Analytical Option Pricing & Implied Volatility
S0 = 100.0
r = 0.05
q = 0.01
T = 0.5  # 6 months maturity
strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]

call_prices = merton.call_price(strikes, T=T, S0=S0, r=r, q=q)
put_prices = merton.put_price(strikes, T=T, S0=S0, r=r, q=q)
implied_vols = merton.implied_vol_smile(strikes, T=T, S0=S0, r=r, q=q)

print(f"\n── 4. Merton Option Pricing & Implied Volatility (T={T} yrs) ──")
print("   Strike K     Call Price     Put Price     Implied Vol     Parity Error")
for strk, cp, pp, iv in zip(strikes, call_prices, put_prices, implied_vols, strict=True):
    parity_err = abs((cp - pp) - (S0 * np.exp(-q * T) - strk * np.exp(-r * T)))
    print(
        "    "
        f"{strk:5.1f}"
        "        "
        f"{cp:8.4f}"
        "       "
        f"{pp:8.4f}"
        "        "
        f"{iv * 100:5.2f}"
        "%         "
        f"{parity_err:.2e}"
    )

# 5. Exact Monte Carlo Simulation Check
n_paths = 50_000
n_steps = 25
rng = np.random.default_rng(2026)

paths = merton.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
terminal_x = paths[:, -1]

mc_mean = float(np.mean(terminal_x))
mc_var = float(np.var(terminal_x))
th_mean = k[1] * T
th_var = k[2] * T

print("\n── 5. Exact Simulation Check (Zero Discretization Error) ──")
print(f"  Simulated Paths  : {n_paths:,} trajectories, {n_steps} steps")
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
    f"  Terminal Var[X_T]: {mc_var:.6f}  (theory: {th_var:.6f}, diff: {abs(mc_var - th_var):.6f})"
)

# 6. Derivation Story Trace
print("\n── 6. Derivation Story Trace ──")
print(merton.__story__())
print("=" * 70)
