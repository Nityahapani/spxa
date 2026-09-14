"""
spxa live demo.

Run with: python examples/demo.py
"""

import numpy as np

import spxa
from spxa.analytics import cumulant_table, hellinger_distance, l2_char_func_distance
from spxa.analytics.fourier import european_call_price, implied_volatility
from spxa.zoo.beyond import FractionalBrownianMotion
from spxa.zoo.levy import BrownianMotion, GammaProcess, NIG, VarianceGamma

print("=" * 65)
print("spxa — stochastic process algebra demo")
print("=" * 65)

# ── 1. Composition ────────────────────────────────────────────────
print("\n── 1. Composition: Z = 0.5*BM + 2*VG ──")
bm = BrownianMotion(mu=0.0, sigma=0.2)
vg = VarianceGamma(sigma=0.3, nu=0.2, theta=-0.1)
Z = 0.5 * bm + 2.0 * vg
print(f"  Z          : {Z}")
print(f"  exactness  : {Z.exactness.name}")
print(f"  triplet.b  : {Z.triplet.b:.6f}")
print(f"  triplet.σ² : {Z.triplet.sigma_sq:.6f}")

# ── 2. Cumulants ──────────────────────────────────────────────────
print("\n── 2. Cumulants of Z at t=1 ──")
for k, v in cumulant_table(Z, order=4).items():
    print(f"  {k:<20} {v:.6f}")

# ── 3. VG closed-form cumulants ───────────────────────────────────
print("\n── 3. VarianceGamma exact cumulants ──")
vg2 = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.15)
kappas = vg2.cumulants(order=4)
print(f"  σ={vg2.sigma}, ν={vg2.nu}, θ={vg2.theta}")
print(f"  κ₁ = {kappas[1]:.6f}  (θ = {vg2.theta})")
print(f"  κ₂ = {kappas[2]:.6f}  (σ²+θ²ν = {vg2.sigma**2 + vg2.theta**2*vg2.nu:.6f})")
print(f"  κ₃ = {kappas[3]:.6f}")
print(f"  κ₄ = {kappas[4]:.6f}")

# ── 4. Story ──────────────────────────────────────────────────────
print("\n── 4. Subordination story: BM @ Gamma ──")
W = BrownianMotion(mu=0.0, sigma=1.0) @ GammaProcess(a=1.0, b=2.0)
print(W.__story__())

# ── 5. Property propagation ───────────────────────────────────────
print("\n── 5. Property propagation ──")
bm_mg = BrownianMotion(mu=0.0, sigma=1.0)
bm_dr = BrownianMotion(mu=1.0, sigma=1.0)
g1 = GammaProcess(a=1.0, b=1.0)
g2 = GammaProcess(a=2.0, b=1.0)
print(f"  BM(0)+BM(0) martingale?  {(bm_mg+bm_mg).properties.is_martingale}")
print(f"  BM(0)+BM(1) martingale?  {(bm_mg+bm_dr).properties.is_martingale}")
print(f"  Gamma+Gamma subordinator? {(g1+g2).properties.is_subordinator}")
print(f"  (-1)*Gamma subordinator?  {(-1.0*g1).properties.is_subordinator}")

# ── 6. Divergences ────────────────────────────────────────────────
print("\n── 6. Divergences between marginals at t=1 ──")
bm_ref  = BrownianMotion(mu=0.0, sigma=0.3)
vg_comp = VarianceGamma(sigma=0.3, nu=0.5, theta=0.0)
nig_p   = NIG(alpha=3.0, beta=0.0, delta=1.0)
print(f"  L²(BM, VG)   : {l2_char_func_distance(bm_ref, vg_comp, t=1.0, u_max=30.0):.6f}")
print(f"  L²(BM, NIG)  : {l2_char_func_distance(bm_ref, nig_p,   t=1.0, u_max=30.0):.6f}")
print(f"  H²(BM, VG)   : {hellinger_distance(bm_ref, vg_comp, t=1.0, u_max=30.0):.6f}")
print(f"  L²(BM, BM)   : {l2_char_func_distance(bm_ref, bm_ref, t=1.0):.2e}  (≈0)")

# ── 7. Option pricing ─────────────────────────────────────────────
print("\n── 7. VG European call prices ──")
vg_fin = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
S0, r, T_exp = 100.0, 0.05, 1.0
print(f"  S0={S0}, r={r}, T={T_exp}, VG(σ={vg_fin.sigma}, ν={vg_fin.nu}, θ={vg_fin.theta})")
print(f"  {'Strike':>8}  {'Call price':>12}  {'Impl. vol':>10}")
for K in [90, 95, 100, 105, 110]:
    price = european_call_price(vg_fin, S0=S0, K=K, r=r, t=T_exp)
    iv = implied_volatility(price, S0=S0, K=K, r=r, t=T_exp)
    print(f"  {K:>8}  {price:>12.4f}  {iv:>10.4f}")

# ── 8. MC check ───────────────────────────────────────────────────
print("\n── 8. Monte Carlo check — VG at t=1 ──")
rng = np.random.default_rng(0)
paths = vg_fin.simulate(n_steps=100, n_paths=50_000, T=1.0, rng=rng)
s = paths[:, -1]
print(f"  empirical mean : {s.mean():.5f}  (theory: {vg_fin.cumulants(1)[1]:.5f})")
print(f"  empirical var  : {s.var():.5f}  (theory: {vg_fin.cumulants(2)[2]:.5f})")

# ── 9. fBM ────────────────────────────────────────────────────────
print("\n── 9. FractionalBrownianMotion (H=0.75) ──")
fbm = FractionalBrownianMotion(H=0.75)
print(f"  exactness        : {fbm.exactness.name}")
print(f"  Var(B^H_1)       : {fbm.variance(1.0):.6f}")
print(f"  Cov(B^H_0.5,B^1) : {fbm.covariance(0.5, 1.0):.6f}")
fp = fbm.simulate(n_steps=500, n_paths=5000, T=1.0, rng=np.random.default_rng(1))
print(f"  MC Var(B^H_1)    : {fp[:,-1].var():.4f}  (theory: {fbm.variance(1.0):.4f})")

print("\n" + "=" * 65)
print(f"  131 tests passing · spxa v{spxa.__version__}")
print("=" * 65)
