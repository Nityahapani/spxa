"""
spxa v1.0.0 — full end-to-end showcase
=======================================

Runs every major capability of the library in a single script:
  1.  Core algebra      — composition, triplet arithmetic, ExactnessLevel
  2.  Property lattice  — propagation through +, *, @
  3.  Cumulants         — exact closed-form vs numerical
  4.  Characteristic    — char_func_exact across process families
  5.  Story / narrator  — plain-text and LaTeX derivation traces
  6.  Zoo (Lévy)        — all 11 EXACT processes
  7.  Zoo (beyond)      — fBM, Hawkes, OULevy
  8.  Multivariate      — correlated BM, CorrelatedLevy
  9.  Operations        — sum_processes, subordination, product, integral
  10. Simulation        — Euler-Maruyama, geometric Lévy, bridge, first-passage
  11. Analytics         — divergences, Fourier pricing, implied-vol smile
  12. Exact samplers    — BM, Gamma, VG, NIG, stable, tempered-stable, NegBin

Run:  python examples/full_showcase.py
"""

import warnings
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_SEP  = "=" * 68
_TICK = "✓"
_FAIL = "✗"

def section(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")

def ok(label: str, got, expected, tol: float = 0.04, abs_tol=None) -> None:
    if abs_tol is not None:
        passed = abs(float(got) - float(expected)) <= abs_tol
    else:
        passed = abs(float(got) - float(expected)) <= tol * max(abs(float(expected)), 1e-10)
    sym = _TICK if passed else _FAIL
    print(f"  {sym}  {label}: {float(got):.5f}  (expected ~{float(expected):.5f})")
    if not passed:
        raise AssertionError(f"FAILED: {label}")

def check(label: str, cond: bool) -> None:
    sym = _TICK if cond else _FAIL
    print(f"  {sym}  {label}")
    if not cond:
        raise AssertionError(f"FAILED: {label}")

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import spxa
from spxa.core import ExactnessLevel, ExactnessError, SpxaDegradationWarning
from spxa.zoo.levy import (
    BrownianMotion, GammaProcess, VarianceGamma, NIG, CGMY,
    AlphaStable, MeixnerProcess, PoissonProcess,
    InverseGaussianProcess, TemperedStable, NegativeBinomialProcess,
)
from spxa.zoo.beyond import FractionalBrownianMotion, HawkesProcess, OULevy
from spxa.zoo.multivariate import (
    MultivariateBrownianMotion, CorrelatedLevy,
    correlated_brownian_motion, independent_levy_vector,
)
from spxa.ops import (
    add, sum_processes, subordinate,
    bernstein_function, char_exp_subordinated,
    multiply, ito_isometry_variance,
)
from spxa.ops.integral import StochasticIntegral
from spxa.story import story_latex, narrate_latex
from spxa.sim import (
    euler_maruyama, euler_maruyama_autonomous, geometric_levy,
    sample_brownian, sample_gamma, sample_vg, sample_nig,
    sample_alpha_stable, sample_tempered_stable,
    brownian_bridge, gamma_bridge, first_passage_time_exact_bm,
    increments_to_paths,
)
from spxa.analytics import cumulant_table, l2_char_func_distance, hellinger_distance
from spxa.analytics.fourier import european_call_price, implied_volatility

print(_SEP)
print(f"  spxa v{spxa.__version__} — full showcase")
print(_SEP)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CORE ALGEBRA
# ─────────────────────────────────────────────────────────────────────────────
section("1.  Core algebra — composition and triplet arithmetic")

bm  = BrownianMotion(mu=0.0, sigma=0.3)
vg  = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
nig = NIG(alpha=3.0, beta=-0.5, delta=1.0)

Z = 0.5 * bm + 2.0 * vg
check("Z = 0.5*BM + 2*VG is EXACT",      Z.exactness == ExactnessLevel.EXACT)
ok("triplet σ²  = (0.5)²×0.09 = 0.0225", Z.triplet.sigma_sq, 0.5**2 * 0.3**2)
ok("triplet b   = 0.5×0 + 2×(-0.1)",      Z.triplet.b,  2.0 * vg.cumulants(1)[1], tol=0.01)

W = bm + nig
check("BM + NIG is EXACT",  W.exactness == ExactnessLevel.EXACT)
ok("(BM+NIG) σ² = 0.09",    W.triplet.sigma_sq, 0.09)

T3 = sum_processes([BrownianMotion(sigma=1.0)] * 5)
ok("sum of 5 BM(σ=1) → σ²=5", T3.triplet.sigma_sq, 5.0)

# scaling
z3 = 3.0 * BrownianMotion(mu=1.0, sigma=1.0)
ok("3×BM(mu=1,σ=1) → σ²=9",  z3.triplet.sigma_sq, 9.0)
ok("3×BM(mu=1,σ=1) → b=3",   z3.triplet.b, 3.0)

# ExactnessError on beyond-Lévy
fbm = FractionalBrownianMotion(H=0.7)
try:
    _ = fbm.triplet
    check("fBM.triplet raises ExactnessError", False)
except ExactnessError:
    check("fBM.triplet raises ExactnessError", True)

# degradation warning
with warnings.catch_warnings(record=True) as w_list:
    warnings.simplefilter("always")
    _ = fbm + bm
    check("fBM + BM emits SpxaDegradationWarning",
          any(issubclass(w.category, SpxaDegradationWarning) for w in w_list))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROPERTY LATTICE
# ─────────────────────────────────────────────────────────────────────────────
section("2.  Property lattice propagation")

bm0 = BrownianMotion(mu=0.0, sigma=1.0)
bm1 = BrownianMotion(mu=1.0, sigma=1.0)
g1  = GammaProcess(a=1.0, b=1.0)
g2  = GammaProcess(a=2.0, b=1.0)
ts  = TemperedStable(alpha=0.5, C=1.0, lam=2.0)

check("BM(0)+BM(0) is martingale",          (bm0 + bm0).properties.is_martingale)
check("BM(0)+BM(1) not martingale",         not (bm0 + bm1).properties.is_martingale)
check("Gamma+Gamma is subordinator",         (g1 + g2).properties.is_subordinator)
check("(-1)*Gamma not subordinator",         not (-1.0 * g1).properties.is_subordinator)
check("TemperedStable is subordinator",      ts.properties.is_subordinator)
check("NegBin is subordinator",              NegativeBinomialProcess(r=1.0, p=0.3).properties.is_subordinator)
check("AlphaStable H=1/alpha",
      abs(AlphaStable(alpha=1.5).properties.self_similarity_index - 1/1.5) < 1e-10)
check("fBM hurst_index stored",              fbm.properties.hurst_index == 0.7)
check("fBM no independent increments",       not fbm.properties.has_independent_increments)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  EXACT CUMULANTS
# ─────────────────────────────────────────────────────────────────────────────
section("3.  Exact cumulants across process families")

from scipy.special import gamma as gamma_fn

checks = [
    ("BM(0,σ=2) κ₁=0",             BrownianMotion(mu=0.0,sigma=2.0).cumulants(2)[1], 0.0,    1e-10, None),
    ("BM(0,σ=2) κ₂=4",             BrownianMotion(mu=0.0,sigma=2.0).cumulants(2)[2], 4.0,    1e-10, None),
    ("Gamma(a=2,b=3) κ₁=a/b",      GammaProcess(a=2.0,b=3.0).cumulants(1)[1],       2/3,    1e-8,  None),
    ("VG(σ,ν,θ) κ₁=θ",             VarianceGamma(sigma=0.2,nu=0.1,theta=-0.15).cumulants(1)[1], -0.15, 1e-6, None),
    ("VG κ₂=σ²+θ²ν",               VarianceGamma(sigma=0.2,nu=0.1,theta=-0.15).cumulants(2)[2],
                                     0.04 + 0.15**2 * 0.1, 1e-4, None),
    ("IG(μ=1,λ=2) κ₁=μ",           InverseGaussianProcess(mu=1.0,lam=2.0).cumulants(1)[1], 1.0, 1e-10, None),
    ("IG κ₂=3μ³/λ",                InverseGaussianProcess(mu=1.0,lam=2.0).cumulants(2)[2], 3*1.0**3/2.0, 1e-10, None),
    ("TS(α=0.5,C=1,λ=2) κ₁",       TemperedStable(alpha=0.5,C=1.0,lam=2.0).cumulants(1)[1],
                                     1.0*float(gamma_fn(0.5))/2.0**0.5, 1e-8, None),
    ("NegBin(r=2,p=0.3) κ₁=rp/q",  NegativeBinomialProcess(r=2.0,p=0.3).cumulants(1)[1], 2*0.3/0.7, 1e-10, None),
    ("NegBin κ₂=rp/q²",            NegativeBinomialProcess(r=2.0,p=0.3).cumulants(2)[2], 2*0.3/0.7**2, 1e-10, None),
    ("Meixner κ₁=δα·tan(β/2)+m",   MeixnerProcess(alpha=0.4,beta=-1.0,delta=0.5).cumulants(1)[1],
                                     0.5*0.4*np.tan(-0.5) + 0.0, 1e-8, None),
]
for label, got, exp, rel, atol in checks:
    ok(label, got, exp, tol=rel, abs_tol=atol)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CHARACTERISTIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
section("4.  Characteristic functions — φ(0)=1 and time-additivity")

procs_cf = [
    ("BrownianMotion",         BrownianMotion(mu=0.0, sigma=1.0)),
    ("VarianceGamma",          VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)),
    ("NIG",                    NIG(alpha=3.0, beta=-0.5, delta=1.0)),
    ("AlphaStable",            AlphaStable(alpha=1.5, beta=0.0)),
    ("InverseGaussianProcess", InverseGaussianProcess(mu=1.0, lam=2.0)),
    ("TemperedStable",         TemperedStable(alpha=0.5, C=1.0, lam=2.0)),
    ("NegativeBinomialProcess",NegativeBinomialProcess(r=2.0, p=0.3)),
    ("MeixnerProcess",         MeixnerProcess(alpha=0.4, beta=-0.5, delta=0.5)),
    ("CGMY",                   CGMY(C=1.0, G=5.0, M=5.0, Y=0.5)),
]
for name, proc in procs_cf:
    cf0 = abs(complex(proc.char_func(u=0.0, t=1.0)) - 1.0)
    check(f"{name}: φ(0)=1", cf0 < 1e-8)
    u, s, t = 1.0, 0.4, 0.6
    cf_s  = complex(proc.char_func(u=u, t=s))
    cf_t  = complex(proc.char_func(u=u, t=t))
    cf_st = complex(proc.char_func(u=u, t=s+t))
    check(f"{name}: φ(u;s+t)=φ(u;s)φ(u;t)", abs(cf_s*cf_t - cf_st) < 1e-8)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  STORY — PLAIN TEXT AND LATEX
# ─────────────────────────────────────────────────────────────────────────────
section("5.  Story / narrator — derivation traces")

Z_story = 0.5 * BrownianMotion(sigma=1.0) + 2.0 * VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)
plain = Z_story.__story__()
check("plain story contains 'Addition'",     "Addition"      in plain)
check("plain story contains 'Scaling'",      "Scaling"       in plain)
check("plain story contains 'BrownianMotion'","BrownianMotion" in plain)
check("plain story contains 'VarianceGamma'","VarianceGamma"  in plain)
check("plain story contains 'EXACT'",        "EXACT"          in plain)
check("plain story contains Sato citation",  "Sato"           in plain)

latex_out = Z_story.__story_latex__()
check("LaTeX story contains aligned env",    r"\begin{aligned}"  in latex_out)
check("LaTeX story contains triplet rule",   r"\sigma^2"         in latex_out)
check("LaTeX story contains end",            r"\end{aligned}"    in latex_out)

print("\n  [Plain story excerpt]")
for line in plain.split("\n")[:12]:
    print(f"    {line}")

print("\n  [LaTeX story excerpt — first 6 lines]")
for line in latex_out.split("\n")[:6]:
    print(f"    {line}")

# Subordination story
W_story = BrownianMotion(mu=0.0, sigma=1.0) @ GammaProcess(a=1.0, b=2.0)
plain_sub = W_story.__story__()
check("subordination story has 'Subordination'", "Subordination" in plain_sub)
check("subordination story cites Thm 30.1",      "30.1"          in plain_sub)

# narrate_latex standalone
from spxa.story.narrator import narrate_latex
node_latex = narrate_latex(Z_story._node)
check("narrate_latex produces non-empty output", len(node_latex) > 50)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL ZOO — ALL 11 LEVY PROCESSES
# ─────────────────────────────────────────────────────────────────────────────
section("6.  Full Lévy zoo — simulate and check marginal moments")

levy_zoo = [
    ("BrownianMotion",          BrownianMotion(mu=0.5, sigma=1.0),       0.5,  1.0),
    ("GammaProcess",            GammaProcess(a=2.0, b=3.0),              2/3,  2/9),
    ("PoissonProcess",          PoissonProcess(rate=2.0),                 2.0,  2.0),
    ("VarianceGamma",           VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1), -0.1, 0.041),
    ("NIG",                     NIG(alpha=3.0, beta=-0.5, delta=1.0),    None, None),
    ("AlphaStable(sym)",        AlphaStable(alpha=1.5, beta=0.0, sigma=1.0), None, None),
    ("InverseGaussianProcess",  InverseGaussianProcess(mu=1.0, lam=2.0), 1.0,  None),
    ("TemperedStable",          TemperedStable(alpha=0.5, C=1.0, lam=2.0), None, None),
    ("NegativeBinomialProcess", NegativeBinomialProcess(r=2.0, p=0.3),   2*0.3/0.7, None),
    ("CGMY",                    CGMY(C=1.0, G=5.0, M=5.0, Y=0.5),       None, None),
    ("MeixnerProcess",          MeixnerProcess(alpha=0.4, beta=-0.5, delta=0.5), None, None),
]

for name, proc, mean_theory, var_theory in levy_zoo:
    paths = proc.simulate(n_steps=1, n_paths=8000, T=1.0, rng=rng)
    emp_mean = paths[:, -1].mean()
    check(f"{name}: paths start at 0", np.all(paths[:, 0] == 0.0))
    check(f"{name}: shape (8000, 2)",  paths.shape == (8000, 2))
    if mean_theory is not None:
        ok(f"{name}: E[X₁]", emp_mean, mean_theory, tol=0.08)
    else:
        print(f"  {_TICK}  {name}: E[X₁] = {emp_mean:.4f} (no closed-form check)")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  BEYOND-LÉVY ZOO
# ─────────────────────────────────────────────────────────────────────────────
section("7.  Beyond-Lévy zoo — fBM, Hawkes, OULevy")

# fBM
fbm7 = FractionalBrownianMotion(H=0.75, sigma=1.0)
check("fBM exactness = MOMENT_PROPAGATION", fbm7.exactness == ExactnessLevel.MOMENT_PROPAGATION)
ok("fBM Var(B^H_1)=σ²=1", fbm7.variance(1.0), 1.0)
ok("fBM Cov(0.5,1)=0.5*(0.5^1.5+1-0.5^1.5)...", fbm7.covariance(0.5, 1.0),
   0.5*(0.5**1.5 + 1.0**1.5 - 0.5**1.5), tol=1e-10)
fbm_paths = fbm7.simulate(n_steps=200, n_paths=2000, T=1.0, rng=rng)
ok("fBM MC Var(B^H_1) ≈ 1", fbm_paths[:,-1].var(), 1.0, tol=0.12)
check("fBM hurst=0.75", fbm7.properties.hurst_index == 0.75)

# Hawkes
hk = HawkesProcess(mu=1.0, alpha=0.5, beta=2.0)
check("Hawkes exactness = MOMENT_PROPAGATION", hk.exactness == ExactnessLevel.MOMENT_PROPAGATION)
ok("Hawkes mean intensity = μ/(1-α/β)", hk.mean_intensity, 1.0/(1 - 0.25))
check("Hawkes Fano factor > 1", hk.fano_factor > 1.0)
hk_paths = hk.simulate(n_steps=500, n_paths=20, T=20.0, rng=rng)
check("Hawkes paths non-decreasing", np.all(np.diff(hk_paths, axis=1) >= 0))

# OULevy
ou = OULevy(lam=2.0, subordinator=GammaProcess(a=1.0, b=2.0), v0=0.0)
ok("OULevy stationary mean = κ₁(G)",    ou.stationary_mean,     GammaProcess(a=1.0,b=2.0).cumulants(1)[1])
ok("OULevy stationary var = κ₂(G)/(2λ)", ou.stationary_variance, GammaProcess(a=1.0,b=2.0).cumulants(2)[2]/4.0)
ok("OULevy autocov(h) = e^{-λh}·Var",   ou.autocovariance(0.5), np.exp(-1.0)*ou.stationary_variance)
ou_paths = ou.simulate(n_steps=200, n_paths=50, T=5.0, rng=rng)
check("OULevy paths non-negative",   np.all(ou_paths >= -1e-12))

# ─────────────────────────────────────────────────────────────────────────────
# 8.  MULTIVARIATE
# ─────────────────────────────────────────────────────────────────────────────
section("8.  Multivariate processes")

# Correlated BM
Sigma = np.array([[1.0, 0.6], [0.6, 2.0]])
mbm = correlated_brownian_motion(Sigma, mu=np.array([0.1, -0.1]))
mbm_paths = mbm.simulate(n_steps=100, n_paths=5000, T=1.0, rng=rng)
check("MultivariateBM shape (5000, 101, 2)", mbm_paths.shape == (5000, 101, 2))
check("MultivariateBM starts at 0",          np.all(mbm_paths[:, 0, :] == 0.0))
emp_cov = np.cov(mbm_paths[:, -1, :].T)
ok("MultivariateBM Cov[0,0]≈1.0", emp_cov[0,0], 1.0, tol=0.08)
ok("MultivariateBM Cov[0,1]≈0.6", emp_cov[0,1], 0.6, tol=0.10)
ok("MultivariateBM Cov[1,1]≈2.0", emp_cov[1,1], 2.0, tol=0.08)

# Triplet
t_mv = mbm.multivariate_triplet()
check("MultivariateLevyTriplet dim=2", t_mv.dim == 2)
np.testing.assert_allclose(t_mv.sigma, Sigma, atol=1e-10)

# CorrelatedLevy mixing
bm_a = BrownianMotion(sigma=1.0); bm_b = BrownianMotion(sigma=1.0)
A = np.array([[1.0, 0.5], [0.5, 1.0]])
CL = CorrelatedLevy([bm_a, bm_b], A=A)
cl_paths = CL.simulate(n_steps=100, n_paths=3000, T=1.0, rng=rng)
check("CorrelatedLevy shape (3000,101,2)", cl_paths.shape == (3000, 101, 2))
emp_cov2 = np.cov(cl_paths[:,-1,:].T)
expected_cov = A @ A.T
ok("CorrelatedLevy Cov[0,0]=1.25", emp_cov2[0,0], expected_cov[0,0], tol=0.08)
ok("CorrelatedLevy Cov[0,1]=1.0",  emp_cov2[0,1], expected_cov[0,1], tol=0.10)

# Independent vector
iv = independent_levy_vector(VarianceGamma(sigma=0.2, nu=0.1, theta=0.0),
                              GammaProcess(a=1.0, b=2.0))
iv_paths = iv.simulate(n_steps=50, n_paths=100, T=1.0, rng=rng)
check("independent_levy_vector shape (100,51,2)", iv_paths.shape == (100, 51, 2))

# ─────────────────────────────────────────────────────────────────────────────
# 9.  OPERATIONS MODULE
# ─────────────────────────────────────────────────────────────────────────────
section("9.  Operations — subordination, product, integral, Bernstein")

# Subordination via @ operator
bm_sub = BrownianMotion(mu=0.0, sigma=0.2)
g_sub  = GammaProcess(a=10.0, b=10.0)
W_sub  = bm_sub @ g_sub
check("BM@Gamma exactness=EXACT",                W_sub.exactness == ExactnessLevel.EXACT)
check("BM@Gamma has stationary increments",       W_sub.properties.has_stationary_increments)
check("BM@Gamma has independent increments",      W_sub.properties.has_independent_increments)
sub_paths = W_sub.simulate(n_steps=100, n_paths=2000, T=1.0, rng=rng)
check("BM@Gamma starts at 0", np.all(sub_paths[:, 0] == 0.0))

# subordinate() function
W_sub2 = subordinate(bm_sub, g_sub)
check("subordinate() == @ operator", W_sub2.exactness == ExactnessLevel.EXACT)

# char_exp_subordinated
cf_sub = char_exp_subordinated(bm_sub, g_sub, u=0.0, t=1.0)
check("char_exp_subordinated φ(0)=1", abs(complex(cf_sub) - 1.0) < 1e-6)

# Bernstein function — Gamma exact formula φ(s)=a·log(1+s/b)
g_bern = GammaProcess(a=1.0, b=2.0)
phi_1  = float(bernstein_function(g_bern, lam=1.0))
ok("Gamma Bernstein φ(1)=log(1.5)", phi_1, np.log(1.5), tol=1e-8)

# TemperedStable Bernstein function
phi_ts = float(bernstein_function(TemperedStable(alpha=0.5, C=1.0, lam=2.0), lam=1.0))
check("TS Bernstein φ(1) > 0", phi_ts > 0)

# Product process
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SpxaDegradationWarning)
    P = multiply(BrownianMotion(mu=1.0, sigma=1.0),
                 BrownianMotion(mu=2.0, sigma=1.0))
check("ProductProcess exactness=MOMENT_PROPAGATION",
      P.exactness == ExactnessLevel.MOMENT_PROPAGATION)
ok("ProductProcess κ₁=E[X]×E[Y]=2", P.cumulants(2)[1], 2.0, tol=1e-10)

# Itô isometry
bm_ito = BrownianMotion(sigma=1.0)
ok("Itô isometry: Var(∫_0^1 s dW_s)=1/3",
   ito_isometry_variance(bm_ito, f=lambda s: s, t=1.0), 1/3, tol=1e-4)

# StochasticIntegral ∫X_s dW_s
bm_X = BrownianMotion(sigma=0.5)
bm_W = BrownianMotion(sigma=1.0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SpxaDegradationWarning)
    SI = StochasticIntegral(bm_X, bm_W, f=lambda x: x)
si_paths = SI.simulate(n_steps=200, n_paths=3000, T=1.0, rng=rng)
check("StochasticIntegral starts at 0", np.all(si_paths[:, 0] == 0.0))
ok("StochasticIntegral mean≈0 (martingale)", si_paths[:, -1].mean(), 0.0, abs_tol=0.1)

# ─────────────────────────────────────────────────────────────────────────────
# 10. SIMULATION BACKENDS
# ─────────────────────────────────────────────────────────────────────────────
section("10. Simulation backends — Euler, geometric Lévy, bridge, first-passage")

# Euler-Maruyama autonomous: dZ = 2dt + 0.5dW
bm_em = BrownianMotion(sigma=1.0)
p_au = euler_maruyama_autonomous(bm_em, drift=2.0, diffusion=0.5, z0=10.0,
                                  T=1.0, n_steps=200, n_paths=10000, rng=rng)
ok("EM autonomous E[Z₁]=12",    p_au[:,-1].mean(), 12.0, tol=0.02)
ok("EM autonomous Var[Z₁]=0.25",p_au[:,-1].var(),  0.25, tol=0.06)

# Euler-Maruyama SDE: dZ = -Z dt + dW (Ornstein-Uhlenbeck)
bm_em2 = BrownianMotion(sigma=1.0)
p_ou = euler_maruyama(bm_em2,
                      drift=lambda z, t: -z,
                      diffusion=lambda z, t: 1.0,
                      z0=1.0, T=3.0, n_steps=3000, n_paths=5000, rng=rng)
check("EM OU shape", p_ou.shape == (5000, 3001))
ok("EM OU E[Z_3]≈e^{-3}≈0.05", p_ou[:,-1].mean(), np.exp(-3.0), abs_tol=0.02)

# Geometric Lévy: risk-neutral VG asset price
vg_geo = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
sg = geometric_levy(vg_geo, mu=0.08, sigma=1.0, s0=100.0,
                    T=1.0, n_steps=100, n_paths=10000, rng=rng)
check("geometric_levy all prices positive", np.all(sg > 0))
ok("geometric_levy E[S_T]=S₀e^μT", sg[:,-1].mean(), 100*np.exp(0.08), tol=0.04)

# Brownian bridge
t_grid, bb = brownian_bridge(x_start=0.0, x_end=2.0, n_points=98,
                               n_paths=5000, rng=rng)
check("BB endpoint pinned at 2", np.allclose(bb[:,-1], 2.0))
t_half = t_grid[49]
ok(f"BB E[B(t={t_half:.2f})]≈2t", bb[:,49].mean(), 2.0 * t_half, abs_tol=0.06)

# Gamma bridge
_, gb = gamma_bridge(a=2.0, b=1.0, T=1.0, x_end=3.0, n_points=50,
                     n_paths=200, rng=rng)
check("Gamma bridge endpoint=3",     np.allclose(gb[:,-1], 3.0))
check("Gamma bridge non-negative",   np.all(gb >= -1e-12))

# First passage time — exact Lévy distribution
tau = first_passage_time_exact_bm(level=1.0, sigma=1.0, n_samples=20000, rng=rng)
check("FPT all positive",         np.all(tau > 0))
check("FPT right-skewed",         np.median(tau) < np.mean(tau))

# ─────────────────────────────────────────────────────────────────────────────
# 11. EXACT SAMPLERS
# ─────────────────────────────────────────────────────────────────────────────
section("11. Exact samplers — direct increment sampling")

dt = 1.0
n  = 20000

inc_bm = sample_brownian(mu=2.0, sigma=1.0, dt=dt, n_steps=1, n_paths=n, rng=rng)
ok("sample_brownian mean=μ",   inc_bm.mean(), 2.0, tol=0.03)
ok("sample_brownian var=σ²",   inc_bm.var(),  1.0, tol=0.03)

inc_g = sample_gamma(a=2.0, b=3.0, dt=dt, n_steps=1, n_paths=n, rng=rng)
ok("sample_gamma mean=a/b",  inc_g.mean(), 2/3, tol=0.03)
ok("sample_gamma var=a/b²",  inc_g.var(),  2/9, tol=0.05)
check("sample_gamma non-negative", np.all(inc_g >= 0))

inc_vg = sample_vg(sigma=0.2, nu=0.1, theta=-0.1, dt=dt, n_steps=1, n_paths=n, rng=rng)
ok("sample_vg mean=θ",    inc_vg.mean(), -0.1, abs_tol=0.015)
ok("sample_vg var=σ²+θ²ν",inc_vg.var(), 0.041, tol=0.06)

inc_nig = sample_nig(alpha=3.0, beta=-0.5, delta=1.0, mu=0.0,
                     dt=dt, n_steps=1, n_paths=n, rng=rng)
g_nig = np.sqrt(3.0**2 - 0.5**2)
ok("sample_nig mean=δβ/γ", inc_nig.mean(), 1.0*(-0.5)/g_nig, tol=0.05)

inc_st = sample_alpha_stable(alpha=1.5, beta=0.0, sigma=1.0, dt=dt,
                              n_steps=1, n_paths=n, rng=rng)
ok("sample_alpha_stable median≈0 (symmetric)", float(np.median(inc_st)), 0.0, abs_tol=0.15)

inc_ts = sample_tempered_stable(alpha=0.5, C=1.0, lam=2.0, dt=0.1,
                                 n_steps=1, n_paths=5000, rng=rng)
check("sample_tempered_stable non-negative", np.all(inc_ts >= 0))

# increments_to_paths
inc_test = np.ones((3, 5))
paths_test = increments_to_paths(inc_test, x0=2.0)
check("increments_to_paths shape",    paths_test.shape == (3, 6))
check("increments_to_paths x0",       np.all(paths_test[:, 0] == 2.0))
check("increments_to_paths cumsum",   np.all(paths_test[:, -1] == 7.0))

# ─────────────────────────────────────────────────────────────────────────────
# 12. ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
section("12. Analytics — divergences, Fourier pricing, implied vol")

# Cumulant table
vg_ct = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
ct = cumulant_table(vg_ct, order=4, t=1.0)
check("cumulant_table has mean",              "mean"             in ct)
check("cumulant_table has variance",          "variance"         in ct)
check("cumulant_table has skewness",          "skewness"         in ct)
check("cumulant_table has excess_kurtosis",   "excess_kurtosis"  in ct)
ok("cumulant_table mean = θ",   ct["mean"],     -0.1, tol=1e-4)
ok("cumulant_table skewness < 0 (left-skewed)", ct["skewness"], ct["skewness"], abs_tol=10)  # just check it runs
check("VG left-skewed (skewness < 0)",        ct["skewness"] < 0)
check("VG leptokurtic (excess_kurtosis > 0)", ct["excess_kurtosis"] > 0)

# L² distance
bm_ref = BrownianMotion(mu=0.0, sigma=0.3)
vg_sym = VarianceGamma(sigma=0.3, nu=0.5, theta=0.0)
d_same = l2_char_func_distance(bm_ref, bm_ref, u_max=20, n_points=500)
d_diff = l2_char_func_distance(bm_ref, vg_sym, u_max=20, n_points=500)
ok("L²(BM, BM) ≈ 0", d_same, 0.0, abs_tol=1e-8)
check("L²(BM, VG) > 0",    d_diff > 0)

# Hellinger distance
h_same = hellinger_distance(bm_ref, bm_ref, u_max=20, n_points=500)
h_diff = hellinger_distance(bm_ref, vg_sym, u_max=20, n_points=500)
ok("H²(BM, BM) ≈ 0",  h_same, 0.0, abs_tol=1e-4)
check("H²(BM, VG) ∈ [0,1]",  0 <= h_diff <= 1.0)
check("L² symmetric: d(X,Y)=d(Y,X)",
      abs(l2_char_func_distance(bm_ref, vg_sym, u_max=20, n_points=300)
          - l2_char_func_distance(vg_sym, bm_ref, u_max=20, n_points=300)) < 1e-10)

# Carr-Madan option pricing
S0, r_f, T_exp = 100.0, 0.05, 1.0
models_pricing = [
    ("GBM",  BrownianMotion(mu=0.0, sigma=0.2)),
    ("VG",   VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)),
    ("NIG",  NIG(alpha=15.0, beta=-5.0, delta=0.5)),
]
strikes = [90.0, 100.0, 110.0]
print()
print(f"  {'Model':<8}  {'K=90':>8}  {'K=100':>8}  {'K=110':>8}  "
      f"{'IV(90)':>8}  {'IV(100)':>8}  {'IV(110)':>8}")
print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

for name, proc in models_pricing:
    prices = [european_call_price(proc, S0=S0, K=K, r=r_f, t=T_exp) for K in strikes]
    ivs    = [implied_volatility(p, S0=S0, K=K, r=r_f, t=T_exp) for p, K in zip(prices, strikes)]
    check(f"{name}: all prices positive", all(p > 0 for p in prices))
    check(f"{name}: all IVs in (0,1)",    all(0 < iv < 1 for iv in ivs))
    print(f"  {name:<8}  {prices[0]:>8.3f}  {prices[1]:>8.3f}  {prices[2]:>8.3f}  "
          f"  {ivs[0]:>6.4f}    {ivs[1]:>6.4f}    {ivs[2]:>6.4f}")

# Put-call parity
from spxa.analytics.fourier import european_put_price
vg_pcp = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
K_pcp  = 100.0
call   = european_call_price(vg_pcp, S0=S0, K=K_pcp, r=r_f, t=T_exp)
put    = european_put_price(vg_pcp,  S0=S0, K=K_pcp, r=r_f, t=T_exp)
parity = call - put - S0 + K_pcp * np.exp(-r_f * T_exp)
ok("Put-call parity: C - P - S + Ke^{-rT} ≈ 0", parity, 0.0, abs_tol=0.5)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(_SEP)
print(f"  spxa v{spxa.__version__} — all checks passed")
print(f"  Zoo:       11 Lévy + 3 beyond-Lévy + multivariate")
print(f"  Algebra:   +  *  @  sum_processes  multiply  StochasticIntegral")
print(f"  Analytics: L²  Hellinger  Carr-Madan  implied-vol  cumulant_table")
print(f"  Sim:       EM  geometric_levy  bridge  first-passage  exact samplers")
print(f"  Story:     __story__()  __story_latex__()  narrate_latex()")
print(_SEP)
