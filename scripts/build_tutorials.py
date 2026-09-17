"""Build the three spxa tutorial notebooks."""

import json
import uuid
from pathlib import Path


def nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": uuid.uuid4().hex[:8]}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
        "id": uuid.uuid4().hex[:8],
    }


# ── Notebook 1: First Composition ─────────────────────────────────────────────
nb1_cells = [
    md("# Tutorial 1 — First Composition\n\nThis notebook introduces the core idea of spxa: stochastic processes as first-class algebraic objects.\nYou write `Z = X + c*Y` or `W = X @ T` and get back a new process with exact analytical properties."),
    code("import numpy as np\nimport matplotlib.pyplot as plt\nimport spxa\nfrom spxa.zoo.levy import BrownianMotion, VarianceGamma, GammaProcess, NIG\nfrom spxa.analytics import cumulant_table\nprint('spxa', spxa.__version__)"),
    md("## 1. Adding two processes\n\nFor independent Lévy processes $X$ and $Y$, the sum $Z = X + Y$ has triplet\n$$\n(b_Z, \\sigma^2_Z, \\nu_Z) = (b_X + b_Y,\\; \\sigma^2_X + \\sigma^2_Y,\\; \\nu_X + \\nu_Y)\n$$\nThis is **exact** — not an approximation."),
    code("bm  = BrownianMotion(mu=0.0, sigma=0.3)   # pure Gaussian, no jumps\nvg  = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)  # jump process\n\nZ = bm + vg\nprint('Z:', Z)\nprint('Exactness:', Z.exactness.name)\nprint('Triplet σ²:', Z.triplet.sigma_sq)          # Gaussian part only (VG has σ²=0)\nprint('Triplet b: ', Z.triplet.b)"),
    md("## 2. Scalar multiplication\n\nFor $Z = c \\cdot X$, the triplet transforms as\n$$\n(b_Z, \\sigma^2_Z, \\nu_Z) = (cb + \\Delta b,\\; c^2 \\sigma^2,\\; \\nu(\\cdot/c))\n$$\nThe drift correction $\\Delta b$ accounts for mass shifting across the unit ball under the truncation function."),
    code("z2 = 2.0 * BrownianMotion(mu=0.0, sigma=1.0)\nprint('2*BM sigma_sq:', z2.triplet.sigma_sq, '  (expected 4.0)')"),
    md("## 3. Cumulant table\n\nEvery EXACT process exposes analytical cumulants via the Lévy–Khintchine formula.\nThe cumulant table also computes skewness and excess kurtosis."),
    code("vg2 = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.15)\ntable = cumulant_table(vg2, order=4, t=1.0)\nfor k, v in table.items():\n    print(f'  {k:<20} {v:.6f}')"),
    md("## 4. Subordination\n\nSubordinating Brownian motion by a Gamma process gives the Variance Gamma model.\nThe `@` operator does this: `W = BM @ Gamma`."),
    code("bm0 = BrownianMotion(mu=0.0, sigma=0.2)\ng   = GammaProcess(a=10.0, b=10.0)\nW   = bm0 @ g   # subordinated process\nprint('Subordinated process:', W)\nprint('Exactness:', W.exactness.name)"),
    md("## 5. The Story: derivation trace\n\n`.__story__()` produces a human-readable derivation of how the current process was constructed."),
    code("Z2 = 0.5 * BrownianMotion(sigma=1.0) + 2.0 * VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)\nprint(Z2.__story__())"),
    md("## 6. LaTeX rendering (Jupyter only)\n\nIn a Jupyter environment, you can render the derivation as LaTeX:"),
    code("# In a Jupyter notebook, use:\n# from IPython.display import display, Math\n# display(Math(Z2.__story_latex__()))\nlatex = Z2.__story_latex__()\nprint(latex[:300], '...')"),
    md("## 7. Property lattice\n\nProperties propagate automatically. The library is honest when they are lost."),
    code("bm_m  = BrownianMotion(mu=0.0, sigma=1.0)   # martingale\nbm_d  = BrownianMotion(mu=1.0, sigma=1.0)   # NOT martingale (drift)\ng1    = GammaProcess(a=1.0, b=1.0)            # subordinator\n\nprint('BM(0)+BM(0) martingale?    ', (bm_m + bm_m).properties.is_martingale)\nprint('BM(0)+BM(1) martingale?    ', (bm_m + bm_d).properties.is_martingale)\nprint('Gamma+Gamma subordinator?  ', (g1 + g1).properties.is_subordinator)\nprint('(-1)*Gamma subordinator?   ', (-1.0 * g1).properties.is_subordinator)"),
    md("## 8. Simulation"),
    code("rng = np.random.default_rng(42)\nvg3 = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)\npaths = vg3.simulate(n_steps=252, n_paths=5, T=1.0, rng=rng)\n\nfig, ax = plt.subplots(figsize=(9, 4))\nt = np.linspace(0, 1, 253)\nfor i in range(5):\n    ax.plot(t, paths[i], lw=0.8)\nax.set_xlabel('Time')\nax.set_ylabel('$X_t$')\nax.set_title('Variance Gamma sample paths')\nplt.tight_layout()\nplt.savefig('vg_paths.png', dpi=120)\nplt.show()\nprint('paths shape:', paths.shape)"),
]

# ── Notebook 2: Finance Models ─────────────────────────────────────────────────
nb2_cells = [
    md("# Tutorial 2 — Finance Models\n\nspxa is purpose-built for quantitative finance. This notebook covers:\n- Exponential Lévy models for asset prices\n- European option pricing via the Carr-Madan FFT method\n- Implied volatility smile from VG and NIG models\n- Comparing models via Hellinger and L² distances"),
    code("import numpy as np\nimport matplotlib.pyplot as plt\nfrom spxa.zoo.levy import BrownianMotion, VarianceGamma, NIG, GammaProcess\nfrom spxa.analytics import cumulant_table, l2_char_func_distance, hellinger_distance\nfrom spxa.analytics.fourier import european_call_price, implied_volatility\nfrom spxa.sim import geometric_levy"),
    md("## 1. Geometric Lévy models\n\nThe exponential Lévy model is $S_t = S_0 \\exp(rt + X_t)$ where $X$ is a Lévy process.\nThe risk-neutral drift is chosen so $\\mathbb{E}[S_t] = S_0 e^{rt}$."),
    code("rng = np.random.default_rng(0)\nS0, r, T = 100.0, 0.05, 1.0\n\nbm_gbm = BrownianMotion(mu=0.0, sigma=0.2)\nvg     = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)\nnig    = NIG(alpha=15.0, beta=-5.0, delta=0.5)\n\nfig, axes = plt.subplots(1, 3, figsize=(13, 4))\nfor ax, proc, name in zip(axes, [bm_gbm, vg, nig], ['GBM', 'Variance Gamma', 'NIG']):\n    paths = geometric_levy(proc, mu=r, sigma=1.0, s0=S0, T=T, n_steps=100, n_paths=8, rng=rng)\n    for i in range(8):\n        ax.plot(np.linspace(0, T, 101), paths[i], lw=0.7, alpha=0.8)\n    ax.axhline(S0, color='k', lw=0.5, ls='--')\n    ax.set_title(name); ax.set_xlabel('Time'); ax.set_ylabel('$S_t$')\nplt.tight_layout()\nplt.savefig('levy_paths.png', dpi=100)\nplt.show()"),
    md("## 2. European option pricing — Carr-Madan FFT\n\nFor a Lévy log-return process $X$ with characteristic function $\\phi(u;t)$,\nthe call price is given by the Carr-Madan formula:\n$$\nC(K) = \\frac{e^{-\\alpha k}}{\\pi} \\int_0^\\infty e^{-iuk} \\psi(u)\\, du\n$$\nwhere $k = \\log(K/S_0)$ and $\\psi$ is a modified version of $\\phi$."),
    code("strikes = np.arange(85, 120, 5.0)\nmodels = [\n    ('GBM (σ=0.2)',  BrownianMotion(mu=0.0, sigma=0.2)),\n    ('VG',           VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)),\n    ('NIG',          NIG(alpha=15.0, beta=-5.0, delta=0.5)),\n]\n\nfig, axes = plt.subplots(1, 2, figsize=(13, 5))\nfor name, proc in models:\n    prices = [european_call_price(proc, S0=S0, K=K, r=r, t=T) for K in strikes]\n    ivs    = [implied_volatility(p, S0=S0, K=K, r=r, t=T) for p, K in zip(prices, strikes)]\n    axes[0].plot(strikes, prices, label=name, marker='o', ms=4)\n    axes[1].plot(strikes, ivs,    label=name, marker='o', ms=4)\n\naxes[0].set_title('Call prices'); axes[0].set_xlabel('Strike'); axes[0].legend()\naxes[1].set_title('Implied vol smile'); axes[1].set_xlabel('Strike'); axes[1].set_ylabel('Impl. vol')\nplt.tight_layout()\nplt.savefig('iv_smile.png', dpi=120)\nplt.show()"),
    md("## 3. Cumulant comparison\n\nThe cumulants directly encode the shape of the distribution — skewness and kurtosis are why Lévy models produce smiles."),
    code("for name, proc in models:\n    t = cumulant_table(proc, order=4)\n    print(f'{name:20}  skew={t[\"skewness\"]:+.4f}  kurt={t[\"excess_kurtosis\"]:+.4f}')"),
    md("## 4. Model distances\n\nspxa computes L² and Hellinger distances between marginal distributions directly from the characteristic functions — no Monte Carlo needed."),
    code("bm_ref = BrownianMotion(mu=0.0, sigma=0.2)\nvg_sym = VarianceGamma(sigma=0.2, nu=0.1, theta=0.0)\nprint('L²(GBM, VG)  :', l2_char_func_distance(bm_ref, vg_sym, u_max=20, n_points=500))\nprint('H²(GBM, VG)  :', hellinger_distance(bm_ref, vg_sym, u_max=20, n_points=500))\nprint('L²(GBM, GBM) :', l2_char_func_distance(bm_ref, bm_ref, u_max=20, n_points=500), ' (expected ≈0)')"),
    md("## 5. Stochastic volatility via subordination\n\nThe Variance Gamma model is exactly a BM subordinated by a Gamma process. The subordinator acts as a stochastic clock — business time vs calendar time."),
    code("bm_sub = BrownianMotion(mu=-0.1, sigma=0.2)\ng_sub  = GammaProcess(a=1.0/0.1, b=1.0/0.1)  # IG(1/nu, 1/nu) for VG\nW_sub  = bm_sub @ g_sub\n\nrng2 = np.random.default_rng(1)\npaths_sub = W_sub.simulate(n_steps=252, n_paths=5000, T=1.0, rng=rng2)\nvg_ref     = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)\npaths_vg   = vg_ref.simulate(n_steps=252, n_paths=5000, T=1.0, rng=rng2)\n\nfig, axes = plt.subplots(1, 2, figsize=(11, 4))\nfor ax, p, label in zip(axes, [paths_sub[:,-1], paths_vg[:,-1]], ['BM@Gamma', 'VG direct']):\n    ax.hist(p, bins=40, density=True, alpha=0.7)\n    ax.set_title(label); ax.set_xlabel('$X_1$')\nprint('BM@Gamma mean:', paths_sub[:,-1].mean().round(4), '  VG mean:', paths_vg[:,-1].mean().round(4))\nplt.tight_layout()\nplt.savefig('subordination_hist.png', dpi=120)\nplt.show()"),
]

# ── Notebook 3: Physics Applications ──────────────────────────────────────────
nb3_cells = [
    md("# Tutorial 3 — Physics Applications\n\nLévy processes appear throughout statistical physics:\n- **Anomalous diffusion** — fractional BM and α-stable processes\n- **First passage times** — exact Lévy distribution\n- **Stochastic volatility via subordination** — random time changes\n- **Hawkes processes** — self-exciting point processes in neuroscience and seismology"),
    code("import numpy as np\nimport matplotlib.pyplot as plt\nfrom spxa.zoo.levy import BrownianMotion, AlphaStable\nfrom spxa.zoo.beyond import FractionalBrownianMotion, HawkesProcess, OULevy\nfrom spxa.zoo.levy import GammaProcess\nfrom spxa.sim import first_passage_time_exact_bm, first_passage_time_bm"),
    md("## 1. Anomalous diffusion: fractional Brownian motion\n\nStandard BM has $\\text{Var}(B_t) = \\sigma^2 t$ (linear in $t$). Fractional BM with Hurst index $H$ has\n$$\\text{Var}(B^H_t) = \\sigma^2 t^{2H}$$\n- $H = 0.5$: standard diffusion (Brownian)\n- $H > 0.5$: superdiffusion (long memory, persistent increments)\n- $H < 0.5$: subdiffusion (anti-persistent increments)"),
    code("rng = np.random.default_rng(42)\nfig, axes = plt.subplots(1, 3, figsize=(13, 4))\nfor ax, H, label in zip(axes, [0.3, 0.5, 0.8], ['H=0.3 (subdiffusion)', 'H=0.5 (BM)', 'H=0.8 (superdiffusion)']):\n    fbm = FractionalBrownianMotion(H=H)\n    paths = fbm.simulate(n_steps=1000, n_paths=5, T=1.0, rng=rng)\n    t = np.linspace(0, 1, 1001)\n    for i in range(5):\n        ax.plot(t, paths[i], lw=0.7)\n    ax.set_title(label)\n    ax.set_xlabel('Time')\nplt.suptitle('Fractional Brownian motion paths')\nplt.tight_layout()\nplt.savefig('fbm_paths.png', dpi=120)\nplt.show()"),
    md("## 2. Mean-square displacement\n\nMSD$(t) = \\mathbb{E}[|X_t - X_0|^2]$ is a key observable in single-particle tracking experiments.\nFor fBM: MSD$(t) = 2\\sigma^2 t^{2H}$."),
    code("rng2 = np.random.default_rng(1)\nt_vals = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])\nfig, ax = plt.subplots(figsize=(8, 5))\nfor H, color in zip([0.3, 0.5, 0.7], ['blue', 'green', 'red']):\n    fbm = FractionalBrownianMotion(H=H, sigma=1.0)\n    msds = []\n    for t in t_vals:\n        paths = fbm.simulate(n_steps=100, n_paths=1000, T=t, rng=rng2)\n        msds.append(np.mean(paths[:, -1]**2))\n    ax.loglog(t_vals, msds, 'o-', color=color, label=f'H={H}')\n    theory = 2.0 * t_vals**(2 * H)\n    ax.loglog(t_vals, theory, '--', color=color, alpha=0.5)\n\nax.set_xlabel('Time $t$')\nax.set_ylabel('MSD')\nax.set_title('Mean-square displacement: empirical (solid) vs $2t^{2H}$ (dashed)')\nax.legend()\nplt.tight_layout()\nplt.savefig('msd.png', dpi=120)\nplt.show()"),
    md("## 3. Heavy-tailed diffusion: α-stable processes\n\nFor $\\alpha < 2$, the $\\alpha$-stable process has power-law tails: no finite variance, no finite mean for $\\alpha \\leq 1$.\nThis describes Lévy flights observed in animal foraging, plasma turbulence, and financial returns."),
    code("rng3 = np.random.default_rng(2)\nfig, axes = plt.subplots(1, 3, figsize=(13, 4))\nfor ax, alpha, name in zip(axes, [2.0, 1.5, 1.0], ['α=2 (Gaussian)', 'α=1.5 (stable)', 'α=1 (Cauchy)']):\n    proc = AlphaStable(alpha=alpha, beta=0.0, sigma=1.0)\n    paths = proc.simulate(n_steps=200, n_paths=5, T=1.0, rng=rng3)\n    t = np.linspace(0, 1, 201)\n    for i in range(5):\n        ax.plot(t, paths[i], lw=0.7)\n    ax.set_title(name)\n    ax.set_xlabel('Time')\nplt.suptitle('α-stable sample paths (decreasing α → heavier tails, larger jumps)')\nplt.tight_layout()\nplt.savefig('stable_paths.png', dpi=120)\nplt.show()"),
    md("## 4. First passage times\n\nThe time for BM to hit a level $a > 0$ follows the Lévy distribution:\n$$P(\\tau \\leq t) = 2\\Phi(-a/(\\sigma\\sqrt{t}))$$\nThis is an $\\alpha=1/2$ stable distribution — heavy-tailed with infinite mean."),
    code("rng4 = np.random.default_rng(3)\ntau_exact = first_passage_time_exact_bm(level=1.0, sigma=1.0, n_samples=20000, rng=rng4)\ntau_sim   = first_passage_time_bm(level=1.0, sigma=1.0, T_max=20.0, n_steps=5000, n_paths=3000, rng=rng4)\ntau_sim   = tau_sim[np.isfinite(tau_sim)]\n\nfig, ax = plt.subplots(figsize=(8, 5))\nfrom scipy.stats import levy\nt_range = np.linspace(0.01, 8, 500)\nax.hist(tau_exact, bins=80, density=True, alpha=0.5, label='Exact samples', color='steelblue')\nax.hist(tau_sim[tau_sim < 8], bins=80, density=True, alpha=0.4, label='Simulation', color='orange')\nc = 1.0  # level=1, sigma=1 → Lévy(0, c=1)\nax.plot(t_range, levy.pdf(t_range, scale=c), 'k-', lw=2, label='Lévy pdf')\nax.set_xlim(0, 8); ax.set_xlabel('First passage time τ'); ax.set_ylabel('Density')\nax.set_title('First passage time of BM to level 1')\nax.legend()\nplt.tight_layout()\nplt.savefig('first_passage.png', dpi=120)\nplt.show()\nprint(f'Empirical median: {np.median(tau_exact):.4f}  (theory: 1.0)')"),
    md("## 5. Hawkes process — self-exciting events\n\nThe Hawkes process models event clustering: each event increases the probability of future events.\nApplications: earthquake aftershocks, neural spike trains, high-frequency order book arrivals."),
    code("rng5 = np.random.default_rng(4)\nhawkes = HawkesProcess(mu=1.0, alpha=0.6, beta=2.0)\nprint(f'Mean intensity λ̄ = {hawkes.mean_intensity:.4f}')\nprint(f'Fano factor F    = {hawkes.fano_factor:.4f}  (>1: overdispersed)')\n\npaths_hk = hawkes.simulate(n_steps=1000, n_paths=3, T=20.0, rng=rng5)\nt_hk = np.linspace(0, 20, 1001)\nfig, ax = plt.subplots(figsize=(11, 4))\nfor i in range(3):\n    ax.plot(t_hk, paths_hk[i], lw=0.8, label=f'Path {i+1}')\nax.set_xlabel('Time'); ax.set_ylabel('N(t)'); ax.set_title('Hawkes process counting paths')\nax.legend()\nplt.tight_layout()\nplt.savefig('hawkes.png', dpi=120)\nplt.show()"),
    md("## 6. OU-Lévy — stochastic volatility\n\nThe Ornstein–Uhlenbeck process driven by a Gamma subordinator is the Barndorff-Nielsen–Shephard stochastic volatility model. The variance process $V_t$ is stationary, non-negative, and mean-reverting."),
    code("rng6 = np.random.default_rng(5)\nlam = 2.0\ng_driver = GammaProcess(a=1.0, b=2.0)\nou = OULevy(lam=lam, subordinator=g_driver, v0=0.5)\n\nprint(f'Stationary mean    : {ou.stationary_mean:.4f}')\nprint(f'Stationary variance : {ou.stationary_variance:.4f}')\nprint(f'Autocov at h=1     : {ou.autocovariance(1.0):.6f}  (theory: e^{{-{lam}}}·Var = {np.exp(-lam)*ou.stationary_variance:.6f})')\n\npaths_ou = ou.simulate(n_steps=500, n_paths=3, T=5.0, rng=rng6)\nt_ou = np.linspace(0, 5, 501)\nfig, ax = plt.subplots(figsize=(10, 4))\nfor i in range(3):\n    ax.plot(t_ou, paths_ou[i], lw=0.8)\nax.axhline(ou.stationary_mean, color='k', ls='--', lw=1, label='Stationary mean')\nax.set_xlabel('Time'); ax.set_ylabel('V(t)'); ax.set_title('OU-Lévy (Barndorff-Nielsen–Shephard variance process)')\nax.legend()\nplt.tight_layout()\nplt.savefig('ou_levy.png', dpi=120)\nplt.show()"),
]


def write_nb(cells: list[dict], path: Path) -> None:
    notebook = nb(cells)
    path.write_text(json.dumps(notebook, indent=1))
    print(f"Written: {path}")


if __name__ == "__main__":
    out = Path("docs/tutorials")
    out.mkdir(parents=True, exist_ok=True)
    write_nb(nb1_cells, out / "01_first_composition.ipynb")
    write_nb(nb2_cells, out / "02_finance_models.ipynb")
    write_nb(nb3_cells, out / "03_physics_applications.ipynb")
    print("All tutorials written.")
