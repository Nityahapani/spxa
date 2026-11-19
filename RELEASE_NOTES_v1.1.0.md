# spxa v1.1.0 — Release Notes

**Tag:** `v1.1.0`  
**Date:** 2026-11-19  
**Previous release:** `v1.0.1` (2026-10-30)

---

## What's new

### Four new modules — the largest feature addition since v1.0.1

**`spxa.pde`** — Finite-difference PDE/PIDE pricing engine
- Black–Scholes: European, digital, American, Bermudan, single/double barriers, discrete dividends, local volatility
- Dupire local volatility from prices; Andreasen–Huge calibration
- Asian options (Večeř), short-rate bond options (Vasicek/CIR), exponential-Lévy PIDEs (Merton, Kou, VG, CGMY, NIG)
- Early exercise: penalty, PSOR, Brennan–Schwartz, Ikonen–Toivanen LCPs
- Two-factor ADI (Douglas, Craig–Sneyd, modified Craig–Sneyd, Hundsdorfer–Verwer) for Heston and two-asset options
- Fokker–Planck density solver, least-squares Monte Carlo cross-checks, grid Greeks
- Convergence studies with Richardson extrapolation and closed-form references

**`spxa.market`** — Market conventions shared by rates and vol
- Dates, tenors, day counts (ACT/360, ACT/365F, ACT/ACT ISDA/ICMA, 30/360 family, BUS/252)
- TARGET, US, UK, joint calendars with Easter rules; business-day adjustment; schedules with stubs and end-of-month rolls
- Yield curves (flat, interpolated discount), equity forwards (dividends, repo, escrowed dividends)
- Black, Bachelier, shifted-Black pricing and Greeks; machine-precision implied vols
- Option quotes/chains, static-arbitrage checks, put-call-parity forwards, implied carry

**`spxa.rates`** — Interest rates on real calendar dates
- Instruments: overnight/term indices with ISDA compounding, deposits, FRAs, futures with Hull–White convexity, OIS/vanilla/basis/amortising swaps, fixed-rate bonds
- Curves: dual-curve bootstrapping (log-linear, cubic, monotone cubic, Hagan–West), turns of year, meeting dates, Nelson–Siegel–Svensson, curve sets; node/quote/key-rate risk, hedging, P&L explain, PCA
- Vol products: caps, floors, swaptions (physical/cash), CMS by replication; SABR (Hagan, Obłój), arbitrage-free SABR density PDE, SABR Monte Carlo, caplet stripping, swaption cube
- Short-rate models: Hull–White (closed forms, exact simulation, trinomial tree, PDE, calibration), G2++, Black–Karasinski, CIR++
- HJM and Cheyette; LIBOR, forward, and SABR-LMM market models
- Bermudan swaptions (PDE, tree, LSMC with Andersen–Broadie bounds); inflation (TIPS reference CPI, seasonal ZCIS, linkers, Jarrow–Yildirim)

**`spxa.vol`** — Volatility surfaces and models
- Surfaces: raw, natural and jump-wings SVI; SSVI and eSSVI with no-arbitrage conditions; quasi-explicit, butterfly-free and surface calibration; Durrleman, calendar and wing diagnostics; sticky-strike/sticky-moneyness Greeks; JSON persistence
- From quotes: `build_vol_surface` with parity forwards, arbitrage repair, Fengler smoothing, implied distributions
- Local vol: Dupire in implied variance, Andreasen–Huge, local-vol Monte Carlo, forward-start options
- SLV: Heston, 2D Fokker–Planck ADI forward solver, leverage by forward induction and particles, SLV Monte Carlo with Brownian-bridge barriers
- Rough vol: rough Heston (fractional Riccati), lifted Heston, rough Bergomi (hybrid scheme); VIX futures/options; joint SPX/VIX calibration
- Martingale optimal transport: marginals from surfaces, LP bounds with dual hedges, entropic (Sinkhorn) MOT, static replication, hedge backtests
- Variance, gamma and volatility swaps; Cont model-risk measures
- Graph nodes: SVI and surface-price nodes, valuing `spxa.risk` options on surfaces for stress testing

### Bug fixes

- **`zoo.levy.CGMY`** — Lévy density, total mass, tail and drift now temper upward jumps with $M$ and downward jumps with $G$ (previously described the mirror-image process). For $1 \le Y < 2$, the drift now includes the $-\int_{|x|>1} x\,\nu(dx)$ correction.
- **`zoo.levy.NIG`** — triplet drift and tail now integrate the Lévy density correctly; drift follows from $\mathbb{E}[X_1] = \mu + \delta\beta/\gamma$.
- **`zoo.levy.KouJumpDiffusion`** — `char_func` now uses the closed-form exponent (previously returned `nan` at complex arguments such as $u = -i$).

### Infrastructure & developer experience

- **Mathematical audits are now first-class**: `audits/` ships with the package (was excluded from the wheel), runs in CI on every push/PR across all 69 stochastic processes / 8 suites (cumulants, characteristic functions, martingale parity, positivity), and is documented in `CONTRIBUTING.md`
- **CI now runs on Python 3.10–3.12** with lint, format, mypy, exactness gate, coverage gate (80%), benchmarks, docs deploy, and PyPI release on tags
- **examples/ and scripts/**: resolved 82 ruff lint errors and formatting issues; `spxa/` and `tests/` remain fully lint-/format-/type-clean
- **README**: rewritten in concise research-library style (xgboost/PyTorch pattern); live GitHub Actions CI badge replaces the static shields.io image

### Stats since v1.0.1 (2026-10-30)

- 346 commits (~229 features, 13 fixes, plus docs/tests/perf/ci)
- 3,032+ tests passing; 86% coverage on `spxa/`
- `ruff check`, `ruff format --check`, and `mypy` all clean on the package source

---

## Installation

```bash
pip install spxa==1.1.0
```

Dev install:

```bash
git clone https://github.com/Nityahapani/spxa.git
cd spxa
pip install -e ".[dev]"
```

---

## Citation

[![CITATION.cff](https://img.shields.io/badge/CITATION.cff-DOI-blue)](CITATION.cff)

```bibtex
@software{spxa,
  author = {Nityahapani},
  title = {spxa: Stochastic Process Algebra \& Computational Graph Engine},
  url = {https://github.com/Nityahapani/spxa},
  version = {1.1.0},
}
```

---

Full changelog: [CHANGELOG.md](CHANGELOG.md)
