# spxa Example Gallery

Browse copy-pasteable examples organized by use case. Every example runs
out of the box with `pip install spxa` — no extra setup required.

All 45 examples live in [`examples/`](https://github.com/Nityahapani/spxa/tree/main/examples)
and are tested in CI.

---

## Core algebra & exact arithmetic

| Example | Description | Tags |
|---------|-------------|------|
| [Process composition & exact triplet arithmetic](https://github.com/Nityahapani/spxa/blob/main/examples/demo.py) | Compose Lévy processes with `+`, `*`, `@`. Characteristic triplets add in closed form; cumulants derive symbolically. Every result carries a strict exactness level (`EXACT` → `MOMENT` → `SIMULATION`). | `algebra` `exactness` `cumulants` |
| [Bochner subordination & derivation traces](https://github.com/Nityahapani/spxa/blob/main/examples/full_showcase.py) | Time-change one process by another via `X @ Y`. Generate plain-text and LaTeX derivation stories with `__story__()` and `narrate_latex()`. | `subordination` `story` |

Run the showcase:

```bash
python examples/full_showcase.py
```

---

## Stochastic volatility

| Example | Description | Tags |
|---------|-------------|------|
| [Heston stochastic volatility](https://github.com/Nityahapani/spxa/blob/main/examples/heston_example.py) | Calibrate and price under the Heston model. Compute machine-precision Greeks (Delta, Gamma, Vega, Rho, Theta) via reverse-mode autodiff — no finite differences. | `volatility` `heston` `greeks` |
| [SABR model & implied volatility smile](https://github.com/Nityahapani/spxa/blob/main/examples/sabr_example.py) | SABR stochastic volatility with Hagan asymptotic implied volatility. Build skew and term structure from forward, alpha, beta, nu, rho. | `volatility` `sabr` `skew` |
| [Bates: Heston + compound Poisson jumps](https://github.com/Nityahapani/spxa/blob/main/examples/bates_example.py) | Stochastic volatility coupled with log-normal Merton jumps. Joint diffusion and jump dynamics for equity smile and term structure. | `volatility` `jumps` `bates` |
| [Rough Bergomi & fractional volatility](https://github.com/Nityahapani/spxa/blob/main/examples/vol_rough_example.py) | Rough Bergomi (rBergomi) with power-law short-time skew blow-up, Riemann–Liouville kernels, and fractional Adams–Bashforth–Moulton Riccati solvers. | `rough-vol` `rbergomi` `fractional` |

---

## Volatility surfaces & calibration

| Example | Description | Tags |
|---------|-------------|------|
| [SVI / SSVI / eSSVI volatility surfaces](https://github.com/Nityahapani/spxa/blob/main/examples/vol_svi_surface_example.py) | Arbitrage-free SVI parametrizations from option quotes. Raw, natural, and jump-wings slices with no-arbitrage conditions, Durrleman diagnostics, and calendar/wing arbitrage checks. | `surface` `svi` `calibration` |
| [Dupire local vol & stochastic local vol](https://github.com/Nityahapani/spxa/blob/main/examples/vol_local_vol_example.py) | Dupire local volatility in implied variance, Andreasen–Huge calibration, and SLV leverage via 2D Fokker–Planck ADI or particle calibration. | `local-vol` `dupire` `slv` |
| [Martingale optimal transport & model risk](https://github.com/Nityahapani/spxa/blob/main/examples/vol_model_risk_example.py) | Model-free bounds via LP or Sinkhorn MOT, static replication, hedge backtests, and Cont's coherent model-risk measures over calibrated model families. | `model-risk` `mot` `bounds` |

---

## Interest rates & curves

| Example | Description | Tags |
|---------|-------------|------|
| [Yield curve bootstrapping](https://github.com/Nityahapani/spxa/blob/main/examples/rates_curve_example.py) | Dual-curve bootstrapping with log-linear, monotone cubic, and Hagan–West monotone convex interpolation. OIS, swaps, bonds, and forward curve construction. | `rates` `curve` `bootstrap` |
| [Hull–White short-rate model](https://github.com/Nityahapani/spxa/blob/main/examples/hull_white_example.py) | Closed forms, exact simulation, trinomial tree, and PDE engines. Calibrate to swaptions and caps; price Bermudan swaptions with Andersen–Broadie bounds. | `rates` `hull-white` `short-rate` |
| [LIBOR & forward market models](https://github.com/Nityahapani/spxa/blob/main/examples/rates_lmm_example.py) | LIBOR market model with abcd volatilities, Rebonato correlations, rank reduction, predictor–corrector simulation, and caplet/swaption calibration. | `rates` `lmm` `libor` |

---

## PDE & PIDE solvers

| Example | Description | Tags |
|---------|-------------|------|
| [Black–Scholes PDE pricing](https://github.com/Nityahapani/spxa/blob/main/examples/pde_black_scholes_example.py) | Finite-difference pricing with Crank–Nicolson + Rannacher start-up. European, digital, American, Bermudan, barriers, discrete dividends, and local volatility. | `pde` `black-scholes` `american` |
| [Heston two-asset ADI PDE](https://github.com/Nityahapani/spxa/blob/main/examples/pde_heston_adi_example.py) | Douglas, Craig–Sneyd, modified Craig–Sneyd, and Hundsdorfer–Verwer ADI schemes for Heston and two-asset options with early-exercise LCPs. | `pde` `heston` `adi` |
| [Exponential-Lévy PIDEs](https://github.com/Nityahapani/spxa/blob/main/examples/pde_levy_and_density_example.py) | Merton, Kou, VG, CGMY, and NIG jump-diffusion PIDEs with moment-matched small jumps and cell-integrated large-jump operators. | `pde` `levy` `pide` |

---

## Simulation & backtesting

| Example | Description | Tags |
|---------|-------------|------|
| [QMC, MLMC & variance reduction](https://github.com/Nityahapani/spxa/blob/main/examples/simulation_backends_example.py) | Quasi-Monte Carlo (Sobol), multilevel Monte Carlo, antithetic variates, control variates, importance sampling, and stratified sampling backends. | `simulation` `qmc` `mlmc` |
| [Bridges & first-passage times](https://github.com/Nityahapani/spxa/blob/main/examples/full_showcase.py) | Brownian and OU bridges via Doob's h-transform, Gamma/Bessel bridges, and exact first-passage-time samplers for Brownian motion. | `simulation` `bridge` `first-passage` |

---

## Visualization & graph

| Example | Description | Tags |
|---------|-------------|------|
| [Computational graph DAG & visualization](https://github.com/Nityahapani/spxa/blob/main/examples/visualization_example.py) | Export pricing pipelines as DAGs (ASCII, DOT, Mermaid), plot Lévy triplet comparisons, implied volatility surfaces, and smile term structures. | `visualization` `dag` `graph` |
| [Full end-to-end showcase](https://github.com/Nityahapani/spxa/blob/main/examples/full_showcase.py) | Every major capability in one script: composition, cumulants, characteristic functions, stories, zoo processes, operations, simulation, analytics, and exact samplers. | `end-to-end` `showcase` |

---

## Run any example

Clone and run locally:

```bash
git clone https://github.com/Nityahapani/spxa.git
cd spxa
pip install -e ".[dev]"
python examples/heston_example.py
```
