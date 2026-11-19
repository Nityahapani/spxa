# spxa Example Gallery

Browse copy-pasteable examples organized by use case. Every example runs
out of the box with `pip install spxa` — no extra setup required.

<style>
.gallery-card {
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 0.5rem;
  padding: 1.25rem;
  margin: 0.75rem 0;
  background: var(--md-default-bg-color--lightest);
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
.gallery-card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  border-color: var(--md-primary-fg-color);
}
.gallery-card h4 {
  margin: 0 0 0.5rem 0;
  font-size: 1.1rem;
  color: var(--md-primary-fg-color);
}
.gallery-card p {
  margin: 0 0 0.75rem 0;
  font-size: 0.9rem;
  color: var(--md-default-fg-color--light);
  line-height: 1.5;
}
.gallery-card code {
  display: block;
  margin-top: 0.5rem;
  padding: 0.6rem;
  border-radius: 0.35rem;
  background: var(--md-code-bg-color);
  font-size: 0.82rem;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
.gallery-tag {
  display: inline-block;
  margin-right: 0.4rem;
  margin-bottom: 0.4rem;
  padding: 0.15rem 0.5rem;
  border-radius: 0.3rem;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  background: var(--md-primary-fg-color--lightest);
  color: var(--md-primary-fg-color);
}
.gallery-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1rem;
}
@media (max-width: 768px) {
  .gallery-grid { flex-direction: column; }
}
</style>

## Core algebra & exact arithmetic

<div class="gallery-grid">

<div class="gallery-card">
<span class="gallery-tag">algebra</span><span class="gallery-tag">exactness</span><span class="gallery-tag">cumulants</span>
<h4>Process composition & exact triplet arithmetic</h4>
<p>Compose Lévy processes with `+`, `*`, `@`. Characteristic triplets add in
closed form; cumulants derive symbolically to any order. Every result carries a
strict exactness level (`EXACT` → `MOMENT` → `SIMULATION`).</p>
<code>Z = 0.5 * VarianceGamma(...) + GammaProcess(...)<br>print(Z.exactness)  # ExactnessLevel.EXACT<br>print(Z.cumulants(4))</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/demo.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">subordination</span><span class="gallery-tag">story</span>
<h4>Bochner subordination & derivation traces</h4>
<p>Time-change one process by another via `X @ Y`. Generate plain-text and
LaTeX mathematical derivation stories with `__story__()` and `narrate_latex()`.</p>
<code>W = bm @ nig<br>print(W.__story__())<br>print(narrate_latex(node._node))</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/full_showcase.py">View source →</a></p>
</div>

</div>

## Stochastic volatility

<div class="gallery-grid">

<div class="gallery-card">
<span class="gallery-tag">volatility</span><span class="gallery-tag">heston</span><span class="gallery-tag">greeks</span>
<h4>Heston stochastic volatility</h4>
<p>Calibrate and price under the Heston model. Compute machine-precision Greeks
(Delta, Gamma, Vega, Rho, Theta) via reverse-mode autodiff — no finite differences.</p>
<code>heston = Heston(v0=0.04, kappa=2.0, theta=0.04, ...)<br>print(heston.exactness.name)<br># Greeks via GradientTape reverse pass</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/heston_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">volatility</span><span class="gallery-tag">sabr</span><span class="gallery-tag">skew</span>
<h4>SABR model & implied volatility smile</h4>
<p>SABR stochastic volatility with Hagan asymptotic implied volatility.
Build skew and term structure from forward, alpha, beta, nu, rho.</p>
<code>sabr = SABRProcess(alpha=0.25, beta=0.5, nu=0.4, rho=-0.3, f0=100.0)<br># implied vol surface from SABR asymptotic formula</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/sabr_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">volatility</span><span class="gallery-tag">jumps</span><span class="gallery-tag">bates</span>
<h4>Bates: Heston + compound Poisson jumps</h4>
<p>Stochastic volatility coupled with log-normal Merton jumps. Joint diffusion
and jump dynamics for equity smile and term structure.</p>
<code>bates = BatesProcess(v0=0.04, kappa=2.0, rate=1.2, jump_mean=-0.06, ...)<br># Heston diffusion + Merton jump diffusion</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/bates_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">rough vol</span><span class="gallery-tag">rbergomi</span><span class="gallery-tag">fractional</span>
<h4>Rough Bergomi & fractional volatility</h4>
<p>Rough Bergomi (rBergomi) with power-law short-time skew blow-up,
Riemann–Liouville kernels, and fractional Adams–Bashforth–Moulton Riccati solvers.</p>
<code># rough Heston with Hurst H < 0.5<br># fractional Riccati ABM solver<br># power-law skew scaling T^(H-1/2)</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/vol_rough_example.py">View source →</a></p>
</div>

</div>

## Volatility surfaces & calibration

<div class="gallery-grid">

<div class="gallery-card">
<span class="gallery-tag">surface</span><span class="gallery-tag">svi</span><span class="gallery-tag">calibration</span>
<h4>SVI / SSVI / eSSVI volatility surfaces</h4>
<p>Arbitrage-free SVI parametrizations from option quotes. Raw, natural, and
jump-wings slices with no-arbitrage conditions, Durrleman diagnostics, and
calendar/wing arbitrage checks.</p>
<code>surface = build_vol_surface(quotes, model="SVI")<br>report = surface.quality_report()<br># Durrleman, butterfly, calendar diagnostics</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/vol_svi_surface_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">local vol</span><span class="gallery-tag">dupire</span><span class="gallery-tag">slv</span>
<h4>Dupire local vol & stochastic local vol</h4>
<p>Dupire local volatility in implied variance, Andreasen–Huge calibration, and
SLV leverage via 2D Fokker–Planck ADI or particle calibration.</p>
<code>lv = DupirePDE(surface)<br>lv_price = lv.price(european_call)<br>slv_leverage = slv.leverage_surface()</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/vol_local_vol_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">model risk</span><span class="gallery-tag">mot</span><span class="gallery-tag">bounds</span>
<h4>Martingale optimal transport & model risk</h4>
<p>Model-free bounds via LP or Sinkhorn MOT, static replication, hedge
backtests, and Cont's coherent model-risk measures over calibrated model families.</p>
<code>bounds = mot.bounds(marginals, method="LP")<br>hedge = mot.dual_hedge(bounds)<br>risk = ModelRisk(models)</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/vol_model_risk_example.py">View source →</a></p>
</div>

</div>

## Interest rates & curves

<div class="gallery-grid">

<div class="gallery-card">
<span class="gallery-tag">rates</span><span class="gallery-tag">curve</span><span class="gallery-tag">bootstrap</span>
<h4>Yield curve bootstrapping</h4>
<p>Dual-curve bootstrapping with log-linear, monotone cubic, and Hagan–West
monotone convex interpolation. OIS, swaps, bonds, and forward curve construction.</p>
<code>curve = bootstrap_ois(deposits, fras, swaps)<br>print(curve.zero_rate(tenor="5Y"))<br># node & quote sensitivities</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/rates_curve_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">rates</span><span class="gallery-tag">hull-white</span><span class="gallery-tag">short rate</span>
<h4>Hull–White short-rate model</h4>
<p>Closed forms, exact simulation, trinomial tree, and PDE engines. Calibrate to
swaptions and caps; price Bermudan swaptions with Andersen–Broadie bounds.</p>
<code>hw = HullWhite(curve, sigma=0.01, a=0.05)<br>price = hw.swaption(swaption)<br>tree = hw.trinomial_tree()</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/hull_white_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">rates</span><span class="gallery-tag">lmm</span><span class="gallery-tag">libor</span>
<h4>LIBOR & forward market models</h4>
<p>LIBOR market model with abcd volatilities, Rebonato correlations, rank
reduction, predictor–corrector simulation, and caplet/swaption calibration.</p>
<code>lmm = LIBORMarketModel(forwards, volatilities, correlations)<br>caplet = lmm.caplet(strike, tenor)<br>swaption = lmm.swaption(...)<br># SABR-LMM extension available</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/rates_lmm_example.py">View source →</a></p>
</div>

</div>

## PDE & PIDE solvers

<div class="gallery-grid">

<div class="gallery-card">
<span class="gallery-tag">pde</span><span class="gallery-tag">black-scholes</span><span class="gallery-tag">american</span>
<h4>Black–Scholes PDE pricing</h4>
<p>Finite-difference pricing with Crank–Nicolson + Rannacher start-up. European,
digital, American, Bermudan, barriers, discrete dividends, and local volatility.</p>
<code>bs = BlackScholesPDE(surface, grid)<br>price = bs.price(american_put)<br>greeks = bs.grid_greeks()</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/pde_black_scholes_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">pde</span><span class="gallery-tag">heston</span><span class="gallery-tag">adi</span>
<h4>Heston two-asset ADI PDE</h4>
<p>Douglas, Craig–Sneyd, modified Craig–Sneyd, and Hundsdorfer–Verwer ADI schemes
for Heston and two-asset options with early-exercise LCPs.</p>
<code>adi = ADISolver(heston_pde, scheme="Hundsdorfer-Verwer")<br>price = adi.price(two_asset_option)</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/pde_heston_adi_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">pde</span><span class="gallery-tag">levy</span><span class="gallery-tag">pide</span>
<h4>Exponential-Lévy PIDEs</h4>
<p>Merton, Kou, VG, CGMY, and NIG jump-diffusion PIDEs with moment-matched small
jumps and cell-integrated large-jump operators.</p>
<code>pide = LevyPIDE(process, grid)<br>price = pide.price(european_call)<br># cell-integrated large jumps + FFT convolution</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/pde_levy_and_density_example.py">View source →</a></p>
</div>

</div>

## Simulation & backtesting

<div class="gallery-grid">

<div class="gallery-card">
<span class="gallery-tag">simulation</span><span class="gallery-tag">qmc</span><span class="gallery-tag">mlmc</span>
<h4>QMC, MLMC & variance reduction</h4>
<p>Quasi-Monte Carlo (Sobol), multilevel Monte Carlo, antithetic variates,
control variates, importance sampling, and stratified sampling backends.</p>
<code>paths = sample_qmc_brownian(n_paths, n_steps)<br>result = mlmc_estimate(payoff, levels=5)<br># antithetic + control variate estimators</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/simulation_backends_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">simulation</span><span class="gallery-tag">bridge</span><span class="gallery-tag">first passage</span>
<h4>Bridges & first-passage times</h4>
<p>Brownian and OU bridges via Doob's h-transform, Gamma/Bessel bridges, and
exact first-passage-time samplers for Brownian motion.</p>
<code>bridge = brownian_bridge(x0, xT, T, n_steps)<br>tau = first_passage_time_exact_bm(boundary)<br># exact (non-Euler) sampling</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/full_showcase.py">View source →</a></p>
</div>

</div>

## Visualization & graph

<div class="gallery-grid">

<div class="gallery-card">
<span class="gallery-tag">visualization</span><span class="gallery-tag">dag</span><span class="gallery-tag">graph</span>
<h4>Computational graph DAG & visualization</h4>
<p>Export pricing pipelines as DAGs (ASCII, DOT, Mermaid), plot Lévy triplet
comparisons, implied volatility surfaces, and smile term structures.</p>
<code>dag_to_dot(graph)<br>plot_iv_surface(surface)<br>plot_levy_density(process)</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/visualization_example.py">View source →</a></p>
</div>

<div class="gallery-card">
<span class="gallery-tag">end-to-end</span><span class="gallery-tag">showcase</span>
<h4>Full end-to-end showcase</h4>
<p>Every major capability in one script: composition, cumulants, characteristic
functions, stories, zoo processes, operations, simulation, analytics, and exact samplers.</p>
<code>python examples/full_showcase.py<br># runs 12 sections with pass/fail checks</code>
<p style="margin-top:0.5rem; font-size:0.8rem;"><a href="https://github.com/Nityahapani/spxa/blob/main/examples/full_showcase.py">View source →</a></p>
</div>

</div>

## Run any example

Clone and run locally:

```bash
git clone https://github.com/Nityahapani/spxa.git
cd spxa
pip install -e ".[dev]"
python examples/heston_example.py
```

All 45 examples are in [`examples/`](https://github.com/Nityahapani/spxa/tree/main/examples).
Each is tested in CI, so you can trust the output matches the library version.
