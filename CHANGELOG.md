# Changelog

All notable changes to spxa are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-09-19

Initial public release. Establishes the algebraic core, the 13-process Lévy zoo, the beyond-Lévy
category, the ops/sim/analytics layers, multivariate support, and the story renderer.

### Added — Core engine

- `core/exactness.py` — `ExactnessLevel` enum (`EXACT` / `MOMENT_PROPAGATION` / `SIMULATION_ONLY`)
  with total ordering and `combine()` rule: operations degrade to the minimum level of their operands
- `core/exceptions.py` — `SpxaError`, `ExactnessError`, `SpxaDegradationWarning`
- `core/levy_measure.py` — `LevyMeasure` hierarchy: `CompoundLevyMeasure`, `DensityLevyMeasure`,
  `ScaledLevyMeasure`
- `core/triplet.py` — immutable `LevyTriplet(b, σ², ν)` with arithmetic (`__add__`, `.scale()`),
  numerical `char_exp()` via the Lévy–Khintchine formula, and `cumulant(n)` (Sato 1999, Thm 25.3)
- `core/properties.py` — `ProcessProperties` lattice tracking stationarity, independence,
  martingale, finite mean/variance, tail index, self-similarity, subordinator flag, Hurst index;
  propagation rules for `+`, scaling, and `@` derived from Sato (1999) Ch. 11 & 30
- `core/process.py` — abstract `Process` base class; operator overloads `+`, `*`, `@`, `-`,
  unary `-`; each returns a new immutable `_ComposedProcess` or `_ScaledProcess` instance;
  `.triplet`, `.cumulants()`, `.char_func()`, `.__story__()`, `.__story_latex__()`
- `story/narrator.py` — `CompositionNode` tree, `narrate()` plain-text derivation engine,
  `narrate_latex()` / `story_latex()` MathJax-ready LaTeX renderer

### Added — Process zoo (`spxa.zoo.levy`)

- `BrownianMotion` — triplet `(μ, σ², 0)`, exact Gaussian simulation, closed-form `char_func_exact()`
- `PoissonProcess` — discrete Lévy measure, exact inter-arrival simulation, `char_func_exact()`
- `GammaProcess` — subordinator, exact `bernstein_function()` `a·log(1+u/b)` (Schilling et al.
  2012, Ex. 3.9), exact cumulants, Gamma simulation
- `VarianceGamma` — Lévy density `C/|x|·exp(Ax−B|x|)`, closed-form cumulants to order 4 and
  series above, exact `char_func_exact()` (Madan, Carr & Chang 1998, eq. 4), exact simulation
  via Gamma subordination
- `NIG` — Bessel Lévy density, exact `char_func_exact()`, closed-form cumulants,
  simulation via InvGaussian subordination (Barndorff-Nielsen 1997)
- `AlphaStable` — exact char_func, self-similarity index `1/α`, Chambers–Mallows–Stuck simulation
  (Samorodnitsky & Taqqu 1994)
- `CGMY` — exact char_func, closed-form cumulants finite for `α < 2`, series-representation
  simulation (Carr, Geman, Madan, Yor 2002; Rosiński 2001)
- `InverseGaussianProcess` — exact Bernstein function, cumulants `(2n−1)!!·μ^{2n-1}/λ^{n-1}`,
  exact `char_func`, Wald simulation (Tweedie 1957)
- `MeixnerProcess` — stable log-cosh characteristic function, exact cumulants to order 4,
  Gil-Pelaez CDF-inversion simulation (Schoutens & Teugels 1998)
- `TemperedStable` — Lévy density `C·x^{-1−α}·exp(−λx)`, exact Bernstein function, cumulants
  `C·Γ(n−α)/λ^{n−α}`, CDF-inversion simulation (Rosiński 2007)
- `NegativeBinomialProcess` — finite-activity discrete subordinator, exact characteristic
  function, closed-form cumulants to order 4, exact NegBin sampler (Quenouille 1949)

### Added — Beyond-Lévy (`spxa.zoo.beyond`)

- `FractionalBrownianMotion` — `MOMENT_PROPAGATION`; Hurst index algebra, Davies–Harte
  simulation; raises `ExactnessError` on `.triplet` (not a semimartingale)
- `HawkesProcess` — `MOMENT_PROPAGATION`; mean intensity, Fano factor, Ogata thinning
  simulation (Hawkes 1971)
- `OULevy` — `MOMENT_PROPAGATION`; stationary mean/variance, autocovariance, exact
  discretisation simulation (Barndorff-Nielsen & Shephard 2001)

### Added — Multivariate (`spxa.zoo.multivariate`)

- `MultivariateBrownianMotion`, `MultivariateLevyTriplet`, `CorrelatedLevy`,
  `correlated_brownian_motion()`, `independent_levy_vector()`

### Added — Operations (`spxa.ops`)

- `addition.py` — `add()`, `sum_processes()`, `triplet_add()`, `exactness_for_addition()`
- `scaling.py` — `scale()`, `negate()`, `subtract()`, `triplet_scale()`, `linear_transform_triplet()`
- `subordination.py` — `subordinate()`, `char_exp_subordinated()`, `bernstein_function()`,
  subordinator validity check
- `product.py` — `ProductProcess`, cumulant/moment conversion, moment-product formula
- `integral.py` — Itô isometry, stochastic convolution char_func, `StochasticIntegral` with
  Euler–Maruyama simulation

### Added — Simulation (`spxa.sim`)

- `euler.py` — Euler–Maruyama scheme, autonomous SDE solver, geometric Lévy, convergence-order
  estimator
- `exact.py` — exact samplers for BM, Gamma, InvGaussian, VG, NIG, stable (CMS),
  compound Poisson, tempered stable
- `bridge.py` — Brownian bridge, Gamma bridge, bridge interpolation, first-passage time
  (simulated and exact)

### Added — Analytics (`spxa.analytics`)

- `divergence.py` — L2, Hellinger, Wasserstein-2, Monte Carlo KL divergence between process
  marginals
- `fourier.py` — Carr–Madan FFT option pricer, put-call parity, implied-vol via bisection;
  routes through `char_func_exact` when available (~1000× speedup for VG/NIG/BM)
- `cumulants.py` — `cumulant_table()`, `skewness()`, `excess_kurtosis()`

### Added — Tests

- `tests/exactness/` — per-process suites verifying closed-form results against published
  formulas: BrownianMotion, GammaProcess, VarianceGamma, NIG, AlphaStable, CGMY,
  InverseGaussianProcess, MeixnerProcess, TemperedStable, NegativeBinomialProcess,
  fBM/Hawkes/OULevy (268 tests total, marked `exactness` — never skippable)
- `tests/unit/` — core engine, ops, sim, multivariate (covering ExactnessLevel, triplet
  arithmetic, composition, properties, story output, all sim algorithms)
- `tests/integration/` — 17 end-to-end composition and analytics tests

### Added — Documentation and tooling

- Three executable Jupyter tutorials: first composition, finance models, physics applications
- `PROCESS_REGISTRY.md` — full table of implemented processes with exactness levels, cumulant
  availability, simulation algorithms, and primary references
- `CONTRIBUTING.md` — contribution guide covering all nine contribution types with exact
  per-type file checklists, commit conventions, versioning policy, and RFC process for
  core changes
- `examples/demo.py`, `examples/full_showcase.py` (100+ checks across all 12 capability areas)
- GitHub Actions workflows: `ci.yml`, `docs.yml`, `benchmarks.yml`, `release.yml`
- `.pre-commit-config.yaml`, `.gitignore`
- `pyproject.toml`: ruff (linting), mypy strict, pytest with `exactness` and `slow` markers,
  coverage config

### Fixed

- `core/triplet.char_exp`: split-integral SciPy compatibility (no `points` kwarg with infinite
  limits)
- `zoo/vg`: triplet drift singularity — switched to tail integrals on `[1, ∞)` anchored to the
  exact mean; `cumulants()` and `char_func()` overridden with exact closed-form formulas
- `zoo/gamma`: triplet drift replaced from quadrature to exact exponential integral `E₁(b)`
- `zoo/nig`: `char_func()` overridden to route through `char_func_exact`, eliminating
  Lévy–Khintchine numerical overflow at large `|u|`
- `zoo/beyond/fbm`: Davies–Harte circulant embedding off-by-one and broadcast error
- `sim/euler.geometric_levy`: switched to MGF-based drift correction for non-Gaussian drivers
- `ops/bernstein_function`: stabilised elementwise computation, avoiding `log(0)`
- `core/_ComposedProcess.simulate`: subordination (`@`) implemented via path interpolation
- `analytics/fourier`: replaced deprecated `np.trapz` with `np.trapezoid` (NumPy 2.x)
- `pyproject.toml`: corrected build backend to `setuptools.build_meta`
