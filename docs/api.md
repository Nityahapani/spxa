# spxa API Reference

Complete reference for spxa v1.0.0. Every entry here was verified against
the running library. All signatures are exact.

---

## Contents

1. [Top-level imports](#top-level-imports)
2. [Core](#core)
   - [ExactnessLevel](#exactnesslevel)
   - [Process](#process)
   - [LevyTriplet](#levytriplet)
   - [LevyMeasure hierarchy](#levymeasure-hierarchy)
   - [ProcessProperties](#processproperties)
   - [Exceptions and warnings](#exceptions-and-warnings)
3. [Zoo — Lévy processes](#zoo--lévy-processes)
   - [BrownianMotion](#brownianmotion)
   - [GammaProcess](#gammaprocess)
   - [VarianceGamma](#variancegamma)
   - [NIG](#nig)
   - [CGMY](#cgmy)
   - [AlphaStable](#alphastable)
   - [MeixnerProcess](#meixnerprocess)
   - [PoissonProcess](#poissonprocess)
   - [InverseGaussianProcess](#inversegaussianprocess)
   - [TemperedStable](#temperedstable)
   - [NegativeBinomialProcess](#negativebinomialprocess)
4. [Zoo — beyond Lévy](#zoo--beyond-lévy)
   - [FractionalBrownianMotion](#fractionalbrownianmotion)
   - [HawkesProcess](#hawkesprocess)
   - [OULevy](#oulevy)
5. [Zoo — multivariate](#zoo--multivariate)
   - [MultivariateLevyTriplet](#multivariatelevytriplet)
   - [MultivariateBrownianMotion](#multivariatebrownianmotion)
   - [CorrelatedLevy](#correlatedlevy)
   - [correlated_brownian_motion](#correlated_brownian_motion)
   - [independent_levy_vector](#independent_levy_vector)
6. [ops — operations](#ops--operations)
   - [addition](#addition)
   - [scaling](#scaling)
   - [subordination](#subordination)
   - [product](#product)
   - [integral](#integral)
7. [analytics](#analytics)
   - [cumulants](#cumulants-module)
   - [divergence](#divergence)
   - [fourier](#fourier)
8. [sim — simulation backends](#sim--simulation-backends)
   - [euler](#euler)
   - [exact samplers](#exact-samplers)
   - [bridge](#bridge)
9. [story — derivation traces](#story--derivation-traces)

---

## Top-level imports

Everything in this section is importable directly from `spxa`.

```python
import spxa

spxa.__version__          # "1.0.0"

# Core types
spxa.Process
spxa.LevyTriplet
spxa.LevyMeasure
spxa.ProcessProperties
spxa.ExactnessLevel
spxa.ExactnessError
spxa.SpxaDegradationWarning

# Lévy processes
spxa.BrownianMotion
spxa.GammaProcess
spxa.VarianceGamma
spxa.NIG
spxa.CGMY
spxa.AlphaStable
spxa.MeixnerProcess
spxa.PoissonProcess
spxa.InverseGaussianProcess
spxa.TemperedStable
spxa.NegativeBinomialProcess

# Beyond-Lévy processes
spxa.FractionalBrownianMotion
spxa.HawkesProcess
spxa.OULevy

# Multivariate
spxa.MultivariateBrownianMotion
spxa.MultivariateLevyTriplet
spxa.CorrelatedLevy
spxa.correlated_brownian_motion
spxa.independent_levy_vector
```

---

## Core

### ExactnessLevel

```python
from spxa.core import ExactnessLevel
```

An enum describing the analytical guarantee attached to a process or result.
All arithmetic between two processes produces the **minimum** level of its operands.

| Member | Meaning |
|---|---|
| `ExactnessLevel.EXACT` | Lévy–Khintchine triplet known in closed form. Triplet arithmetic is analytically exact. |
| `ExactnessLevel.MOMENT_PROPAGATION` | No closed-form triplet. Finite moments tracked where computable. |
| `ExactnessLevel.SIMULATION_ONLY` | No analytical handle. Sample paths only. |

**Ordering:** `EXACT` < `MOMENT_PROPAGATION` < `SIMULATION_ONLY` (where `<` means "more informative than").

```python
ExactnessLevel.combine(a, b)  # → ExactnessLevel
# Returns the minimum (most degraded) of two levels.
```

**Comparison operators** `<`, `<=`, `>`, `>=` work as expected.

---

### Process

```python
from spxa.core import Process  # abstract base class
```

All stochastic processes in spxa inherit from `Process`. Every instance is
**immutable** — operators return new process objects, never mutate in place.

#### Properties

```python
process.exactness    # → ExactnessLevel
process.triplet      # → LevyTriplet   (raises ExactnessError if not EXACT)
process.properties   # → ProcessProperties
```

#### Operators

```python
Z = X + Y            # sum of independent processes
Z = X - Y            # X + (-1)*Y
Z = c * X            # scalar multiplication, c ∈ ℝ \ {0}
Z = X * c            # same as c * X
Z = -X               # negation
Z = X @ T            # subordination: Z_t = X_{T_t}  (T must be a subordinator)
Z = sum([X, Y, ...]) # Python's built-in sum() works via __radd__
```

All operators propagate `ExactnessLevel` and `ProcessProperties` automatically.
If `ExactnessLevel` degrades, `SpxaDegradationWarning` is emitted with a
mathematical reason.

#### Methods

```python
process.cumulants(order: int) -> dict[int, float]
```
Return cumulants κ₁, …, κ_order of X₁ (per unit time). Available for
`EXACT` and `MOMENT_PROPAGATION`. Keys are integers 1..order, values are floats.

```python
process.char_func(u: float | np.ndarray, t: float = 1.0) -> np.ndarray
```
Evaluate E[e^{iuX_t}]. Available for `EXACT`. For processes with a
closed-form formula this delegates to `char_func_exact`; otherwise uses
Lévy–Khintchine numerical integration.

```python
process.simulate(
    n_steps: int,
    n_paths: int = 1,
    T: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Simulate sample paths. Returns shape `(n_paths, n_steps + 1)`. All paths
start at 0: `paths[:, 0] == 0` always. For multivariate processes returns
shape `(n_paths, n_steps + 1, d)`.

```python
process.__story__() -> str
```
Plain-text derivation trace: how this process was composed, with citations to
Sato (1999) and other primary sources. Suitable for terminal printing.

```python
process.__story_latex__() -> str
```
MathJax-ready LaTeX derivation. Returns a `\begin{aligned}...\end{aligned}`
block. Use in Jupyter:
```python
from IPython.display import display, Math
display(Math(Z.__story_latex__()))
```

---

### LevyTriplet

```python
from spxa.core import LevyTriplet
```

Immutable dataclass representing the Lévy–Khintchine triplet $(b, \sigma^2, \nu)$.

```python
LevyTriplet(b: float, sigma_sq: float, nu: LevyMeasure)
```

| Field | Type | Description |
|---|---|---|
| `b` | `float` | Drift, defined w.r.t. truncation function h(x)=𝟙_{|x|≤1} |
| `sigma_sq` | `float` | Gaussian variance σ² ≥ 0 |
| `nu` | `LevyMeasure` | Lévy measure on ℝ\{0} |

#### Methods

```python
triplet + other           # → LevyTriplet  (b adds, σ² adds, ν adds)
triplet.scale(c: float)   # → LevyTriplet  (image measure ν(·/c), drift corrected)

triplet.cumulant(n: int) -> float
# n-th cumulant of X₁:
#   κ₁ = b + ∫_{|x|>1} x ν(dx)
#   κ₂ = σ² + ∫ x² ν(dx)
#   κ_n = ∫ xⁿ ν(dx)  for n ≥ 3

triplet.char_exp(u: float | np.ndarray, t: float = 1.0) -> np.ndarray
# Evaluate E[e^{iuX_t}] = exp(t·ψ(u)) via numerical Lévy–Khintchine integration.
# For processes with char_func_exact, prefer that instead.
```

---

### LevyMeasure hierarchy

```python
from spxa.core import LevyMeasure, DensityLevyMeasure, CompoundLevyMeasure, ScaledLevyMeasure
```

#### `LevyMeasure` (abstract base)

```python
measure.density(x: np.ndarray) -> np.ndarray   # Lévy density k(x): ν(dx)=k(x)dx
measure.total_mass() -> float                   # ν(ℝ\{0}), np.inf for infinite activity
measure.tail(x: float) -> float                 # ν((x,∞)) for x > 0
measure.moment(n: int) -> float                 # ∫ xⁿ ν(dx), n ≥ 1
measure.is_finite_activity() -> bool            # total_mass() < ∞
measure + other                                 # → CompoundLevyMeasure
```

#### `DensityLevyMeasure`

Lévy measure specified by a density function. Used for all named processes
(Gamma, NIG, VG, CGMY, etc.). The density, total mass, tail function, and
moment formulas are provided at construction time.

```python
DensityLevyMeasure(
    _density_fn: Callable[[np.ndarray], np.ndarray],
    _total_mass: float,
    _tail_fn: Callable[[float], float] | None = None,
    _moment_fns: dict[int, Callable[[], float]] = {},
)
```

#### `CompoundLevyMeasure`

Sum of multiple Lévy measures. Produced by `measure1 + measure2`.
All methods delegate to components.

#### `ScaledLevyMeasure`

Image measure ν_Z where Z=cX: ν_Z(B)=ν_X(B/c). Produced by `LevyTriplet.scale(c)`.

---

### ProcessProperties

```python
from spxa.core import ProcessProperties
```

A dataclass carrying mathematical properties of a process. Propagated automatically
under all operations. Use `process.properties` to access.

| Field | Type | Meaning |
|---|---|---|
| `has_stationary_increments` | `bool` | X_{t+s}−X_s ∼ X_t in distribution |
| `has_independent_increments` | `bool` | Increments on disjoint intervals independent |
| `is_martingale` | `bool` | E[X_t \| F_s] = X_s a.s. |
| `has_finite_variance` | `bool` | Var(X_t) < ∞ |
| `has_finite_mean` | `bool` | E[\|X_t\|] < ∞ |
| `self_similarity_index` | `float \| None` | H such that X_{ct} ∼ c^H X_t |
| `tail_index` | `float \| None` | α such that ν((x,∞)) ∼ x^{−α} |
| `is_subordinator` | `bool` | Paths are non-decreasing a.s. |
| `hurst_index` | `float \| None` | Hurst index (fBM only) |
| `notes` | `list[str]` | Human-readable notes appended during composition |

**Propagation rules** — automatically applied by operators:

| Operation | Stationary incr. | Indep. incr. | Martingale | Subordinator |
|---|---|---|---|---|
| X + Y | X and Y both | X and Y both | X and Y both | X and Y both |
| c·X | preserved | preserved | preserved (c≠0) | c > 0 only |
| X @ T | always True | always True | False (check specific) | False |

---

### Exceptions and warnings

```python
from spxa.core import SpxaError, ExactnessError, SpxaDegradationWarning
```

#### `SpxaError`
Base class for all spxa exceptions.

#### `ExactnessError(SpxaError)`
Raised when a method requiring a higher `ExactnessLevel` is called. Always
includes a mathematical explanation of what is missing and why.

```python
# Attributes
err.method    # str — name of the method that was called
err.required  # str — ExactnessLevel name required
err.actual    # str — ExactnessLevel name of the process
err.reason    # str — mathematical explanation
```

Common triggers: `.triplet` on a beyond-Lévy process; `.char_func` on
`SIMULATION_ONLY`; `.cumulants` on `SIMULATION_ONLY`.

#### `SpxaDegradationWarning(UserWarning)`
Emitted when an operation degrades `ExactnessLevel`. Always includes the
mathematical reason (which Lévy property was violated and why).

```python
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    Z = fbm + bm   # emits SpxaDegradationWarning
```

---

## Zoo — Lévy processes

All processes in `spxa.zoo.levy` have `ExactnessLevel.EXACT`.
All inherit from `Process` and implement `triplet`, `cumulants`,
`char_func`, and `simulate`.

### BrownianMotion

```python
from spxa.zoo.levy import BrownianMotion

BrownianMotion(mu: float = 0.0, sigma: float = 1.0)
```

Standard Brownian motion with drift. Triplet: (μ, σ², 0) — no jump component.

| Parameter | Constraint | Meaning |
|---|---|---|
| `mu` | ℝ | Drift per unit time |
| `sigma` | > 0 | Diffusion coefficient |

**Cumulants:** κ₁ = μ, κ₂ = σ², κ_n = 0 for n ≥ 3.

**Extra methods:**
```python
bm.char_func_exact(u, t=1.0)  # exp(iμtu − σ²tu²/2)
```

**Properties:** self-similarity index H=0.5; martingale iff μ=0.

**Reference:** Wiener (1923); Sato (1999), Example 8.3.

---

### GammaProcess

```python
from spxa.zoo.levy import GammaProcess

GammaProcess(a: float, b: float)
```

Pure-jump subordinator with independent Gamma increments.
Lévy density: k(x) = a·x⁻¹·e^{−bx}·𝟙_{x>0}.

| Parameter | Constraint | Meaning |
|---|---|---|
| `a` | > 0 | Shape rate (jump frequency) |
| `b` | > 0 | Inverse scale (jump size decay) |

**Cumulants:** κ_n = a·(n−1)! / b^n.

**Extra methods:**
```python
gp.cumulant_exact(n: int, t: float = 1.0) -> float
gp.bernstein_function(lam: float | np.ndarray) -> np.ndarray
# Exact: φ(s) = a·log(1 + s/b)
```

**Properties:** is_subordinator=True; infinite activity.

**Reference:** Moran (1968); Cont & Tankov (2004), Table 4.2.

---

### VarianceGamma

```python
from spxa.zoo.levy import VarianceGamma

VarianceGamma(sigma: float, nu: float, theta: float = 0.0)
```

Brownian motion with drift subordinated by a Gamma process.
Infinite activity, finite variation. Standard model for equity log-returns.

| Parameter | Constraint | Meaning |
|---|---|---|
| `sigma` | > 0 | Volatility of BM component |
| `nu` | > 0 | Variance rate of Gamma subordinator |
| `theta` | ℝ | Drift of BM component (controls skewness) |

**Cumulants (Madan, Carr & Chang 1998):**
- κ₁ = θ
- κ₂ = σ² + θ²ν
- κ₃ = 2θ³ν² + 3σ²θν
- κ₄ = 3σ⁴ν + 12σ²θ²ν² + 6θ⁴ν³

**Extra methods:**
```python
vg.char_func_exact(u, t=1.0)
# ((1 − iuθν + σ²νu²/2))^{−t/ν}
```

**Simulation:** Gamma subordination: G ~ Gamma(Δt/ν, ν); ΔX = θG + σ√G·Z.

**Reference:** Madan, Carr & Chang (1998).

---

### NIG

```python
from spxa.zoo.levy import NIG

NIG(alpha: float, beta: float, delta: float, mu: float = 0.0)
```

Normal Inverse Gaussian process. BM subordinated by an Inverse Gaussian process.
Semi-heavy tails (exponential decay). Parameter space: α > |β|.

| Parameter | Constraint | Meaning |
|---|---|---|
| `alpha` | > 0, > \|β\| | Tail heaviness (larger → lighter tails) |
| `beta` | ∈ (−α, α) | Asymmetry (β < 0 → left skew) |
| `delta` | > 0 | Scale |
| `mu` | ℝ | Location shift |

Let γ = √(α² − β²).

**Cumulants (Barndorff-Nielsen 1997):**
- κ₁ = μ + δβ/γ
- κ₂ = δα²/γ³
- κ₃ = 3δα²β/γ⁵
- κ₄ = 3δα²(α²+4β²)/γ⁷

**Extra methods:**
```python
nig.char_func_exact(u, t=1.0)
# exp(t·(iμu + δ(γ − √(α² − (β+iu)²))))
```

**Simulation:** Inverse Gaussian subordination.

**Reference:** Barndorff-Nielsen (1997).

---

### CGMY

```python
from spxa.zoo.levy import CGMY

CGMY(C: float, G: float, M: float, Y: float)
```

Four-parameter pure-jump process. Nests VG (Y=0) and α-stable (G=M→0).
Controls tail heaviness, jump activity, and asymmetry independently.
Lévy density: C·e^{−Gx}/x^{1+Y}·𝟙_{x>0} + C·e^{−M|x|}/|x|^{1+Y}·𝟙_{x<0}.

| Parameter | Constraint | Meaning |
|---|---|---|
| `C` | > 0 | Jump intensity |
| `G` | ≥ 0 | Right-tail decay rate |
| `M` | ≥ 0 | Left-tail decay rate |
| `Y` | < 2 | Fine structure: Y<0 finite activity, 0≤Y<1 infinite activity finite variation, 1≤Y<2 infinite variation |

**Cumulants (Carr et al. 2002, eq. 2):**
κ_n = C·Γ(n−Y)·(M^{Y−n} + (−1)^n·G^{Y−n}) — valid for n > Y.

**Extra methods:**
```python
cgmy.cumulant_exact(n: int, t: float = 1.0) -> float
cgmy.char_func_exact(u, t=1.0)
# exp(t·C·Γ(−Y)·((M−iu)^Y − M^Y + (G+iu)^Y − G^Y))
```

**Reference:** Carr, Geman, Madan & Yor (2002).

---

### AlphaStable

```python
from spxa.zoo.levy import AlphaStable

AlphaStable(alpha: float, beta: float = 0.0, sigma: float = 1.0, mu: float = 0.0)
```

α-stable Lévy process. Only class of Lévy processes closed under addition
and scaling. Power-law tails for α < 2.

| Parameter | Constraint | Meaning |
|---|---|---|
| `alpha` | ∈ (0, 2] | Stability index. α=2 → Gaussian, α=1 → Cauchy |
| `beta` | ∈ [−1, 1] | Skewness. β=0 → symmetric |
| `sigma` | > 0 | Scale |
| `mu` | ℝ | Location |

**Moments:** E[|X₁|^p] < ∞ iff p < α. Mean finite iff α > 1. Variance finite iff α = 2.

**Extra methods:**
```python
stable.char_func_exact(u, t=1.0)
# exp(t·(−σ^α|u|^α(1 − iβ·sign(u)·tan(πα/2)) + iμu))  for α ≠ 1
```

**Properties:** self_similarity_index = 1/α.

**Simulation:** Chambers–Mallows–Stuck (1976) algorithm — exact.

**Reference:** Samorodnitsky & Taqqu (1994); Nolan (2020).

---

### MeixnerProcess

```python
from spxa.zoo.levy import MeixnerProcess

MeixnerProcess(alpha: float, beta: float, delta: float, m: float = 0.0)
```

Pure-jump process with hypergeometric characteristic function and orthogonal
polynomial structure (Meixner–Pollaczek polynomials are eigenfunctions of its generator).

| Parameter | Constraint | Meaning |
|---|---|---|
| `alpha` | > 0 | Scale |
| `beta` | ∈ (−π, π) | Asymmetry. β=0 → symmetric |
| `delta` | > 0 | Shape (jump intensity) |
| `m` | ℝ | Location shift |

**Cumulants (Schoutens & Teugels 1998):**
- κ₁ = δα·tan(β/2) + m
- κ₂ = δα²/(2cos²(β/2))
- κ₃ = δα³·sin(β/2)/(2cos³(β/2))
- κ₄ = δα⁴(2+cosβ)/(8cos⁴(β/2))

**Extra methods:**
```python
mx.char_func_exact(u, t=1.0)
# exp(imu)·(cos(β/2)/cosh((αu−iβ)/2))^{2δt}

mx.cumulant_exact(n: int, t: float = 1.0) -> float
```

**Simulation:** Gil-Pelaez CDF inversion.

**Reference:** Schoutens & Teugels (1998).

---

### PoissonProcess

```python
from spxa.zoo.levy import PoissonProcess

PoissonProcess(rate: float, jump_size: float = 1.0)
```

Homogeneous Poisson process with constant jump size. Finite activity, discrete jumps.

| Parameter | Constraint | Meaning |
|---|---|---|
| `rate` | > 0 | Arrival rate λ (jumps per unit time) |
| `jump_size` | ℝ\{0} | Size of each jump c |

**Cumulants:** κ_n(X_t) = λ·c^n·t for all n ≥ 1.

**Properties:** is_subordinator=True iff jump_size > 0; finite activity.

**Simulation:** Poisson-distributed increment counts.

**Reference:** Kingman (1993); Sato (1999), Example 8.4.

---

### InverseGaussianProcess

```python
from spxa.zoo.levy import InverseGaussianProcess

InverseGaussianProcess(mu: float, lam: float)
```

Subordinator with independent IG-distributed increments. Fundamental in NIG
construction. Lévy density: √(λ/2π)·x^{−3/2}·exp(−λ(x−μ)²/(2μ²x)) for x > 0.

| Parameter | Constraint | Meaning |
|---|---|---|
| `mu` | > 0 | Mean of X₁ |
| `lam` | > 0 | Shape; Var(X₁) = μ³/λ |

**Cumulants (Tweedie 1957, eq. 2.5):**
κ_n = (2n−1)!!·μ^{2n−1}/λ^{n−1}  where (2n−1)!! = 1·3·5·…·(2n−1).

Explicitly: κ₁ = μ, κ₂ = 3μ³/λ, κ₃ = 15μ⁵/λ², κ₄ = 105μ⁷/λ³.

**Extra methods:**
```python
ig.cumulant_exact(n: int, t: float = 1.0) -> float
ig.char_func_exact(u, t=1.0)
# exp(t·(λ/μ)·(1 − √(1 − 2iuμ²/λ)))

ig.bernstein_function(lam: float | np.ndarray) -> np.ndarray
# Exact: φ(s) = (λ/μ)(1 − √(1 − 2μ²s/λ))
```

**Properties:** is_subordinator=True; infinite activity.

**Reference:** Tweedie (1957); Schilling et al. (2012), Example 3.12.

---

### TemperedStable

```python
from spxa.zoo.levy import TemperedStable

TemperedStable(alpha: float, C: float, lam: float)
```

Pure-jump subordinator with Lévy density C·x^{−1−α}·exp(−λx) for x>0.
All moments finite; infinite activity. Bridges between stable and Gamma.

| Parameter | Constraint | Meaning |
|---|---|---|
| `alpha` | ∈ (0, 1) | Stability index (controls small-jump behaviour) |
| `C` | > 0 | Jump intensity |
| `lam` | > 0 | Tempering (exponential decay of large jumps) |

**Cumulants (Rosiński 2007, Prop. 2.1):**
κ_n = C·Γ(n−α)/λ^{n−α} for all n ≥ 1.

**Extra methods:**
```python
ts.cumulant_exact(n: int, t: float = 1.0) -> float
ts.char_func_exact(u, t=1.0)
# exp(t·C·Γ(−α)·((λ−iu)^α − λ^α))

ts.bernstein_function(lam: float | np.ndarray) -> np.ndarray
# Exact: φ(s) = −C·Γ(−α)·((λ+s)^α − λ^α)  [>0 since Γ(−α)<0 for α∈(0,1)]
```

**Properties:** is_subordinator=True; infinite activity; all moments finite.

**Simulation:** Gil-Pelaez CDF inversion.

**Reference:** Rosiński (2007); Schilling et al. (2012), Example 3.8.

---

### NegativeBinomialProcess

```python
from spxa.zoo.levy import NegativeBinomialProcess

NegativeBinomialProcess(r: float, p: float)
```

Finite-activity pure-jump subordinator with NegBin-distributed increments.
Discrete Lévy measure: ν = Σ_{k=1}^∞ (r·p^k/k)·δ_k. Equivalent to a
Poisson process with Gamma-distributed rate.

| Parameter | Constraint | Meaning |
|---|---|---|
| `r` | > 0 | Shape (dispersion) |
| `p` | ∈ (0, 1) | Success probability |

Mean = rp/(1−p), Var = rp/(1−p)² > Mean (overdispersed).

**Cumulants (Johnson et al. 2005, Table 5.2):**
- κ₁ = rp/(1−p)
- κ₂ = rp/(1−p)²
- κ₃ = rp(1+p)/(1−p)³
- κ₄ = rp(1+4p+p²)/(1−p)⁴

**Extra methods:**
```python
nb.cumulant_exact(n: int, t: float = 1.0) -> float
nb.char_func_exact(u, t=1.0)
# ((1−p)/(1−p·e^{iu}))^{rt}
```

**Properties:** is_subordinator=True; finite activity; integer-valued paths.

**Simulation:** `numpy.negative_binomial(n=r·Δt, p=1−p)` — exact.

**Reference:** Quenouille (1949); Johnson, Kemp & Kotz (2005), Chapter 5.

---

## Zoo — beyond Lévy

Processes in `spxa.zoo.beyond` have `ExactnessLevel.MOMENT_PROPAGATION`.
Calling `.triplet` always raises `ExactnessError` with a mathematical explanation.

### FractionalBrownianMotion

```python
from spxa.zoo.beyond import FractionalBrownianMotion

FractionalBrownianMotion(H: float, sigma: float = 1.0)
```

Centred Gaussian process with covariance
Cov(B^H_s, B^H_t) = σ²/2·(s^{2H} + t^{2H} − |s−t|^{2H}).
Not a semimartingale for H ≠ 0.5. No Lévy–Khintchine triplet.

| Parameter | Constraint | Meaning |
|---|---|---|
| `H` | ∈ (0, 1) | Hurst index. H=0.5 → standard BM |
| `sigma` | > 0 | Scale (std dev at t=1) |

**Extra methods:**
```python
fbm.variance(t: float) -> float
# σ²·t^{2H}

fbm.covariance(s: float, t: float) -> float
# σ²/2·(s^{2H} + t^{2H} − |s−t|^{2H})
```

**Properties:** self_similarity_index=H; hurst_index=H; has_independent_increments=False for H≠0.5.

**Simulation:** Davies–Harte circulant embedding — exact in distribution, O(n log n).

**Reference:** Mandelbrot & Van Ness (1968).

---

### HawkesProcess

```python
from spxa.zoo.beyond import HawkesProcess

HawkesProcess(mu: float, alpha: float, beta: float)
```

Self-exciting point process with conditional intensity
λ(t) = μ + α·Σ_{t_i<t} exp(−β(t−t_i)).
Requires subcriticality: α < β (α/β < 1).

| Parameter | Constraint | Meaning |
|---|---|---|
| `mu` | > 0 | Background intensity |
| `alpha` | > 0, < β | Excitation amplitude |
| `beta` | > 0 | Exponential decay rate |

**Extra properties (on the instance):**
```python
hawkes.mean_intensity        # float: μ/(1−α/β)
hawkes.asymptotic_variance_rate  # float: λ̄·(1+α/β)/(1−α/β)²
hawkes.fano_factor           # float: (1+α/β)/(1−α/β)² — always ≥ 1
```

**Simulation:** Ogata's thinning algorithm.

**Reference:** Hawkes (1971); Hawkes & Oakes (1974); Ogata (1981).

---

### OULevy

```python
from spxa.zoo.beyond import OULevy

OULevy(lam: float, subordinator: Process | None = None, v0: float = 0.0)
```

Ornstein–Uhlenbeck process driven by a Lévy subordinator:
dV_t = −λV_t dt + dL_{λt}. Default subordinator: GammaProcess(a=1, b=1).

| Parameter | Constraint | Meaning |
|---|---|---|
| `lam` | > 0 | Mean reversion rate λ |
| `subordinator` | must be a subordinator | Driving Lévy process L |
| `v0` | ℝ | Initial value V₀ |

**Extra properties (on the instance):**
```python
ou.stationary_mean        # float: κ₁(L₁)
ou.stationary_variance    # float: κ₂(L₁)/(2λ)

ou.autocovariance(h: float) -> float
# e^{−λh}·Var(V∞)   [Barndorff-Nielsen & Shephard 2001, eq. 17]
```

**Simulation:** Exact Euler discretisation.

**Reference:** Barndorff-Nielsen & Shephard (2001).

---

## Zoo — multivariate

### MultivariateLevyTriplet

```python
from spxa.zoo.multivariate import MultivariateLevyTriplet

MultivariateLevyTriplet(b: np.ndarray, sigma: np.ndarray, nu: LevyMeasure, dim: int)
```

d-dimensional Lévy–Khintchine triplet.

| Field | Type | Constraint |
|---|---|---|
| `b` | `np.ndarray` | shape (d,) |
| `sigma` | `np.ndarray` | shape (d,d), symmetric PSD |
| `nu` | `LevyMeasure` | Lévy measure on ℝᵈ\{0} |
| `dim` | `int` | Dimension d |

```python
t1 + t2           # component-wise addition (same dim required)
t.transform(A)    # → MultivariateLevyTriplet  A: ℝᵈ→ℝᵏ

t.cumulant(n: int, direction: np.ndarray) -> float
# n-th cumulant of the projection ⟨direction, X₁⟩
```

---

### MultivariateBrownianMotion

```python
from spxa.zoo.multivariate import MultivariateBrownianMotion

MultivariateBrownianMotion(
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    dim: int = 2,
)
```

d-dimensional BM with drift μ and covariance matrix Σ.
Simulation via Cholesky decomposition: ΔX = μ·Δt + L·Z·√Δt where L=chol(Σ).

**Extra methods:**
```python
mbm.covariance(t: float = 1.0) -> np.ndarray   # t·Σ
mbm.correlation() -> np.ndarray                 # correlation matrix of X₁
mbm.multivariate_triplet() -> MultivariateLevyTriplet
```

**Simulation:** returns shape `(n_paths, n_steps+1, dim)`.

---

### CorrelatedLevy

```python
from spxa.zoo.multivariate import CorrelatedLevy

CorrelatedLevy(processes: list[Process], A: np.ndarray)
```

d-dimensional Lévy process Z_t = A·(X¹_t, …, Xᵈ_t)ᵀ, mixing d independent
1-d processes with matrix A ∈ ℝ^{k×d}. Covariance: Σ_Z = A·diag(σ²)·Aᵀ.

```python
cl.multivariate_triplet() -> MultivariateLevyTriplet
```

**Simulation:** returns shape `(n_paths, n_steps+1, dim_out)`.

---

### correlated_brownian_motion

```python
from spxa.zoo.multivariate import correlated_brownian_motion

correlated_brownian_motion(
    sigma_matrix: np.ndarray,
    mu: np.ndarray | None = None,
) -> MultivariateBrownianMotion
```

Convenience constructor: d-dimensional BM with target covariance Σ.

---

### independent_levy_vector

```python
from spxa.zoo.multivariate import independent_levy_vector

independent_levy_vector(*processes: Process) -> CorrelatedLevy
```

Stack d independent 1-d processes into a d-dimensional vector process
with identity mixing matrix A=I_d.

---

## ops — operations

### addition

```python
from spxa.ops import add, sum_processes, triplet_add, exactness_for_addition
```

```python
add(X: Process, Y: Process) -> Process
```
Equivalent to `X + Y`. Returns a new process representing the sum of two
independent processes. Triplet: (b_X+b_Y, σ²_X+σ²_Y, ν_X+ν_Y) when both EXACT.

```python
sum_processes(processes: Iterable[Process]) -> Process
```
Sum of an iterable of independent processes. Raises `ValueError` if empty.

```python
triplet_add(t1: LevyTriplet, t2: LevyTriplet) -> LevyTriplet
```
Add two triplets component-wise. Useful when constructing custom processes.

```python
exactness_for_addition(X: Process, Y: Process) -> tuple[ExactnessLevel, str]
```
Determine the `ExactnessLevel` of X+Y and the reason if it degrades.
Returns `(level, reason)` where `reason` is `""` if no degradation.

---

### scaling

```python
from spxa.ops import scale, negate, subtract, triplet_scale
```

```python
scale(X: Process, c: float) -> Process       # c·X (c ≠ 0)
negate(X: Process) -> Process                # −X
subtract(X: Process, Y: Process) -> Process  # X − Y
triplet_scale(triplet: LevyTriplet, c: float) -> LevyTriplet
```

`triplet_scale` implements the exact image-measure formula:
b_Z = c·b + drift_correction, σ²_Z = c²σ², ν_Z(B) = ν_X(B/c).

---

### subordination

```python
from spxa.ops import subordinate, char_exp_subordinated, bernstein_function, is_valid_subordination
```

```python
subordinate(parent: Process, subordinator: Process) -> Process
```
Equivalent to `parent @ subordinator`. Returns Z_t = X_{T_t}.

```python
char_exp_subordinated(
    parent: Process,
    subordinator: Process,
    u: float | np.ndarray,
    t: float = 1.0,
) -> np.ndarray
```
Characteristic function of Z = X @ T using the composition formula
ψ_Z(u) = −φ(−ψ_X(u)) where φ is the Laplace exponent of T.

```python
bernstein_function(
    subordinator: Process,
    lam: float | np.ndarray,
) -> np.ndarray
```
Evaluate the Laplace exponent φ(s) = −log E[e^{−sT₁}].
Dispatches to a closed-form `bernstein_function` method on the process
when available (GammaProcess, InverseGaussianProcess, TemperedStable),
falling back to numerical evaluation via the characteristic function.

```python
is_valid_subordination(parent: Process, subordinator: Process) -> tuple[bool, str]
```
Check validity. Returns `(True, "")` or `(False, reason)`.

---

### product

```python
from spxa.ops import multiply, ProductProcess, cumulants_to_raw_moments, raw_moments_to_cumulants
```

```python
multiply(X: Process, Y: Process, max_moment_order: int = 4) -> ProductProcess
```
Product process Z_t = X_t·Y_t. Always `MOMENT_PROPAGATION`. Emits
`SpxaDegradationWarning`. Moments computed via E[X^n·Y^n] = E[X^n]·E[Y^n]
(independence).

```python
cumulants_to_raw_moments(kappas: dict[int, float]) -> dict[int, float]
raw_moments_to_cumulants(moments: dict[int, float]) -> dict[int, float]
```
Exact conversion between cumulants and raw moments via the cumulant-moment
recurrence (Stuart & Ord 1994). Roundtrip error < 1e-12.

`ProductProcess` exposes `.cumulants(order)` and `.simulate(...)`.

---

### integral

```python
from spxa.ops import ito_isometry_variance, stochastic_convolution_char_func, StochasticIntegral
```

```python
ito_isometry_variance(
    integrator: Process,
    f: Callable[[float], float],
    t: float = 1.0,
    n_quad: int = 1000,
) -> float
```
Exact variance of ∫₀ᵗ f(s) dX_s via the Itô isometry:
Var = κ₂(X₁)·∫₀ᵗ f(s)² ds. Valid for square-integrable Lévy martingale X
and deterministic f. (Applebaum 2009, Theorem 4.2.3.)

```python
stochastic_convolution_char_func(
    integrator: Process,
    f: Callable[[float], float],
    u: float | np.ndarray,
    t: float = 1.0,
    n_quad: int = 500,
) -> np.ndarray
```
Characteristic function of ∫₀ᵗ f(s) dX_s for deterministic f.
Formula: E[exp(iu∫₀ᵗ f(s)dX_s)] = exp(∫₀ᵗ ψ_X(u·f(s)) ds).
Requires `EXACT` integrator.

```python
StochasticIntegral(X: Process, Y: Process, f: Callable[[float], float])
```
Process Z_t = ∫₀ᵗ f(X_s) dY_s for random integrand f(X_s).
Always `MOMENT_PROPAGATION`. Emits `SpxaDegradationWarning`.
Simulation via Euler–Maruyama: Z_{t_{k+1}} ≈ Z_{t_k} + f(X_{t_k})·ΔY_k.

---

## analytics

### cumulants module

```python
from spxa.analytics import cumulant_table, skewness, excess_kurtosis
```

```python
cumulant_table(process: Process, order: int = 4, t: float = 1.0) -> dict[str, float]
```
Returns a dict with keys:
`mean`, `variance`, `std`, `skewness`, `excess_kurtosis`, `kappa_1`, …, `kappa_order`.
All values are for X_t (cumulants scale by t). Requires at least `MOMENT_PROPAGATION`.

```python
skewness(process: Process, t: float = 1.0) -> float
# κ₃(X_t) / κ₂(X_t)^{3/2}

excess_kurtosis(process: Process, t: float = 1.0) -> float
# κ₄(X_t) / κ₂(X_t)²
```

---

### divergence

```python
from spxa.analytics import (
    l2_char_func_distance,
    hellinger_distance,
    wasserstein2_distance,
    kl_divergence_mc,
)
```

All divergences compare marginal distributions at time t. No Monte Carlo
needed for L² and Hellinger — they use characteristic functions directly.

```python
l2_char_func_distance(
    X: Process, Y: Process,
    t: float = 1.0,
    u_max: float = 50.0,
    n_points: int = 2000,
) -> float
```
L² distance between marginals: (1/2π)·∫|φ_X(u;t) − φ_Y(u;t)|² du.
Equal to 0 iff X_t and Y_t have the same distribution (Plancherel theorem).
Symmetric.

```python
hellinger_distance(
    X: Process, Y: Process,
    t: float = 1.0,
    u_max: float = 50.0,
    n_points: int = 2000,
) -> float
```
Squared Hellinger distance H²(X_t, Y_t) ∈ [0, 1].
H²=0 iff distributions identical; H²=1 iff mutually singular.

```python
wasserstein2_distance(
    X: Process, Y: Process,
    t: float = 1.0,
    u_max: float = 100.0,
    n_points: int = 4000,
) -> float
```
Wasserstein-2 distance W₂(X_t, Y_t). Computed via Gil-Pelaez CDF inversion
followed by quantile function comparison. Returns W₂ (not squared).

```python
kl_divergence_mc(
    X: Process, Y: Process,
    t: float = 1.0,
    n_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> float
```
Monte Carlo estimate of KL(X_t ‖ Y_t) via importance sampling.
Approximate — density recovered by Gil-Pelaez inversion. Always ≥ 0.

---

### fourier

```python
from spxa.analytics.fourier import european_call_price, european_put_price, implied_volatility
```

```python
european_call_price(
    process: Process,
    S0: float,
    K: float,
    r: float,
    t: float,
    alpha: float = 1.5,
    n_points: int = 4096,
    eta: float = 0.25,
) -> float
```
European call price assuming S_t = S₀·exp(rt + X_t) where X is the
log-return process. Uses the Carr-Madan (1999) FFT formula. Dispatches to
`char_func_exact` when available (fast path for BM, VG, NIG, CGMY, etc.).

| Parameter | Meaning |
|---|---|
| `process` | Log-return Lévy process |
| `S0` | Current asset price |
| `K` | Strike |
| `r` | Risk-free rate |
| `t` | Time to expiry |
| `alpha` | Dampening factor α > 0 (typically 1–2) |
| `n_points` | FFT grid size (power of 2) |
| `eta` | Frequency spacing |

```python
european_put_price(process, S0, K, r, t, **kwargs) -> float
```
Put price via put-call parity: C − S₀ + K·e^{−rt}.

```python
implied_volatility(
    price: float,
    S0: float,
    K: float,
    r: float,
    t: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float
```
Black-Scholes implied volatility from an option price via bisection.
Returns `float("nan")` if no solution found in `[1e-6, 10]`.

---

## sim — simulation backends

### euler

```python
from spxa.sim import euler_maruyama, euler_maruyama_autonomous, geometric_levy, convergence_order_estimate
```

```python
euler_maruyama(
    driver: Process,
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    z0: float = 0.0,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Euler–Maruyama for dZ_t = a(Z_t, t) dt + b(Z_t, t) dX_t.
`drift(z, t)` and `diffusion(z, t)` are Python callables.
Returns shape `(n_paths, n_steps+1)`.

```python
euler_maruyama_autonomous(
    driver: Process,
    drift: float,
    diffusion: float,
    z0: float = 0.0,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Vectorised Euler for constant-coefficient SDEs: dZ = a dt + b dX.
Exact for constant coefficients (no discretisation error).
Returns shape `(n_paths, n_steps+1)`.

```python
geometric_levy(
    driver: Process,
    mu: float,
    sigma: float,
    s0: float = 1.0,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Simulate S_t = s₀·exp(drift_correction·t + σ·X_t) where the drift correction
is chosen so E[S_t] = s₀·e^{μt} (risk-neutral). Drift correction computed
from the MGF via the characteristic function. Exact — no discretisation.
Returns shape `(n_paths, n_steps+1)`.

```python
convergence_order_estimate(
    driver: Process,
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    z0: float = 0.0,
    T: float = 1.0,
    n_paths_ref: int = 5000,
    step_sizes: tuple[int, ...] = (50, 100, 200, 400),
    rng_seed: int = 0,
) -> dict[str, object]
```
Empirical strong-convergence order via log-log regression against a
fine-grid reference solution. Returns dict with keys:
`errors`, `dt_values`, `estimated_order`, `log_log_slope`, `step_sizes`.

---

### exact samplers

All return increments of shape `(n_paths, n_steps)`.
Use `increments_to_paths(increments, x0=0.0)` to convert to paths.

```python
from spxa.sim import (
    sample_brownian, sample_gamma, sample_inverse_gaussian,
    sample_vg, sample_nig, sample_alpha_stable,
    sample_compound_poisson, sample_tempered_stable,
    increments_to_paths,
)
```

| Function | Distribution | Algorithm |
|---|---|---|
| `sample_brownian(mu, sigma, dt, n_steps, n_paths, rng)` | N(μΔt, σ²Δt) | Direct Gaussian |
| `sample_gamma(a, b, dt, n_steps, n_paths, rng)` | Gamma(aΔt, 1/b) | numpy Gamma |
| `sample_inverse_gaussian(mu_ig, lambda_ig, dt, n_steps, n_paths, rng)` | IG(μΔt, λΔt²) | numpy Wald |
| `sample_vg(sigma, nu, theta, dt, n_steps, n_paths, rng)` | VG(σ,ν,θ,Δt) | Gamma subordination |
| `sample_nig(alpha, beta, delta, mu, dt, n_steps, n_paths, rng)` | NIG(α,β,δ,μ,Δt) | IG subordination |
| `sample_alpha_stable(alpha, beta, sigma, dt, n_steps, n_paths, rng)` | S_α(σΔt^{1/α},β,0) | Chambers–Mallows–Stuck |
| `sample_compound_poisson(rate, jump_sampler, dt, n_steps, n_paths, rng)` | CP(rate,jump_dist) | Thinning |
| `sample_tempered_stable(alpha, C, lam, dt, n_steps, n_paths, rng)` | TS(α,CΔt,λ) | Rejection from stable (Rosiński 2007) |

```python
increments_to_paths(increments: np.ndarray, x0: float = 0.0) -> np.ndarray
# Prepend x0 and cumsum. (n_paths, n_steps) → (n_paths, n_steps+1).
```

---

### bridge

```python
from spxa.sim import (
    brownian_bridge, gamma_bridge,
    brownian_bridge_interpolate,
    first_passage_time_bm, first_passage_time_exact_bm,
)
```

```python
brownian_bridge(
    t_start: float = 0.0,
    t_end: float = 1.0,
    x_start: float = 0.0,
    x_end: float = 0.0,
    sigma: float = 1.0,
    n_points: int = 100,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]
```
Exact conditional BM bridge X(t) | X(t_start)=x_start, X(t_end)=x_end.
Returns `(time_grid, paths)` with `time_grid.shape=(n_points+2,)` and
`paths.shape=(n_paths, n_points+2)`. Endpoint constraint is exact.
Conditional mean at t: x_start + (x_end−x_start)·(t−t_start)/(t_end−t_start).

```python
gamma_bridge(
    a: float,
    b: float,
    T: float = 1.0,
    x_end: float | None = None,
    n_points: int = 100,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]
```
Gamma process bridge via the Beta distribution identity:
G(t)|G(T)=g_T ~ g_T·Beta(at, a(T−t)).
Returns `(time_grid, paths)`. If `x_end=None`, samples g_T from Gamma(aT, b).

```python
brownian_bridge_interpolate(
    coarse_paths: np.ndarray,
    coarse_times: np.ndarray,
    fine_times: np.ndarray,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Refine coarse BM paths to a finer time grid by inserting bridge-sampled
interior points. Coarse path values at coarse_times are preserved exactly.
Returns shape `(n_paths, len(fine_times))`.

```python
first_passage_time_bm(
    level: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    T_max: float = 10.0,
    n_steps: int = 10000,
    n_paths: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Estimate first passage times τ = inf{t: X_t ≥ level} by simulation.
Returns shape `(n_paths,)`. Paths not reaching `level` by `T_max`
return `np.inf`.

```python
first_passage_time_exact_bm(
    level: float,
    sigma: float = 1.0,
    n_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Exact samples from the first passage time distribution of BM(0, σ) to
a positive level. Uses the Lévy distribution (one-sided stable, α=1/2):
P(τ ≤ t) = 2Φ(−level/(σ√t)). Returns shape `(n_samples,)`.
Requires `level > 0`.

---

## story — derivation traces

```python
from spxa.story import CompositionNode, NodeKind, narrate, narrate_latex, story_latex
```

### CompositionNode

```python
CompositionNode(
    kind: NodeKind,
    name: str,
    children: list[CompositionNode] = [],
    math_note: str = "",
    reference: str = "",
)
```

Node in the composition tree. Every `Process` carries one at `process._node`.

### NodeKind

```python
NodeKind.PRIMITIVE      # leaf: a named process
NodeKind.ADDITION       # from X + Y
NodeKind.SCALING        # from c * X
NodeKind.SUBORDINATION  # from X @ T
NodeKind.MOMENT_APPROX  # from non-Lévy operations
```

### narrate

```python
narrate(node: CompositionNode, depth: int = 0) -> str
```
Recursively produce a plain-text derivation. Called internally by
`process.__story__()`.

### narrate_latex

```python
narrate_latex(node: CompositionNode, depth: int = 0) -> str
```
Produce a LaTeX derivation fragment from a node. Output is suitable for
embedding inside a `\begin{aligned}` environment. Called internally by
`story_latex` and `process.__story_latex__()`.

### story_latex

```python
story_latex(process: Process) -> str
```
Full standalone LaTeX derivation for a process. Returns a
`\begin{aligned}...\end{aligned}` block containing:
the composition tree (via `narrate_latex`), exactness level, and all
`ProcessProperties` fields. Render in Jupyter:

```python
from IPython.display import display, Math
display(Math(story_latex(Z)))
```
