# Process Registry

This catalogue describes the implemented code, not planned classes.
`EXACT` means a Lévy analytical representation is supported, **not exact floating-point
arithmetic or a guarantee that every sampler is exact in distribution**.
`MOMENT_PROPAGATION` denotes explicit marginal moment formulas; unsupported formulas raise.
`SIMULATION_ONLY` denotes simulation without a general analytical moment implementation.

## Lévy processes (`spxa.zoo.levy`)

| Class | Cumulants | Simulation | Primary reference |
|---|---|---|---|
| `BrownianMotion` | All orders; variance `sigma²*t` | Gaussian increments; `sigma=0` is deterministic drift | Wiener (1923) |
| `PoissonProcess` | `rate*jump_size**n*t` | Poisson increment counts | Kingman (1993) |
| `CompoundPoisson` | `rate*t*E[Y**n]` where finite | Vectorized compounding; discrete/continuous jump laws | Kingman (1993); Cont & Tankov (2004) |
| `MertonJumpDiffusion` (`MertonProcess`, `Merton76`) | Gaussian plus compound-Poisson cumulants | Gaussian/Poisson-normal increments | Merton (1976) |
| `GammaProcess` | `a*t*Gamma(n)/b**n` | Gamma increments, shape/rate convention | Moran (1968) |
| `VarianceGamma` | Finite orders via Gamma subordination | Gaussian/Gamma mixture | Madan, Carr & Chang (1998) |
| `NIG` | Finite orders via its triplet | Gaussian/IG mixture | Barndorff-Nielsen (1997) |
| `CGMY` | All orders with positive G/M; untempered tails obey the tail-moment criterion | **Approximate** finite-grid CF inversion | Carr, Geman, Madan & Yor (2002) |
| `AlphaStable` | Integer cumulants only below alpha, except Gaussian alpha=2 | Chambers–Mallows–Stuck, S1 parameterization with alpha=1 location correction | Chambers, Mallows & Stuck (1976); Nolan (2020) |
| `InverseGaussianProcess` | `t*(2n-3)!!*mu**(2n-1)/lam**(n-1)` | Wald increments | Tweedie (1957) |
| `MeixnerProcess` | Derivative-polynomial recurrence for all requested finite orders | **Approximate** truncated Fourier density/CDF interpolation | Schoutens & Teugels (1998) |
| `TemperedStable` | `t*C*Gamma(n-alpha)/lam**(n-alpha)` | Exponential tilting of positive stable increments; bounded proposal/time-subdivision budgets, no approximate fallback | Rosiński (2007) |
| `NegativeBinomialProcess` | Closed form to order four; convergent series above | NumPy negative-binomial increments | Quenouille (1949) |

All thirteen classes above have `ExactnessLevel.EXACT`. `CompoundPoisson` is now an
implemented/exported process; `sample_compound_poisson` remains a separate increment utility.

CGMY follows the conventional tail naming: **M tempers positive jumps, G negative jumps**.
Its Y=0 and Y=1 characteristic-exponent limits are implemented explicitly.

## Beyond Lévy (`spxa.zoo.beyond`)

| Class | Capability | Marginal moments / simulation | Primary reference |
|---|---|---|---|
| `FractionalBrownianMotion` | MOMENT_PROPAGATION | Gaussian cumulants; variance `sigma²*t**(2H)`; Davies–Harte grid sampling | Mandelbrot & Van Ness (1968) |
| `HawkesProcess` | MOMENT_PROPAGATION | Empty-history finite-horizon moment ODEs, orders 1–8; thinning simulation. Stationary intensity and asymptotic Fano factor are separate properties | Hawkes (1971); Hawkes & Oakes (1974) |
| `OULevy` | MOMENT_PROPAGATION | Transient cumulants and stationary moments of `dV=-lam*V dt+dL_(lam*t)`; **approximate**, mean-preserving kernel quadrature | Barndorff-Nielsen & Shephard (2001) |
| `MarkovModulatedPoissonProcess` | MOMENT_PROPAGATION | Finite-horizon matrix-exponential moments/cumulants with conditioning guards; exact event-driven CTMC/Poisson sampling | Fischer & Meier-Hellstern (1993) |

| `CIRProcess` (`CoxIngersollRoss`) | MOMENT_PROPAGATION | Noncentral chi-square conditional law; stationary Gamma law; Riccati bonds | Cox, Ingersoll & Ross (1985) |
| `OrnsteinUhlenbeckProcess` (`OrnsteinUhlenbeck`, `Vasicek`) | MOMENT_PROPAGATION | Conditional Gaussian law; exact Gaussian path transitions; Vasicek bonds | Uhlenbeck & Ornstein (1930); Vasicek (1977) |
| `HestonProcess` (`Heston`) | MOMENT_PROPAGATION | Log-return affine CF; numerical Riccati cumulants through order 12; **approximate** path simulation | Heston (1993); Albrecher et al. (2007) |
| `CEVProcess` (`CEV`, `ConstantElasticityOfVariance`) | MOMENT_PROPAGATION | Exact absorbing/reciprocal-Bessel or GBM transitions; finite-difference Greeks; infinite variance for 1<gamma<=3/2; gamma>1 fully collateralized calls | Cox (1975); Schroder (1989) |

Nondecreasing counts do not make Hawkes or MMPP a Lévy subordinator. OU values are not
nondecreasing. These classes are not eligible clocks for `@`.

## Composition and multivariate support

- Independent sums and scalar images preserve Lévy laws when the inputs are Lévy.
- BM/Gamma and BM/IG subordination expose VG/NIG triplets and samplers. Other valid Lévy
  pairs expose exact exponent composition and finite cumulant composition, but retain
  `MOMENT_PROPAGATION` with a warning until an explicit triplet representation is supported.
- Independent products propagate time-specific moments. Arbitrary stochastic integrals are
  simulation-only; Gaussian-integrand/Lévy-martingale cases support numerical Itô-isometry
  moments under the stated integrability assumptions.
- `MultivariateBrownianMotion` supports Gaussian vector laws. `CorrelatedLevy` retains the
  scalar jump measures along their actual vector directions; projections/transforms include
  truncation drift corrections. General multivariate jump densities are not supplied.

New process proposals: use the **Process Proposal** issue template and follow `CONTRIBUTING.md`.
