# Numerical contracts and limits

## What EXACT means

`ExactnessLevel.EXACT` describes analytical Lévy capability. It does not certify floating-point
quadrature, a truncated Fourier grid, or every sampler. No finite test suite proves every
parameter combination. Process parameters, composition trees and stored parameter arrays are
immutable; construct a new object to change a parameter.

`cumulants(order)` means the time-one marginal. Use `cumulants_at(order,t)` for other times.
A Lévy law has `kappa_n(t)=t*kappa_n(1)`; fBM, products, OU, CIR, Heston, CEV, Hawkes and MMPP do **not** in general.
Stationary OU moments, stationary Hawkes intensity and asymptotic Hawkes Fano factor are separate
properties. Nonexistent stable/tail moments raise instead of returning numerical artifacts.

## Lévy measures and exponents

The convention is `h(x)=1_{|x|<=1}` in the compensated `x*h(x)` term. Integrate against a
`LevyMeasure`, not its density, when atoms may occur. The density of a pure atomic measure is
zero almost everywhere, not its probability-mass function. Vector measures live along their
actual jump directions. Projection drift depends on the **original Euclidean radius**.

Use a continuous characteristic exponent, not `log(char_func(u))`: the principal logarithm
loses winding information. Complex frequencies are meaningful only within the exponential
moment strip. Analytic continuation is not an expectation outside that strip.

## Samplers

- Gaussian, Poisson, Gamma, VG, NIG, IG and negative-binomial increments use their named laws.
- Stable sampling uses the S1 convention, including the alpha=1 logarithmic location correction.
- Tempered stable sampling uses positive-stable exponential tilting. It does not truncate small
  jumps. A block is split until its acceptance probability is reasonable. More than 10,000 time
  subdivisions raises `ValueError`; exhausting `n_rejection_attempts*n_paths*n_steps` proposals **per block** raises
  `RuntimeError` (default `n_rejection_attempts=50`). Extreme
  parameters can therefore be unsupported instead of silently approximated. Floating-point
  underflow/overflow remains possible close to limiting parameters, as with other numeric laws.
- **CGMY and Meixner simulation remain approximate**: finite frequency/spatial support and CDF
  interpolation affect tails and accuracy. They are not certified arbitrary-tolerance samplers.
- **OU–Lévy simulation is approximate**: it applies a mean-preserving kernel weight to clock
  increments. The analytical transient moments and stationary properties are separate formulas.
- Brownian bridges use the full conditional Gaussian covariance. Gamma bridges use joint
  Dirichlet increments, not independent Beta marginals at each interior time.

## Composition/integrals

Canonical Brownian/Gamma and Brownian/IG clocks expose VG/NIG triplets. A deterministic
nonnegative Brownian clock rescales Brownian time exactly; a zero-volatility parent
reduces to its drift times the clock. Brownian self-similarity excludes mixed drift/noise. Other valid independent
Lévy clock pairs have CF/exponent and Bell-polynomial cumulant composition but **no general
triplet API**; a warning and `MOMENT_PROPAGATION` make this limitation explicit.

Arbitrary `StochasticIntegral(X,Y,f)` is simulation-only. Brownian X plus a square-integrable
Lévy martingale Y supports mean zero and a numerically integrated Itô-isometry variance when
the Gaussian expectation of `f(X_s)**2` converges. It uses the actual callable f; a divergent or
unresolved expectation is rejected. Higher cumulants are not generally known. A shared X/Y
object shares its simulated path inside an integral, rather than inventing independence.

## Distribution comparison

Hellinger is the density/PMF overlap `1-integral sqrt(p*q)`, not an overlap of square roots of
characteristic functions. Gaussian marginals use closed-form expressions. Known primitive
marginals use SciPy densities, PMFs and quantiles; W2 integrates the squared quantile difference.

Unknown smooth continuous marginals use an **explicitly approximate** Fourier fallback with a
warning. Density normalization, nonfinite output or excessive negative mass causes failure.
Unknown atom/continuous mixtures and general unknown discrete laws are unsupported. KL is
asymmetric and is infinite when the target assigns zero mass/density to sampled source support.
`kl_divergence_mc` remains a seeded Monte Carlo estimate, not a deterministic exact divergence.
`l2_char_func_distance` is a finite-band characteristic-function metric; Plancherel gives the
full density L2 norm only when densities are square-integrable and the frequency range is infinite.

## Option pricing and geometric assets

Fourier call/put pricing treats X as raw log-return noise and normalizes
`S_t=S0*exp(X_t+r*t-log(E[exp(X_t)]))`.
Already-risk-neutral input is unchanged by this convention. A finite exponential first moment
is required; damped FFT additionally requires the moment at `alpha+1`.
No Gaussian drift approximation is substituted when the true exponential moment is absent.
Finite FFT grid, damping and truncation cause numerical error; refine/check a grid for the
specific law. Implied volatility validates no-arbitrage bounds before root solving.

The strong-convergence utility uses common nested driver increments for coarse/fine paths.
Independent simulations are not a valid strong-error estimate. Convergence slopes remain
statistical diagnostics, not mathematical proofs of an SDE scheme's order.


## Current-upstream process additions

- Compound Poisson jump laws are immutable, including discrete arrays. Zero-size events are
  not Lévy atoms. Primitive, triplet and image-measure exponents use stable small-jump formulas.
- Gaussian OU and CIR have exact conditional transition laws and separate stationary APIs.
  Gaussian OU is **not** the approximate OU–Lévy sampler described above.
- Heston log-return cumulants use numerical Riccati ODEs through order twelve; requests above
  the budget raise. Pricing overrides the drift with r-q. Euler path schemes remain approximate.
- CEV's default uses exact absorbing/reciprocal Bessel or lognormal transitions, not a hidden
  Milstein fallback. Explicit Euler/Milstein alternatives are approximate. Greeks are numerical.
  For gamma>1 the discounted state is a strict local martingale; for 1<gamma<=3/2 its variance
  is infinite. Higher-than-second CEV cumulants are unsupported. The gamma>1 call convention
  is fully collateralized, not the minimal discounted expected call payoff.
- Merton's Poisson option series checks both ordinary and stock-measure truncation tails;
  increase max_jumps if its explicit budget is insufficient.
- Pass n_paths, T and rng by keyword for portable simulation. CEV's original dt/time_horizon
  keywords and positional layout are preserved; T is an added keyword-only alias.

The [reconciliation and derivation](refresh_mathematics.md) states these formulas and limits.
