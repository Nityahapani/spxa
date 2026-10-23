# RFC 0001 — Restore mathematical capability contracts

**Status:** local proposal and implementation draft; **not approved upstream**.
The repository contribution guide requires an RFC and two maintainer approvals for core
implementation changes. This local working branch does not imply that either approval exists.

## Motivation

The original review of main at `78cf452` demonstrated conflicts between primitive characteristic
functions, their triplets, numerical moments, and documented exactness. It also found that
composition lost middle-capability moments and that non-Lévy clocks were accepted.

## Proposed behavior

1. Integrate functions against a `LevyMeasure`, including atomic measures. A density is not
   a universal representation. Preserve unit-ball endpoint atoms during drift corrections.
2. Scale drift by the complete truncation correction
   `integral c*x*(1_{|c*x|<=1}-1_{|x|<=1}) nu(dx)`.
3. Expose a continuous `characteristic_exponent(u)` rather than reconstructing one with the
   principal logarithm of a CF. Reject exponential moments outside the distribution's strip.
4. Add `cumulants_at(order,t)` without changing the existing `cumulants(order)` signature.
   Lévy cumulants scale linearly. Beyond-Lévy classes must give their own time law or reject
   the request. `cumulants(order)` describes time one, not stationary/asymptotic quantities.
5. Preserve CF/moment composition through sums and scalar wrappers. Validate both parent and
   clock Lévy increment properties for subordination.
6. Expose canonical BM/Gamma and BM/IG triplets. Generic subordination keeps correct CFs,
   finite cumulants and conditional-increment sampling, but explicitly lacks a generic
   triplet API and therefore uses `MOMENT_PROPAGATION` with a degradation warning.
7. Enforce post-construction parameter immutability, defensively copy arrays, freeze
   composition trees, and return defensive property objects.
8. Retain vector jump directions and apply radial truncation corrections for multivariate
   transforms/projections. Reject an operation if the vector measure cannot represent it.

## Compatibility and migration

- Existing operator and method call signatures are retained. New exponent/time methods and
  measure integration are additive. Mathematical parameter names remain unchanged.
- Corrected numerical results are **intentional bug fixes**, not compatibility shims for
  erroneous formulas. Existing exactness tests which encoded the wrong IG cumulants, OU
  normalization or tempered-tail moment condition are replaced by independent references.
- Assignment to process parameters now raises; construct a new object instead.
- Gaussian `sigma=0` and Hawkes `alpha=0` are supported degenerate limits.
- Generic subordination and unsupported stochastic-integral analytics now report conservative
  capabilities rather than promising an unavailable/incorrect analytical operation.
- Arbitrary process products remain independent copies even for the same Python object.
  In stochastic integrals only, identical X/Y objects deliberately share their simulated
  path, as required to represent an integral with respect to itself.
- Explicitly stationary OU properties remain available; its ordinary cumulants now describe
  the transient law from `v0` at time one. Hawkes asymptotic properties are likewise separate
  from finite-horizon count cumulants.

## Validation

The regression suite checks published law identities, SciPy reference distributions,
independent high-precision differentiation, conditional covariance, tail moment existence,
seeded empirical CFs, and known Black–Scholes prices. Numerical approximation methods remain
identified as approximate; no global all-parameter accuracy theorem is claimed.

## Out of scope

A universal explicit Bochner Lévy-measure representation, arbitrary random-integrand moment
closure, general multivariate Lebesgue densities, and formally error-certified CGMY/Meixner
inverse-CDF samplers are not implemented by this proposal. Unsupported requests fail explicitly.


## Refresh onto ef80f28

The local proposal now reconciles the six upstream additions rather than overwriting them.
Compound Poisson atoms, stable small-jump exponents, immutable jump laws, nonzero initial
states and portable simulation calls extend the same contracts. CIR/Gaussian OU ordinary
cumulants describe their configured time-one law; stationary APIs are retained. The two-sided
OU spectrum is normalized to stationary variance. Heston uses differentiated Riccati ODEs
and correct pricing drift overrides. CEV has exact absorbing/reciprocal Bessel transitions,
immutable Greek bumps, actual transient moments and explicit strict-local-martingale limits.
Merton option-series truncation is checked rather than silently hidden. See
[the derivation](../refresh_mathematics.md). These are still unapproved core changes.
