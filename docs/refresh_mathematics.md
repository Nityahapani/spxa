# Current-upstream mathematical reconciliation

This is part of the **unapproved local proposal** based on upstream
`ef80f282dcf8f342814442dbc769b1b48a6de851`. The six newly added process families,
exports, aliases, examples and tests are retained. The earlier proposal was based
on `78cf452`; its test totals are not evidence for this revision.

## Common marginal and composition contract

`cumulants(order)` describes the configured process at time **one**.
`cumulants_at(order,t)` describes its actual marginal at `t`; stationary quantities
have separate `stationary_*` methods/properties. A CIR or Gaussian OU process
started from a fixed state is not initialized in its stationary distribution.
The first simulated column is the configured initial state, not universally zero.
Heston's default output is log return, so its initial value is zero, not `s0`.

Internal simulation calls use `n_paths=`, `T=` and `rng=` keywords. This preserves
CEV's existing positional `dt`/`time_horizon` interface while adding keyword-only
`T` for composition. Contradictory horizons or invalid dimensions raise.

## Compound Poisson and image measures

For arrival rate `lambda` and jump law `F`, the Lévy measure is
`nu(dx)=lambda F(dx)` **on x != 0**. Zero-size events do not contribute to its mass:
`nu(R\{0})=lambda P(Y != 0)`. For finite moments,

\[
\kappa_n(X_t)=\lambda t\,E[Y^n],\qquad
\psi(u)=\lambda(\phi_Y(u)-1).
\]

Small jump laws use stable `expm1`, rational, or sinc-series representations of
`phi_Y-1`. Computing a rounded CF and then subtracting one can erase an entire
high-rate law. Built-in jump distributions and their stored arrays are immutable.
Discrete measures integrate their atoms, not a putative Lebesgue density.
Continuous built-ins use CDF/PPF and SF/ISF probability-space quadrature so a narrow
shifted density is not missed. Generic custom laws retain explicit support limits.

For `c != 0`, set

\[
a_c=\int cx\bigl(1_{|cx|\le1}-1_{|x|\le1}\bigr)\nu(dx).
\]

The transformed jump integral satisfies
`I_scaled(u)=I_original(c*u)-i*u*a_c`; using the original integral preserves atoms
and narrow continuous peaks. Integration splits at the physical truncation
boundaries and removes duplicated endpoint atoms. The first cumulant integrates
large jumps on the two explicit tail domains, excluding boundary atoms at ±1.

References: Sato (1999), Chapters 8, 11 and 25; Cont & Tankov (2004), Section 3.4.

## CIR and Gaussian OU

CIR conditional cumulants come from a scaled noncentral chi-square distribution.
Writing `e=exp(-kappa*t)`, `q=sigma²*(1-e)/(2*kappa)` and
`a=2*kappa*theta/sigma²`, a stable form is

\[
\kappa_n(r_t)=(n-1)!\{a q^n+n r_0 e q^{n-1}\}.
\]

`expm1` retains the variance at very small times; density evaluation uses the
noncentral chi-square implementation instead of potentially overflowing unscaled
Gamma/Bessel prefactors. The complex CF rejects moments outside its finite strip.

For Gaussian OU, the two-sided spectral convention is

\[
f(\omega)=\frac{\sigma^2}{2\pi(\theta^2+\omega^2)},\quad
C(h)=\int_{\mathbb R}e^{i\omega h}f(\omega)\,d\omega.
\]

Consequently `integral f = sigma²/(2*theta)`. This is not the one-sided density,
which is twice as large on nonnegative frequencies. Both new families have direct
SciPy reference-marginal adapters, including their deterministic initial atoms.

References: Cox, Ingersoll & Ross (1985); Uhlenbeck & Ornstein (1930); Vasicek (1977).

## Heston

The configured SDE uses asset drift `mu`; option-pricing `r,q` replace that drift
with `r-q`, rather than merely changing discount factors. A divided-exponential
Riccati expression avoids the removable `g`-ratio singularity at `u=-i`, including
`kappa-rho*sigma_v <= 0`. Stable complex logarithms preserve the small-vol-of-vol
Black–Scholes limit. Exponential moments beyond the Riccati blow-up time raise;
analytic continuation outside the moment domain is not an expectation.

For log-MGF `K=C+D*v0+mu*z*t`, the affine Riccati equation is

\[
D'=\tfrac12\sigma_v^2D^2+(\rho\sigma_v z-\kappa)D+
\tfrac12(z^2-z),\qquad C'=\kappa\theta D.
\]

Differentiating this polynomial equation gives finite-time cumulant ODEs without
finite differences of a floating-point CF. Numerical DOP853 integration supports
orders 1–12; higher requests raise, not NaN-filled dictionaries. Tests compare
orders 1–6 against independently differentiated 70-digit affine formulas.
Heston remains `MOMENT_PROPAGATION`; Fourier pricing, ODE integration and Euler
path schemes are numerical, not globally error-certified. Full truncation keeps
its untruncated auxiliary variance state but reports its nonnegative positive part.

References: Heston (1993); Albrecher et al. (2007); Lord et al. (2010).

## Exact CEV transition: derivation of the absorbing mixture

For `dS=mu*S*dt+sigma*S^gamma*dW`, let `a=1-gamma != 0`,
`Y=exp(-mu*t)*S` and

\[
\tau(t)=\int_0^t e^{-2\mu a s}\,ds,\qquad
R=\frac{Y^{2a}}{\sigma^2a^2}.
\]

In the changed time, `R` is squared Bessel with dimension `delta=2-1/a`.
For `a>0`, use its **absorbing/killed** kernel, not a reflecting extension.
Set `nu=1/(2*a)`, `c=sigma²*a²*tau`, `q=S_initial^(2*a)/(2*c)`,
and `Z=R_next/(2*tau)`. Expanding the modified Bessel function `I_nu` gives

\[
p_Z(z)=\sum_{k=0}^{\infty}
  \underbrace{\frac{e^{-q}q^{k+\nu}}{\Gamma(k+\nu+1)}}_{w_k}
  \underbrace{\frac{e^{-z}z^k}{\Gamma(k+1)}}_{\text{Gamma}(k+1,1)\ \text{density}}.
\]

The missing mass is the absorbing atom. The following finite random construction
has precisely these subprobability weights:

1. Draw `H ~ Gamma(nu,1)`. If `H >= q`, return zero.
2. Otherwise draw `K ~ Poisson(q-H)` and `Z ~ Gamma(K+1,1)`.
3. Return `exp(mu*dt)*(2*c*Z)^(1/(2*a))`.

Indeed,

\[
P(K=k,H<q)=\int_0^q
\frac{e^{-h}h^{\nu-1}}{\Gamma(\nu)}
\frac{e^{-(q-h)}(q-h)^k}{\Gamma(k+1)}\,dh
=\frac{e^{-q}q^{k+\nu}}{\Gamma(k+\nu+1)}.
\]

The final equality is the beta integral. This samples both the continuous law and
its absorbing atom, with no Milstein fallback. For `a<0`, `delta>2` and the reciprocal
state uses an ordinary noncentral chi-square transition. For `gamma=1`, the exact
law is lognormal. Explicit Euler/Milstein methods remain approximate. Floating-point
and NumPy random-generator parameter limits still apply; “exact transition” means
no time-discretization approximation, not exact real-number computation.

The regression suite checks nine elasticities and both drift signs using four
coarse steps against noncentral-chi-square/Bessel CDFs, as well as the Gaussian,
branching and reciprocal-BES(3) reference identities.

References: Revuz & Yor (1999), *Continuous Martingales and Brownian Motion*,
3rd ed., Chapter XI; Schroder (1989), *Journal of Finance* 44(1), 211–219.

### CEV moments and collateralization for gamma > 1

The discounted process is then a **strict local martingale**. Its mean is less than
`s0*exp(mu*t)`; the incomplete-Gamma factor is included. For reciprocal BES(3)
(`gamma=2`, `mu=0`, `sigma=s0=1`), `E[S_t]=erf(1/sqrt(2*t))`.
The density has tail proportional to `s^(-2*gamma)`, so the second moment is infinite
for `1<gamma<=3/2`. Properties and cumulant requests respect that fact. General finite
variance uses centered positive-tail quadrature over the full support, including
absorption, rather than a fixed upper cutoff. An explicit `s0` override selects that
initial law throughout the calculation. CEV cumulants above order two are not supplied.

The existing `gamma>1` **fully collateralized call** convention is preserved and made
explicit: its price equals discounted expected call payoff **plus the martingale
defect**. Puts are discounted expected put payoffs. This convention, not the minimal
uncollateralized call, satisfies the displayed ordinary put–call parity. Greeks are
finite differences on new parameter-bump objects; they never mutate the source law.

## Merton series

Use direct ordinary and stock-numeraire Poisson PMFs, not a recursion starting from
an underflowed `exp(-lambda*T)`. Both omitted probability tails must be below `tol`;
an inadequate `max_jumps` raises with instructions to increase the budget. It does
not silently return intrinsic value. Fourier inversion independently checks a
1,000-arrivals-per-unit-time case. Merton's path transitions remain exact Gaussian
plus Poisson-normal increments; its vanilla option series remains numerical.


## Executable first-passage density illustration

The physics tutorial now uses fixed displayed bins and density weights `1/(N * bin_width)`.
For grid-monitored paths, N is the original path count, including paths with no observed hit.
Thus the displayed histogram integrates to the observed unconditional hit fraction in the
window, not one. Filtering by the displayed window and using `density=True` would instead
plot a conditional density against an unconditional Lévy PDF. Discrete monitoring remains
approximate because it misses between-grid crossings. A deterministic censored-data
regression checks the actual histogram expressions used by the notebook.
