# Exact derivatives through the process algebra

Prices, cumulants, likelihoods and calibration residuals all depend on a process's parameters. spxa computes their derivatives exactly, in the sense of automatic differentiation: the derivative of the number the code computes, accurate to rounding, with no difference quotients. Every derivative is labelled with how it was obtained:

| Label | Meaning |
| :--- | :--- |
| `EXACT` | forward-mode jets or authored adjoints through closed forms |
| `NUMERICAL` | a finite difference somewhere on the route (a fallback that is reported) |
| `STOCHASTIC` | an unbiased Monte Carlo estimate, returned with standard errors |

```python
import numpy as np
import spxa
from spxa.zoo.levy import VarianceGamma

vg = spxa.DiffProcess(VarianceGamma(0.2, 0.3, -0.1))

def exkurt(th):
    k = vg.cumulants(4, th)
    return k[3] / k[1] ** 2

spxa.grad(exkurt)(vg.values)     # d(excess kurtosis) / d(sigma, nu, theta), exact
```

## 1. Jets: forward mode for NumPy code

A jet carries a value $v$, its gradient $\nabla v \in \mathbb C^n$ and, optionally, its Hessian. It implements NumPy's `__array_ufunc__` and `__array_function__` protocols and has rules for the SciPy special functions used by the zoo: $\Gamma$, $\log\Gamma$, $\psi$, $\psi^{(k)}$, the error functions, $\log\Phi$ and the Bessel functions $K_\nu$ and $K_\nu e^{x}$. The rules for $K_\nu$ cover its *order* derivative, which the generalised hyperbolic and variance gamma densities need. Code written with these operations differentiates without change.

Jets refuse to lose information. `float()` of a jet, an operation without a rule, or an object array raises `JetError`. A derivative is exact or the call fails; it is never silently a difference quotient.

`spxa.grad`, `spxa.value_and_grad`, `spxa.jacobian` and `spxa.hessian` wrap any such function, in the style of JAX. A complex output is differentiated as a holomorphic function of the real argument.

## 2. The algebra is closed under differentiation

`DiffProcess(X)` parses a zoo expression built with $+$, scaling and $@$ into a tree of differentiable closed forms. Every primitive Lévy family has its exponent $\psi(u;\theta)$ written for jets. The composition rules are the same as for values:

$$
\psi_{X+Y} = \psi_X + \psi_Y, \qquad \psi_{cX}(u) = \psi_X(cu), \qquad \psi_{X\circ T}(u) = \psi_T\big(-i\psi_X(u)\big),
$$

so the chain rule gives, for example,

$$
\partial_{\theta_X}\psi_{X\circ T}(u) = -i\,\psi_T'(w)\,\partial_{\theta_X}\psi_X(u), \qquad w = -i\psi_X(u).
$$

**Cumulants.** $\kappa_n = n!\,[z^n]\,K(z)$ with $K(z) = \psi(-iz)$. spxa evaluates $K$ on a truncated Taylor series in $z$ whose coefficients are jets in $\theta$. The cumulants and their full Jacobian then come from one pass, with no contour integral and no differences.

**Hessians.** Second-order jets give $\partial^2_\theta\psi$ and Hessians of anything built from it.

The `GRADIENTS` laws (§6) check every rule against Richardson differences of the zoo's own closed forms, which are a separate code path.

## 3. Fourier prices

`spxa.graph.autodiff.fourier` prices European options with the risk-neutral convention

$$
S_T = S_0\exp\big((r-q)T + \omega T + X_T\big), \qquad \omega = -\psi(-i),
$$

by COS (Fang & Oosterlee 2008), Carr–Madan (1999) or Lewis (2001). `price_and_gradient` returns the prices and their exact Jacobian, and Hessians on request. Drift in $X$ cancels in $X_T + \omega T$, so its derivative is exactly zero.

The derivative is of the discretised price. Carr–Madan and Lewis use composite Gauss–Legendre nodes that do not depend on $\theta$. The COS interval

$$
[a, b] = c_1 \pm L\sqrt{c_2 + \sqrt{c_4}}
$$

does depend on $\theta$, through the cumulants. With `track_interval=True` (the default) that dependence is differentiated as well, so the gradient agrees with differences of `cos_price` to rounding. A frozen interval (`track_interval=False`) misses the truncation sensitivity. For variance gamma fits this was up to $4\times10^{-4}$ in the option-fit Jacobian, enough to stall a least-squares fit. Brownian motion has $c_4 = 0$, where $\sqrt{\cdot}$ is not differentiable; there the term is treated as the constant 0, which is its exact value.

## 4. Calibration

| Calibrator | Exact derivative | Option |
| :--- | :--- | :--- |
| `discovery.fit_structure` | binned likelihood or cumulant distance (L-BFGS-B gradient) | `gradient="exact"` |
| `discovery.fit_options` | vega-weighted COS residuals (least-squares Jacobian) | `jac="exact"` |
| `rates.calibrate_sabr` | Hagan and Obłój implied vols | `jac="exact"` |
| `LeastSquaresOptimizer.minimize` | any residual Jacobian you supply | `jac=` |

Each defaults to the exact derivative; `"2-point"` restores SciPy's differences.

**Implicit parameters.** When `calibrate_sabr` pins $\alpha$ to the at-the-money vol, $\alpha(\rho,\nu)$ solves $g(\alpha,\rho,\nu) = \sigma_{ATM}(\alpha,\rho,\nu) - \sigma = 0$. The implicit function theorem gives $\partial\alpha = -\partial g/g_\alpha$, which is exact to first order. A consequence is checked by the tests: the pinned at-the-money vol has zero gradient.

**Objectives with kinks.** The binned likelihood clips probabilities at a floor and takes a running maximum. Jets differentiate the branch taken, which is the derivative almost everywhere. On the penalty plateau the gradient is zero. If a gradient is not finite, `fit_structure` falls back to a difference for that step, and `fit_options` zeroes the Jacobian rows whose derivatives are not finite or whose residual sits at the penalty.

## 5. Monte Carlo gradients

`spxa.graph.autodiff.montecarlo` returns $V(\theta) = e^{-rT}E[f(S_T)]$, its gradient and per-coordinate standard errors. The result is labelled `STOCHASTIC`.

**Pathwise.** $X_T = g(\theta,\xi)$ is differentiated per path. Gamma and inverse Gaussian variates use *implicit reparameterisation* (Figurnov, Mohamed & Mnih 2018). For $G = F^{-1}(U;\theta)$,

$$
\frac{\partial G}{\partial\theta} = -\frac{\partial_\theta F(G;\theta)}{f(G;\theta)}.
$$

For a unit Gamma($a$) variate, the lower series $\partial_a P(a,x) = \sum_n t_n\big(\log x - \psi(a+n+1)\big)$, with $t_n = e^{-x}x^{a+n}/\Gamma(a+n+1)$, is used for $x \le a + 1 + 4\sqrt a$. In the upper tail, $P \to 1$ and that sum cancels catastrophically: it lost up to 4 digits at $x \approx a + 4\sqrt a$ for $a = 1000$. There spxa uses instead

$$
\frac{\partial G}{\partial a} = \int_0^\infty \big(\log(x+s) - \psi(a)\big)\Big(1 + \frac sx\Big)^{a-1}e^{-s}\,ds,
$$

an integral without cancellation, by Gauss–Laguerre quadrature after the substitution $s = u/r$, $r = 1 - (a-1)/x$. The two branches agree with mpmath to $10^{-11}$ for $a \in [0.02, 1000]$ and $U \in [10^{-6}, 1 - 10^{-12}]$.

Poisson counts are not differentiable in their rate, so the rate enters through the score $N/\lambda - T$. The resulting hybrid pathwise/score estimator is still unbiased. Pathwise needs a payoff that is Lipschitz in $S_T$; a digital is refused.

**Likelihood ratio.** With $Y = X_T + \omega T$, whose density is $p_T(y - \omega T;\theta)$,

$$
\partial_\theta V = e^{-rT}E\big[f(S_T)\,\partial_\theta\log p_T(Y - \omega T;\theta)\big].
$$

The score includes the martingale correction and comes from jets through the closed densities of Brownian motion, Gamma, inverse Gaussian, variance gamma and NIG. Any payoff works.

**Malliavin.** For a Brownian component $\sigma W$ plus independent jumps, the delta weight is $W_T/(S_0\sigma T)$ and the vega weight is $W_T^2/(\sigma T) - W_T - 1/\sigma$ (Fournié et al. 1999).

Tests pool fixed seeds against COS and Black–Scholes references. They check that the estimators are unbiased (pooled $|z| < 4$) and that the standard errors are calibrated (per-seed $z$-scores have unit spread).

## 6. What is tested

- **Jet rules.** Every rule is checked against the complex step, mpmath or high-order differences.
- **The `GRADIENTS` laws** in `spxa.laws` (seven laws):
  - $\partial\psi$, $\partial\kappa$ and COS Jacobians match differences of independent code;
  - the sum, scaling and subordination chain rules hold;
  - Hessians are symmetric and consistent.

  Over 12 seeds × 300 examples per law no law failed. Mutation tests show that a derivative off by $10^{-4}$ is caught.
- **Differences use steps of $10^{-4}$ relative to the parameter.** At $10^{-3}$, the Richardson truncation error near CGMY's $\Gamma(-Y)$ pole ($Y = 1.9$) reached $1.2\times10^{-6}$ and falsified a jet derivative that mpmath confirms.
- **Discovery gradients.** The exponents and cumulants in free coordinates match differences to $10^{-9}$ relative, for all eleven families and four composites. The likelihood and cumulant objectives match differences to $10^{-8}$ of the largest gradient entry on five structures, and the option-residual Jacobian does so on six.

**Known limitations.**

- The generalised hyperbolic Lévy density uses a truncated quadrature, so its triplet mean $\kappa_1$ is off by about $2.5\times10^{-9}$. The characteristic exponent and its derivatives are unaffected.
- `jacobian` and `price_and_gradient` use forward mode, so they cost about $n+1$ evaluations for $n$ parameters. For the calibrators here, $n \le 10$.
- The Hull–White, Heston-joint and SVI calibrators still use differences.

## References

- Carr, P. & Madan, D. (1999). Option valuation using the fast Fourier transform. *J. Comput. Finance* 2(4).
- Fang, F. & Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM J. Sci. Comput.* 31(2).
- Figurnov, M., Mohamed, S. & Mnih, A. (2018). Implicit reparameterization gradients. *NeurIPS*.
- Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L. & Touzi, N. (1999). Applications of Malliavin calculus to Monte Carlo methods in finance. *Finance Stoch.* 3.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer, ch. 7.
- Griewank, A. & Walther, A. (2008). *Evaluating Derivatives*, 2nd ed. SIAM.
- Lewis, A. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes. SSRN 282110.
- Madan, D., Carr, P. & Chang, E. (1998). The variance gamma process and option pricing. *Eur. Finance Rev.* 2.
