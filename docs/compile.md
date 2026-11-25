# Compiled kernels (`spxa.compile`)

`spxa.compile` turns a Lévy process expression into generated code that evaluates its
characteristic exponent, the exact parameter gradient and Hessian, the cumulants and their
Jacobian, a fused exact sampler, and COS option prices with their exact Jacobian. One
expression gives one kernel, for NumPy, Numba or JAX, and either one parameter vector or a
batch of them.

```python
import numpy as np
from spxa.zoo.levy import BrownianMotion, CGMY, GammaProcess, KouJumpDiffusion

X = (
    BrownianMotion(0.05, 0.2) + 0.5 * KouJumpDiffusion(0.05, 0.2, 2.0, 0.4, 8.0, 5.0)
) @ GammaProcess(4.0, 4.0) + CGMY(1.0, 5.0, 6.0, 0.5)

cp = X.compile()                    # backend="numpy" | "numba" | "jax"
cp.names                            # ('BrownianMotion.mu', ..., 'scale', 'GammaProcess.a', ...)
psi, grad = cp.char_exponent_and_grad(np.linspace(-20, 20, 1001))   # (1001,), (1001, 15)
cp.cumulants(4)                     # [-0.064064, 0.219823, -0.014432, 0.026146]

K = np.linspace(80, 120, 9)
prices, jac = cp.cos_price(100.0, K, 0.5, rate=0.03, grad=True)      # (9,), (9, 15)
theta = np.tile(cp.values, (64, 1))                                  # 64 parameter sets
P, J = cp.cos_price(100.0, K, 0.5, rate=0.03, theta=theta, grad=True)  # (64, 9), (64, 9, 15)

paths = cp.simulate(252, 1000, 1.0, np.random.default_rng(1))       # (1000, 253), exact
source = cp.export("model.py")      # standalone NumPy module: psi, char_func, psi_and_grad
```

`X.compile(free=["GammaProcess.a", "CGMY.C"])` exposes only those parameters and folds the
others in as constants. `cp.to_process(theta)` rebuilds the zoo expression at any parameter
vector, and `cp.story()` prints it.

## 1. Lowering

Every zoo Lévy family in `spxa.compile.supported_families()` has a lowering rule that writes
its exponent as an expression over named parameters. The families are `AlphaStable`,
`BrownianMotion`, `CGMY`, `CompoundPoisson` (normal, exponential, double-exponential,
uniform and discrete jumps), `Drift`, `GammaProcess`, `GeneralizedHyperbolicProcess`,
`InverseGaussianProcess`, `KouJumpDiffusion`, `MeixnerProcess`, `MertonJumpDiffusion`, `NIG`,
`NegativeBinomialProcess`, `NormalTemperedStableProcess`, `PoissonProcess`, `TemperedStable`
and `VarianceGamma`. Compositions lower as

$$
\psi_{X+Y}(u) = \psi_X(u) + \psi_Y(u),\qquad
\psi_{cX}(u) = \psi_X(cu),\qquad
\psi_{X\circ T}(u) = \psi_T\!\left(-i\,\psi_X(u)\right),
$$

the last being subordination of $X$ by a subordinator $T$ (`X @ T`). Parameters form one flat
vector, named `"<Family>.<param>"`. Repeated families are numbered (`"NIG[1].alpha"`) and scale
factors are named `scale`, `scale[1]`, …; the layout is the same as `DiffProcess`.

The intermediate representation (`spxa.compile.ir`) is a hash-consed DAG of real, complex
and Boolean nodes. Building a node folds constants and applies algebraic identities, and equal
subexpressions are stored once. Numerically delicate forms are rules, not rewrites: the
overflow-free $\log\cosh$, `expm1`/`log1p` where they matter, $\mathrm{sinc}$ with a series
near zero, the CGMY series bands around the $\Gamma(-Y)$ poles at $Y = 0, 1$, and the order
derivative of $K_\nu$ for the generalised hyperbolic family.

## 2. Derivatives

`spxa.compile.diff.derivative(roots, w)` differentiates symbolically with respect to a
parameter or to $u$ and shares nodes with the value. Gradients, Hessians, $\partial\psi/\partial u$
and its parameter gradient are all such derivatives, emitted together, so common
subexpressions are computed once per evaluation.

Cumulants are $\kappa_n = \mathrm{Re}\,(-i)^n\psi^{(n)}(0)$, and their Jacobian is
$\partial_\theta\kappa_n = \mathrm{Re}\,(-i)^n\partial_\theta\psi^{(n)}(0)$. Up to order four
both come from one generated program. A `where` takes its branch from the condition at
$u = 0$, which is also what Taylor-mode arithmetic does, so the two agree node by node. Taylor
mode (`spxa.compile.evaluate.taylor`) handles the rest: exponents that use `sinc`, `sinc_d1`,
the Bessel order derivative or an opaque call in $u$ (their dedicated series handle the
removable singularity at 0), orders above four, and any non-finite generated value.

Exponents that use $\mathrm{Re}$, $\mathrm{Im}$, $|\cdot|$ or $\mathrm{sign}$ of a
$u$-dependent value outside a comparison (α-stable components) are not analytic at 0.
`cp.analytic` is then `False`, and cumulants and COS prices raise `CompileError`.

## 3. Backends

`spxa.compile.codegen.generate(outputs, n_params, backend)` emits straight-line code:

| backend | code | notes |
|---|---|---|
| `"numpy"` | vectorised NumPy | bit-identical to the reference interpreter `evaluate` |
| `"numba"` | one `@njit` loop over points (and parameter rows), `parallel=` optional | scalar complex helpers for `exp`, `expm1`, `**` and division |
| `"jax"` | `jax.numpy` under `jax.jit`, 64-bit | no `kve` (generalised hyperbolic) or opaque calls: `BackendUnsupported` |

Numba and JAX agree with NumPy to a few $10^{-16}$ relative. Edge cases that change finite
results are matched to NumPy explicitly: $0^{y}$ for complex $y$ with $\mathrm{Re}\,y > 0$ is $0$
on all three, not `nan`, and Numba's complex division by zero follows NumPy. Non-finite results
can differ in the imaginary part only. At overflow, complex `expm1` on the real axis is the exact
real value `inf + 0j` on Numba and JAX, where NumPy forms `inf + nan j`. JAX also returns
`inf + 0j` for a complex value divided by zero.

Parameter-only branch conditions (such as the CGMY bands) are evaluated before the call. The
kernel is specialised on the pattern, so a parameter vector or batch on one side of every
branch runs straight-line code.

JAX compiles once per input shape. Arrays are padded to power-of-two lengths on every non-unit
axis, repeating edge values, and the outputs are sliced back. A calibration loop that changes
its batch size, strike count or frequency grid therefore compiles a logarithmic number of
times. After the first call, single-parameter COS prices on JAX take about as long as on NumPy.

## 4. Sampling

`cp.simulate(n_steps, n_paths=1, T=1.0, rng=None)`, `cp.sample_at(times, rng)` and
`cp.paths(times, n_paths, rng)` draw exact increments from the lowered tree:

- Gaussian components of a sum are merged into one normal draw.
- Compound Poisson jump sums are aggregated per step: draw the jump count, then the sum of that
  many jumps in one call.
- Subordination draws clock increments, then the subordinated process at those random times
  (vectorised `sample_at`).
- Inverse Gaussian variates use the Michael–Schucany–Haas transform in a form free of
  cancellation for small shape-to-mean ratios. This also fixes the zoo `NIG` and
  `InverseGaussianProcess` samplers.
- `theta` may be a `(n_paths, n_free)` batch: one parameter set per path.

`cp.sampler_exact()` reports whether every component has an exact sampler.

## 5. COS prices with the exact Jacobian

`cp.cos_price(spot, strikes, maturities, rate=0.0, dividend=0.0, call=True, n_terms=1024,
width=12.0, theta=None, grad=False)` prices European options under
$S_T = S_0 e^{(r-q)T + \omega T + X_T}$ with $\omega = -\psi(-i)$. It uses the COS expansion on
$[a, b] = c_1 \pm L\sqrt{c_2 + \sqrt{c_4}}$, where $c_n = \kappa_n T$ and $L$ = `width`.
Quotes with the same maturity share one expansion, and `theta` may be a batch.

The Jacobian differentiates everything that depends on $\theta$, including the truncation
interval. With $u_k = k\pi/(b-a)$, the exponent of the $k$-th term is
$g_k = iu_k\left((r-q+\omega)T - a\right) + T\psi(u_k)$, and

$$
\frac{\partial u_k}{\partial\theta} = -u_k\,\frac{\partial_\theta b - \partial_\theta a}{b - a},
$$

so the term derivative combines $\partial_\theta\psi(u_k)$, $\partial_u\psi(u_k)$, the
derivatives of $\omega$, of $a$ and $b$ (through the cumulant Jacobian), and the closed-form
derivatives of the payoff coefficients in the end points. Prices and the Jacobian match the
graph COS pricer (`spxa.graph.autodiff.fourier`, `track_interval=True`) to $10^{-12}$. The
Jacobian also agrees with central differences of the computed price to $2\times10^{-7}$, the
error of the differences themselves.

## 6. Standalone export

`cp.export(path=None, grad=True)` returns, and optionally writes, the source of a module that
needs only NumPy and, for Gamma, digamma and Bessel functions, SciPy. It defines `NAMES`,
`THETA0`, `psi(u, theta)`, `char_func(u, t, theta)` and `psi_and_grad(u, theta)`. The exported
functions are the general programs: they are valid for every $\theta$, and they agree bit for
bit with the in-process NumPy kernel. Opaque leaves and a free generalised hyperbolic
`lambda_param` (its $K_\lambda$ order derivative needs spxa) raise `CompileError`.

## 7. Verification

`tests/unit/compile` holds more than 500 tests:

- each lowering rule against the zoo exponent;
- gradients and Hessians against `DiffProcess`;
- the Numba and JAX backends against NumPy;
- the samplers' empirical characteristic functions;
- COS prices and Jacobians against the graph pricer and against differences;
- exported modules, imported with spxa blocked.

The conformance suite (`test_conformance.py`) builds 24 random composition trees from a fixed
seed. They cover sums, scalings, subordination by every subordinator family, and clocks of
clocks up to depth three, and each tree is checked end to end:

- $\psi$ against the zoo, to $10^{-9}$ (scaled);
- the gradient against `DiffProcess`, or against fourth-order differences of the rebuilt
  process for compound Poisson, to $10^{-6}$;
- cumulants 1–4 against the zoo, to $10^{-8}$ relative;
- the sampler's empirical characteristic function, within $4.5/\sqrt n$;
- Numba and JAX against NumPy, to $10^{-12}$.

## 8. Performance

`python benchmarks/compile_report.py` prints the table below for every installed backend
(2-CPU Linux machine, NumPy backend, 4096 frequencies, composite = the example above). Each
baseline is the fastest uncompiled route: a leaf's closed-form `char_exponent`, a composite's
`_char_exponent` (the path that sums, scalings and subordination use), `DiffProcess` for
gradients, and the graph COS pricer once per parameter set for batches.

| quantity | uncompiled (ms) | compiled (ms) | speed-up |
|---|---:|---:|---:|
| Merton $\psi$ (leaf, closed form) | 0.08 | 0.13 | 0.6× |
| composite $\psi$ (Kou has no vectorised zoo exponent) | 51.5 | 0.61 | 85× |
| composite $\psi$, vectorised families only | 0.46 | 0.43 | 1.1× |
| composite $\psi$ + gradient (15 parameters) | 12.8 | 1.61 | 7.9× |
| vectorised composite $\psi$ + gradient (16 parameters) | 10.1 | 1.98 | 5.1× |
| COS prices + Jacobian, 9 strikes | 6.67 | 2.56 | 2.6× |
| COS prices + Jacobian, 9 strikes × 64 parameter sets | 427 | 141 | 3.0× |

The gains come from fusing the whole expression, from generated derivatives, and from batching.
A single closed-form family is already as fast as NumPy allows, and the compiled call overhead
makes it slower. The sampler is bound by random-number generation: fused simulation is
comparable to the zoo for single families and up to about 2.5× faster for sums of several
components. Numba and JAX give similar numbers on CPU (JAX was up to about 15% faster on the
gradient rows on this machine). `benchmarks/test_compile_benchmarks.py` tracks the same
quantities in the benchmark workflow.
