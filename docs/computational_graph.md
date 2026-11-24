# Stochastic Computational Graph Engine (`spxa.graph`)

## 1. Overview and Core Philosophy

In quantitative finance, models have historically been siloed across distinct libraries:
- One implementation for closed-form analytics (e.g. characteristic function quadrature).
- Another implementation for Monte Carlo simulation paths.
- Finite difference routines for Greek sensitivities.
- Specialized optimization loops for model calibration.

`spxa.graph` solves this fragmentation by treating stochastic processes, analytical transforms, payoff functions, and risk metrics as nodes in a shared **Directed Acyclic Graph (DAG)**. A stochastic model is defined **once**; the execution engine then dynamically routes evaluation across:
1. **Analytical Backend**: Closed-form solutions and exact moments.
2. **Numerical Backend**: Fourier transforms (Carr-Madan, Lewis, Cosine methods).
3. **Simulation Backend**: Pathwise Monte Carlo sampling.
4. **Accelerated Backend**: Vectorized batch evaluation across thousands of strikes and maturities.
5. **Autodiff Tape**: Exact, zero-truncation sensitivities (Delta, Gamma, Vega, Rho, Theta) via reverse-mode vector-Jacobian products (VJPs) and forward-mode dual numbers.

---

## 2. Architecture

```
                    ┌───────────────────────────┐
                    │  StochasticProcessNode    │
                    │  (GBM, Heston, SABR, ...) │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
          ▼                       ▼                        ▼
┌──────────────────┐   ┌──────────────────────┐   ┌─────────────────┐
│ CharacteristicFn │   │ Monte Carlo Paths    │   │  ProcessMoment  │
│      Node        │   │        Node          │   │      Node       │
└─────────┬────────┘   └──────────┬───────────┘   └────────┬────────┘
          │                       │                        │
          ▼                       ▼                        ▼
┌──────────────────┐   ┌──────────────────────┐            │
│ Numerical Fourier│   │ Payoff Evaluation    │            │
│   (Carr-Madan)   │   │  (Call / Put / Asian)│            │
└─────────┬────────┘   └──────────┬───────────┘            │
          │                       │                        │
          └───────────────────────┼────────────────────────┘
                                  ▼
                      ┌───────────────────────┐
                      │    Pricing Result     │
                      └───────────┬───────────┘
                                  │
                      ┌───────────▼───────────┐
                      │  GradientTape Autodiff│
                      │ (Delta, Gamma, Vega)  │
                      └───────────────────────┘
```

---

## 3. Quickstart Examples

### A. European Option Pricing and Automated Greeks

```python
from spxa.graph import (
    build_black_scholes_pricing_graph,
    compute_graph_greeks,
)

# Build a Black-Scholes pricing DAG
graph = build_black_scholes_pricing_graph(
    s0=100.0,
    k=100.0,
    t=1.0,
    r=0.05,
    sigma=0.2,
    is_call=True,
)

# Evaluate closed-form analytical price
results = graph.evaluate()
print("Call Price:", results["price"])

# Greeks: first order from the tape, gamma/vanna/volga from an exact jet Hessian
greeks = compute_graph_greeks(graph)
print("Delta:", greeks["delta"])
print("Gamma:", greeks["gamma"])
print("Vega :", greeks["vega"])
print("Theta:", greeks["theta"])
print("Rho  :", greeks["rho"])
print("Vanna:", greeks["vanna"], "Volga:", greeks["volga"])

# graph_greeks also reports whether every Greek is exact
from spxa.graph import graph_greeks

res = graph_greeks(graph, strict=True)  # raises if any Greek would be a difference quotient
print(res.exactness)
```

### B. Graph-Based Model Calibration

```python
import numpy as np
from spxa.graph import (
    ComputationalGraph,
    GraphCalibrator,
    MSELossNode,
    ParameterNode,
)

# Build parameter recovery problem
param_a = ParameterNode(value=0.0, name="slope")
param_b = ParameterNode(value=0.0, name="intercept")

# Setup synthetic targets
x_data = np.linspace(0.0, 1.0, 50)
y_target = 3.5 * x_data + 1.2

calibrator = GraphCalibrator()
loss_node = MSELossNode(prediction_node=..., target=y_target)
```

---

## 4. Graph Optimization Passes

The `GraphCompiler` optimizes the DAG before execution:
- **Constant Folding**: Subgraphs depending purely on `ConstantNode` are pre-evaluated at compile time.
- **Common Subexpression Elimination (CSE)**: Identical mathematical operations with identical inputs are merged into single nodes.
- **Dead Code Elimination**: Pruning of nodes unreferenced by any downstream target.
- **Execution Caching**: Thread-safe evaluation cache for instant reuse of stationary nodes.

---

## 5. Universal Implementation Layer (`spxa.graph.adapters`)

Rather than requiring manually handwritten graph nodes for every one of the ~200 processes in `spxa.zoo` or objects in other submodules, `spxa.graph` provides an automatic **reflection and adaptation layer**:

### A. Zero-Boilerplate Zoo Process Adaptation

Any process from `spxa.zoo` (e.g. `BatesModel`, `VarianceGamma`, `NIG`, `CIRProcess`, `BNSModel`, etc.), as well as any composed process ($X + Y$, $c \cdot X$, $X @ T$), can be directly used in computational graphs:

```python
from spxa.zoo import BatesModel, VarianceGamma

# 1. Any zoo process immediately has .to_graph()
bates = BatesModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.5)
bates_node = bates.to_graph()

# 2. Parameters are automatically discovered and mutable
bates_node.params["v0"].value = 0.05
bates_node.sync_parameters_to_process()

# 3. Graph nodes for simulation and characteristic function are auto-generated
cf_node = bates_node.char_func_node(u=0.5, t=1.0)
sim_node = bates_node.simulate_node(t=1.0, n_paths=1000, n_steps=100)
```

### B. Submodule Integration (`risk`, `diff`, `calculus`, `copula`)

- **Risk Instruments (`spxa.risk`)**: Instruments (`EuropeanOption`, `BarrierOption`, `AsianOption`) and `Portfolio` have `.to_graph()` returning automated payoff and sensitivity aggregation graphs.
- **Differentiable SDEs (`spxa.diff`)**: `DifferentiableSDE` models have `.to_graph()` providing drift, diffusion, and Euler step graph nodes.
- **Copulas (`spxa.copula`)**: Copulas have `.to_graph()` providing coupled uniform sampling nodes.
- **Generators (`spxa.calculus`)**: `InfinitesimalGenerator` adapts into graph operator nodes evaluating $\mathcal{A} f(x)$.

---

## 6. Deep Mathematical Calculus on Stochastic Processes

The computational graph serves as a first-class algebraic engine for advanced stochastic calculus:

### A. Automated Itô Lemma (`ItoTransformNode`)
Applies Itô's Lemma to derive the exact SDE of $Y_t = f(X_t, t)$ automatically using graph derivatives:
$$dY_t = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} dW_t$$

### B. Measure Changes (`GirsanovTransformNode`, `EsscherTransformNode`)
- **Girsanov Drift Shifts**: Shifts diffusion drift under market price of risk $\lambda(x, t)$: $\mu^\mathbb{Q} = \mu^\mathbb{P} - \sigma \lambda$.
- **Esscher Transforms**: Shifts characteristic exponents for jump processes: $\Psi_\theta(u) = \Psi(u - i\theta) - \Psi(-i\theta)$.

### C. Bochner Subordination & Cumulant Generation (`BochnerSubordinationNode`, `AutomatedCumulantNode`)
- Composes parent Lévy exponent with subordinator Laplace exponent: $\Psi_Z(u) = -\Phi_T(-\Psi_X(u))$.
- Automatically extracts cumulants $\kappa_1, \kappa_2, \kappa_3, \kappa_4$ to arbitrary order by real central differentiation of the CGF $K(s) = \Psi(-is)$.

### D. Infinitesimal Generators & Martingale Verification (`InfinitesimalGeneratorNode`, `MartingaleVerificationNode`)
- Evaluates Dynkin's formula $\mathcal{A} f(x) = \mu(x) f'(x) + \frac{1}{2}\sigma^2(x) f''(x)$.
- Algebraically checks whether $f(X_t)$ is a true martingale by proving $\mathcal{A} f(x) \equiv 0$ across the state space.

### E. Speed / Scale Density & Feller Boundary Classification (`ScaleDensityNode`, `SpeedDensityNode`, `FellerBoundaryClassifierNode`)
- Computes scale density $s(x) = \exp\left(-\int \frac{2\mu}{\sigma^2}\right)$ and speed density $m(x) = \frac{1}{\sigma^2 s}$.
- Classifies 1D diffusion boundaries into Feller types: **Regular, Exit, Entrance, or Natural**.

### F. Malliavin Calculus & Bismut Integration by Parts (`MalliavinBismutDeltaNode`)
- Simulates the first variation process $Y_t = \partial X_t / \partial x_0$ alongside $X_t$.
- Evaluates pathwise Bismut weights $\pi_T = \frac{1}{T} \int_0^T \frac{Y_t}{\sigma(X_t)} dW_t$, computing exact Greek sensitivities of discontinuous payoffs without payoff mollification.

---

## 7. Advanced Stochastic Foundations

The engine includes three theoretical pillars bridging stochastic analysis and computational graphs:

### A. Semimartingale Decomposition Wires & Algebra (`spxa.graph.semimartingale`)
In continuous-time stochastic calculus, semimartingales form the largest class of processes with respect to which stochastic integrals can be defined.
- **Typed Semimartingale Wires (`SemimartingaleWire`)**: Graph edges carry continuous-time processes decomposed into their canonical components:
  $$X_t = X_0 + A_t + M_t^c + J_t$$
  where $A_t$ is continuous of finite variation, $M_t^c$ is a continuous local martingale, and $J_t$ is a pure jump process.
- **Quadratic Covariation Operator (`QuadraticCovariationNode`)**: Computes the pathwise and predictable bracket $[X, Y]_t$ and $\langle X^c, Y^c \rangle_t$:
  $$[X, Y]_t = \lim_{|\Pi| \to 0} \sum_{i} (X_{t_{i+1}} - X_{t_i})(Y_{t_{i+1}} - Y_{t_i})$$
- **Stochastic Product & Integration by Parts (`StochasticProductNode`)**: Evaluates the product $Z_t = X_t Y_t$ verifying the stochastic integration by parts formula:
  $$d(X_t Y_t) = X_{t-} dY_t + Y_{t-} dX_t + d[X, Y]_t$$
  achieving machine-precision bracket balance ($\epsilon \sim 10^{-15}$).
- **Predictable Itô Integrals (`StochasticIntegralNode`)**: Computes $(H \cdot X)_t = \int_0^t H_s dX_s$ with exact Itô isometry $\mathbb{E}[(H \cdot M)_t^2] = \mathbb{E}\left[\int_0^t H_s^2 d\langle M \rangle_s\right]$.

### B. Four-Facet Representation Duality Engine (`spxa.graph.duality`)
Processes are mathematically dual across four equivalent representations:
1. **SDE Facet (`SDEFacet`)**: $dX_t = \mu(X_t) dt + \sigma(X_t) dW_t + J_t$.
2. **Infinitesimal Generator Facet (`GeneratorFacet`)**: $\mathcal{A} f(x) = \mu(x) f'(x) + \frac{1}{2}\sigma^2(x) f''(x)$.
3. **Fourier Characteristic Exponent Facet (`FourierFacet`)**: $\mathbb{E}[e^{i u X_t}] = \exp(t \Psi(u))$.
4. **Speed / Scale Density Facet (`DensityFacet`)**: Invariant measure $\pi(x) \propto m(x)$.

The `ProcessDualityCore` automatically synthesizes missing representations:
- Given an SDE, derives the infinitesimal generator and speed/scale density.
- Given an infinitesimal generator, inverts for the drift $\mu(x) = \mathcal{A}(\text{id})(x)$ and diffusion $\sigma^2(x) = \mathcal{A}(x^2) - 2x\mathcal{A}(x)$.
- **Lévy-Khintchine Triplet Compiler (`compile_levy_khintchine`, `triplet_to_fourier_facet`)**: Directly compiles arbitrary triplets $(b, \sigma^2, \nu)$ into executable characteristic exponents:
  $$\Psi(u) = i b u - \frac{1}{2} \sigma^2 u^2 + \int_{\mathbb{R}} \left( e^{i u y} - 1 - i u y \mathbf{1}_{|y|\le 1} \right) \nu(dy)$$

### C. Algebraic Process Surgery Operators (`spxa.graph.surgery`)
Advanced operators that transform stochastic paths and boundary conditions algebraically:
- **Doléans Exponential & Stochastic Logarithm (`DoleansExponentialNode`, `StochasticLogarithmNode`)**:
  Computes the stochastic exponential martingale:
  $$\mathcal{E}(X)_t = \exp\left(X_t - X_0 - \frac{1}{2}\langle X^c \rangle_t\right) \prod_{s \le t} (1 + \Delta X_s) e^{-\Delta X_s}$$
  and its inverse, the stochastic logarithm $\mathcal{L}(Z)_t = \int \frac{dZ_s}{Z_{s-}}$.
- **Tanaka Decomposition & Local Time Extraction (`TanakaDecompositionNode`)**:
  Decomposes positive convex envelopes and isolates local time $L_t^a(X)$ at strike $a$:
  $$(X_t - a)^+ = (X_0 - a)^+ + \int_0^t \mathbf{1}_{\{X_s > a\}} dX_s + \frac{1}{2} L_t^a(X)$$
- **Doob $h$-Transform (`DoobHTransformNode`)**:
  Conditions diffusion paths on staying in a sub-domain or reaching a target via excessive harmonic functions $h$:
  $$\mu^h(t, x) = \mu(t, x) + \sigma^2(t, x) \frac{h'(x)}{h(x)}$$
  (e.g., standard Brownian motion conditioned to stay positive produces the 3D Bessel process drift $1/x$).
- **Stratonovich $\leftrightarrow$ Itô Conversion (`StratonovichToItoNode`, `ItoToStratonovichNode`)**:
  Transitions between Itô and Stratonovich calculus by adding or subtracting the midpoint drift correction $\frac{1}{2}\sigma(x)\sigma'(x)$.

---

## 8. Statistical Information Geometry & Manifold Calibration (`spxa.geom`, `spxa.graph.geom`)

Information geometry equips stochastic process parameter spaces $\mathcal{M} = \{\theta \in \Theta\}$ with intrinsic Riemannian manifolds rather than flat Euclidean spaces:

### A. Fisher Information Metric (FIM)
The Fisher metric tensor measures informational distance between stochastic transition densities:
$$g_{ij}(\theta) = \mathbb{E}_\theta \left[ \frac{\partial \log p}{\partial \theta_i} \frac{\partial \log p}{\partial \theta_j} \right] = - \mathbb{E}_\theta \left[ \frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j} \right]$$
- **Exact Closed Forms**: Computed analytically for Gaussians $g = \text{diag}(1/\sigma^2, 2/\sigma^2)$ and Ornstein-Uhlenbeck transition densities.
- **Numerical & Empirical Score Covariance**: Automatically evaluated via finite-difference Hessians and pathwise score vectors.

### B. Connections & Geodesic Flow
- **Levi-Civita Connection & Christoffel Symbols**: $\Gamma^k_{ij}(\theta) = \frac{1}{2} g^{kl} (\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$.
- **Amari $\alpha$-Connections**: $\Gamma^{(\alpha), k}_{ij} = \Gamma^k_{ij} - \frac{\alpha}{2} g^{kl} T_{ijl}$ where $T_{ijl}$ is the skewness tensor ($\alpha = 0$ Riemannian, $\alpha = 1$ exponential, $\alpha = -1$ mixture).
- **Rao Hyperbolic Distance (`GeodesicFlow`)**: Integrates geodesic paths $\ddot{\theta}^k + \Gamma^k_{ij} \dot{\theta}^i \dot{\theta}^j = 0$ via shooting solvers to measure true model distance $d_R(\theta_1, \theta_2) = \int \sqrt{g_{ij} \dot{\theta}^i \dot{\theta}^j} dt$.

### C. Natural Gradient Optimization (`NaturalGradientCalibratorNode`)
Replaces naive Euclidean gradient descent with Riemannian manifold updates:
$$\tilde{\nabla}_\theta \mathcal{L} = g^{-1}(\theta) \nabla_\theta \mathcal{L}, \quad \theta_{t+1} = \theta_t - \eta \tilde{\nabla}_\theta \mathcal{L}$$
eliminating ill-conditioned ravines, Feller boundary sticking, and parameterization distortions.

---

## 9. Rough Paths & Tensor Signature Calculus (`spxa.signature`, `spxa.graph.signature`)

Path signatures provide a non-parametric, coordinate-free encoding of continuous and jump paths:

### A. Truncated Tensor Signature & Chen's Identity (`PathSignature`)
For a multidimensional path $X: [0, T] \to \mathbb{R}^d$, the truncated signature of order $M$ is:
$$S(X)_{s,t}^{(M)} = \bigoplus_{k=0}^M \int_{s < u_1 < \dots < u_k < t} dX_{u_1} \otimes \dots \otimes dX_{u_k} \in T^{(M)}(\mathbb{R}^d)$$
Satisfies Chen's multiplicative identity:
$$S(X)_{0,T} = S(X)_{0,t} \otimes S(X)_{t,T}$$

### B. Shuffle Algebra & Free Lie Log-Signature (`SignatureAlgebra`)
- **Shuffle Product**: Iterated integrals multiply via shuffles: $S^u \cdot S^v = \sum_{w \in u \sqcup v} S^w$.
- **Log-Signature Projection**: Projects signatures onto the free Lie algebra $\mathfrak{g}^{(M)}(\mathbb{R}^d)$ via Campbell-Baker-Hausdorff / Dynkin operators, reducing dimensionality according to Witt's formula (e.g. $15 \to 5$ at depth 3 for 2D paths, a 66.7% compression without information loss).

### C. Signature Kernels via 2D Hyperbolic Goursat PDE (`SignatureKernel`)
Computes the infinite-dimensional inner product $K(X, Y) = \langle S(X), S(Y) \rangle$ without tensor truncation by solving the hyperbolic PDE:
$$\frac{\partial^2 K}{\partial s \partial t}(s, t) = K(s, t) \langle \dot{X}_s, \dot{Y}_t \rangle, \quad K(s, 0) = 1, \quad K(0, t) = 1$$
Runs in polynomial $O(N_X \cdot N_Y)$ time.

### D. Universal Linear Pricing (`SignatureLinearPricerNode`)
Represents arbitrary path-dependent payoffs (Asian, lookbacks, barriers) as linear functionals on the expected signature:
$$\text{Price} \approx \langle \mathbf{w}, \mathbb{E}[S(X)_{0,T}] \rangle$$

---

## 10. Rough Volatility & Volterra Stochastic Calculus (`spxa.rough`, `spxa.graph.rough`)

Extends stochastic modeling to sub-diffusive, non-Markovian volatility surfaces:

### A. Fractional Singular Kernels & Hybrid Convolution (`FractionalKernel`)
- Riemann-Liouville fractional power kernel with Hurst exponent $H \in (0, 1/2)$:
  $$K_H(t) = \frac{\sqrt{2H}}{\Gamma(H + 1/2)} t^{H - 1/2}$$
- **Bennedsen-Lunde-Pakkanen Hybrid Scheme**: Decomposes $\int_0^t (t-s)^{H-1/2} dW_s$ into exact analytic integration for the singular neighborhood $s \approx t$ and FFT-accelerated convolution for the regular tail.
- **Exact Fractional Brownian Motion**: Cholesky covariance sampling of $R_H(t, s) = \frac{1}{2}(t^{2H} + s^{2H} - |t - s|^{2H})$.

### B. Rough Bergomi Model (`RoughBergomi`)
Asset and variance dynamics:
$$S_t = S_0 \mathcal{E}\left( \int_0^t \sqrt{v_s} dW_s \right), \quad v_t = \xi_0 \exp\left( \eta \sqrt{2H} \tilde{W}_t^H - \frac{1}{2} \eta^2 t^{2H} \right)$$
Reproduces the empirical power-law blow-up of short-time at-the-money implied volatility skews:
$$\psi(\tau) = \left|\frac{\partial \sigma_{BS}}{\partial k}\right|_{k=0} \approx \frac{|\rho| \eta}{2} \frac{\sqrt{2H}}{H + 1/2} \tau^{H - 1/2}$$

### C. Fractional Adams-Bashforth-Moulton Riccati Solver (`FractionalRiccatiSolver`)
Solves the non-local fractional Riccati Volterra integral equation:
$$h(t) = \int_0^t \frac{(t-s)^{H-1/2}}{\Gamma(H + 1/2)} F(u, h(s)) ds$$
providing exact, highly stable characteristic functions $\mathbb{E}[e^{i u X_T}]$ for the Rough Heston model.




