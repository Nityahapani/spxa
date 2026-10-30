# Rough Volatility

Time series of realised volatility behave locally like fractional Brownian motion with Hurst exponent $H \approx 0.1$, far below the Brownian value $H = 1/2$ (Gatheral, Jaisson & Rosenbaum 2018). Implied volatility surfaces show the matching signature: the at-the-money skew grows like $\tau^{H-1/2}$ as maturity $\tau \to 0$. Classical Markovian stochastic volatility models cannot produce that power law. `spxa.rough` implements the two reference rough models, rough Bergomi and rough Heston, together with the fractional kernels they are built on.

| Object | Implementation |
| :--- | :--- |
| Power kernels, fBM and Volterra simulation | `spxa.rough.FractionalKernel` |
| Rough Bergomi model | `spxa.rough.RoughBergomi` |
| Rough Heston fractional Riccati solver | `spxa.rough.FractionalRiccatiSolver` |
| Graph nodes | `spxa.graph.rough.RoughKernelNode`, `RoughBergomiNode`, `FractionalRiccatiNode` |
| JAX-differentiable variants | `spxa.diff.simulate_diff_fbm`, `spxa.diff.simulate_diff_rough_heston` |

---

## 1. Fractional processes

### 1.1 Fractional Brownian motion

Fractional Brownian motion $B^H$ is the centred Gaussian process with covariance

$$
\mathbb{E}[B^H_s B^H_t] = \tfrac12\left(s^{2H} + t^{2H} - |t-s|^{2H}\right),\qquad H\in(0,1).
$$

It has stationary increments and is self-similar ($B^H_{ct} \overset{d}{=} c^H B^H_t$). Its paths are Hölder continuous of every order below $H$. For $H < 1/2$ the increments are negatively correlated, and $B^H$ is neither a semimartingale nor Markov. `FractionalKernel.sample_fbm_cholesky` samples it exactly on a grid by Cholesky factorisation of this covariance, at $O(n^3)$ cost.

### 1.2 Riemann–Liouville (Volterra) processes

Rough volatility models are driven by the *Volterra* process

$$
\tilde W_t = \int_0^t (t-s)^{H-1/2}\, dZ_s, \qquad \operatorname{Var}(\tilde W_t) = \frac{t^{2H}}{2H},
$$

where $Z$ is a standard Brownian motion. $\tilde W$ has the same local roughness and scaling as fBM but no longer has stationary increments. What makes it convenient is that it is *adapted to $Z$*, so it can be correlated with the Brownian motion that drives the asset price. The joint law of $(\tilde W, Z)$ on a grid is Gaussian, with

$$
\begin{aligned}
\operatorname{Cov}(\tilde W_s, \tilde W_t) &= \frac{s^{a+1}\,t^{a}}{a+1}\;{}_2F_1\!\Big(-a,\,1;\,a+2;\,\tfrac{s}{t}\Big), \qquad s\le t,\ a = H-\tfrac12, \\
\operatorname{Cov}(\tilde W_t, Z_s) &= \frac{t^{a+1} - \big(t-\min(s,t)\big)^{a+1}}{a+1}, \\
\operatorname{Cov}(Z_s, Z_t) &= \min(s,t).
\end{aligned}
$$

The first line follows from writing $\int_0^s (s-u)^a (t-u)^a\,du$ as an Euler integral. At $s=t$ the hypergeometric series converges because $c-a-b = 2H > 0$. `FractionalKernel.volterra_joint_covariance` assembles the $2n\times 2n$ matrix and `sample_volterra_cholesky` samples it exactly (Bayer, Friz & Gatheral 2016, §4). The test suite checks every entry against direct quadrature.

### 1.3 Hybrid convolution

`FractionalKernel.hybrid_convolution(dw, dt, b_cutoff)` evaluates $\int_0^{t_i}(t_i-s)^{H-1/2}dW_s$ from a given sequence of Brownian increments in $O(n^2)$:

* On the $b$ cells nearest the singularity, it uses the exact cell-average of the kernel, $\frac{1}{\Delta}\int_{(k-1)\Delta}^{k\Delta} s^{a}\,ds$.
* On the remaining cells it uses the kernel value at the cell's left endpoint (a Riemann sum).

On the singular cells this is the conditional expectation of the Wiener integral given the increment. That is the best estimate available from `dw` alone, but it misses the variance of the residual. For $H = 0.1$ and $\Delta = 0.01$ it captures about 85% of the true variance $t^{2H}/2H$. The full Bennedsen–Lunde–Pakkanen (2017) hybrid scheme recovers the missing part by sampling the local Wiener integrals jointly with $dW$. **When exact distributional properties matter, use `sample_volterra_cholesky`**, which is what `RoughBergomi` does.

---

## 2. Rough Bergomi

Bayer, Friz & Gatheral (2016) model the forward variance curve $\xi_0(t) = \mathbb{E}[v_t]$ as a lognormal Volterra functional:

$$
\begin{aligned}
v_t &= \xi_0(t)\,\exp\!\Big(\eta\sqrt{2H}\,\tilde W_t - \tfrac12\eta^2 t^{2H}\Big),\\
\frac{dS_t}{S_t} &= \sqrt{v_t}\,dW_t,\qquad dW_t = \rho\,dZ_t + \sqrt{1-\rho^2}\,dZ^\perp_t .
\end{aligned}
$$

Because $\operatorname{Var}(\sqrt{2H}\tilde W_t) = t^{2H}$, the exponential is a mean-one lognormal, so $\mathbb{E}[v_t] = \xi_0(t)$ holds exactly. `RoughBergomi` uses a flat curve $\xi_0(t) \equiv \xi_0$.

### 2.1 Simulation

`RoughBergomi.simulate_paths` proceeds in three steps:

1. Sample $(\tilde W_{t_i}, Z_{t_i})_{i\le n}$ jointly and exactly (§1.2).
2. Set $v_{t_i} = \xi_0\exp(\eta\sqrt{2H}\,\tilde W_{t_i} - \tfrac12\eta^2 t_i^{2H})$. This is exact on the grid.
3. Advance $\log S$ with a left-point (Itô) Euler step, using $\Delta W_i = \rho\,\Delta Z_i + \sqrt{1-\rho^2}\,\Delta Z^\perp_i$ where $\Delta Z_i$ are the increments of the *same* $Z$ that drives $\tilde W$.

Step 3 is what produces the leverage effect: with $\rho < 0$, returns and variance are negatively correlated. The model has no Markovian state, and exact simulation costs $O(n^3)$ for the factorisation and $O(n^2)$ per path. Moderate grids ($n \lesssim 500$) are therefore the intended use.

### 2.2 Short-time ATM skew

To first order in the vol-of-vol, the ATM implied volatility skew is (Fukasawa 2017; Bayer, Friz & Gatheral 2016)

$$
\left.\frac{\partial\sigma_{BS}(k,\tau)}{\partial k}\right|_{k=0}
\;\approx\; \frac{\rho\,\eta\,\sqrt{2H}}{2\,(H+\tfrac12)(H+\tfrac32)}\;\tau^{H-1/2}, \qquad \tau\to0 .
$$

`RoughBergomi.theoretical_short_time_skew(tau)` returns the absolute value of this expression. The $1/(H+\tfrac32)$ factor is the rough-volatility counterpart of the "one-half rule" relating implied and local volatility skews (Bourgey, De Marco, Friz & Pigato 2023). In the test suite, a Monte Carlo estimate of the skew with $H=0.1$, $\eta=0.5$, $\rho=-0.7$, $\tau = 0.1$ agrees with the asymptotic formula to within about 5–10%. At 60,000 paths the Monte Carlo error dominates, and the test allows 15%.

```python
from spxa.rough import RoughBergomi

model = RoughBergomi(hurst=0.1, eta=1.9, rho=-0.9, xi0=0.04, s0=100.0)
sim = model.simulate_paths(t_mat=1.0, n_steps=100, n_paths=20_000, seed=1)
v, s = sim["variance_paths"], sim["spot_paths"]
print(v.mean(axis=0)[[25, 50, 100]])   # ≈ 0.04 at every date (flat forward variance)
print(s[:, -1].mean())                 # ≈ 100 (martingale)
```

---

## 3. Rough Heston

El Euch & Rosenbaum (2019) introduced a rough version of the Heston model. Let $\alpha = H + \tfrac12 \in (\tfrac12, 1]$ and $X_t = \log(S_t/S_0)$:

$$
\begin{aligned}
dX_t &= -\tfrac12 V_t\,dt + \sqrt{V_t}\,dW_t,\\
V_t &= V_0 + \frac{1}{\Gamma(\alpha)}\int_0^t (t-s)^{\alpha-1}\Big[\lambda(\theta - V_s)\,ds + \nu\sqrt{V_s}\,dB_s\Big],
\qquad d\langle W,B\rangle_t = \rho\,dt .
\end{aligned}
$$

The model is no longer Markov, but it remains *affine*. Its characteristic function is

$$
\log\mathbb{E}\big[e^{iuX_T}\big] = \lambda\theta\,I^1 h(u,T) + V_0\,I^{1-\alpha}h(u,T),
$$

where $I^\beta f(t) = \frac{1}{\Gamma(\beta)}\int_0^t (t-s)^{\beta-1}f(s)\,ds$ is the Riemann–Liouville fractional integral and $h$ solves the **fractional Riccati equation**

$$
D^\alpha h = F(u,h),\qquad I^{1-\alpha}h(0) = 0, \qquad
F(u,h) = -\tfrac12(u^2 + iu) + (i\rho\nu u - \lambda)\,h + \tfrac12\nu^2 h^2,
$$

or equivalently the Volterra integral equation $h = I^\alpha F(u,h)$.

### 3.1 Numerical solution

`FractionalRiccatiSolver.solve` integrates $h = I^\alpha F(u,h)$ with the fractional Adams–Bashforth–Moulton predictor–corrector of Diethelm, Ford & Freed (2002). On a uniform grid $t_k = k\Delta$, the predictor is

$$
h^P_{n} = \frac{\Delta^\alpha}{\Gamma(\alpha+1)}\sum_{j=0}^{n-1} b_{j,n}\,F(u,h_j),
\qquad b_{j,n} = (n-j)^\alpha - (n-1-j)^\alpha,
$$

and the corrector is

$$
h_{n} = \frac{\Delta^\alpha}{\Gamma(\alpha+2)}\Big(F(u,h^P_{n}) + \sum_{j=0}^{n-1} a_{j,n}\,F(u,h_j)\Big),
$$

with the standard product-trapezoidal weights $a_{0,n} = (n-1)^{\alpha+1} - (n-1-\alpha)\,n^\alpha$ and $a_{j,n} = (n-j+1)^{\alpha+1} - 2(n-j)^{\alpha+1} + (n-j-1)^{\alpha+1}$.

### 3.2 Characteristic function without singular quadrature

The term $I^{1-\alpha}h$ has a weakly singular kernel. Fractional integrals form a semigroup ($I^{\beta}I^{\gamma} = I^{\beta+\gamma}$), and $h = I^\alpha F$, so

$$
I^{1-\alpha}h = I^{1-\alpha}I^{\alpha}F = I^1 F(u,h) = \int_0^T F(u,h(s))\,ds .
$$

`rough_heston_char_func` therefore evaluates

$$
\log\phi(u,T) = \lambda\theta\int_0^T h(u,s)\,ds + V_0\int_0^T F\big(u,h(u,s)\big)\,ds
$$

with the trapezoidal rule. Both integrands are regular, so no singular quadrature is needed. The implementation has three consistency checks:

* **Classical limit.** At $H = \tfrac12$ ($\alpha = 1$), $I^{0}h = h$ and the Riccati equation becomes the Heston ODE $\psi' = F(u,\psi)$. The solver matches the closed-form Heston characteristic function (`spxa.zoo.beyond.heston.HestonProcess.char_func`) to about $10^{-7}$ with 400 steps.
* **Martingale property.** At $u = -i$, $F(-i, 0) = 0$, so $h \equiv 0$ and $\phi(-i,T) = \mathbb{E}[S_T/S_0] = 1$ exactly.
* **Convergence.** Results stabilise as the grid is refined, with differences below $10^{-5}$ between 200 and 400 steps at $H=0.1$.

Option prices follow from $\phi$ by standard Fourier inversion, for example the Carr–Madan damped transform used in `spxa.analytics.fourier` or the Lewis formula.

---

## References

* Bayer, C., Friz, P. & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887–904.
* Bennedsen, M., Lunde, A. & Pakkanen, M. S. (2017). Hybrid scheme for Brownian semistationary processes. *Finance and Stochastics*, 21(4), 931–965.
* Bourgey, F., De Marco, S., Friz, P. K. & Pigato, P. (2023). Local volatility under rough volatility. *Mathematical Finance*.
* Diethelm, K., Ford, N. J. & Freed, A. D. (2002). A predictor-corrector approach for the numerical solution of fractional differential equations. *Nonlinear Dynamics*, 29, 3–22.
* El Euch, O. & Rosenbaum, M. (2019). The characteristic function of rough Heston models. *Mathematical Finance*, 29(1), 3–38.
* Fukasawa, M. (2017). Short-time at-the-money skew and rough fractional volatility. *Quantitative Finance*, 17(2), 189–198.
* Fukasawa, M. (2020). Short-term at-the-money asymptotics under stochastic volatility models. arXiv:1801.08675 (rough Bergomi skew constant, Theorem 5.1).
* Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933–949.
