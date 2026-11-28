# Beyond the Lévy Regime

spxa is built on Lévy process arithmetic, but many processes used in practice fall outside the Lévy class. This document explains how spxa handles them, what guarantees are maintained, and what is lost.

## Why the Lévy boundary matters

The Lévy–Khintchine triplet is defined only for processes with stationary and independent increments. Many important processes violate one or both:

| Process | Stationary increments | Independent increments | Lévy? |
|---|---|---|---|
| Brownian motion | yes | yes | yes |
| Poisson process | yes | yes | yes |
| Variance Gamma | yes | yes | yes |
| Fractional BM ($H \neq 1/2$) | yes | **no** | **no** |
| Hawkes process | **no** | **no** | **no** |
| OU-Lévy | **no** | **no** | **no** |
| Brownian bridge | **no** | yes | **no** |

The loss of independent increments means the Lévy–Khintchine formula no longer applies. No triplet exists. Arithmetic cannot be exact.

## Fractional Brownian motion

**Definition** (Mandelbrot & Van Ness 1968). The fractional Brownian motion (fBM) $B^H = (B^H_t)_{t \geq 0}$ with Hurst index $H \in (0, 1)$ is the unique (up to scaling) centered Gaussian process with covariance

$$\mathbb{E}[B^H_s B^H_t] = \frac{1}{2}\left(|s|^{2H} + |t|^{2H} - |s-t|^{2H}\right)$$

For $H = 1/2$, fBM is standard Brownian motion. For $H \neq 1/2$, fBM has **long memory** (for $H > 1/2$) or **anti-persistence** (for $H < 1/2$), and its increments are correlated. It is **not a semimartingale** for $H \neq 1/2$, which means Itô calculus does not apply and the Lévy–Khintchine representation does not exist.

**What spxa tracks for fBM:**

- The Hurst index $H$ as the fundamental parameter
- Self-similarity: $B^H_{ct} \stackrel{d}{=} c^H B^H_t$, so $H$ propagates under scaling as $c \cdot B^H \sim c^1$ in scale, Hurst index preserved
- Variance: $\text{Var}(B^H_t) = t^{2H}$
- The Hölder regularity: paths are $(H - \varepsilon)$-Hölder continuous for every $\varepsilon > 0$

**What spxa does not and cannot track exactly:**

- A distributional triplet (does not exist)
- Cumulants of compositions involving fBM (only approximated via moment propagation)
- Addition of fBM with a Lévy process: the result is not fBM and not a Lévy process; spxa degrades to `MOMENT_PROPAGATION` and issues a warning

**spxa's Hurst algebra.** For compositions of fBM processes:

- $c \cdot B^H$: Hurst index preserved, variance scaled by $c^2$
- $B^{H_1} + B^{H_2}$ (independent): mean and variance propagate; Hurst index of the sum is $\max(H_1, H_2)$ asymptotically (the dominant long-memory component governs long-run autocorrelation), but this is an approximation and is flagged as such

## Hawkes processes

**Definition** (Hawkes 1971). A Hawkes process $N = (N_t)_{t \geq 0}$ is a point process with conditional intensity

$$\lambda(t) = \mu + \int_0^t h(t - s)\, dN_s$$

where $\mu > 0$ is the background rate and $h : [0, \infty) \to [0, \infty)$ is the **excitation kernel** satisfying $\int_0^\infty h(s)\, ds < 1$ (subcriticality condition). The process is self-exciting: each event increases the probability of future events.

A Hawkes process does not have stationary increments (conditional on the history, increments are non-stationary) and is not a Lévy process.

**What spxa tracks for Hawkes processes:**

- The mean intensity: $\bar{\lambda} = \mu / (1 - \|h\|_1)$ where $\|h\|_1 = \int_0^\infty h(s)\, ds$
- The variance of $N_t$: $\text{Var}(N_t) \approx \bar{\lambda} t \cdot (1 - \|h\|_1)^{-1}$ for large $t$ (asymptotic result, Hawkes & Oakes 1974)
- The Fano factor: $F = \text{Var}(N_t) / \mathbb{E}[N_t]$, which is always $\geq 1$, measuring burstiness
- Stationarity of the intensity in the stationary regime (when $t \to \infty$)

**spxa's exactness for Hawkes:** `MOMENT_PROPAGATION`. Simulation uses Ogata's thinning algorithm.

## OU-Lévy processes

**Definition** (Barndorff-Nielsen & Shephard 2001). The Ornstein–Uhlenbeck process driven by a Lévy subordinator $L$ is defined by the SDE

$$dV_t = -\lambda V_t\, dt + dL_{\lambda t}$$

with $\lambda > 0$ and $L$ a subordinator (non-decreasing Lévy process). The solution is

$$V_t = e^{-\lambda t} V_0 + \int_0^t e^{-\lambda(t-s)}\, dL_{\lambda s}$$

Unlike a Lévy process, $V$ does not have independent increments (the decay structure creates serial dependence). However, it does have a **stationary distribution** which is infinitely divisible, with a known Lévy–Khintchine triplet. The stationary distribution of $V$ has Lévy measure $\nu_V(dx) = \int_x^\infty \nu_L(dy)/y$ (Barndorff-Nielsen & Shephard 2001, Proposition 2.1).

**What spxa tracks for OU-Lévy:**

- The stationary distribution exactly (triplet is known)
- The autocovariance: $\text{Cov}(V_t, V_{t+h}) = e^{-\lambda h} \text{Var}(V_\infty)$
- The mean reversion rate $\lambda$
- The marginal moments exactly from the stationary distribution's triplet

**spxa's exactness for OU-Lévy:** `MOMENT_PROPAGATION` for path-level operations; the stationary distribution is `EXACT`.

## CARMA processes

**Definition** (Brockwell 2001). A Lévy-driven CARMA(p, q) process solves, formally, $a(D)Y_t = b(D)\,DL_t$ with

$$a(z) = z^p + a_1z^{p-1} + \dots + a_p, \qquad b(z) = b_0 + b_1z + \dots + b_qz^q, \qquad q < p .$$

It is made precise by a state-space form. With $A$ the companion matrix of $a$, $e_p$ the last unit vector and $b = (b_0, \dots, b_q, 0, \dots, 0)$:

$$dX_t = AX_t\,dt + e_p\,dL_t, \qquad Y_t = b^\top X_t .$$

CARMA(1, 0) is the Lévy-driven Ornstein–Uhlenbeck process. Higher orders give autocovariances that are sums of damped exponentials and damped oscillations, matching the shapes of ARMA models in continuous time and on irregular grids.

**Stationary version.** If every zero $\lambda_i$ of $a$ has negative real part, then

$$Y_t = \int_{-\infty}^t g(t-s)\,dL_s, \qquad g(t) = b^\top e^{At}e_p = \sum_i \frac{b(\lambda_i)}{a'(\lambda_i)}\,e^{\lambda_it}$$

(for distinct zeros). Each jump of $L$ adds a copy of the kernel $g$ to the path. Everything stationary follows from $g$:

- $E[Y] = \kappa_1(L_1)\,b_0/a_p$;
- $\gamma(h) = \kappa_2(L_1)\,b^\top e^{A|h|}\Sigma b$, where $A\Sigma + \Sigma A^\top + e_pe_p^\top = 0$;
- $f(\omega) = \frac{\kappa_2(L_1)}{2\pi}|b(i\omega)|^2/|a(i\omega)|^2$;
- $\kappa_n(Y) = \kappa_n(L_1)\int_0^\infty g(s)^n\,ds$, which is a finite sum of residues: $\int_0^\infty g^n = \sum_{i_1..i_n}\prod_k c_{i_k}\big/\big(-\sum_k\lambda_{i_k}\big)$;
- $\log E[e^{iuY}] = \int_0^\infty \psi_L(u\,g(s))\,ds$.

The second-order structure sees the driver only through $\kappa_2(L_1)$. Jumps show up in the higher cumulants and in the paths.

**What spxa tracks for CARMA:** all of the above exactly (`CARMAProcess`). CARMA(1, 0) reproduces the OU class to $10^{-14}$. The spectral density integrates to $\gamma(0)$, and its cosine transform gives $\gamma(h)$ to $10^{-7}$ on a finite grid. The residue sums match quadrature, and with a Brownian driver the characteristic function matches the Gaussian closed form to $10^{-9}$.

**Simulation.** With a Brownian driver, the discretised state $X_{k+1} = e^{A\Delta}X_k + m_\Delta + \varepsilon_k$ is exact, with $\varepsilon_k \sim N(0, \sigma^2(\Sigma - e^{A\Delta}\Sigma e^{A^\top\Delta}))$. Any other driver is simulated on sub-intervals, each increment weighted by the kernel averaged over its sub-interval. This keeps the mean exact and makes the variance error $O(\delta^2)$. One step of a Gamma-driven CARMA(2, 1) has a variance error of $-10\%$ with one substep, $-1.6\%$ with two, and below 1% from four on.

**spxa's exactness for CARMA:** `MOMENT_PROPAGATION`. There is no triplet, because the increments are dependent.

**Estimation.** Sampled CARMA data form a linear state-space model with exact transitions. `fit_carma` fits it by the Kalman-filter Gaussian quasi-likelihood, on even or irregular grids; see [Calibration §3.4](calibration.md#34-carma-processes-an-exact-kalman-filter).

## Degradation policy

When an operation mixes process types or leaves the Lévy regime, spxa:

1. Sets `ExactnessLevel` to the lowest level among operands
2. Issues a `SpxaDegradationWarning` with a message stating: which operation caused the degradation, what mathematical property was lost, and what is still being tracked exactly
3. Raises `ExactnessError` if a method requiring a higher exactness level is called on the result (e.g., calling `.triplet` on a fBM + Gamma sum)

This design ensures the library never silently returns a wrong answer by pretending an approximate result is exact.

## References

- Mandelbrot, B. B. & Van Ness, J. W. (1968). Fractional Brownian motions, fractional noises and applications. *SIAM Review*, 10(4), 422–437.
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83–90.
- Hawkes, A. G. & Oakes, D. (1974). A cluster process representation of a self-exciting process. *Journal of Applied Probability*, 11(3), 493–503.
- Barndorff-Nielsen, O. E. & Shephard, N. (2001). Non-Gaussian Ornstein–Uhlenbeck-based models and some of their uses in financial economics. *Journal of the Royal Statistical Society B*, 63(2), 167–241.
- Brockwell, P. J. (2001). Lévy-driven CARMA processes. *Annals of the Institute of Statistical Mathematics*, 53(1), 113–124.
- Brockwell, P. J., Davis, R. A. & Yang, Y. (2011). Estimation for non-negative Lévy-driven CARMA processes. *Journal of Business & Economic Statistics*, 29(2), 250–259.
- Applebaum, D. (2009). *Lévy Processes and Stochastic Calculus*, 2nd ed. Cambridge University Press. Chapter 4.
