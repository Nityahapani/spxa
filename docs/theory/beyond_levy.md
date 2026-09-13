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
- Applebaum, D. (2009). *Lévy Processes and Stochastic Calculus*, 2nd ed. Cambridge University Press. Chapter 4.
