# Lévy–Khintchine Formula and the Triplet

## The central bijection

A stochastic process $X = (X_t)_{t \geq 0}$ is a **Lévy process** if it satisfies:

1. $X_0 = 0$ almost surely
2. Independent increments: for $0 \leq s < t$, $X_t - X_s$ is independent of $\mathcal{F}_s$
3. Stationary increments: $X_t - X_s \sim X_{t-s}$ in distribution
4. Stochastic continuity: $\lim_{s \to t} P(|X_s - X_t| > \varepsilon) = 0$ for all $\varepsilon > 0$
5. Càdlàg paths: right-continuous with left limits almost surely

The foundational result (Lévy 1934, Khintchine 1938) is a **bijection**:

$$\text{Lévy processes on } \mathbb{R}^d \longleftrightarrow \text{Infinitely divisible distributions on } \mathbb{R}^d$$

A distribution $\mu$ is infinitely divisible if for every $n \in \mathbb{N}$ there exist i.i.d. $X_1, \ldots, X_n$ with $X_1 + \cdots + X_n \sim \mu$.

Given a Lévy process $X$, the distribution of $X_1$ is infinitely divisible. Conversely, given any infinitely divisible distribution $\mu$, there exists a unique (in law) Lévy process with $\mathcal{L}(X_1) = \mu$.

## The Lévy–Khintchine formula

Every Lévy process $X$ on $\mathbb{R}^d$ has characteristic function

$$\mathbb{E}\left[e^{i\langle u, X_t \rangle}\right] = e^{t \psi(u)}, \quad u \in \mathbb{R}^d$$

where the **characteristic exponent** $\psi : \mathbb{R}^d \to \mathbb{C}$ takes the form

$$\psi(u) = i\langle b, u \rangle - \frac{1}{2}\langle u, \Sigma u \rangle + \int_{\mathbb{R}^d} \left(e^{i\langle u, x \rangle} - 1 - i\langle u, x \rangle \mathbf{1}_{\|x\| \leq 1}\right) \nu(dx)$$

The triple $(b, \Sigma, \nu)$ is the **Lévy–Khintchine triplet** (or characteristic triplet) of $X$, where:

- $b \in \mathbb{R}^d$ is the **drift vector**
- $\Sigma \in \mathbb{R}^{d \times d}$ is a symmetric positive semidefinite **diffusion matrix**; in one dimension this is $\sigma^2 \geq 0$
- $\nu$ is the **Lévy measure** on $\mathbb{R}^d \setminus \{0\}$, satisfying $\int (1 \wedge \|x\|^2)\, \nu(dx) < \infty$

The representation is **unique**: two Lévy processes have the same law if and only if their triplets are identical (Sato 1999, Theorem 8.1).

## The Lévy–Itô decomposition

The triplet has a constructive interpretation. Every Lévy process decomposes as

$$X_t = bt + \sigma W_t + \int_{\|x\| > 1} x\, N(ds, dx) \cdot t + \int_{\|x\| \leq 1} x\, \tilde{N}(ds, dx)$$

where $W$ is a standard Brownian motion, $N$ is the Poisson random measure of jumps of $X$, and $\tilde{N}(ds, dx) = N(ds, dx) - \nu(dx)\, ds$ is the compensated measure. The four components are independent.

The Lévy measure $\nu(B) = \mathbb{E}[\#\{t \in [0,1] : \Delta X_t \in B\}]$ counts the expected number of jumps per unit time of size in $B$. It encodes the entire jump structure of the process.

## The truncation function

The centering term $\mathbf{1}_{\|x\| \leq 1}$ is a choice of **truncation function**. Different choices give different values of $b$ but the same $\Sigma$ and $\nu$, so the triplet is only canonical relative to a fixed truncation. spxa uses $h(x) = \mathbf{1}_{\|x\| \leq 1}$ throughout. When accepting external triplets from the literature, verify the truncation convention.

## Cumulants from the triplet

The $n$-th cumulant of $X_t$ (when it exists) is $t$ times the $n$-th cumulant of $X_1$. For $X_1$:

$$\kappa_1 = b + \int_{\|x\| > 1} x\, \nu(dx)$$

$$\kappa_2 = \sigma^2 + \int x^2\, \nu(dx)$$

$$\kappa_n = \int x^n\, \nu(dx), \quad n \geq 3$$

These exist if and only if $\int_{\|x\| > 1} \|x\|^n\, \nu(dx) < \infty$. The Lévy measure's tail behavior completely controls the existence of moments.

## Infinite divisibility of the triplet

The set of valid triplets is closed under:

- Convex combination of Lévy measures: $\nu = \lambda \nu_1 + (1-\lambda) \nu_2$ with $\lambda \in [0,1]$
- Scaling of $b$ and $\sigma^2$ by positive constants
- Image measures: if $T : \mathbb{R}^d \to \mathbb{R}^k$ is linear, the image of a Lévy process under $T$ is a Lévy process with triplet $(Tb, T\Sigma T^\top, \nu \circ T^{-1})$

These are the mathematical facts that make spxa's arithmetic exact.

## References

- Lévy, P. (1934). Sur les intégrales dont les éléments sont des variables aléatoires indépendantes. *Ann. Scuola Norm. Sup. Pisa*, 3, 337–366.
- Khintchine, A. (1938). Limit laws of sums of independent random variables. *ONTI*, Moscow.
- Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*. Cambridge University Press. Theorems 7.10, 8.1.
- Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC. Chapter 4.
- Applebaum, D. (2009). *Lévy Processes and Stochastic Calculus*, 2nd ed. Cambridge University Press. Chapter 1.
