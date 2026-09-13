# Triplet Arithmetic Under Composition

This document derives the triplet rules that underpin spxa's operator overloading. Every formula here has a direct code counterpart in `spxa/ops/`.

## Addition of independent Lévy processes

**Theorem** (Sato 1999, Proposition 11.10). Let $X$ and $Y$ be independent Lévy processes on $\mathbb{R}^d$ with triplets $(b_X, \Sigma_X, \nu_X)$ and $(b_Y, \Sigma_Y, \nu_Y)$ respectively. Then $Z = X + Y$ is a Lévy process with triplet

$$b_Z = b_X + b_Y, \quad \Sigma_Z = \Sigma_X + \Sigma_Y, \quad \nu_Z = \nu_X + \nu_Y$$

**Proof sketch.** The characteristic exponent is additive: $\psi_Z(u) = \psi_X(u) + \psi_Y(u)$, which follows directly from independence and the definition of the characteristic function. Reading off the triplet components from the Lévy–Khintchine form of $\psi_X + \psi_Y$ gives the result.

This is the core fact that makes spxa's `X + Y` exact. No approximation is involved for independent Lévy processes.

## Scalar multiplication

**Theorem.** If $X$ has triplet $(b, \sigma^2, \nu)$ and $c \in \mathbb{R}$, then $Z = cX$ has triplet

$$b_Z = cb + \int_{\mathbb{R}} x\left(\mathbf{1}_{\|x\| \leq 1} - \mathbf{1}_{\|cx'\| \leq 1}\right) \nu(dx'/c), \quad \sigma^2_Z = c^2 \sigma^2, \quad \nu_Z(B) = \nu(B/c)$$

In the one-dimensional case with the standard truncation, the drift transforms as

$$b_Z = cb + \int_1^{1/|c|} x\, \mathrm{sgn}(c)\, \nu(dx) \cdot \mathbf{1}_{|c| \neq 1}$$

when $|c| < 1$, and symmetrically when $|c| > 1$. The image measure $\nu_Z(B) = \nu(B/c)$ is exact in all cases.

**Derivation.** The characteristic function of $cX_t$ is $\mathbb{E}[e^{iucX_t}] = e^{t\psi_X(cu)}$. Substituting $cu$ for $u$ in the Lévy–Khintchine formula and changing variables $x \mapsto x/c$ in the integral gives the result.

## Linear transformation

**Theorem.** If $X$ is a $d$-dimensional Lévy process with triplet $(b, \Sigma, \nu)$ and $A : \mathbb{R}^d \to \mathbb{R}^k$ is a linear map, then $Z = AX$ is a $k$-dimensional Lévy process with triplet

$$b_Z = Ab + \int_{\mathbb{R}^d} \left(Ax\, \mathbf{1}_{\|Ax\| \leq 1} - Ax\, \mathbf{1}_{\|x\| \leq 1}\right) \nu(dx), \quad \Sigma_Z = A\Sigma A^\top, \quad \nu_Z = \nu \circ A^{-1}$$

where $(\nu \circ A^{-1})(B) = \nu(A^{-1}(B))$ is the image measure.

## Subordination

**Definition.** Let $X = (X_t)_{t \geq 0}$ be a Lévy process on $\mathbb{R}^d$ (the **parent process**) and $T = (T_t)_{t \geq 0}$ be an independent **subordinator** — a non-decreasing Lévy process with $T_0 = 0$. The **subordinated process** is $Z_t = X_{T_t}$.

A subordinator has Laplace exponent $\phi(\lambda) = -\log \mathbb{E}[e^{-\lambda T_1}]$, which is a Bernstein function: $\phi(\lambda) = \delta \lambda + \int_0^\infty (1 - e^{-\lambda s})\, \rho(ds)$ with $\delta \geq 0$ and $\rho$ the Lévy measure of $T$ (a measure on $(0, \infty)$ with $\int (1 \wedge s)\, \rho(ds) < \infty$).

**Theorem** (Sato 1999, Theorem 30.1; Cont & Tankov 2004, Proposition 4.4). $Z = X \circ T$ is a Lévy process with characteristic exponent

$$\psi_Z(u) = -\phi(-\psi_X(u)) = \delta \psi_X(u) + \int_0^\infty \left(e^{s\psi_X(u)} - 1\right) \rho(ds)$$

The triplet of $Z$ in one dimension is:

$$b_Z = \delta b_X + \int_0^\infty \left(\int_{\mathbb{R}} x\, e^{s\psi_X}\, \text{(via inversion)}\right) \rho(ds)$$

In practice, computing the triplet of $Z$ explicitly requires either a closed-form inversion (available for named process pairs like BM-under-Gamma = Variance Gamma) or numerical evaluation of the Lévy measure via

$$\nu_Z(B) = \delta \nu_X(B) + \int_0^\infty P(X_s \in B)\, \rho(ds)$$

spxa computes subordination exactly for known pairs (documented per process) and falls back to numerical quadrature otherwise, with `ExactnessLevel.MOMENT_PROPAGATION` and an explicit warning.

**The `@` operator in spxa.** `Z = X @ T` denotes subordination: $Z_t = X_{T_t}$. The left operand is the parent process, the right operand must be a subordinator.

## Cumulant propagation under composition

For any operation producing a Lévy process, cumulants follow from the triplet:

$$\kappa_n(Z_t) = t \cdot \kappa_n(Z_1) = t \cdot \int x^n\, \nu_Z(dx) + t \cdot \sigma_Z^2 \cdot \mathbf{1}_{n=2}$$

This is computed symbolically by spxa when $\nu_Z$ has a closed-form density, and numerically otherwise with a stated error bound.

## Property propagation

Operations propagate properties through the lattice defined in `spxa/core/properties.py`. Key rules:

| Property | Under $X + Y$ | Under $cX$ | Under $X \circ T$ |
|---|---|---|---|
| Stationary increments | preserved | preserved | preserved |
| Independent increments | preserved | preserved | preserved |
| Finite variance | preserved if both have it | preserved | depends on $T$ |
| Finite mean | preserved if both have it | preserved | preserved if $T$ has finite mean |
| Self-similarity index $H$ | lost unless $X = Y$ | preserved | lost in general |
| Martingale | preserved if both are | preserved if $c \neq 0$ and $b=0$ | depends |
| Tail index $\alpha$ | $\min(\alpha_X, \alpha_Y)$ | preserved | depends on $T$ |

Cells marked "depends" cause spxa to check the specific triplet values rather than propagating symbolically.

## What breaks outside the Lévy regime

The above arithmetic is **exact only for independent Lévy processes**. The following cases require approximation:

- **Fractional Brownian motion**: not a semimartingale, not a Lévy process, no triplet. The Hurst index $H$ is the relevant parameter; self-similarity and long memory are propagated heuristically.
- **Products $X \cdot Y$**: not a Lévy process in general even for Lévy $X$, $Y$. spxa computes moments of the product when $X$ and $Y$ are independent using the moment product formula.
- **Stochastic integrals $\int f(X_s)\, dY_s$**: exact only for deterministic $f$; for random integrands, Itô isometry gives the variance and spxa propagates moments only.

In all approximate cases, `ExactnessLevel` is set to `MOMENT_PROPAGATION` or `SIMULATION_ONLY`, and calling `.triplet` raises `ExactnessError` with an explanation.

## References

- Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*. Cambridge University Press. Proposition 11.10, Theorem 30.1.
- Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC. Propositions 4.1, 4.4.
- Schilling, R., Song, R. & Vondraček, Z. (2012). *Bernstein Functions*, 2nd ed. De Gruyter. Chapters 3, 5.
