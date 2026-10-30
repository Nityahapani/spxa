# Rough Paths and Path Signatures

The **signature** of a path is the sequence of its iterated integrals. It is a graded, coordinate-free summary that determines the path up to tree-like equivalence (Hambly & Lyons 2010). It also linearises every continuous function of the path: any such function can be approximated by a linear functional of the signature. `spxa.signature` implements truncated signatures, the shuffle and Lie algebra structure, and the signature kernel computed by solving a Goursat PDE.

| Object | Implementation |
| :--- | :--- |
| Truncated signature $S^{(M)}(X)$, Chen products | `spxa.signature.PathSignature` |
| Shuffles, log-signature, Witt dimensions | `spxa.signature.SignatureAlgebra` |
| Signature kernel $\langle S(X), S(Y)\rangle$ | `spxa.signature.SignatureKernel` |
| Graph nodes | `spxa.graph.signature.PathSignatureNode`, `SignatureKernelNode`, `ExpectedSignatureNode`, `SignatureLinearPricerNode` |

---

## 1. The tensor algebra and the signature

Let $X:[0,T]\to\mathbb{R}^d$ have bounded variation. Its signature over $[s,t]$ is the element of the tensor algebra $T((\mathbb{R}^d)) = \prod_{k\ge0}(\mathbb{R}^d)^{\otimes k}$

$$
S(X)_{s,t} = \Big(1,\; \int_{s<u_1<t} dX_{u_1},\; \int_{s<u_1<u_2<t} dX_{u_1}\otimes dX_{u_2},\; \dots\Big).
$$

Coordinate $(i_1,\dots,i_k)$ at level $k$ is $S^{i_1\cdots i_k}_{s,t} = \int_{s<u_1<\dots<u_k<t} dX^{i_1}_{u_1}\cdots dX^{i_k}_{u_k}$. Truncating at depth $M$ leaves

$$
\dim T^{(M)}(\mathbb{R}^d) = \sum_{k=0}^M d^k = \frac{d^{M+1}-1}{d-1}
$$

coordinates (`PathSignature.signature_dimension`). Levels are stored as flattened arrays in row-major (lexicographic word) order. For example, `sig[2].reshape(d, d)[i, j]` is $S^{ij}$.

### 1.1 Linear segments

For a single linear segment with increment $\Delta$, every iterated integral can be computed in closed form:

$$
S(\Delta) = \exp_\otimes(\Delta) = \sum_{k\ge0}\frac{\Delta^{\otimes k}}{k!}.
$$

### 1.2 Chen's identity

Concatenating paths corresponds to multiplying their signatures in the tensor algebra:

$$
S(X)_{s,u} = S(X)_{s,t}\otimes S(X)_{t,u},
\qquad
(A\otimes B)^{(k)} = \sum_{j=0}^k A^{(j)}\otimes B^{(k-j)} .
$$

`PathSignature.signature` evaluates the signature of a piecewise-linear interpolation of the sample points exactly, as the product $\bigotimes_i \exp_\otimes(\Delta X_i)$. `verify_chen_identity` checks the identity numerically by splitting a path. No discretisation error is introduced. The only approximation is the piecewise-linear interpolation of the data.

### 1.3 Properties used in practice

* **Invariance under reparametrisation.** $S(X\circ\phi) = S(X)$ for increasing $\phi$, so the signature ignores the speed at which a path is traversed. To keep time information, add time as an extra coordinate: $\hat X_t = (t, X_t)$.
* **Uniqueness.** Up to tree-like equivalence, $S(X)$ determines $X$. For time-augmented paths it determines $X$ exactly.
* **Universality.** On compact sets of paths, linear functionals $\langle \ell, S(X)\rangle$ are dense in the continuous functions. This is why signature regression works for path-dependent payoffs.
* **Factorial decay.** $\|S^{(k)}(X)\| \le \|X\|_{1\text{-var}}^k / k!$, so truncation error decays super-exponentially in $M$.

---

## 2. Shuffle algebra and log-signature

### 2.1 Shuffle identity

For words $u, v$, the shuffle product $u \sqcup\!\sqcup\, v$ is the sum of all interleavings that keep the order within $u$ and within $v$ (`SignatureAlgebra.shuffles`). The signature is a *character* of the shuffle algebra:

$$
\langle S(X), u\rangle\,\langle S(X), v\rangle = \langle S(X), u \sqcup\!\sqcup\, v\rangle .
$$

In particular $S^i S^j = S^{ij} + S^{ji}$. Products of signature coordinates are therefore again linear in the signature, which is the algebraic basis of universality.

### 2.2 Group-like elements and the free Lie algebra

Signatures are *group-like*: $\log_\otimes S(X)$ lies in the free Lie algebra $\mathfrak{g}(\mathbb{R}^d)$, the space spanned by iterated commutators $[e_i, e_j] = e_i\otimes e_j - e_j\otimes e_i$. Up to level 3,

$$
\log S = S^{(1)} + \Big(S^{(2)} - \tfrac12 S^{(1)}\otimes S^{(1)}\Big)
+ \Big(S^{(3)} - \tfrac12\big(S^{(1)}\otimes S^{(2)} + S^{(2)}\otimes S^{(1)}\big) + \tfrac13 (S^{(1)})^{\otimes3}\Big) + \cdots
$$

For $d = 2$, `SignatureAlgebra.log_signature_2d` returns coordinates in the Hall basis:

| Level | Basis | Coordinate |
| :--- | :--- | :--- |
| 1 | $e_1, e_2$ | $S^1, S^2$ (total increment) |
| 2 | $[e_1,e_2]$ | Lévy area $A^{12} = \tfrac12(S^{12}-S^{21})$ |
| 3 | $[[e_1,e_2],e_1],\ [[e_1,e_2],e_2]$ | $\tfrac16\langle S^{(3)}, 2\cdot121 - 112 - 211\rangle,\ \tfrac16\langle S^{(3)}, 122 - 2\cdot212 + 221\rangle$ |

The level-3 formulas rely on two facts. First, $\langle S, P\rangle = \langle\log S, P\rangle$ for every Lie polynomial $P$. Second, the two level-3 basis polynomials are orthogonal, each with squared norm 6. The test suite checks the reconstruction $\log S^{(3)} = c_1[[e_1,e_2],e_1] + c_2[[e_1,e_2],e_2]$ against the tensor logarithm to machine precision.

### 2.3 Witt's formula

The dimension of the level-$k$ component of the free Lie algebra on $d$ generators is the necklace polynomial

$$
L_d(k) = \frac1k\sum_{m\mid k}\mu(m)\, d^{k/m},
$$

where $\mu$ is the Möbius function. `log_signature_dimension(d, M)` returns $\sum_{k\le M}L_d(k)$. For $d=2$ the level dimensions are $2,1,2,3,6,9,18,30,56,99,\dots$, so at depth 10 the log-signature has 226 coordinates against 2046 for the signature.

---

## 3. The signature kernel

For paths $X$ and $Y$, the signature kernel $K(X,Y) = \langle S(X), S(Y)\rangle$ (using the inner product induced by the Euclidean inner product on each level) solves a linear hyperbolic **Goursat problem** (Salvi, Cass, Foster, Lyons & Yang 2021):

$$
\frac{\partial^2 K}{\partial s\,\partial t}(s,t) = \langle \dot X_s, \dot Y_t\rangle\, K(s,t),
\qquad K(s,0) = K(0,t) = 1,
$$

and $K(X,Y) = K(T_X, T_Y)$. This is the *untruncated* inner product, and evaluating it costs $O(NM\,d)$ for paths with $N$ and $M$ segments.

### 3.1 Discretisation

For piecewise-linear paths, $\langle\dot X,\dot Y\rangle$ is constant on each grid cell, with cell value $M_{ij} = \langle\Delta X_i, \Delta Y_j\rangle$ after scaling to unit cells. Integrating the PDE over a cell gives the exact identity

$$
K_{i+1,j+1} - K_{i+1,j} - K_{i,j+1} + K_{i,j} = M_{ij}\cdot\overline{K}_{\text{cell}} .
$$

Approximating the cell mean $\overline K$ by the average of the four corners gives the implicit trapezoidal scheme implemented in `SignatureKernel.solve_goursat_pde`:

$$
K_{i+1,j+1} = \frac{K_{i+1,j} + K_{i,j+1} - K_{i,j} + \tfrac14 M_{ij}\,(K_{i,j}+K_{i+1,j}+K_{i,j+1})}{1 - \tfrac14 M_{ij}} .
$$

The scheme is second-order accurate. For one-dimensional linear paths $X_t = Y_t = t$ the exact kernel is $\sum_k 1/(k!)^2 = I_0(2)$. The solver gives $2.33$ on a single cell, $2.27994$ on 10 cells and $2.2795888$ on 100 cells, against $I_0(2) = 2.2795853$. For coarse samples with large increments, refine the paths dyadically before solving.

### 3.2 Derived quantities

* **Gram matrices.** `compute_gram_matrix(paths)` returns $G_{ij} = K(X^{(i)}, X^{(j)})$. $G$ is positive semi-definite, so it can be used directly in kernel ridge regression, SVMs, MMD two-sample tests and Gaussian processes on path space.
* **Feature-space distance.** `normalized_distance(X, Y)` returns
  $$\|S(X)-S(Y)\| = \sqrt{K(X,X) - 2K(X,Y) + K(Y,Y)} .$$
  This is a metric on tree-like equivalence classes of paths.

```python
import numpy as np
from spxa.signature import PathSignature, SignatureKernel

rng = np.random.default_rng(0)
x = np.cumsum(rng.normal(0, 0.1, (30, 2)), axis=0)
y = np.cumsum(rng.normal(0, 0.1, (40, 2)), axis=0)

ps = PathSignature(depth=10)
truncated = ps.flatten_signature(ps.signature(x)) @ ps.flatten_signature(ps.signature(y))
print(truncated, SignatureKernel.evaluate(x, y))   # agree to ~1e-4 relative
```

### 3.3 Expected signature and linear pricing

Universality (§1.3) means a path-dependent payoff can be approximated as $F(X) \approx \langle \ell, S^{(M)}(\hat X)\rangle$. Expectation is linear, so

$$
\mathbb{E}^{\mathbb{Q}}[F(X)] \approx \big\langle \ell,\ \mathbb{E}^{\mathbb{Q}}[S^{(M)}(\hat X)]\big\rangle .
$$

Once the *expected signature* has been computed from simulated or model paths (`ExpectedSignatureNode`), pricing any payoff in the span reduces to a dot product (`SignatureLinearPricerNode`). The weights $\ell$ are fitted by regression on sample paths. Under mild conditions the expected signature characterises the law of the path (Chevyrev & Lyons 2016).

---

## 4. Relation to rough paths

For Brownian paths and other paths with finite $p$-variation, $p \in [2,3)$, the iterated integrals at level 2 are not defined pathwise. Lyons' theory makes them so by *lifting* $X$ to a rough path $(X, \mathbb{X})$, where $\mathbb{X}_{s,t}$ plays the role of $S^{(2)}_{s,t}$ and Chen's identity is imposed as an axiom. Solutions of rough differential equations then depend continuously on the lift. For Brownian motion the Stratonovich lift $\mathbb{X}^{ij}_{s,t} = \int_s^t (W^i_u - W^i_s)\circ dW^j_u$ is the limit of the signatures of piecewise-linear interpolations. This limit is exactly what `PathSignature` computes on sampled data. The antisymmetric part of $\mathbb{X}$ is the Lévy area, the level-2 log-signature coordinate above.

---

## References

* Chevyrev, I. & Lyons, T. (2016). Characteristic functions of measures on geometric rough paths. *Annals of Probability*, 44(6), 4049–4082.
* Chen, K.-T. (1957). Integration of paths, geometric invariants and a generalized Baker–Hausdorff formula. *Annals of Mathematics*, 65(1), 163–178.
* Hambly, B. & Lyons, T. (2010). Uniqueness for the signature of a path of bounded variation and the reduced path group. *Annals of Mathematics*, 171(1), 109–167.
* Lyons, T. (1998). Differential equations driven by rough signals. *Revista Matemática Iberoamericana*, 14(2), 215–310.
* Reutenauer, C. (1993). *Free Lie Algebras*. Oxford University Press.
* Salvi, C., Cass, T., Foster, J., Lyons, T. & Yang, W. (2021). The signature kernel is the solution of a Goursat PDE. *SIAM Journal on Mathematics of Data Science*, 3(3), 873–899.
* Friz, P. & Hairer, M. (2020). *A Course on Rough Paths* (2nd ed.). Springer.
