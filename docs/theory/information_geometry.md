# Information Geometry of Statistical Models

A parametric family of probability laws $\{p(\cdot\,;\theta) : \theta \in \Theta \subset \mathbb{R}^n\}$ is a smooth manifold. If the score vectors are linearly independent, it carries a canonical Riemannian metric, the **Fisher information metric**. Because the metric is invariant under reparametrisation, distances, geodesics and curvature describe the model and do not depend on the coordinates chosen. `spxa.geom` provides these objects for the laws of stochastic processes. They are the foundation of geodesic model morphing ([Model Morphing §3](model_morphing.md#3-information-geometric-geodesics)) and natural-gradient calibration.

| Object | Implementation |
| :--- | :--- |
| Fisher metric $g_{ij}(\theta)$ | `spxa.geom.FisherInformationMetric` |
| Levi-Civita and Amari $\alpha$-connections, curvature | `spxa.geom.AffineConnection` |
| Geodesics, exponential map, Rao distance | `spxa.geom.GeodesicFlow` |
| Natural-gradient descent on the DAG | `spxa.graph.geom.NaturalGradientCalibratorNode` |

---

## 1. The Fisher information metric

Let $\ell(\theta) = \log p(X;\theta)$. Under the usual regularity conditions (differentiation under the integral sign is allowed), the score has mean zero and

$$
g_{ij}(\theta) \;=\; \mathbb{E}_\theta\!\left[\partial_i \ell \,\partial_j \ell\right]
\;=\; -\,\mathbb{E}_\theta\!\left[\partial_i \partial_j \ell\right].
$$

$g$ is symmetric positive semi-definite, and positive definite exactly when the model is locally identifiable. By the **Čencov theorem**, it is (up to scale) the only Riemannian metric on statistical manifolds that is invariant under sufficient statistics (Markov morphisms). `FisherInformationMetric.is_positive_definite` checks the identifiability condition numerically.

### 1.1 Univariate Gaussian

In coordinates $\theta = (\mu, \sigma)$:

$$
g(\mu,\sigma) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 2/\sigma^2 \end{pmatrix},
\qquad
ds^2 = \frac{d\mu^2 + 2\,d\sigma^2}{\sigma^2}.
$$

Substituting $\mu = \sqrt{2}\,x$ gives $ds^2 = 2\,(dx^2 + d\sigma^2)/\sigma^2$, which is twice the Poincaré half-plane metric. The manifold is therefore a hyperbolic plane with constant Gaussian curvature $K = -\tfrac12$ and scalar curvature $R = 2K = -1$. `AffineConnection.ricci_scalar` reproduces $R = -1$ to about $10^{-7}$ at every point.

### 1.2 Gaussian transition densities (Ornstein–Uhlenbeck)

For any model whose likelihood is Gaussian, $X \sim \mathcal{N}(m(\theta), v(\theta))$, the Fisher metric is

$$
g_{ij} = \frac{\partial_i m\,\partial_j m}{v} + \frac{\partial_i v\,\partial_j v}{2v^2}.
$$

`FisherInformationMetric.ou_transition_fisher` applies this formula to the exact OU transition $X_{\Delta}\mid X_0 = x_0$ of $dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t$, where

$$
m = x_0 e^{-\theta\Delta} + \mu(1 - e^{-\theta\Delta}),\qquad
v = \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta\Delta}\right),
$$

in coordinates $(\theta, \mu, \sigma)$. The information of $N$ independent transitions is the sum of the per-transition matrices.

### 1.3 Empirical and observed information

When no closed form exists:

* `empirical_fisher(scores)` returns $\frac1N\sum_k s_k s_k^\top$ with $s_k = \nabla_\theta \log p(x_k;\theta)$, the outer-product-of-gradients estimator.
* `numerical_hessian_fisher(nll, theta)` returns the central-difference Hessian of the negative log-likelihood. This is the observed information, which is consistent for $g$ at the MLE.

If the model is misspecified the two estimators differ, and the sandwich covariance of [Calibration §4](calibration.md#4-inference-standard-errors-and-model-selection) should be used.

---

## 2. Connections and curvature

### 2.1 Levi-Civita connection

The Christoffel symbols of the second kind are

$$
\Gamma^k_{ij} = \tfrac12\, g^{kl}\left(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}\right).
$$

`AffineConnection(metric_fn, dim)` computes them by central differences of a user-supplied metric $\theta \mapsto g(\theta)$. The result has shape `(dim, dim, dim)`, and index `(k, i, j)` is $\Gamma^k_{ij}$.

### 2.2 Amari $\alpha$-connections

Statistical manifolds carry a one-parameter family of torsion-free connections (Amari & Nagaoka 2000):

$$
\Gamma^{(\alpha),k}_{ij} = \Gamma^k_{ij} - \frac{\alpha}{2}\, g^{kl}\, T_{ijl},
\qquad T_{ijl} = \mathbb{E}_\theta\!\left[\partial_i \ell\,\partial_j \ell\,\partial_l \ell\right].
$$

* $\alpha = 0$ gives the Levi-Civita connection, which is metric-compatible.
* $\alpha = 1$ gives the **e-connection**, which is flat for exponential families. Natural parameters are affine coordinates.
* $\alpha = -1$ gives the **m-connection**, which is flat for mixture families. Expectation parameters are affine coordinates.

The $\pm\alpha$ connections are dual with respect to $g$: $\partial_k g_{ij} = \Gamma^{(\alpha)}_{ki,j} + \Gamma^{(-\alpha)}_{kj,i}$. This duality underlies the generalised Pythagorean theorem for KL divergence. Pass the skewness tensor $T$ to `amari_alpha_connection(theta, alpha, skewness_tensor)`.

### 2.3 Curvature

$$
R^l{}_{ijk} = \partial_j \Gamma^l_{ik} - \partial_k \Gamma^l_{ij} + \Gamma^m_{ik}\Gamma^l_{jm} - \Gamma^m_{ij}\Gamma^l_{km},
\qquad
R_{ik} = R^l{}_{ilk},\qquad R = g^{ik}R_{ik}.
$$

Curvature measures how far the model is from being flat. With negative curvature, as on the Gaussian manifold, geodesics diverge. Linear interpolation of parameters then differs noticeably from the geodesic, and the Fréchet mean of several models is unique.

---

## 3. Geodesics and Rao distance

A geodesic $\theta(t)$ solves

$$
\ddot\theta^k + \Gamma^k_{ij}(\theta)\,\dot\theta^i\dot\theta^j = 0 .
$$

`GeodesicFlow` integrates this equation as a first-order system in $(\theta, \dot\theta)$ using adaptive RK45.

* **Exponential map.** `exponential_map(theta0, v0)` returns $\operatorname{Exp}_{\theta_0}(v_0) = \theta(1)$ together with the whole trajectory.
* **Logarithm by shooting.** `geodesic_shooting(theta0, theta1)` solves the two-point boundary value problem $\theta(0)=\theta_0,\ \theta(1)=\theta_1$. It minimises $\|\operatorname{Exp}_{\theta_0}(v) - \theta_1\|^2$ over $v$, starting from the Euclidean guess $v = \theta_1 - \theta_0$.
* **Rao distance.**
  $$d_R(\theta_0,\theta_1) = \int_0^1 \sqrt{g_{ij}(\theta(t))\,\dot\theta^i\dot\theta^j}\,dt .$$
  Speed is conserved along a geodesic, so this equals $\sqrt{v_0^\top g(\theta_0) v_0}$. The trapezoidal integral therefore also measures the integrator's accuracy.

### Closed form on the Gaussian manifold

Using the half-plane isometry of §1.1,

$$
d_R\big((\mu_0,\sigma_0),(\mu_1,\sigma_1)\big)
= \sqrt2\,\ln\frac{1+\delta}{1-\delta} = 2\sqrt2\,\operatorname{artanh}\delta,
\qquad
\delta = \sqrt{\frac{(\mu_1-\mu_0)^2 + 2(\sigma_1-\sigma_0)^2}{(\mu_1-\mu_0)^2 + 2(\sigma_1+\sigma_0)^2}} .
$$

This is `GeodesicFlow.gaussian_exact_rao_distance` (and `spxa.morph.gaussian_rao_distance`). Geodesics are semi-ellipses centred on the $\mu$-axis (vertical lines when $\mu_0=\mu_1$). Along them a change of mean is "paid for" by a temporary increase in volatility.

```python
import numpy as np
from spxa.geom import FisherInformationMetric, AffineConnection, GeodesicFlow

conn = AffineConnection(lambda th: FisherInformationMetric.gaussian_fisher(th[0], th[1]), dim=2)
flow = GeodesicFlow(conn)

print(conn.ricci_scalar(np.array([0.3, 1.2])))                        # ≈ -1.0
print(flow.rao_distance(np.array([0.0, 1.0]), np.array([1.0, 2.0])))  # ≈ 1.18936 (shooting)
print(GeodesicFlow.gaussian_exact_rao_distance((0.0, 1.0), (1.0, 2.0)))  # 1.18938 (closed form)
```

---

## 4. Natural-gradient calibration

For a loss $\mathcal{L}(\theta)$ on a statistical manifold, the direction of steepest descent with respect to the Fisher metric is the natural gradient (Amari 1998):

$$
\tilde\nabla_\theta \mathcal{L} = g(\theta)^{-1}\nabla_\theta \mathcal{L},
\qquad
\theta_{k+1} = \theta_k - \eta\,(g(\theta_k) + \lambda I)^{-1}\nabla_\theta \mathcal{L}(\theta_k).
$$

The update is invariant to smooth reparametrisation, to first order in $\eta$. When $\mathcal{L}$ is a negative log-likelihood, the natural gradient step with $\eta = 1$ is Fisher scoring. `NaturalGradientCalibratorNode` implements the damped update above. It takes Tikhonov damping $\lambda$, central-difference Euclidean gradients and parameter-bound clipping, and works on the `ParameterNode`s of a computational graph.

---

## References

* Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*, 10(2), 251–276.
* Amari, S. & Nagaoka, H. (2000). *Methods of Information Geometry*. AMS / Oxford University Press.
* Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
* Atkinson, C. & Mitchell, A. F. S. (1981). Rao's distance measure. *Sankhyā A*, 43, 345–365.
* Čencov, N. N. (1982). *Statistical Decision Rules and Optimal Inference*. AMS.
* Rao, C. R. (1945). Information and the accuracy attainable in the estimation of statistical parameters. *Bull. Calcutta Math. Soc.*, 37, 81–91.
