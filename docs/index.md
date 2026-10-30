# spxa

**Stochastic processes as first-class algebraic objects.**

```python
from spxa.zoo.levy import VarianceGamma, GammaProcess

X = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
Y = GammaProcess(a=1.0, b=2.0)

Z = X + 0.5 * Y
Z.triplet          # exact Lévy–Khintchine triplet
Z.cumulants(4)     # exact symbolic cumulants
Z.__story__()      # derivation trace
```

## Where to start

- **[Theory: Lévy–Khintchine](theory/levy_khintchine.md)** — the bijection that makes arithmetic exact
- **[Theory: Triplet arithmetic](theory/triplet_arithmetic.md)** — the rules behind every operator
- **[Theory: Beyond Lévy](theory/beyond_levy.md)** — fBM, Hawkes, OU-Lévy and the degradation policy
- **[Theory: Model Morphing](theory/model_morphing.md)** — exact interpolation between processes along triplet, generator, Fisher–Rao and Wasserstein paths
- **[Theory: Information geometry](theory/information_geometry.md)** — Fisher metric, α-connections, curvature, Rao distance and natural gradient
- **[Theory: Rough paths & signatures](theory/rough_paths_signatures.md)** — tensor algebra, log-signatures, the signature kernel and expected signatures
- **[Theory: Rough volatility](theory/rough_volatility.md)** — Volterra fBM, rough Bergomi and the fractional Riccati equation of rough Heston
- **[Theory: Copulas](theory/copulas.md)** — Sklar, dependence measures, copula families, estimation and coupled simulation
- **[Theory: Stochastic control](theory/stochastic_control.md)** — HJB, LQG, Merton, Almgren–Chriss, Avellaneda–Stoikov and finite-difference HJB
- **[Theory: Estimation & calibration](theory/calibration.md)** — exact MLE, GMM/ECF, filtering, surface calibration and inference
- **[Process registry](https://github.com/Nityahapani/spxa/blob/main/PROCESS_REGISTRY.md)** — all implemented processes with exactness levels and references
- **[Contributing](contributing.md)** — how to add processes, operations, and core changes
