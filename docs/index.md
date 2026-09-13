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
- **[Process registry](process_registry.md)** — all implemented processes with exactness levels and references
- **[Contributing](contributing.md)** — how to add processes, operations, and core changes
