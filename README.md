# spxa — Stochastic Process Algebra

**spxa** is a Python library where stochastic processes are first-class algebraic objects.

```python
from spxa.zoo.levy import VarianceGamma, BrownianMotion, GammaProcess
from spxa.zoo.beyond import FractionalBrownianMotion

X = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
Y = GammaProcess(a=1.0, b=2.0)

Z = X + 0.5 * Y          # exact: triplets add
W = X @ Y                 # exact: subordination via Bernstein functions

Z.triplet                 # LevyTriplet(b=..., sigma=..., nu=...)
Z.cumulants(order=4)      # exact symbolic cumulants
Z.char_func(u=1.0)        # characteristic function E[e^{iuZ_t}]
Z.__story__()             # human-readable derivation of Z's properties
```

Write `Z = X + c*Y`, get back a new process with its Lévy–Khintchine triplet computed exactly, cumulants derived symbolically, and a property lattice tracking what remains true (stationarity of increments, martingale property, tail class, self-similarity index) and what was invalidated by the operation. The library is loudly honest when you leave the Lévy regime, degrading gracefully to moment propagation rather than silently lying.

## Why spxa

Every stochastic modeling library treats processes as simulation engines. spxa treats them as algebraic objects. The Lévy–Khintchine bijection — between Lévy processes and infinitely divisible distributions — means triplet arithmetic is exact for independent Lévy processes. This is a mathematical fact that no existing software exploits compositionally.

Researchers in quantitative finance, statistical physics, and computational biology currently re-derive combinations by hand each time, or simulate everything at the cost of closed-form insight. spxa closes that gap.

## Core guarantees

- Every operation returns a new immutable process object — no mutation
- Every process carries an `ExactnessLevel`: `EXACT`, `MOMENT_PROPAGATION`, or `SIMULATION_ONLY`
- Operations between exactness levels degrade to the lower level with an explicit warning and mathematical reason
- The property lattice propagates automatically and never makes silent incorrect claims
- `tests/exactness/` verifies closed-form results against published formulas and blocks any PR that breaks them

## Installation

```bash
pip install spxa
```

Requires Python ≥ 3.10.

## Documentation

- [Theory: Lévy–Khintchine and triplet arithmetic](docs/theory/levy_khintchine.md)
- [Theory: Triplet arithmetic under composition](docs/theory/triplet_arithmetic.md)
- [Theory: Beyond the Lévy regime](docs/theory/beyond_levy.md)
- [Tutorial: First composition](docs/tutorials/01_first_composition.ipynb)
- [Tutorial: Finance models](docs/tutorials/02_finance_models.ipynb)
- [Tutorial: Physics applications](docs/tutorials/03_physics_applications.ipynb)
- [Process registry](PROCESS_REGISTRY.md)

## Process zoo

See [PROCESS_REGISTRY.md](PROCESS_REGISTRY.md) for the full table of implemented processes with exactness levels, available cumulants, simulation algorithms, and primary references.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contribution tiers:

- **New process to the zoo**: requires triplet, cumulants, one primary citation
- **New operation**: requires exactness proof or explicit approximation justification with cited error bounds
- **Changes to `core/`**: requires RFC document and two maintainer approvals

## License

MIT
