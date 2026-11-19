# spxa

Stochastic processes as first-class algebraic objects. Exact Lévy–Khintchine
arithmetic, a differentiable computational graph, and reverse-mode Greeks.

[![CI](https://github.com/Nityahapani/spxa/actions/workflows/ci.yml/badge.svg)](https://github.com/Nityahapani/spxa/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/spxa)](https://pypi.org/project/spxa/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/spxa/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/style-ruff-000000)](https://github.com/astral-sh/ruff)

spxa is a Python library for quantitative research. It treats continuous-time
stochastic processes as first-class algebraic objects: compose them with `+`,
`*`, `@`, and their Lévy–Khintchine triplets add in closed form. A computational
graph engine compiles pricing pipelines and routes to analytical, Fourier, batch,
or Monte Carlo backends, with reverse-mode autodiff for machine-precision Greeks.

## Install

```bash
pip install spxa
```

Dev install:

```bash
git clone https://github.com/Nityahapani/spxa.git
cd spxa
pip install -e ".[dev]"
```

## Quickstart

```python
import spxa
from spxa import VarianceGamma, GammaProcess

X = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
Y = GammaProcess(a=1.0, b=2.0)
Z = X + 0.5 * Y            # exact triplet arithmetic

print(Z.exactness)          # ExactnessLevel.EXACT
print(Z.triplet.sigma_sq)   # 0.0
print(float(Z.triplet.b))   # 0.04072233534646699
print(Z.exactness == spxa.ExactnessLevel.EXACT)  # True
```

| Module | Purpose |
|--------|---------|
| `spxa.core` | Process algebra, Lévy–Khintchine triplets, exactness lattice |
| `spxa.zoo` | 69 stochastic processes |
| `spxa.graph` | Differentiable computational graph & autodiff tape |
| `spxa.diff` | Greeks, pricers, neural SDEs |
| `spxa.calibration` | MLE, GMM, Kalman/particle filters, surface calibration |
| `spxa.copula` | Archimedean, elliptical, extreme-value copulas |
| `spxa.control` | Merton portfolio, market making, HJB solvers |
| `spxa.pde` | Finite-difference PDE/PIDE solvers |
| `spxa.rates` / `spxa.market` | Curves, swaps, bonds, bootstrapping |
| `spxa.vol` | SVI/SSVI surfaces, local vol, rough vol |
| `spxa.rough` | Fractional volatility, rBergomi |
| `spxa.morph` | Model morphing & barycenters |
| `spxa.geom` | Information geometry |
| `spxa.signature` | Rough path signatures |

## Docs

- [API reference](https://nityahapani.github.io/spxa)
- [Process registry](https://github.com/Nityahapani/spxa/blob/main/PROCESS_REGISTRY.md)
- [Theory guides](docs/theory/)
- [Tutorials](docs/tutorials/)

## Citation

[![CITATION.cff](https://img.shields.io/badge/CITATION.cff-DOI-blue)](CITATION.cff)

```bibtex
@software{spxa,
  author = {Nityahapani},
  title = {spxa: Stochastic Process Algebra \& Computational Graph Engine},
  url = {https://github.com/Nityahapani/spxa},
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Tests and the full audit suite must pass:

```bash
python -m pytest tests/ -q
python audits/run_master_audit.py
```

## License

[MIT](LICENSE)
