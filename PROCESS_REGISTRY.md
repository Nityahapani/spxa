# Process Registry

Every process implemented in spxa. The exactness level, available cumulants, simulation algorithm, and primary reference are listed for each entry.

**Exactness levels**
- `EXACT` — Lévy–Khintchine triplet is known in closed form; all arithmetic is exact
- `MOMENT` — closed-form triplet unavailable or inapplicable; finite moments are propagated
- `SIM` — no analytical handle; simulation only

---

## Lévy processes (`spxa.zoo.levy`)

| Class | Exactness | Cumulants | Simulation | Reference |
|---|---|---|---|---|
| `BrownianMotion` | EXACT | all orders | exact (Gaussian increments) | Wiener 1923 |
| `PoissonProcess` | EXACT | all orders | exact (exponential inter-arrivals) | Poisson 1837 |
| `CompoundPoisson` | EXACT | all orders (if jump dist. known) | exact (thinning) | Kingman 1993 |
| `GammaProcess` | EXACT | all orders | exact (shape-rate parametrisation) | Moran 1968 |
| `VarianceGamma` | EXACT | all orders | BM subordinated by Gamma | Madan, Carr, Chang 1998 |
| `NIG` | EXACT | all orders | BM subordinated by InvGaussian | Barndorff-Nielsen 1997 |
| `CGMY` | EXACT | all finite (α < 2) | series representation (Rosiński 2001) | Carr, Geman, Madan, Yor 2002 |
| `AlphaStable` | EXACT | order < α only | Chambers–Mallows–Stuck | Samorodnitsky & Taqqu 1994 |
| `TemperedStable` | EXACT | all orders | rejection from stable | Rosiński 2007 |
| `MeixnerProcess` | EXACT | all orders | acceptance-rejection | Schoutens & Teugels 1998 |
| `InverseGaussianProcess` | EXACT | all orders | exact (Wald distribution) | Tweedie 1957 |
| `TemperedStable` | EXACT | all orders (C·Γ(n-α)/λ^{n-α}) | Gil-Pelaez CDF inversion | Rosiński 2007 |
| `NegativeBinomialProcess` | EXACT | all orders (closed form to n=4, series above) | exact (numpy NegBin sampler) | Quenouille 1949 |
| `MeixnerProcess` | EXACT | all orders (closed form to n=4, numerical above) | Gil-Pelaez CDF inversion | Schoutens & Teugels 1998 |

## Beyond Lévy (`spxa.zoo.beyond`)

| Class | Exactness | Notes | Reference |
|---|---|---|---|
| `FractionalBrownianMotion` | MOMENT | not a semimartingale; Hurst algebra for self-similarity | Mandelbrot & Van Ness 1968 |
| `HawkesProcess` | MOMENT | self-exciting; mean intensity and variance propagated | Hawkes 1971 |
| `OULevy` | MOMENT | OU driven by Lévy subordinator; stationary dist. exact | Barndorff-Nielsen & Shephard 2001 |

---

New process proposals: open an issue using the **Process Proposal** template.
