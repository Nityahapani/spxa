# Property-based testing of stochastic algebra

`spxa.laws` turns the axioms of stochastic process algebra into executable tests. You declare a law, for example

$$
E[X + Y] = E[X] + E[Y], \qquad \varphi_{X \circ T}(u) = e^{\psi_T(-i\psi_X(u))}, \qquad \text{martingale}(X) \Rightarrow \text{martingale}(cX),
$$

and spxa searches randomised processes and parameters for a counterexample. If it finds one, it shrinks it to a minimal one and reports it.

```python
from spxa.laws import compositions, expect_close, law
from spxa.laws.quantities import mean

@law(X=compositions(max_depth=1), Y=compositions(max_depth=1))
def mean_is_additive(X, Y):
    """E[X + Y] = E[X] + E[Y]"""
    expect_close(mean(X + Y), mean(X) + mean(Y))

print(mean_is_additive.check(n_examples=200))
# mean_is_additive: VERIFIED (no counterexample in 200 examples)
```

The ideas are those of QuickCheck (Claessen & Hughes 2000) and Hypothesis (MacIver, Hatfield-Dodds et al. 2019), specialised to processes. Strategies generate Lévy processes and their compositions, and the library states the theory of Lévy processes as laws. A statistical layer covers laws about random output, such as simulators. [`examples/09_property_based_laws.py`](https://github.com/Nityahapani/spxa/blob/main/examples/09_property_based_laws.py) walks through all of it.

## 1. Strategies and shrinking

A `Strategy` draws an example from a `numpy.random.Generator`, together with the ways the example can be simplified. Each example is a rose tree (`Tree`): the root is the value and the children are simpler values. Because the tree is built alongside the value, shrinking works through `map`, `filter`, `flatmap` and composite strategies with no extra code. This is the "integrated shrinking" of Hedgehog and of Hypothesis's internal representation.

Shrinking is a greedy descent. The children of the current counterexample are tried simplest first, and the first child that fails *in the same way* replaces it: a violation must stay a violation, and an error must raise the same exception type. The result is locally minimal, meaning none of its direct simplifications fails.

Floats shrink towards a target (0, or `simplest=`). A float counts as simpler when its shortest decimal representation is shorter. Counterexamples therefore read `Drift(mu=0.1)`, not `Drift(mu=1e-16)`.

**Processes.** `PRIMITIVES` holds one strategy per Lévy family in the zoo, with parameters that shrink to round values. `compositions(max_depth=…)` builds expression trees with `+`, `c * X`, `-X` and `X @ T`. A composite counterexample shrinks first to its operands, so a law that fails for `X + Y` because of `X` alone is reported with `X`. Generic draws almost never have mean exactly zero, so `martingales()` centres them as $X_t - E[X_1]t$. Otherwise laws with a martingale hypothesis would hold vacuously. α-stable processes have no finite variance and are included only on request (`levy_processes(heavy_tails=True)`, `alpha_stables()`).

## 2. Laws and their outcomes

A `Law` pairs a predicate with one strategy per argument. Inside a law:

- `expect_close(a, b, rtol, atol)` requires $|a - b| \le \text{atol} + \text{rtol}\max(|a|, |b|)$ elementwise, with exact equality for infinities;
- `expect(condition, message, **details)` and `implies(premise, conclusion)`;
- `assume(condition)` discards examples outside the law's domain.

`Law.check(n_examples, seed)` returns a `LawResult` with one of four statuses:

| Status | Meaning |
| :--- | :--- |
| `verified` | no counterexample in `n_examples` examples. This is evidence, not proof. |
| `falsified` | the predicate failed. The shrunk and the original counterexamples are kept. |
| `error` | evaluation raised another exception. This is a counterexample too. |
| `unsatisfiable` | `assume` discarded too many examples. Nothing was learned. |

Every example comes from one generator seeded by `seed`, so a check, and every counterexample it reports, is reproducible.

**A false law.** `martingale(X) => martingale(X + c)` is false for $c \ne 0$, because $X + c\,t$ has mean $c\,t$. Checking it gives at once

```text
martingale_plus_constant: FALSIFIED after 0 passing examples (seed 1)
  counterexample (shrunk in 6 steps):
    X = Drift(mu=0.0)
    c = -0.1
```

The process shrinks to the simplest martingale and the constant to a short non-zero number.

## 3. The library

`spxa.laws.library` states the theory as 46 laws in seven groups. The default tolerances for closed-form identities are a relative $10^{-6}$ and an absolute $10^{-9}$, at frequencies $u \in [-4, 4]$.

| Group | Laws | What it states |
| :--- | ---: | :--- |
| `AXIOMS` | 5 | $\varphi(0) = 1$, $\lvert\varphi\rvert \le 1$, $\varphi(-u) = \overline{\varphi(u)}$, $\varphi_{s+t} = \varphi_s\varphi_t$, even cumulants $\ge 0$ |
| `ALGEBRA` | 10 | cumulants and cfs of sums, scalings and negation: additivity, commutativity, associativity, $\kappa_n(cX) = c^n\kappa_n(X)$, … |
| `SUBORDINATION` | 7 | $\psi_{X\circ T}(u) = \psi_T(-i\psi_X(u))$, the subordinated cumulants $\kappa_1 = \kappa_1^T\kappa_1^X$, $\kappa_2 = \kappa_1^T\kappa_2^X + \kappa_2^T(\kappa_1^X)^2$, $\kappa_3$, deterministic and additive clocks, associativity |
| `TRIPLET` | 6 | a primitive's closed forms against its Lévy–Khintchine triplet and its Lévy measure (moments, tails, truncated first moments, jump exponent) |
| `CUMULANTS` | 2 | cumulants recovered from the cf by Cauchy's integral formula, convexity of the cumulant generating function |
| `PROPERTIES` | 9 | martingale, subordinator, moment and self-similarity flags against the quantities they claim, and their closure under the operators |
| `SIMULATION` | 7 | simulated paths and `sample_at` draws against the cf and the cumulants (§4) |

The subordination laws rest on Bochner's formula: for an independent subordinator $T$, $E[e^{iuX_{T_t}}] = E[e^{T_t\psi_X(u)}] = e^{t\psi_T(-i\psi_X(u))}$ (Sato 1999, Thm 30.1). Its cumulant form follows from composing the cumulant generating functions, $K_{X\circ T} = K_T\circ K_X$. A Lévy process is a martingale exactly when $E\lvert X_1\rvert < \infty$ and $E[X_1] = 0$. That is why `martingale_flag_is_complete` matters: a `False` flag is also a claim.

## 4. Laws about random output

A law about a simulator cannot demand equality, but it can demand that a test with a known false-alarm rate does not reject. `spxa.laws.stats` provides such tests, and `expect_not_rejected(p)` fails the law when $p$ is below `LEVEL` $= 10^{-6}$.

**Empirical characteristic function.** For $n$ draws, $\bar\varphi_n(u) = n^{-1}\sum_k e^{iuX_k}$ has

$$
\operatorname{Var}\operatorname{Re}\bar\varphi_n(u) = \frac{1}{n}\Big[\frac{1 + \operatorname{Re}\varphi(2u)}{2} - (\operatorname{Re}\varphi(u))^2\Big],\qquad
\operatorname{Var}\operatorname{Im}\bar\varphi_n(u) = \frac{1}{n}\Big[\frac{1 - \operatorname{Re}\varphi(2u)}{2} - (\operatorname{Im}\varphi(u))^2\Big],
$$

because $\cos^2 x = (1 + \cos 2x)/2$ (Feuerverger & Mureika 1977). The summands are bounded, so the normal approximation holds for every law, heavy-tailed ones included. Several frequencies are combined by Bonferroni. The other tests are a z-test of the sample mean, one of the sample variance using $\operatorname{Var}s^2 = \kappa_4/n + 2\kappa_2^2/(n-1)$, and a test of independence through the joint empirical cf of two increments against $\varphi(u)\varphi(v)$.

**False alarms.** With level $\alpha$ per example, a check of $n$ examples of a correct law fails with probability at most $n\alpha$. At 100 examples that is once in $10^4$ runs. Each example draws its own seed from the law's strategies, so a counterexample replays exactly. Pure drifts have no sampling noise: every z-score divides by the standard error plus $10^{-10}$ times the scale of the quantity tested, so rounding never rejects.

`SIMULATION` uses these tests on the terminal value, on single increments, on the independence of increments, on the mean and variance, and on `sample_at` at mixed horizons. Its processes include α-stables and depth-1 compositions, so the simulators of sums, multiples and time changes are tested too.

## 5. Suites, reports, pytest and the shell

```python
from spxa.laws import Suite, default_suite
from spxa.laws.library import ALGEBRA, SUBORDINATION

report = Suite("algebra", ALGEBRA, SUBORDINATION).check(n_examples=50, seed=0)
print(report)                  # summary line, table, every counterexample
report.to_markdown()           # table plus collapsible counterexamples
report.to_dict()               # JSON-ready
report.raise_if_failed()       # SuiteFailure (an AssertionError) unless all verified

default_suite().select(tags=["subordination"], exclude_tags=["simulation"])
```

Under pytest, `law_tests` produces one parametrised case per law. A failing case prints the full report with the shrunk counterexample and its seed:

```python
# tests/test_laws.py
from spxa.laws import law_tests
from spxa.laws.library import ALGEBRA

test_algebra = law_tests(ALGEBRA, n_examples=50)
```

`SPXA_LAW_EXAMPLES` and `SPXA_LAW_SEED` override the example count and the seed without editing the tests, for example in a nightly job.

From the shell, `spxa-laws` (or `python -m spxa.laws`) checks the library. It exits with 0 when every law is verified, 1 otherwise and 2 on bad arguments:

```text
spxa-laws --tags algebra,subordination --examples 300 --seed 7
spxa-laws --exclude-tags simulation --format markdown -o laws.md
spxa-laws --list
```

## 6. What the laws found in spxa

Running the library over the zoo while it was being written found defects in spxa itself. Each one was fixed in its own commit with a regression test:

- **Triplets and measures.** The drift of `LevyTriplet.scale` and of the Gamma, Poisson and Meixner triplets; the inverse Gaussian Lévy measure; the NTS triplet, which carried the wrong measure; the α-stable triplet; negative binomial and compound Poisson measures inside sums and scalings.
- **Characteristic functions.** The cf and cumulants of `X @ T` and of composites containing it went through a triplet that does not exist for `@`, and raised. `char_exp_subordinated` took $\psi_X$ from the principal logarithm. The GH exponent wrapped by $2\pi$ once $\delta\lvert\beta\rvert$ was of order one, which flipped the sign of $\varphi(u, t)$ at non-integer $t$. The Meixner cf collapsed to 0 once $\cosh$ overflowed. CGMY needed closed forms at the poles $Y = 0, 1$.
- **Flags.** Self-similarity was claimed for processes that are not strictly self-similar. Martingales run on a clock with finite mean were flagged as non-martingales, and so were sums whose means cancel.
- **Simulators.** The CGMY, Meixner and tempered stable simulators inverted the cf on fixed grids that aliased or under-resolved the law. The NTS clock was a gamma matched to two moments. The α = 1 stable simulator missed the $\frac{2}{\pi}\beta\sigma t\log t$ term. `X @ T` interpolated the parent path linearly. Tempered stable draws were unresolved at short horizons, and GH draws came from an ad hoc grid. Tempered stable, NTS and CGMY with $0 < Y < 1$ now have exact samplers. The tempered stable sampler accepts draws of a tilted positive stable law (Kanter 1975; Kawai & Masuda 2011).

## References

- Claessen, K. & Hughes, J. (2000). QuickCheck: a lightweight tool for random testing of Haskell programs. *Proc. ICFP*.
- Feuerverger, A. & Mureika, R. A. (1977). The empirical characteristic function and its applications. *Ann. Statist.* 5.
- Kanter, M. (1975). Stable densities under change of scale and total variation inequalities. *Ann. Probab.* 3.
- Kawai, R. & Masuda, H. (2011). On simulation of tempered stable random variates. *J. Comput. Appl. Math.* 235.
- MacIver, D. R., Hatfield-Dodds, Z. et al. (2019). Hypothesis: a new approach to property-based testing. *J. Open Source Software* 4(43).
- MacIver, D. R. & Donaldson, A. F. (2020). Test-case reduction via test-case generation: insights from the Hypothesis reducer. *Proc. ECOOP*.
- Sato, K. (1999). *Lévy Processes and Infinitely Divisible Distributions*. Cambridge.
