# Contributing to spxa

spxa is research infrastructure for stochastic process algebra. Correctness
and mathematical honesty are non-negotiable. Every contribution must be
grounded in a published result, and the library must never silently return
an incorrect answer.

This document is the authoritative guide for every type of contribution.
Read the section for what you want to add, then follow it exactly.

---

## Contents

1. [Setup](#setup)
2. [Codebase map](#codebase-map)
3. [Contribution types](#contribution-types)
   - [A. New Lévy process (zoo/levy)](#a-new-lévy-process-zoolevy)
   - [B. New beyond-Lévy process (zoo/beyond)](#b-new-beyond-lévy-process-zoobeyond)
   - [C. New operation (ops/)](#c-new-operation-ops)
   - [D. New analytic tool (analytics/)](#d-new-analytic-tool-analytics)
   - [E. New simulation algorithm (sim/)](#e-new-simulation-algorithm-sim)
   - [F. New multivariate process (zoo/multivariate)](#f-new-multivariate-process-zoomultivariate)
   - [G. Core change (core/)](#g-core-change-core)
   - [H. Bug fix](#h-bug-fix)
   - [I. Documentation](#i-documentation)
4. [The exactness test suite](#the-exactness-test-suite)
5. [Code style](#code-style)
6. [Commit conventions](#commit-conventions)
7. [Versioning and deprecation](#versioning-and-deprecation)

---

## Setup

```bash
git clone https://github.com/Nityahapani/spxa.git
cd spxa
pip install -e ".[dev]"
pre-commit install
```

Dev dependencies: `pytest`, `pytest-cov`, `sympy`, `numpy`, `scipy`,
`ruff`, `mypy`, `pre-commit`.

Run the full test suite before and after any change:

```bash
python -m pytest tests/ -q
```

All 235+ tests must pass. Never submit a PR that breaks existing tests.

---

## Codebase map

```
spxa/
├── core/               # The algebraic engine — change with extreme care
│   ├── exactness.py    # ExactnessLevel enum and combine()
│   ├── exceptions.py   # ExactnessError, SpxaDegradationWarning
│   ├── levy_measure.py # LevyMeasure hierarchy
│   ├── process.py      # Base Process class, operators, _ScaledProcess, _ComposedProcess
│   ├── properties.py   # ProcessProperties lattice and propagation rules
│   └── triplet.py      # LevyTriplet dataclass and arithmetic
├── story/
│   └── narrator.py     # CompositionNode, narrate(), narrate_latex(), story_latex()
├── zoo/
│   ├── levy/           # Lévy processes (EXACT triplet)
│   │   ├── brownian.py
│   │   ├── gamma.py
│   │   ├── vg.py
│   │   ├── nig.py
│   │   ├── cgmy.py
│   │   ├── stable.py
│   │   ├── meixner.py
│   │   ├── poisson.py
│   │   ├── inverse_gaussian.py
│   │   └── __init__.py
│   ├── beyond/         # Non-Lévy processes (MOMENT_PROPAGATION)
│   │   ├── fbm.py
│   │   ├── hawkes.py
│   │   ├── ou_levy.py
│   │   └── __init__.py
│   ├── multivariate.py # Multivariate processes
│   └── __init__.py
├── ops/                # Operations on processes
│   ├── addition.py
│   ├── scaling.py
│   ├── subordination.py
│   ├── product.py
│   ├── integral.py
│   └── __init__.py
├── analytics/          # Analytical tools
│   ├── cumulants.py
│   ├── divergence.py
│   ├── fourier.py
│   └── __init__.py
├── sim/                # Simulation backends
│   ├── euler.py
│   ├── exact.py
│   ├── bridge.py
│   └── __init__.py
└── __init__.py         # Top-level public API

tests/
├── exactness/          # SACRED — closed-form formulas vs published results
├── integration/        # End-to-end pipelines
└── unit/               # Module-level unit tests

docs/
└── theory/             # Mathematical derivations (Markdown with LaTeX)

PROCESS_REGISTRY.md     # Table of all implemented processes — keep up to date
```

---

## Contribution types

---

### A. New Lévy process (zoo/levy)

**Open a Process Proposal issue first.** The proposal must contain:
1. The Lévy–Khintchine triplet $(b, \sigma^2, \nu)$ in closed form, with derivation or citation
2. Cumulant formulas (at minimum $\kappa_1$ and $\kappa_2$) with citation
3. The primary reference (original journal paper — not a textbook)
4. Proposed `ExactnessLevel` (must be `EXACT` for a Lévy process with known triplet)
5. The simulation algorithm with complexity per step

**Files to create or modify:**

**1. Create `spxa/zoo/levy/<name>.py`**

The file must contain:

```python
# Module-level docstring with full mathematical specification:
# - Lévy density k(x)
# - Triplet (b, σ², ν)
# - Cumulant formulas
# - All primary references

def _<name>_levy_measure(params) -> DensityLevyMeasure:
    """
    Lévy density function.
    Cite the exact equation number and reference.
    """
    def density(x: np.ndarray) -> np.ndarray: ...
    def tail(xv: float) -> float: ...
    def moment_n() -> float: ...   # for each n you know exactly

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=np.inf,        # or finite for compound Poisson
        _tail_fn=tail,
        _moment_fns={1: moment_1, 2: moment_2, ...},
    )


class MyProcess(Process):
    def __init__(self, param1: float, param2: float) -> None:
        # validate parameters
        # store self.param1, self.param2
        # build CompositionNode with math_note and reference
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        # Return LevyTriplet(b=..., sigma_sq=..., nu=...)
        # b must be consistent with the truncation h(x) = 1_{|x|≤1}
        # If b is computed by quadrature, use an exact formula (like E_1)
        # when available. See gamma.py for the pattern.

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=...,      # True only if drift=0 and ∫_{|x|>1} x ν(dx) = 0
            has_finite_variance=...,
            has_finite_mean=...,
            self_similarity_index=None,   # fill if self-similar
            tail_index=...,               # α such that ν((x,∞)) ~ x^{-α}, or None
            is_subordinator=...,          # True only if support ⊂ (0,∞)
        )

    def cumulants(self, order: int) -> dict[int, float]:
        # Override with exact closed-form formulas where known.
        # Fall back to super().cumulants() only if no formula exists.
        ...

    def char_func_exact(self, u, t=1.0) -> np.ndarray:
        # If the characteristic function has a closed form, implement it here
        # and override char_func() to call it.
        ...

    def char_func(self, u, t=1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)  # if closed form exists

    def simulate(self, n_steps, n_paths=1, T=1.0, rng=None) -> np.ndarray:
        # Use the most efficient exact algorithm known.
        # Return shape (n_paths, n_steps + 1), paths[:, 0] == 0.
        ...

    def __repr__(self) -> str:
        return f"MyProcess(param1={self.param1}, param2={self.param2})"
```

**2. Add to `spxa/zoo/levy/__init__.py`**

```python
from spxa.zoo.levy.<name> import MyProcess
# add to __all__
```

**3. Add to `spxa/zoo/__init__.py`**

```python
from spxa.zoo.levy import ..., MyProcess
# add to __all__
```

**4. Add to `spxa/__init__.py`**

```python
from spxa.zoo import ..., MyProcess
# add to __all__
```

**5. Create `tests/exactness/test_<name>.py`**

Every test must have `@pytest.mark.exactness` and cite the formula being
verified. Minimum required tests:

```python
@pytest.mark.exactness
def test_<name>_no_gaussian_component() -> None:
    """σ²=0 for pure-jump processes. <Primary reference>."""
    ...

@pytest.mark.exactness
def test_<name>_char_func_at_zero() -> None:
    """φ(0;t)=1. Definition of characteristic function."""
    ...

@pytest.mark.exactness
def test_<name>_char_func_time_additivity() -> None:
    """φ(u;s+t)=φ(u;s)·φ(u;t). Sato (1999), Definition 1.6."""
    ...

@pytest.mark.exactness
def test_<name>_cumulant_1() -> None:
    """κ₁ = <formula>. <Reference, equation number>."""
    ...

@pytest.mark.exactness
def test_<name>_cumulant_2() -> None:
    """κ₂ = <formula>. <Reference, equation number>."""
    ...

@pytest.mark.exactness
def test_<name>_simulate_mean() -> None:
    """Empirical mean converges to κ₁. Law of large numbers."""
    ...

@pytest.mark.exactness
def test_<name>_simulate_variance() -> None:
    """Empirical variance converges to κ₂."""
    ...
```

If the process is a subordinator, also add:

```python
@pytest.mark.exactness
def test_<name>_is_subordinator() -> None:
    """Paths are non-decreasing. <Reference>."""
    ...

@pytest.mark.exactness
def test_<name>_simulate_non_decreasing() -> None:
    """All increments non-negative."""
    ...
```

**6. Add to `PROCESS_REGISTRY.md`**

Add a row to the table with: class name, exactness level, available
cumulants, simulation algorithm, primary reference.

**Pattern to copy:** `spxa/zoo/levy/gamma.py` and
`tests/exactness/test_gamma.py` are the reference implementation.

---

### B. New beyond-Lévy process (zoo/beyond)

Beyond-Lévy processes (fBM, Hawkes, OU-Lévy) violate stationarity or
independence of increments and therefore have no Lévy–Khintchine triplet.

**Files to create or modify:**

**1. Create `spxa/zoo/beyond/<name>.py`**

Key differences from a Lévy process:

```python
class MyBeyondProcess(Process):
    def __init__(self, ...) -> None:
        # ExactnessLevel is MOMENT_PROPAGATION, not EXACT
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        # MUST raise ExactnessError with a clear explanation of
        # which Lévy property is violated and why no triplet exists.
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason="<process name> violates <stationary/independent> increments "
                   "because <mathematical reason>. No Lévy–Khintchine triplet exists. "
                   "Use <what to use instead>.",
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=...,   # likely False or process-specific
            has_independent_increments=...,  # likely False
            ...
            notes=[
                "Human-readable explanation of why this is beyond Lévy.",
                "What analytical quantities ARE available.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        # Implement what you CAN compute: typically κ₁ and κ₂ exactly,
        # and float('nan') for higher orders that are intractable.
        result = {}
        if order >= 1:
            result[1] = ...
        if order >= 2:
            result[2] = ...
        for n in range(3, order + 1):
            result[n] = float('nan')
        return result

    # Add domain-specific analytical methods beyond what Process provides:
    def variance(self, t: float) -> float: ...
    def covariance(self, s: float, t: float) -> float: ...
    def autocovariance(self, h: float) -> float: ...
```

**2. Add to `spxa/zoo/beyond/__init__.py`, `spxa/zoo/__init__.py`, `spxa/__init__.py`**

Same pattern as Lévy processes.

**3. Create `tests/exactness/test_beyond.py` entries**

Add a class `TestMyBeyondProcess` to the existing
`tests/exactness/test_beyond.py`. Required tests:

- `test_<name>_triplet_raises` — verifies `ExactnessError` is raised
- `test_<name>_<analytical_property>` — verifies each formula you implement
  against its cited source
- `test_<name>_simulate_*` — verifies simulation statistics

**Pattern to copy:** `spxa/zoo/beyond/fbm.py` (fBM) and the
`TestFractionalBrownianMotion` class in `tests/exactness/test_beyond.py`.

---

### C. New operation (ops/)

Operations are functions that take Process objects and return a new Process,
or take a LevyTriplet and return a new LevyTriplet.

**Before implementing, open an issue stating:**
- The mathematical definition of the operation
- Whether the result is exact for independent Lévy processes, or approximate
- The theorem or cited result justifying the computation
- The ExactnessLevel propagation rule

**Files to create or modify:**

**1. Create `spxa/ops/<operation>.py`**

```python
"""
<Operation name>.

Mathematical basis
------------------
State the theorem. Cite author, year, theorem number.

Exact for: <conditions under which result is EXACT>
Approximate for: <conditions under which it degrades and why>
"""

def <operation>(X: Process, Y: Process, ...) -> Process:
    """
    Brief description.

    Parameters / Returns / Raises / References sections.
    """
    ...
```

Rules:
- If the operation is exact for independent Lévy processes, include a
  proof sketch or theorem citation in the docstring
- If the result degrades ExactnessLevel, emit `SpxaDegradationWarning`
  with a mathematical reason — never silently degrade
- If the operation requires conditions (e.g. Y must be a subordinator),
  check them explicitly and raise `ValueError` with a helpful message

**2. Add to `spxa/ops/__init__.py`**

Add the function/class to the imports and `__all__`.

**3. Add tests to `tests/unit/test_ops.py`**

Add a new test class. At minimum:

- Correctness tests: verify the output against known exact values
- Exactness tests: verify `ExactnessLevel` is set correctly
- Error tests: verify that invalid inputs raise the right exceptions
- Warning tests: verify that degradation emits `SpxaDegradationWarning`

If the operation touches the triplet, add a test that verifies the triplet
arithmetic against Sato (1999) directly.

**Pattern to copy:** `spxa/ops/subordination.py` and the
`TestSubordination` class in `tests/unit/test_ops.py`.

---

### D. New analytic tool (analytics/)

Analytic tools compute quantities derived from the characteristic function
or cumulants of a process.

**Files to create or modify:**

**1. Add to an existing `spxa/analytics/<module>.py` or create a new one**

Every function must:
- State clearly whether the result is exact or approximate
- Say which processes it works for (EXACT only? MOMENT_PROPAGATION too?)
- Raise `ExactnessError` if called on a process whose exactness level
  is insufficient
- Cite the mathematical result being implemented

```python
def my_analytic_function(
    process: Process,
    ...,
) -> float:
    """
    Brief description.

    Exact for EXACT processes via <formula>.
    Requires: <conditions>.

    Parameters
    ----------
    ...

    References
    ----------
    Author (year), equation/theorem number.
    """
    if process.exactness == ExactnessLevel.SIMULATION_ONLY:
        raise ExactnessError(...)
    ...
```

**2. Add to `spxa/analytics/__init__.py`**

**3. Add tests to `tests/unit/test_ops.py` or a new `tests/unit/test_analytics.py`**

At minimum: a test against a known value, a test for the correct error
when called on an unsupported process.

**Pattern to copy:** `spxa/analytics/divergence.py` and the
`TestAnalyticsPipeline` class in `tests/integration/test_composition.py`.

---

### E. New simulation algorithm (sim/)

Simulation functions produce sample paths or random variates. The bar is
high: if you claim a sampler is "exact", it must be exact in distribution —
not an approximation.

**Files to create or modify:**

**1. Add to an existing `spxa/sim/<module>.py` or create a new one**

```python
def sample_<name>(
    param1: float,
    param2: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exact increments of <process name> via <algorithm name>.

    Algorithm (<primary citation>):
      1. ...
      2. ...

    Parameters
    ----------
    ...

    Returns
    -------
    np.ndarray
        Increments, shape (n_paths, n_steps). Call increments_to_paths()
        to prepend x0 and convert to paths.

    References
    ----------
    <Full citation, year, algorithm number if applicable>
    """
    ...
```

Conventions:
- Standalone samplers return **increments** of shape `(n_paths, n_steps)`,
  not paths. Use `increments_to_paths()` from `sim/exact.py` to convert.
- If the algorithm is approximate (e.g. Euler-Maruyama approximation to
  an SDE), state the strong convergence order and conditions in the docstring.
- If the algorithm is exact, cite the theorem guaranteeing this.

**2. Add to `spxa/sim/__init__.py`**

**3. Add tests to `tests/unit/test_sim.py`**

Required:
- Shape test: `assert increments.shape == (n_paths, n_steps)`
- Mean test: empirical mean vs theoretical mean (use `pytest.approx`)
- Variance test: empirical variance vs theoretical variance
- If the process is a subordinator: non-negativity test, monotonicity test

Statistical tests must use `n_paths ≥ 5000` and loose tolerances
(`rel=0.05` or `abs=0.1`) — they are Monte Carlo, not exactness tests.

**Pattern to copy:** `spxa/sim/exact.py` (`sample_gamma`, `sample_vg`)
and `tests/unit/test_sim.py` (`TestExactSamplers`).

---

### F. New multivariate process (zoo/multivariate)

Multivariate processes live in `spxa/zoo/multivariate.py` and use
`MultivariateLevyTriplet` (matrix $\Sigma$) rather than `LevyTriplet`
(scalar $\sigma^2$).

**Files to create or modify:**

**1. Add to `spxa/zoo/multivariate.py`**

```python
class MyMultivariateProcess(Process):
    def __init__(self, dim: int, ...) -> None:
        self.dim = dim
        ...
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        # Return a 1-d projection triplet (e.g. trace of Σ as σ²)
        # for compatibility with the base class.
        ...

    def multivariate_triplet(self) -> MultivariateLevyTriplet:
        # Return the full d-dimensional triplet.
        ...

    def simulate(self, n_steps, n_paths=1, T=1.0, rng=None) -> np.ndarray:
        # Return shape (n_paths, n_steps + 1, dim).
        # paths[:, 0, :] == 0.
        ...
```

**2. Add to `spxa/zoo/__init__.py`**

**3. Add tests to `tests/unit/test_multivariate.py`**

Required:
- Shape test: `assert paths.shape == (n_paths, n_steps + 1, dim)`
- Starts at zero: `np.testing.assert_array_equal(paths[:, 0, :], 0.0)`
- Covariance recovery: for a BM-based process, empirical covariance should
  match the theoretical $\Sigma$ (use `np.testing.assert_allclose(..., atol=0.1)`)

**Pattern to copy:** `MultivariateBrownianMotion` in `multivariate.py` and
`TestMultivariateBrownianMotion` in `tests/unit/test_multivariate.py`.

---

### G. Core change (core/)

`core/` is the algebraic foundation. Any change requires:

1. **Open an RFC issue** using the RFC template in `.github/ISSUE_TEMPLATE/rfc.md`.
   The RFC must state: motivation, proposed change, backward compatibility
   impact, and alternatives considered.
2. **Two maintainer approvals** before implementation begins.
3. **Full backward compatibility** or a deprecation plan spanning two minor
   versions (see [Versioning](#versioning-and-deprecation)).

**Files in `core/` and what they own:**

- `exactness.py` — the `ExactnessLevel` enum and `combine()`. Only add a
  new level if it is mathematically distinct from the three existing ones.
- `exceptions.py` — `SpxaError`, `ExactnessError`, `SpxaDegradationWarning`.
  New exception subclasses must have a specific mathematical reason.
- `levy_measure.py` — `LevyMeasure` hierarchy. New subclasses must implement
  `density()`, `total_mass()`, `tail()`, and `moment()`.
- `process.py` — `Process` base class and operators. The operator signatures
  (`__add__`, `__mul__`, `__matmul__`) are frozen after `1.0.0`.
- `properties.py` — `ProcessProperties` and propagation rules. Adding a new
  property requires updating all three `combine_*` static methods and all
  existing processes that set that property.
- `triplet.py` — `LevyTriplet`. The `b`, `sigma_sq`, `nu` fields and the
  `__add__`, `scale()`, `char_exp()`, `cumulant()` methods are frozen after
  `1.0.0`.

**Test requirements for core changes:**

- All existing tests must still pass
- New behavior must be covered by tests in `tests/unit/test_core.py`
- If the change affects exactness propagation, add a test that verifies
  the exact ExactnessLevel of a composed process after the change

---

### H. Bug fix

**Open a bug report issue first** using the bug report template. If the bug
is an exactness violation (spxa returned an analytically wrong value), add
the label `exactness-violation` — this is the highest priority category.

**Files to modify:**

1. Fix the bug in the relevant source file
2. Add a test in the appropriate test file that would have caught the bug.
   For exactness violations, the test goes in `tests/exactness/`.
3. Add an entry to `CHANGELOG.md` under `## Unreleased` → `### Fixed`

**For exactness violations specifically:**

The new test must cite the published formula that contradicts the wrong
value spxa was returning. This test must go in `tests/exactness/` and
carry `@pytest.mark.exactness`. The fix must be verified against the
published formula in the test, not just against the "fixed" code.

---

### I. Documentation

**Theory docs** (`docs/theory/`) are formal mathematical write-ups. They
must be self-contained, cite primary sources, and match the code exactly —
every formula in a theory doc must be implemented in the code and verified
by an exactness test.

**Tutorials** (`docs/tutorials/`) are Jupyter notebooks. They must be
runnable (`jupyter nbconvert --to notebook --execute`) and committed with
outputs cleared.

**API docs** are auto-generated from docstrings via `mkdocstrings`. Improve
API docs by improving the docstring in the source file, not by editing
generated files.

**Docstring format:** NumPy style throughout. Every public function and
class must have: a one-line summary, a `Parameters` section, a `Returns`
section, and a `References` section citing the primary source. Equation
numbers in citations are required for any non-obvious formula.

---

## The exactness test suite

`tests/exactness/` is sacred. It is the mechanism by which spxa guarantees
it never silently returns a wrong mathematical result.

**Invariants that must never be violated:**

1. No test in `tests/exactness/` may carry `@pytest.mark.skip`,
   `@pytest.mark.skipif`, or `@pytest.mark.xfail` for any reason. If a
   formula is wrong, fix the formula. If a test is wrong, fix the test.
   The `scripts/check_exactness_markers.py` script enforces this in CI.

2. Every test must have `@pytest.mark.exactness`.

3. Every test docstring must cite the exact theorem, equation, or section
   being verified. The citation must include author, year, and location
   (equation number, theorem number, or page). Example:
   ```python
   """κ₂(X₁) = σ² + θ²ν for VG. Madan, Carr & Chang (1998), Section 2."""
   ```

4. The test must verify a closed-form result, not just "it runs without
   error" or "the output is positive". Compare against the mathematical
   formula, not against what the code produces.

5. Statistical tests (empirical mean/variance) belong in
   `tests/exactness/` only when verifying that the simulation converges
   to a theoretically known value. Use `n_paths ≥ 10000` and tolerances
   `abs=0.05` or `rel=0.05` — small enough to catch real bugs, loose
   enough that they don't flake.

---

## Code style

**Linting:** `ruff` for linting and formatting — enforced by pre-commit
and CI. Configuration in `pyproject.toml`.

**Type checking:** `mypy` in strict mode — enforced by CI.

**Naming:**
- Process classes: `PascalCase` (e.g. `VarianceGamma`, `NIG`)
- Functions: `snake_case`
- Private helpers: `_snake_case`
- Lévy measure constructors: `_<processname>_levy_measure()`

**No comment line breaks.** Lines like `# ====` or `# ----` are
prohibited. Section structure comes from function/class definitions.

**No commented-out code in PRs.** Use git history.

**No unnecessary comments.** A comment that restates what the code
obviously does is noise. Comments should explain *why*, especially for
non-obvious mathematical choices.

**Docstrings are not optional.** Every public function, class, and
method must have a NumPy-style docstring with a References section.

---

## Commit conventions

Use conventional commits:

```
feat(zoo): add TemperedStable — exact cumulants, rejection simulation
feat(ops): add exponential() — stochastic exponential of a semimartingale
fix(zoo/vg): eliminate singularity warning in triplet drift
test(exactness): VG — char_func vs time-additivity
docs(theory): add derivation of subordination triplet formula
chore: update PROCESS_REGISTRY with TemperedStable entry
```

Scope is the module: `core`, `zoo`, `zoo/levy`, `zoo/beyond`, `ops`,
`analytics`, `sim`, `story`, `docs`, `test`, `chore`.

**Commit granularity:** commit after each logical unit of work. Do not
batch unrelated changes. Do not commit broken states.

After each commit that adds a feature or fix, push to the remote. The
CI runs on every push and a failing CI is a blocking issue.

---

## Versioning and deprecation

spxa follows semantic versioning strictly.

**`core/` and `ops/` APIs are frozen after `1.0.0`.** Adding parameters
with defaults is allowed. Removing parameters, changing return types, or
changing operator semantics requires a major version bump.

**Deprecations** must:
1. Emit `DeprecationWarning` with a clear message stating what to use instead
2. Remain in the codebase for at least two minor versions before removal
3. Have an entry in `CHANGELOG.md` under `### Deprecated`

**Breaking changes** require:
1. A major version bump
2. An entry in `CHANGELOG.md` under `### Removed`
3. A migration note in the changelog explaining what to do

**Every PR** must add an entry to `CHANGELOG.md` under `## Unreleased`
in the appropriate section (`Added`, `Changed`, `Fixed`, `Deprecated`,
`Removed`).
