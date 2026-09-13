# Contributing to spxa

spxa is research infrastructure. Correctness and mathematical honesty are non-negotiable. This document explains how to contribute at each tier.

## Setup

```bash
git clone https://github.com/Nityahapani/spxa.git
cd spxa
pip install -e ".[dev]"
pre-commit install
```

Dev dependencies: `pytest`, `pytest-cov`, `sympy`, `numpy`, `scipy`, `ruff`, `mypy`, `pre-commit`.

## Contribution tiers

### Tier 1: New process to the zoo

Open a **Process Proposal** issue first. The proposal must include:

1. The Lévy–Khintchine triplet `(b, σ², ν)` in closed form, or an explicit statement of why one does not exist and what exactness level is appropriate
2. Cumulant formulas (at least mean and variance) with derivation or citation
3. The primary reference (journal paper, not a textbook entry)
4. Proposed `ExactnessLevel`

Once the proposal is approved, implement in `spxa/zoo/levy/` or `spxa/zoo/beyond/`. Requirements:

- Inherit from `LevyProcess` or `BeyondLevyProcess`
- Implement `triplet()`, `cumulants(order)`, `char_func(u, t)`, `simulate(n_steps, n_paths, T)`
- Every docstring must cite the primary reference
- Add a row to `PROCESS_REGISTRY.md`
- Add at least one test in `tests/exactness/` verifying a cumulant or characteristic function value against the published formula

### Tier 2: New operation

Operations live in `spxa/ops/`. Requirements:

- If the operation is exact (closed-form result for independent Lévy processes), provide a proof or cite the theorem
- If the operation is approximate, document the approximation scheme, its error bounds, and the conditions under which it applies
- Update `ExactnessLevel` propagation in `spxa/core/properties.py`
- Add tests in `tests/integration/`

### Tier 3: Changes to `core/`

`core/` is the algebraic foundation. Any change requires:

1. An RFC document in `docs/rfc/` describing the motivation, the proposed change, and alternatives considered
2. Approval from two maintainers before implementation begins
3. Full backward compatibility or a deprecation plan spanning two major versions

## The exactness test suite

`tests/exactness/` is sacred. These tests verify closed-form results against published formulas from primary references. Rules:

- No test in this directory may be skipped or marked `xfail` without a maintainer approval and a linked issue explaining the mathematical reason
- Every test must include a comment citing the exact theorem or formula being verified (equation number and reference)
- A failing exactness test blocks all merges

## Code style

- `ruff` for linting and formatting — enforced by CI
- `mypy` for type checking — enforced by CI
- No comment line breaks (`# ====`, `# ----`, etc.)
- No commented-out code in PRs
- Docstrings use NumPy style

## Versioning and deprecation

spxa follows semantic versioning strictly.

- `core/` and `ops/` APIs are frozen after `1.0.0`
- Deprecations must emit `DeprecationWarning` for at least two minor versions before removal
- Breaking changes require a major version bump and an entry in `CHANGELOG.md`

## Changelog

Every PR must add an entry to `CHANGELOG.md` under `## Unreleased` in the appropriate section (`Added`, `Changed`, `Fixed`, `Deprecated`, `Removed`).
