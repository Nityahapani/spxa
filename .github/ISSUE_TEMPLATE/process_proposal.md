---
name: Process proposal
about: Propose a new stochastic process for the zoo
labels: process-proposal
---

## Process name and class name

<!-- e.g. Normal Inverse Gaussian, class NIG -->

## Lévy–Khintchine triplet

Provide the triplet $(b, \sigma^2, \nu)$ in closed form, or explain why it does not exist.

**Drift $b$:**

**Diffusion coefficient $\sigma^2$:**

**Lévy measure $\nu(dx)$:**

If no triplet exists, state which Lévy property is violated (stationary increments / independent increments) and what the proposed exactness level is.

## Cumulant formulas

At minimum, mean and variance. Cite the source for each formula.

**Mean $\kappa_1$:**

**Variance $\kappa_2$:**

**Higher cumulants (if known):**

## Proposed exactness level

- [ ] `EXACT` — triplet known in closed form
- [ ] `MOMENT_PROPAGATION` — moments trackable, no full triplet
- [ ] `SIMULATION_ONLY`

## Simulation algorithm

Which algorithm will be used to simulate sample paths? State complexity per step.

## Primary reference

Full citation (authors, year, journal, DOI). Textbook entries are not sufficient — provide the original paper.

## Domain motivation

Where is this process used? (finance, physics, biology, etc.) Why does it belong in spxa's zoo?
