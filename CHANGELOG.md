# Changelog

All notable changes to spxa are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

## Unreleased

### Added
- `TemperedStable` — pure-jump subordinator with Lévy density C·x^{-1-α}·exp(-λx); exact Bernstein function, cumulants, char_func, CDF inversion simulation (Rosiński 2007)
- `NegativeBinomialProcess` — finite-activity discrete subordinator; exact characteristic function, cumulants to order 4, NegBin simulation (Quenouille 1949)
- 34 exactness tests for both new processes

## 0.1.0 — TBD

Initial release. Core algebraic engine, Lévy zoo, and analytics module.
