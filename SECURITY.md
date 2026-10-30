# Security Policy

## Supported versions

Security fixes are applied to the latest minor release line of `spxa`. Older lines receive fixes only at the maintainers' discretion.

| Version | Supported          |
| :------ | :----------------- |
| 1.x (latest minor) | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue, pull request, or discussion for security problems.**

Report vulnerabilities privately through one of the following channels:

1. **GitHub private vulnerability reporting** (preferred): go to the [Security tab](https://github.com/Nityahapani/spxa/security) of the repository and choose **Report a vulnerability**.
2. **Email**: send the details to **nityahapani.tges@gmail.com** with the subject line `[spxa security]`.

Please include as much of the following as you can:

- the affected version(s) or commit hash, Python version, and operating system;
- the component involved (e.g. `spxa.graph.serialization`, a specific calibrator or backend);
- a minimal reproducer or proof of concept;
- the impact you believe the issue has (code execution, data exposure, denial of service, …).

## What to expect

- **Acknowledgement** within 5 business days.
- **Initial assessment** (confirmed / not reproducible / not a security issue) within 14 days.
- For confirmed issues, a fix is developed in a private security advisory, released as a patch version, and the advisory is published with a CVE where appropriate. Reporters are credited unless they prefer to remain anonymous.

Please give us a reasonable window to release a fix before any public disclosure.

## Scope

In scope:

- Code in the `spxa` package published on PyPI and in this repository.
- Deserialisation of computational graphs and other artefacts (`spxa.graph.serialization`): loading untrusted serialised graphs must not lead to arbitrary code execution.
- The release and CI pipeline in `.github/workflows/` (e.g. supply-chain or token-exposure issues).

Out of scope:

- **Numerical or mathematical inaccuracies** (wrong prices, biased estimators, convergence failures). These are important, but they are correctness bugs, not security issues. Please report them through the [bug report template](https://github.com/Nityahapani/spxa/issues/new/choose).
- Vulnerabilities in third-party dependencies (NumPy, SciPy, JAX, PyTorch, …) that are not specific to how `spxa` uses them; please report those upstream.
- Denial of service caused by deliberately requesting enormous simulations (e.g. billions of Monte Carlo paths) on your own machine.

## Safe usage notes

- Treat serialised models, graphs, and calibration artefacts from untrusted sources like executable code. Only load files you trust.
- `spxa` does not make network requests at runtime; if you observe one, please report it.
