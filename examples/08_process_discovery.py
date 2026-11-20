"""
Example 08: Process discovery, from returns or option prices to a composition.

Demonstrates:
1. Writing a CSV of weekly returns drawn from VG + 0.5 * Gamma, then
   recovering a composition from the file alone with ``discover``.
2. Reading the report: the headline sentence, the ranked table with
   BIC weights, goodness-of-fit p-values and CDF gaps, and the flags for
   degenerate fits.
3. Standard errors for the winner, and a Vuong test against the runner-up.
4. Bootstrap selection frequencies: how often each structure wins on
   resampled data.
5. Discovery from summary statistics alone (mean, variance, skewness,
   kurtosis), and why it cannot separate the candidates.
6. Discovery from a surface of European option prices.

The same search is available from the shell::

    spxa-discover returns returns.csv --families BM,VG,NIG,Gamma

Run with ``python examples/08_process_discovery.py [--quick]``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spxa import GammaProcess, VarianceGamma  # noqa: E402
from spxa.discovery import (  # noqa: E402
    VARIANCE_GAMMA,
    Model,
    OptionQuotes,
    bootstrap,
    discover,
    discover_from_options,
    from_statistics,
    grammar_from_names,
    leaf,
    model_prices,
    summarize,
)

DT = 1 / 52


def simulate_returns(n: int, seed: int = 11) -> np.ndarray:
    """Weekly increments of VG(0.2, 0.25, -0.3) + 0.5 * Gamma(1, 2)."""
    truth = VarianceGamma(sigma=0.2, nu=0.25, theta=-0.3) + 0.5 * GammaProcess(a=1.0, b=2.0)
    path = truth.simulate(n_steps=n, n_paths=1, T=n * DT, rng=np.random.default_rng(seed))
    return np.diff(np.asarray(path).reshape(-1))


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main(quick: bool = False) -> None:
    n = 1500 if quick else 4000
    families = ["BM", "VG", "Gamma"] if quick else ["BM", "VG", "NIG", "Gamma", "NormalJumps"]
    grammar = grammar_from_names(families, max_terms=2)

    section("1. Returns CSV  ->  composition")
    x = simulate_returns(n)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "returns.csv"
        csv_path.write_text("return\n" + "\n".join(f"{v:.17g}" for v in x) + "\n")
        report = discover(str(csv_path), dt=DT, grammar=grammar, n_starts=2)
    print("truth: VarianceGamma(sigma=0.2, nu=0.25, theta=-0.3) + 0.5 * GammaProcess(a=1, b=2)")
    print(report)

    section("2. Uncertainty for the winner and a Vuong test against the runner-up")
    print(report.uncertainty().table())
    v = report.compare(0, 1)
    print(
        f"\nVuong z = {v.z:.2f}, p = {v.p_value:.3f}: preferred = {v.preferred} "
        "(BIC-corrected; 'neither' means the data cannot separate them)"
    )

    section("3. Bootstrap: how stable is the selection?")
    boot = bootstrap(report, n_boot=10 if quick else 40, top=3, seed=1)
    print(boot.table())

    section("4. Summary statistics only")
    s = summarize(x, DT)  # per-unit-time cumulants; the table quotes weekly moments
    weekly = (s.mean * DT, s.variance * DT, s.skewness(), s.excess_kurtosis())
    print("weekly mean %.5f, variance %.6f, skewness %.3f, excess kurtosis %.2f" % weekly)
    stats_only = discover(from_statistics(*weekly, dt=DT, n=n), grammar=grammar)
    print(stats_only)

    section("5. Option prices  ->  risk-neutral composition")
    strikes = np.tile(np.linspace(80.0, 120.0, 9), 3)
    maturities = np.repeat([0.25, 0.5, 1.0], 9)
    calls = strikes >= 100.0
    shell = OptionQuotes.from_arrays(100.0, strikes, maturities, np.ones(27), calls, rate=0.03)
    rn_truth = Model(leaf(VARIANCE_GAMMA), [(0.15, 0.3, -0.2)])
    quotes = OptionQuotes.from_arrays(
        100.0, strikes, maturities, model_prices(rn_truth, shell, 4096), calls, rate=0.03
    )
    print("risk-neutral truth: VarianceGamma(sigma=0.15, nu=0.3, theta=-0.2)")
    print(
        discover_from_options(quotes, grammar=grammar_from_names(["BM", "VG", "NIG"], max_terms=1))
    )


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
