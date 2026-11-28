"""
Example 10: inference beyond a single best fit, live monitoring, and CARMA.

Demonstrates:
1. Posterior probabilities over discovered structures (``report.posterior``),
   with Laplace and SMC evidence for a weakly identified extra component.
2. Model-averaged option prices that carry structure uncertainty.
3. Monitoring new returns against the discovered law (``report.monitor``):
   an exact false-alarm guarantee, an alarm after a volatility rise, and the
   estimated change point.
4. A Lévy-driven CARMA(2, 1) process: its exact stationary structure, a
   Gaussian quasi-likelihood fit by Kalman filter, and order selection.

Run with ``python examples/10_inference_monitoring_carma.py [--quick]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spxa import CARMAProcess, GammaProcess  # noqa: E402
from spxa.calibration import fit_carma, select_carma_order  # noqa: E402
from spxa.discovery import discover, grammar_from_names  # noqa: E402

DT = 1.0 / 252.0


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main(quick: bool = False) -> None:
    rng = np.random.default_rng(7)
    n = 1500 if quick else 4000

    section("1. Posterior over structures")
    returns = rng.normal(0.06 * DT, 0.2 * np.sqrt(DT), n)
    grammar = grammar_from_names(["BM", "VG", "NormalJumps"], max_terms=2)
    report = discover(returns, dt=DT, grammar=grammar, n_starts=2, seed=1)
    posterior = report.posterior(top=4)
    print(posterior.table(top=4))
    print(f"95% credible set: {[e.structure.key() for e in posterior.credible_set(0.95)]}")
    if not quick:
        smc = report.posterior(top=4, evidence="smc", n_draws=1000, rng=np.random.default_rng(2))
        print("with SMC evidence:")
        print(smc.table(top=4))

    section("2. Model-averaged option prices")
    strikes = np.array([90.0, 100.0, 110.0])
    prices = posterior.option_prices(100.0, strikes, 0.5, rate=0.03)
    for k, m, s in zip(strikes, prices.mean, prices.sd):
        print(f"  call K={k:5.1f}: {m:7.4f}  (posterior sd {s:.4f})")

    section("3. Monitoring new returns")
    monitor = report.monitor(arl0=10_000.0)
    print(
        f"Shiryaev-Roberts over {len(monitor.alternatives)} local alternatives; "
        f"in-control run length at least {monitor.guaranteed_arl0:.0f} days, "
        f"P(false alarm within a year) <= {monitor.false_alarm_bound(252):.3f}"
    )
    calm = rng.normal(0.06 * DT, 0.2 * np.sqrt(DT), 250)
    stressed = rng.normal(0.06 * DT, 0.3 * np.sqrt(DT), 250)
    alarm = monitor.run(np.concatenate([calm, stressed])).first_alarm
    if alarm is None:
        print("no alarm")
    else:
        print(
            f"alarm on day {alarm.time} (volatility rose on day 250); "
            f"estimated change point: day {alarm.change_point}"
        )

    section("4. CARMA(2, 1) driven by a Gamma process")
    truth = CARMAProcess([1.2, 1.0], [0.8, 1.0], driver=GammaProcess(a=3.0, b=2.0))
    print(f"AR zeros {np.round(truth.ar_roots, 3)}, MA zero {np.round(truth.ma_roots, 3)}")
    print(
        f"stationary mean {truth.stationary_mean:.4f}, variance {truth.stationary_variance:.4f}, "
        f"skewness {truth.stationary_cumulant(3) / truth.stationary_variance**1.5:.3f}"
    )
    step = 0.1
    y = truth.simulate(2000 if quick else 5000, 1, T=(2000 if quick else 5000) * step, rng=rng)[0]
    fit = fit_carma(y, 2, 1, step, n_starts=2)
    print(fit.summary())
    print(f"true ar {truth.ar}, ma {truth.ma}, sigma {np.sqrt(3.0 / 4.0):.4f}")
    ranked = select_carma_order(y, step, max_p=2 if quick else 3, n_starts=2)
    print("orders by BIC: " + ", ".join(f"({f.p},{f.q}) {f.bic:.1f}" for f in ranked))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
