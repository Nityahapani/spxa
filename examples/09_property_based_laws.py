"""
Example 09: Property-based testing of stochastic algebra.

Demonstrates:
1. Declaring a law, E[X + Y] = E[X] + E[Y], and checking it over randomised
   processes and parameters.
2. A false law, martingale(X) => martingale(X + c). spxa falsifies it and
   shrinks the counterexample to a minimal one.
3. The subordination identity phi_{X @ T}(u) = exp(psi_T(-i psi_X(u))) from
   the library, checked over composite processes.
4. A statistical law: the simulator of a process against its
   characteristic function, at a known false-alarm rate.
5. A suite of library laws with its report, and how the same laws run
   under pytest and from the shell::

       spxa-laws --tags algebra,subordination --examples 200

Run with ``python examples/09_property_based_laws.py [--quick]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spxa.laws import (  # noqa: E402
    Suite,
    compositions,
    expect_close,
    expect_not_rejected,
    floats,
    implies,
    integers,
    law,
    levy_processes,
    martingales,
)
from spxa.laws.library import ALGEBRA, SIMULATION, SUBORDINATION  # noqa: E402
from spxa.laws.quantities import mean  # noqa: E402
from spxa.laws.stats import ecf_test  # noqa: E402
from spxa.zoo.levy import Drift  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# 1. A law is a predicate plus a strategy per argument ---------------------


@law(X=compositions(max_depth=1), Y=compositions(max_depth=1))
def mean_is_additive(X, Y):
    """E[X + Y] = E[X] + E[Y]"""
    mx, my = mean(X), mean(Y)
    if np.isfinite(mx) and np.isfinite(my):
        expect_close(mean(X + Y), mx + my)


# 2. A law that is false ---------------------------------------------------


def _is_martingale(X) -> bool:
    k1 = mean(X)
    return bool(np.isfinite(k1) and abs(k1) <= 1e-9 * (1 + abs(k1)))


@law(X=martingales(), c=floats(-1.0, 1.0, simplest=0.0))
def martingale_plus_constant(X, c):
    """martingale(X) => martingale(X + c t)"""
    implies(_is_martingale(X), lambda: _is_martingale(X + Drift(c)), "E[X_1 + c] = c")


# 4. A statistical law ------------------------------------------------------


@law(X=levy_processes(), seed=integers(0, 2**31 - 1, simplest=0))
def terminal_value_matches_cf(X, seed):
    """the empirical cf of simulated X_1 matches phi_1"""
    x = X.simulate(4, 4000, 1.0, np.random.default_rng(seed))[:, -1]
    expect_not_rejected(ecf_test(x, lambda u: X.char_func(u, 1.0), [0.5, 2.0]), what="ecf")


def main(quick: bool = False) -> None:
    n = 20 if quick else 200

    section("1. Declaring a law")
    print(mean_is_additive.check(n_examples=n, seed=1))

    section("2. Falsifying a law, with shrinking")
    result = martingale_plus_constant.check(n_examples=n, seed=1)
    print(result)
    X, c = result.counterexample["X"], result.counterexample["c"]
    print(f"\nE[X_1] = {mean(X):.3g}, but E[(X + Drift({c}))_1] = {mean(X + Drift(c)):.3g}.")
    print("The law holds only for c = 0. Shrinking reduced X to the simplest")
    print("martingale and c to a short non-zero number.")

    section("3. The subordination identity, from the library")
    subordination_cf = next(L for L in SUBORDINATION if L.name == "subordination_cf")
    print(f"{subordination_cf.name}: {subordination_cf.description}")
    print(subordination_cf.check(n_examples=n // 2, seed=2))

    section("4. A statistical law: simulator against the cf")
    print(terminal_value_matches_cf.check(n_examples=n // 4, seed=3))
    print("Each example is a test at level 1e-6, so a correct simulator fails")
    print(f"at most {n // 4} x 1e-6 of the time.")

    section("5. A suite and its report")
    suite = Suite("example", ALGEBRA, SUBORDINATION).select(exclude_tags=["moments"])
    report = suite.check(n_examples=10 if quick else 50, seed=4)
    print(report)
    print(f"\nSIMULATION holds {len(SIMULATION)} more laws; `spxa-laws --list` shows all of them.")
    print("Under pytest:  test_algebra = law_tests(ALGEBRA, n_examples=50)")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
