"""Simulation laws: hold on the zoo, catch broken simulators, replay counterexamples."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from spxa.laws import Law
from spxa.laws.library.simulation import (
    SIMULATION,
    paths_start_at_zero,
    sample_at_matches_cf,
    simulated_cf,
    simulated_increments_independent,
    simulated_variance,
)
from spxa.laws.strategies import just
from spxa.zoo.levy import BrownianMotion, GammaProcess, TemperedStable


@pytest.mark.parametrize("law_", SIMULATION, ids=lambda law_: law_.name)
def test_holds_on_the_zoo(law_: Law) -> None:
    result = law_.check(n_examples=12, seed=11)
    assert result.ok, str(result)


def _on(law_: Law, X: object) -> Law:
    return Law(law_.name, law_.fn, {**law_.strategies, "X": just(X)})


class _WrongScale(BrownianMotion):
    """Simulates 1.1 sigma: a 10% scale error, as from a sqrt(dt) slip."""

    def simulate(self, n_steps, n_paths=1, T=1.0, rng=None):  # type: ignore[no-untyped-def]
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        z = rng.normal(self.mu * dt, 1.1 * self.sigma * np.sqrt(dt), (n_paths, n_steps))
        return np.concatenate([np.zeros((n_paths, 1)), np.cumsum(z, axis=1)], axis=1)


class _Dependent(BrownianMotion):
    """Second increment repeats half of the first: right marginals, dependent increments."""

    def simulate(self, n_steps, n_paths=1, T=1.0, rng=None):  # type: ignore[no-untyped-def]
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        z = rng.standard_normal((n_paths, n_steps))
        z[:, 1:] = 0.5 * z[:, :-1] + np.sqrt(0.75) * z[:, 1:]
        inc = self.mu * dt + self.sigma * np.sqrt(dt) * z
        return np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)


class _NonZeroStart(BrownianMotion):
    def simulate(self, n_steps, n_paths=1, T=1.0, rng=None):  # type: ignore[no-untyped-def]
        return super().simulate(n_steps, n_paths, T, rng) + 1.0


class _BadSampleAt(GammaProcess):
    """sample_at ignores the horizon beyond t = 1."""

    def sample_at(self, times, rng: Optional[np.random.Generator] = None):  # type: ignore[no-untyped-def]
        return super().sample_at(np.minimum(times, 1.0), rng)


def test_scale_error_is_caught() -> None:
    result = _on(simulated_cf, _WrongScale(0.0, 1.0)).check(n_examples=5)
    assert result.status == "falsified"
    assert "ecf" in str(result)


def test_variance_law_catches_the_scale_error() -> None:
    assert _on(simulated_variance, _WrongScale(0.1, 0.5)).check(n_examples=5).status == "falsified"


def test_dependent_increments_are_caught() -> None:
    result = _on(simulated_increments_independent, _Dependent(0.0, 1.0)).check(n_examples=5)
    assert result.status == "falsified"
    assert "independence" in str(result)


def test_paths_must_start_at_zero() -> None:
    result = _on(paths_start_at_zero, _NonZeroStart(0.0, 1.0)).check(n_examples=3)
    assert result.status == "falsified"


def test_sample_at_horizon_is_checked() -> None:
    assert _on(sample_at_matches_cf, _BadSampleAt(2.0, 3.0)).check(n_examples=3).status == (
        "falsified"
    )


def test_tempered_stable_clock_counterexample_now_holds() -> None:
    """Falsified before the exact tempered stable sampler (p = 7e-12)."""
    from spxa.laws.library.simulation import simulated_mean

    X = TemperedStable(0.6, 1.0, 1.0) @ GammaProcess(1.0, 2.5)
    law_ = Law(
        "replay",
        simulated_mean.fn,
        {"X": just(X), "T": just(1.0), "seed": just(949413399)},
    )
    assert law_.check(n_examples=1).ok
