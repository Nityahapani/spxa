r"""
Simulation against the closed forms.

Every process can be simulated and also has a characteristic function.
These laws check that the two describe the same law, using the
statistical expectations of :mod:`spxa.laws.stats`. Each example draws its
own seed, so runs are reproducible, and the false-alarm rate is at most
``n_examples * LEVEL``.

===================================  ================================================================
``simulated_cf``                     the empirical cf of :math:`X_T` matches :math:`\varphi_T`
``simulated_increments_cf``          the first increment over :math:`\Delta t = T/n` matches :math:`\varphi_{\Delta t}`
``simulated_increments_independent``  :math:`E e^{i(u\Delta_1 + v\Delta_2)} = \varphi_{\Delta t}(u)\varphi_{\Delta t}(v)`
``simulated_mean``                   :math:`\bar X_T` is consistent with :math:`\kappa_1 T` (finite variance)
``simulated_variance``               :math:`s^2(X_T)` is consistent with :math:`\kappa_2 T` (finite :math:`\kappa_4`)
``sample_at_matches_cf``             ``sample_at`` draws at mixed horizons match :math:`\varphi_t`
``paths_start_at_zero``              every path starts at 0 and has ``n_steps + 1`` finite values
===================================  ================================================================

Frequencies are scaled by the standard deviation of :math:`X_T` (when it is
finite), so a test probes the bulk of the law rather than a region where
:math:`\varphi` is 1 or 0.

The processes include heavy-tailed α-stable laws and depth-1 compositions,
so the simulators of sums, multiples and time changes are covered as well.
"""

from __future__ import annotations

import warnings

import numpy as np

from spxa.core.process import Process
from spxa.laws.expect import assume, expect
from spxa.laws.law import Law, law
from spxa.laws.processes import compositions, levy_processes
from spxa.laws.quantities import char_func
from spxa.laws.stats import (
    ecf_test,
    expect_not_rejected,
    independence_test,
    mean_test,
    variance_test,
)
from spxa.laws.strategies import floats, integers, sampled_from

#: Paths per example: enough for z-tests, few enough for quick checks.
N_PATHS = 4000

_X = compositions(levy_processes(heavy_tails=True), max_depth=1)
_SEED = integers(0, 2**31 - 1, simplest=0)
_T = floats(0.25, 2.0, simplest=1.0)


def _simulate(X: Process, n_steps: int, T: float, seed: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(X.simulate(n_steps, N_PATHS, T, np.random.default_rng(seed)))


def _scale(X: Process, t: float) -> float:
    """Standard deviation of X_t when finite and positive, else 1."""
    try:
        k2 = float(X.cumulants(2)[2])
    except Exception:  # noqa: BLE001  (heavy tails: cumulants may not exist)
        return 1.0
    return float(np.sqrt(k2 * t)) if np.isfinite(k2) and k2 > 0 else 1.0


def _phi(X: Process, t: float):  # type: ignore[no-untyped-def]
    return lambda u: complex(char_func(X, u, t))


@law(X=_X, T=_T, seed=_SEED, tags=("simulation", "cf"))
def simulated_cf(X: Process, T: float, seed: int) -> None:
    """empirical cf of simulated X_T = phi_T"""
    x = _simulate(X, 2, T, seed)[:, -1]
    s = _scale(X, T)
    expect_not_rejected(ecf_test(x, _phi(X, T), [0.5 / s, 2.0 / s]), what="ecf of X_T")


@law(X=_X, n_steps=sampled_from([2, 5]), seed=_SEED, tags=("simulation", "cf"))
def simulated_increments_cf(X: Process, n_steps: int, seed: int) -> None:
    """empirical cf of the first increment = phi_dt"""
    dt = 1.0 / n_steps
    x = _simulate(X, n_steps, 1.0, seed)
    s = _scale(X, dt)
    expect_not_rejected(
        ecf_test(x[:, 1] - x[:, 0], _phi(X, dt), [0.5 / s, 2.0 / s]), what="ecf of an increment"
    )


@law(X=_X, seed=_SEED, tags=("simulation", "independence"))
def simulated_increments_independent(X: Process, seed: int) -> None:
    """joint ecf of two increments = product of marginals"""
    x = _simulate(X, 2, 1.0, seed)
    d = np.diff(x, axis=1)
    s = _scale(X, 0.5)
    phi = _phi(X, 0.5)
    pairs = [(1.0 / s, 1.0 / s), (1.0 / s, -1.0 / s)]
    expect_not_rejected(
        independence_test(d[:, 0], d[:, 1], phi, phi, pairs), what="increment independence"
    )


@law(X=_X, T=_T, seed=_SEED, tags=("simulation", "moments"))
def simulated_mean(X: Process, T: float, seed: int) -> None:
    """mean of simulated X_T consistent with kappa_1 T"""
    assume(X.properties.has_finite_variance, "needs a finite variance")
    k = X.cumulants(2)
    x = _simulate(X, 2, T, seed)[:, -1]
    expect_not_rejected(mean_test(x, k[1] * T, k[2] * T), what="sample mean")


@law(X=_X, T=_T, seed=_SEED, tags=("simulation", "moments"))
def simulated_variance(X: Process, T: float, seed: int) -> None:
    """sample variance of simulated X_T consistent with kappa_2 T"""
    assume(X.properties.has_finite_variance, "needs finite moments")
    k = X.cumulants(4)
    assume(np.isfinite(k[4]), "needs a finite fourth cumulant")
    x = _simulate(X, 2, T, seed)[:, -1]
    expect_not_rejected(variance_test(x, k[2] * T, k[4] * T), what="sample variance")


@law(X=_X, seed=_SEED, tags=("simulation", "sample_at"))
def sample_at_matches_cf(X: Process, seed: int) -> None:
    """sample_at at mixed horizons matches phi_t at each horizon"""
    times = np.tile([0.3, 1.7], N_PATHS // 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = X.sample_at(times, np.random.default_rng(seed))
    for t in (0.3, 1.7):
        s = _scale(X, t)
        expect_not_rejected(
            ecf_test(x[times == t], _phi(X, t), [0.5 / s, 2.0 / s]), what=f"sample_at({t})"
        )


@law(X=_X, n_steps=integers(1, 6, simplest=1), seed=_SEED, tags=("simulation", "shape"))
def paths_start_at_zero(X: Process, n_steps: int, seed: int) -> None:
    """paths have shape (n_paths, n_steps + 1), start at 0 and are finite"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = np.asarray(X.simulate(n_steps, 3, 1.0, np.random.default_rng(seed)))
    expect(x.shape == (3, n_steps + 1), f"shape {x.shape}")
    expect(bool(np.all(x[:, 0] == 0.0)), "paths do not start at 0", first=x[:, 0])
    expect(bool(np.all(np.isfinite(x))), "non-finite path values")


SIMULATION: tuple[Law, ...] = (
    simulated_cf,
    simulated_increments_cf,
    simulated_increments_independent,
    simulated_mean,
    simulated_variance,
    sample_at_matches_cf,
    paths_start_at_zero,
)
