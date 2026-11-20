"""FFT inversion sampler and the zoo simulators built on it.

Statistical checks use the empirical characteristic function. For n draws,
Re and Im of mean(exp(iuX)) have known variances (1 ± Re φ(2u))/2 − (Re/Im φ(u))²,
so each z-score is standard normal under the null. The |z| < 4.5 threshold has
a false-alarm rate of about 7e-6 per comparison, and the seeds are fixed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats  # type: ignore[import-untyped]

from spxa.sim.cf_sampling import cf_grid_cdf, sample_from_cf
from spxa.zoo.levy import CGMY, GammaProcess, MeixnerProcess, TemperedStable

Z_MAX = 4.5


def _ecf_z(x: np.ndarray, phi, u: float) -> float:
    p = complex(np.atleast_1d(phi(np.array([u])))[0])
    p2 = complex(np.atleast_1d(phi(np.array([2 * u])))[0])
    m = np.mean(np.exp(1j * u * x))
    var_re = max((1 + p2.real) / 2 - p.real**2, 1e-30)
    var_im = max((1 - p2.real) / 2 - p.imag**2, 1e-30)
    n = len(x)
    return max(
        abs(m.real - p.real) / np.sqrt(var_re / n), abs(m.imag - p.imag) / np.sqrt(var_im / n)
    )


def test_grid_cdf_matches_gamma() -> None:
    X = GammaProcess(20.0, 3.0)
    k = X.cumulants(2)
    x, cdf = cf_grid_cdf(lambda u: X.char_func(u, 0.25), k[1] * 0.25, np.sqrt(k[2] * 0.25))
    exact = stats.gamma(5.0, scale=1 / 3.0).cdf(x)
    assert np.max(np.abs(cdf - exact)) < 1e-6


def test_grid_refines_until_the_cf_has_decayed() -> None:
    X = GammaProcess(4.0, 3.0)  # cf decays like |u|^-1 at t = 0.25, so N must grow
    k = X.cumulants(2)
    x, _ = cf_grid_cdf(lambda u: X.char_func(u, 0.25), k[1] * 0.25, np.sqrt(k[2] * 0.25))
    assert len(x) > 2**14


def test_sampler_passes_ks_on_gamma() -> None:
    X = GammaProcess(20.0, 3.0)
    k = X.cumulants(2)
    draws = sample_from_cf(
        lambda u: X.char_func(u, 0.25),
        k[1] * 0.25,
        np.sqrt(k[2] * 0.25),
        20_000,
        np.random.default_rng(0),
    )
    assert stats.kstest(draws, stats.gamma(5.0, scale=1 / 3.0).cdf).pvalue > 1e-3


def test_rejects_degenerate_scale() -> None:
    with pytest.raises(ValueError, match="positive std"):
        cf_grid_cdf(lambda u: np.ones_like(u, dtype=complex), 0.0, 0.0)


@pytest.mark.parametrize(
    "X",
    [
        CGMY(1.07, 9.55, 2.73, 1.81),  # the old fixed grid aliased this: z = 533
        CGMY(1.31, 3.43, 1.85, 0.13),
        MeixnerProcess(1.31, -1.15, 0.178, -0.19),
        MeixnerProcess(0.6, -1.0, 1.65, -0.16),
        TemperedStable(0.9, 0.2, 0.5),
        TemperedStable(0.31, 0.67, 8.2),
    ],
    ids=repr,
)
def test_simulated_increments_match_the_cf(X) -> None:
    dt = 0.25
    x = X.simulate(4, 50_000, T=1.0, rng=np.random.default_rng(3))
    incr = x[:, 1]
    final = x[:, -1]
    for u in (0.5, 2.0, 8.0):
        assert _ecf_z(incr, lambda v: X.char_func(v, dt), u) < Z_MAX
        assert _ecf_z(final, lambda v: X.char_func(v, 1.0), u) < Z_MAX


def test_lower_support_bound_is_respected() -> None:
    X = TemperedStable(0.31, 0.67, 8.2)
    paths = X.simulate(50, 2_000, T=1.0, rng=np.random.default_rng(1))
    assert np.all(np.diff(paths, axis=1) >= 0.0)


def test_lattice_law_is_reported_not_inverted() -> None:
    from spxa.sim.cf_sampling import CharacteristicFunctionNotDecayed
    from spxa.zoo.levy import PoissonProcess

    X = PoissonProcess(2.0)  # integer valued: |phi| is periodic and never decays
    with pytest.raises(CharacteristicFunctionNotDecayed):
        cf_grid_cdf(lambda u: X.char_func(u), 2.0, np.sqrt(2.0), max_points=2**15, fail_above=1e-4)
