"""Exact gradients of the discovery objectives (spxa.graph.autodiff.discovery)."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pytest

from spxa.discovery import families as F
from spxa.discovery.empirical import from_statistics, summarize
from spxa.discovery.expr import Model, Node, leaf
from spxa.discovery.fit import BinnedData, _Objective, fit_structure, initial_params
from spxa.discovery.options import OptionQuotes, _vegas, fit_options, model_prices
from spxa.graph.autodiff.discovery import (
    FreeModel,
    cumulant_value_and_grad,
    likelihood_value_and_grad,
    options_residual_jacobian,
)
from spxa.graph.autodiff.jet import variable_vector


def _rich(f: Callable[[np.ndarray], Any], x: np.ndarray, k: int, h: float) -> np.ndarray:
    def d(hh: float) -> np.ndarray:
        e = np.zeros_like(x)
        e[k] = hh
        return (np.asarray(f(x + e)) - np.asarray(f(x - e))) / (2 * hh)

    return (4 * d(h / 2) - d(h)) / 3


STRUCTURES = [leaf(f) for f in F.FAMILIES.values()] + [
    leaf(F.VARIANCE_GAMMA) + leaf(F.DRIFT),
    leaf(F.BROWNIAN) @ leaf(F.GAMMA),
    0.5 * leaf(F.NIG_FAMILY) + leaf(F.NORMAL_JUMPS),
    leaf(F.BROWNIAN) @ leaf(F.INVERSE_GAUSSIAN) + leaf(F.DRIFT),
]


@pytest.mark.parametrize("s", STRUCTURES, ids=[s.key() for s in STRUCTURES])
def test_log_mgf_and_cumulant_gradients(s: Node) -> None:
    m = Model(s, initial_params(s, 0.2))
    x = m.to_free()
    fm = FreeModel(s, x)
    th = variable_vector(x)
    z = np.array([-0.4, 0.1j, 0.3 + 0.5j, 1.2j, -2j])
    K = fm.log_mgf(z, th)
    np.testing.assert_allclose(K.value, m.log_mgf(z), rtol=1e-14, atol=1e-15)
    kap = fm.cumulants(4, th)
    ref = m.cumulants(4)
    np.testing.assert_allclose(kap.value, ref, rtol=1e-13, atol=1e-15)
    for k in range(x.size):
        gK = _rich(lambda t: Model.from_free(s, t).log_mgf(z), x, k, 1e-3)
        np.testing.assert_allclose(K.grad[:, k], gK, rtol=1e-9, atol=1e-11)
        gk = _rich(lambda t: Model.from_free(s, t).cumulants(4), x, k, 1e-3)
        np.testing.assert_allclose(kap.grad[:, k], gk, rtol=1e-9, atol=1e-11)


@pytest.fixture(scope="module")
def sample_summary() -> Any:
    rng = np.random.default_rng(3)
    true = Model(leaf(F.VARIANCE_GAMMA) + leaf(F.DRIFT), [(0.2, 0.3, -0.1), (0.01,)])
    paths = true.to_process().simulate(2000, 1, T=2000 / 52, rng=rng)
    return summarize(np.diff(paths[0]), dt=1 / 52)


OBJ_STRUCTURES = [
    leaf(F.VARIANCE_GAMMA) + leaf(F.DRIFT),
    leaf(F.NIG_FAMILY) + leaf(F.DRIFT),
    leaf(F.BROWNIAN) + leaf(F.NORMAL_JUMPS) + leaf(F.DRIFT),
    leaf(F.MEIXNER) + leaf(F.DRIFT),
    leaf(F.CGMY_FAMILY) + leaf(F.DRIFT),
]


@pytest.mark.parametrize("s", OBJ_STRUCTURES, ids=[s.key() for s in OBJ_STRUCTURES])
def test_likelihood_and_cumulant_objective_gradients(s: Node, sample_summary: Any) -> None:
    warnings.simplefilter("ignore")
    summ = sample_summary
    x = Model(s, initial_params(s, float(np.sqrt(summ.variance)))).to_free() * 1.1 + 0.05
    bd = BinnedData.from_sample(summ.sample, summ.dt)
    obj = _Objective(s, summ, "likelihood", bd)
    v, g = likelihood_value_and_grad(s, bd, x)
    assert v == pytest.approx(obj(x), rel=1e-14)
    fd = np.array([_rich(obj, x, k, 1e-4) for k in range(x.size)])
    np.testing.assert_allclose(g, fd, rtol=0, atol=1e-8 * np.max(np.abs(fd)))
    k, dt = summ.cumulants, summ.dt
    var = k[1] * dt
    plain = from_statistics(k[0] * dt, var, k[2] * dt / var**1.5, k[3] * dt / var**2, dt=dt)
    for sm in (summ, plain):  # weighted by sampling variances, and moment errors
        oc = _Objective(s, sm, "cumulants", None)
        v, g = cumulant_value_and_grad(s, sm.cumulants, x, sm.dt, oc.weights, oc.drift_at)
        assert v == pytest.approx(oc(x), rel=1e-13)
        fd = np.array([_rich(oc, x, k, 1e-4) for k in range(x.size)])
        np.testing.assert_allclose(g, fd, rtol=0, atol=1e-9 * max(1.0, np.max(np.abs(fd))))


@pytest.fixture(scope="module")
def quotes() -> OptionQuotes:
    true = Model(leaf(F.NIG_FAMILY) + leaf(F.DRIFT), [(8.0, -2.0, 0.5), (0.0,)])
    K = np.tile([70, 85, 95, 100, 105, 115, 130.0], 2)
    T = np.repeat([0.25, 1.0], 7)
    q0 = OptionQuotes.from_arrays(100.0, K, T, np.ones(14), K >= 100, 0.02, 0.01)
    return OptionQuotes.from_arrays(100.0, K, T, model_prices(true, q0), K >= 100, 0.02, 0.01)


OPT_STRUCTURES = [
    leaf(F.NIG_FAMILY) + leaf(F.DRIFT),
    leaf(F.VARIANCE_GAMMA),
    leaf(F.BROWNIAN) + leaf(F.DOUBLE_EXP_JUMPS),
    leaf(F.CGMY_FAMILY),
    leaf(F.MEIXNER),
    leaf(F.BROWNIAN) @ leaf(F.INVERSE_GAUSSIAN),
]


@pytest.mark.parametrize("s", OPT_STRUCTURES, ids=[s.key() for s in OPT_STRUCTURES])
def test_option_residual_jacobian(s: Node, quotes: OptionQuotes) -> None:
    warnings.simplefilter("ignore")
    vega = _vegas(quotes, quotes.implied_vols())
    x = Model(s, initial_params(s, 0.2)).to_free()

    def resid(t: np.ndarray) -> np.ndarray:
        return (model_prices(Model.from_free(s, t), quotes) - quotes.prices) / vega

    J = options_residual_jacobian(s, quotes, x, vega)
    fd = np.array([_rich(resid, x, k, 1e-4) for k in range(x.size)]).T
    np.testing.assert_allclose(J, fd, rtol=0, atol=1e-8 * np.max(np.abs(fd)))


def test_exact_and_difference_fits_agree(quotes: OptionQuotes, sample_summary: Any) -> None:
    warnings.simplefilter("ignore")
    s = leaf(F.NIG_FAMILY)
    a = fit_options(s, quotes, jac="exact", n_starts=1)
    b = fit_options(s, quotes, jac="2-point", n_starts=1)
    assert a.objective <= b.objective * (1 + 1e-6) + 1e-20
    np.testing.assert_allclose(a.model.to_free(), b.model.to_free(), rtol=1e-4, atol=1e-6)
    s2 = leaf(F.VARIANCE_GAMMA)
    c = fit_structure(s2, sample_summary, gradient="exact", n_starts=1)
    d = fit_structure(s2, sample_summary, gradient="2-point", n_starts=1)
    assert c.objective <= d.objective + 1e-8
    np.testing.assert_allclose(c.model.to_free(), d.model.to_free(), rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError):
        fit_options(s, quotes, jac="3-point")
    with pytest.raises(ValueError):
        fit_structure(s2, sample_summary, gradient="adjoint")
