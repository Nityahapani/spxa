"""The exact Hessian of the binned discovery likelihood."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from spxa.discovery import families as F
from spxa.discovery.empirical import summarize
from spxa.discovery.expr import Model, Node, leaf
from spxa.discovery.fit import BinnedData, _Objective, fit_structure, initial_params
from spxa.graph.autodiff.discovery import (
    likelihood_value_and_grad,
    likelihood_value_grad_hess,
)


@pytest.fixture(scope="module")
def data() -> Any:
    rng = np.random.default_rng(3)
    true = Model(leaf(F.VARIANCE_GAMMA) + leaf(F.DRIFT), [(0.2, 0.3, -0.1), (0.01,)])
    paths = true.to_process().simulate(1500, 1, T=1500 / 52, rng=rng)
    summ = summarize(np.diff(paths[0]), dt=1 / 52)
    return summ, BinnedData.from_sample(summ.sample, summ.dt)


STRUCTURES = [
    leaf(F.VARIANCE_GAMMA) + leaf(F.DRIFT),
    leaf(F.NIG_FAMILY) + leaf(F.DRIFT),
    leaf(F.BROWNIAN) + leaf(F.NORMAL_JUMPS) + leaf(F.DRIFT),
    leaf(F.MEIXNER) + leaf(F.DRIFT),
    leaf(F.CGMY_FAMILY) + leaf(F.DRIFT),
    leaf(F.BROWNIAN) @ leaf(F.GAMMA) + leaf(F.DRIFT),
    0.5 * leaf(F.NIG_FAMILY) + leaf(F.DOUBLE_EXP_JUMPS) + leaf(F.DRIFT),
]


def _grad(s: Node, bd: BinnedData, x: np.ndarray) -> np.ndarray:
    return likelihood_value_and_grad(s, bd, x)[1]


def _rich_jac(s: Node, bd: BinnedData, x: np.ndarray, h: float) -> np.ndarray:
    """Richardson-extrapolated central differences of the exact gradient."""
    n = x.size
    out = np.empty((n, n))
    for k in range(n):

        def d(hh: float, k: int = k) -> np.ndarray:
            e = np.zeros(n)
            e[k] = hh
            return (_grad(s, bd, x + e) - _grad(s, bd, x - e)) / (2 * hh)

        out[:, k] = (4 * d(h / 2) - d(h)) / 3
    return out


@pytest.mark.parametrize("s", STRUCTURES, ids=[s.key() for s in STRUCTURES])
def test_hessian_matches_differences_of_the_exact_gradient(s: Node, data: Any) -> None:
    warnings.simplefilter("ignore")
    summ, bd = data
    x = Model(s, initial_params(s, float(np.sqrt(summ.variance)))).to_free() * 1.1 + 0.05
    v1, g1 = likelihood_value_and_grad(s, bd, x)
    v2, g2, H = likelihood_value_grad_hess(s, bd, x)
    assert v2 == v1
    np.testing.assert_allclose(g2, g1, rtol=1e-12, atol=1e-12 * np.max(np.abs(g1)))
    assert v2 == pytest.approx(_Objective(s, summ, "likelihood", bd)(x), rel=1e-14)
    np.testing.assert_array_equal(H, H.T)
    fd = _rich_jac(s, bd, x, 1e-4)
    scale = np.max(np.abs(fd))
    np.testing.assert_allclose(H, fd, rtol=0, atol=1e-8 * scale)


def test_hessian_at_the_optimum_is_positive_definite(data: Any) -> None:
    summ, bd = data
    s = leaf(F.VARIANCE_GAMMA)
    res = fit_structure(s, summ, binned=bd, n_starts=2)
    structure = res.model.structure
    x = res.model.to_free()
    _, g, H = likelihood_value_grad_hess(structure, bd, x)
    w = np.linalg.eigvalsh(H)
    assert w.min() > 0
    # first-order optimality, relative to the curvature: the Newton step is tiny
    step = np.linalg.solve(H, g)
    assert np.max(np.abs(step)) < 1e-3


def test_hessian_is_the_observed_information_of_a_gaussian(data: Any) -> None:
    r"""BM + drift: the curvature in ``log sigma`` is about ``2n`` at the optimum.

    For i.i.d. Gaussian data the observed information of :math:`\log\sigma` at
    the MLE is :math:`2n`. The binned, smoothed likelihood loses a little
    information to binning, so the exact Hessian lands slightly below that.
    """
    rng = np.random.default_rng(11)
    r = 0.01 + 0.2 * np.sqrt(1 / 52) * rng.standard_normal(4000)
    summ = summarize(r, dt=1 / 52)
    bd = BinnedData.from_sample(summ.sample, summ.dt)
    res = fit_structure(leaf(F.BROWNIAN), summ, binned=bd, n_starts=1)
    _, _, H = likelihood_value_grad_hess(res.model.structure, bd, res.model.to_free())
    n = summ.n
    assert 0.85 * 2 * n < H[0, 0] < 1.01 * 2 * n
    # the drift's information is n dt^2 / (sigma^2 dt) = n dt / sigma^2
    sigma = res.model.params[0][0]
    info_mu = n * summ.dt / sigma**2
    assert 0.9 * info_mu < H[1, 1] < 1.01 * info_mu
