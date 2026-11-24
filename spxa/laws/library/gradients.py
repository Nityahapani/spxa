r"""
Laws of exact parameter derivatives through the process algebra.

:class:`~spxa.graph.autodiff.process.DiffProcess` differentiates the
characteristic exponent, the cumulants and Fourier prices with forward-mode
jets. These laws set those derivatives against independent computations:
Richardson-extrapolated central differences of the *zoo* closed forms (a
separate code path), and the chain rules of the algebra.

==========================================  =====================================================
``psi_gradient_matches_differences``        :math:`\partial_\theta\psi` equals differences of the zoo :math:`\psi`
``cumulant_jacobian_matches_differences``   :math:`\partial_\theta\kappa_n` equals differences of ``cumulants()``
``gradient_of_sum_splits``                  :math:`\partial\psi_{X+Y} = (\partial_{\theta_X}\psi_X, \partial_{\theta_Y}\psi_Y)`
``scaling_chain_rule``                      :math:`\partial_\theta\psi_{cX}(u) = \partial_\theta\psi_X(cu)`, :math:`\partial_c\psi_{cX} = u\,\psi_X'(cu)`
``subordination_chain_rule``                :math:`\partial\psi_{X@T} = \psi_T'(w)(-i)\partial\psi_X`, :math:`w = -i\psi_X(u)`
``hessian_is_symmetric_and_consistent``     :math:`\partial^2_\theta\psi` is symmetric and differentiates :math:`\partial_\theta\psi`
``cos_gradient_is_derivative_of_price``     the COS Jacobian equals differences of the COS price
==========================================  =====================================================

Processes without a differentiable closed form (compound Poisson with a
general jump law) are skipped by assumption. Differences use steps relative
to the parameter, :math:`h = 10^{-4}\max(|\theta_k|, 0.05)`, with one Richardson
step (larger steps let the truncation error near poles, e.g. CGMY's
:math:`\Gamma(-Y)` at :math:`Y \to 2`, reach :math:`10^{-6}`); parameters whose perturbation leaves the valid domain are
skipped.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

from spxa.core.process import Process
from spxa.laws.expect import assume, expect_close
from spxa.laws.law import Law, law
from spxa.laws.library._common import constants, frequencies
from spxa.laws.processes import compositions, levy_processes, subordinators
from spxa.laws.quantities import char_exponent, cumulants, exponential_moment_strip
from spxa.laws.strategies import floats, one_of

_X = compositions(max_depth=1, operations=("+", "*", "-", "@"))


def _diff(X: Process) -> Any:
    from spxa.graph.autodiff.process import DiffProcess, UnsupportedProcessError

    try:
        return DiffProcess(X)
    except UnsupportedProcessError:
        assume(False, "no differentiable closed form")


def _richardson(dp: Any, k: int, f: Callable[[Process], Any]) -> Any:
    """Central difference of ``f(process)`` in free parameter ``k``, one Richardson step."""
    th = np.asarray(dp.values, dtype=float)
    h = 1e-4 * max(abs(th[k]), 0.05)

    def at(step: float) -> Any:
        e = np.zeros_like(th)
        e[k] = step
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                up, dn = f(dp.to_process(th + e)), f(dp.to_process(th - e))
        except (ValueError, ZeroDivisionError, OverflowError):
            assume(False, "perturbed parameters leave the domain")
        return (np.asarray(up) - np.asarray(dn)) / (2 * step)

    return (4 * at(h / 2) - at(h)) / 3


def _u_grid(u: float) -> np.ndarray:
    return np.array([u, 0.37 * u - 0.5, 2.1])


@law(X=_X, u=frequencies(), tags=("gradients", "cf"))
def psi_gradient_matches_differences(X: Process, u: float) -> None:
    """d psi / d theta (jets) = differences of the zoo exponent"""
    dp = _diff(X)
    uu = _u_grid(u)
    psi, g = dp.char_exponent_and_grad(uu)
    scale = 1.0 + np.abs(psi)
    for k in range(len(dp.names)):
        fd = _richardson(dp, k, lambda P: char_exponent(P, uu))
        th = max(abs(float(dp.values[k])), 1.0)
        expect_close(
            g[:, k] * th / scale,
            fd * th / scale,
            rtol=1e-6,
            atol=1e-7,
            what=f"d psi / d {dp.names[k]}",
        )


@law(X=_X, tags=("gradients", "cumulants"))
def cumulant_jacobian_matches_differences(X: Process) -> None:
    """d kappa_n / d theta (Taylor jets) = differences of cumulants(), n <= 4"""
    dp = _diff(X)
    assume(dp.analytic, "cumulants need finite moments")
    lo, hi = exponential_moment_strip(X)
    assume(hi - lo > 1e-3, "exponential moment strip too narrow")
    kap, J = dp.cumulants_and_jacobian(4)
    scale = 1.0 + np.abs(kap)
    for k in range(len(dp.names)):
        fd = _richardson(dp, k, lambda P: cumulants(P, 4))
        th = max(abs(float(dp.values[k])), 1.0)
        expect_close(
            J[:, k] * th / scale,
            fd * th / scale,
            rtol=1e-6,
            atol=1e-7,
            what=f"d kappa / d {dp.names[k]}",
        )


_PRIM = levy_processes()


@law(X=_PRIM, Y=_PRIM, u=frequencies(), tags=("gradients", "algebra", "addition"))
def gradient_of_sum_splits(X: Process, Y: Process, u: float) -> None:
    """d psi_{X+Y} = (d psi_X / d theta_X, d psi_Y / d theta_Y)"""
    dx, dy = _diff(X), _diff(Y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dz = _diff(X + Y)
    uu = _u_grid(u)
    _, gz = dz.char_exponent_and_grad(uu)
    _, gx = dx.char_exponent_and_grad(uu)
    _, gy = dy.char_exponent_and_grad(uu)
    expect_close(gz, np.concatenate([gx, gy], axis=1), rtol=1e-13, atol=1e-14, what="gradient")


@law(X=_PRIM, c=constants(), u=frequencies(), tags=("gradients", "algebra", "scaling"))
def scaling_chain_rule(X: Process, c: float, u: float) -> None:
    """d psi_{cX}(u)/d theta = d psi_X(cu)/d theta; d psi_{cX}/dc = u psi_X'(cu)"""
    from spxa.graph.autodiff.jet import Jet

    dx = _diff(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dz = _diff(c * X)
    uu = _u_grid(u)
    _, gz = dz.char_exponent_and_grad(uu)
    _, gx = dx.char_exponent_and_grad(c * uu)
    n = len(dx.names)
    expect_close(gz[:, :n], gx, rtol=1e-12, atol=1e-13, what="d / d theta_X")
    # psi_X'(cu) from a jet in the frequency
    w = Jet(c * uu.astype(complex), np.ones((uu.size, 1)))
    dpsi = np.asarray(dx.char_exponent(w).grad)[:, 0]
    expect_close(gz[:, n], uu * dpsi, rtol=1e-12, atol=1e-13, what="d / d c")


@law(
    X=one_of(levy_processes(["Drift", "BrownianMotion", "VarianceGamma", "NIG", "CGMY"])),
    T=subordinators(),
    u=frequencies(),
    tags=("gradients", "subordination"),
)
def subordination_chain_rule(X: Process, T: Process, u: float) -> None:
    """d psi_{X@T} = psi_T'(w) (-i) d psi_X (theta_X); d psi_T(w) (theta_T); w = -i psi_X(u)"""
    from spxa.graph.autodiff.jet import Jet

    dx, dt = _diff(X), _diff(T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dz = _diff(X @ T)
    uu = _u_grid(u)
    _, gz = dz.char_exponent_and_grad(uu)
    psi_x, gx = dx.char_exponent_and_grad(uu)
    w = -1j * psi_x
    _, gt = dt.char_exponent_and_grad(w)
    dpsi_t = np.asarray(dt.char_exponent(Jet(w, np.ones((uu.size, 1)))).grad)[:, 0]
    n = len(dx.names)
    expect_close(
        gz[:, :n], dpsi_t[:, None] * (-1j) * gx, rtol=1e-12, atol=1e-13, what="d / d theta_X"
    )
    expect_close(gz[:, n:], gt, rtol=1e-12, atol=1e-13, what="d / d theta_T")


@law(X=_X, u=frequencies(), tags=("gradients", "hessian"))
def hessian_is_symmetric_and_consistent(X: Process, u: float) -> None:
    """d2 psi / d theta2 is symmetric and equals differences of the jet gradient"""
    dp = _diff(X)
    uu = _u_grid(u)
    psi, g, H = dp.char_exponent_hessian(uu)
    scale = (1.0 + np.abs(psi))[:, None, None]
    expect_close(H / scale, np.swapaxes(H, 1, 2) / scale, rtol=1e-12, atol=1e-13, what="symmetry")
    th0 = np.asarray(dp.values, dtype=float)
    for k in range(len(dp.names)):
        h = 1e-4 * max(abs(th0[k]), 0.05)

        def d(step: float) -> np.ndarray:
            e = np.zeros_like(th0)
            e[k] = step
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    up = dp.char_exponent_and_grad(uu, th0 + e)[1]
                    dn = dp.char_exponent_and_grad(uu, th0 - e)[1]
            except (ValueError, ZeroDivisionError, OverflowError):
                assume(False, "perturbed parameters leave the domain")
            return np.asarray((up - dn) / (2 * step))

        fd = (4 * d(h / 2) - d(h)) / 3
        tk = max(abs(float(th0[k])), 1.0)
        expect_close(
            H[:, :, k] * tk / scale[:, :, 0],
            fd * tk / scale[:, :, 0],
            rtol=1e-6,
            atol=1e-7,
            what=f"d grad / d {dp.names[k]}",
        )


_PRICEABLE = levy_processes(
    ["BrownianMotion", "VarianceGamma", "NIG", "MeixnerProcess", "CGMY", "MertonJumpDiffusion"]
)


@law(
    X=_PRICEABLE,
    moneyness=floats(-0.3, 0.3, simplest=0.0),
    T=floats(0.25, 2.0, simplest=1.0),
    tags=("gradients", "pricing"),
)
def cos_gradient_is_derivative_of_price(X: Process, moneyness: float, T: float) -> None:
    """the COS Jacobian (interval tracked) = differences of the COS price"""
    from spxa.graph.autodiff.fourier import cos_price, price_and_gradient

    dp = _diff(X)
    lo, hi = exponential_moment_strip(X)
    assume(hi > 1.05, "E[exp(X)] must be finite for the martingale correction")
    S, r, q = 100.0, 0.02, 0.01
    K = S * np.exp(np.array([moneyness, 0.0, -moneyness / 2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        J = price_and_gradient(dp, S, K, T, method="cos", rate=r, dividend=q).jacobian
    th0 = np.asarray(dp.values, dtype=float)
    for k in range(len(dp.names)):
        h = 1e-4 * max(abs(th0[k]), 0.05)

        def d(step: float) -> np.ndarray:
            e = np.zeros_like(th0)
            e[k] = step
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    up = cos_price(dp, S, K, T, r, q, theta=th0 + e)
                    dn = cos_price(dp, S, K, T, r, q, theta=th0 - e)
            except (ValueError, ZeroDivisionError, OverflowError):
                assume(False, "perturbed parameters leave the domain")
            return np.asarray((up - dn) / (2 * step))

        fd = (4 * d(h / 2) - d(h)) / 3
        tk = max(abs(float(th0[k])), 1.0)
        expect_close(
            J[:, k] * tk, fd * tk, rtol=1e-6, atol=1e-7 * S, what=f"d price / d {dp.names[k]}"
        )


GRADIENTS: tuple[Law, ...] = (
    psi_gradient_matches_differences,
    cumulant_jacobian_matches_differences,
    gradient_of_sum_splits,
    scaling_chain_rule,
    subordination_chain_rule,
    hessian_is_symmetric_and_consistent,
    cos_gradient_is_derivative_of_price,
)
