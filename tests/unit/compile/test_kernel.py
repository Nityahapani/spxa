"""Compiled exponent kernels: values, gradients, Hessians, batching and specialisation."""

from __future__ import annotations


import numpy as np
import pytest

from spxa.compile.ir import CompileError
from spxa.compile.kernel import ExponentKernel, resolve_free
from spxa.compile.lower import lower
from spxa.graph.autodiff.process import DiffProcess
from spxa.zoo import levy as L

U = np.linspace(-40.0, 40.0, 81)
UC = np.linspace(-4.0, 4.0, 9) - 0.25j


def _backends() -> list:
    out = ["numpy"]
    for mod, be in (("numba", "numba"), ("jax", "jax")):
        try:
            __import__(mod)
            out.append(be)
        except ImportError:  # pragma: no cover - depends on the environment
            pass
    return out


BACKENDS = _backends()


def _close(got: np.ndarray, ref: np.ndarray, rtol: float) -> None:
    ref = np.asarray(ref)
    got = np.broadcast_to(got, ref.shape)
    err = np.abs(got - ref) / np.maximum(1.0, np.abs(ref))
    assert np.max(err) <= rtol, f"max scaled error {np.max(err):.3g}"


def _cases() -> dict:
    bm = L.BrownianMotion(0.05, 0.2)
    return {
        "vg_gamma": L.VarianceGamma(0.2, 0.3, -0.1) + 0.5 * L.GammaProcess(2.0, 3.0),
        "subordinated": (bm + L.VarianceGamma(0.2, 0.3, -0.1)) @ L.GammaProcess(4.0, 4.0),
        "cgmy": L.CGMY(1.0, 5.0, 10.0, 0.5) + L.BrownianMotion(0.0, 0.1),
        "kou_nig": L.KouJumpDiffusion(0.0, 0.1, 2.0, 0.4, 10.0, 5.0) + 0.5 * L.NIG(10.0, -2.0, 0.5),
        "gh": L.GeneralizedHyperbolicProcess(-0.5, 10.0, -2.0, 0.5, 0.0),
    }


CASES = _cases()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_value_and_gradient_match_diff_process(backend: str, name: str) -> None:
    proc = CASES[name]
    if backend != "numpy" and name == "gh":
        pytest.skip("kve of u is NumPy-only")
    k = ExponentKernel(lower(proc), backend)
    dp = DiffProcess(proc)
    assert k.names == tuple(dp.names)
    for uu in (U, UC):
        v_ref, g_ref = dp.char_exponent_and_grad(uu)
        psi, grad = k.psi_and_grad(uu)
        _close(psi, v_ref, 5e-14)
        _close(grad, g_ref, 1e-12)
        _close(k.psi(uu), v_ref, 5e-14)
    _close(k.char_func(U, 0.5), np.exp(0.5 * dp.char_exponent(U)), 5e-14)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["vg_gamma", "subordinated", "cgmy"])
def test_hessian_matches_diff_process(backend: str, name: str) -> None:
    proc = CASES[name]
    k = ExponentKernel(lower(proc), backend)
    v_ref, g_ref, h_ref = DiffProcess(proc).char_exponent_hessian(UC)
    psi, grad, hess = k.psi_grad_hess(UC)
    _close(psi, v_ref, 5e-14)
    _close(grad, g_ref, 1e-12)
    _close(hess, h_ref, 1e-10)
    np.testing.assert_array_equal(hess, np.swapaxes(hess, -1, -2))


@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_parameters_equal_row_by_row(backend: str) -> None:
    k = ExponentKernel(lower(CASES["subordinated"]), backend)
    rng = np.random.default_rng(7)
    theta = k.theta0 * (1.0 + 0.1 * rng.standard_normal((5, k.n_free)))
    psi_b, grad_b = k.psi_and_grad(U, theta)
    assert psi_b.shape == (5,) + U.shape and grad_b.shape == (5,) + U.shape + (k.n_free,)
    for r in range(5):
        psi_r, grad_r = k.psi_and_grad(U, theta[r])
        _close(psi_b[r], psi_r, 1e-15)
        _close(grad_b[r], grad_r, 1e-15)
    _, _, hess_b = k.psi_grad_hess(U[:3], theta)
    assert hess_b.shape == (5, 3, k.n_free, k.n_free)


def test_multidimensional_frequencies() -> None:
    k = ExponentKernel(lower(CASES["vg_gamma"]))
    uu = U.reshape(9, 9)
    np.testing.assert_array_equal(k.psi(uu), k.psi(U).reshape(9, 9))
    assert k.psi(2.0).shape == ()


@pytest.mark.parametrize("backend", BACKENDS)
def test_free_subset_and_fixed_values(backend: str) -> None:
    proc = CASES["vg_gamma"]
    low = lower(proc)
    k = ExponentKernel(low, backend, free=["VarianceGamma.theta", "scale"])
    assert k.names == ("VarianceGamma.theta", "scale")
    np.testing.assert_array_equal(k.theta0, [-0.1, 0.5])
    dp = DiffProcess(proc, free=["VarianceGamma.theta", "scale"])
    v_ref, g_ref = dp.char_exponent_and_grad(U)
    psi, grad = k.psi_and_grad(U)
    _close(psi, v_ref, 5e-14)
    _close(grad, g_ref, 1e-12)
    k2 = ExponentKernel(low, backend, free=[2, 5], fixed={"GammaProcess.a": 4.0})
    full = k2.full_theta()
    assert full[3] == 4.0 and full[2] == -0.1
    ref = low.build(full)._char_exponent(U)
    _close(k2.psi(U), ref, 5e-14)
    batch = k2.full_theta(np.array([[0.0, 1.0], [0.1, 2.0]]))
    assert batch.shape == (2, 6) and batch[1, 5] == 2.0 and batch[0, 3] == 4.0


def test_argument_validation() -> None:
    low = lower(CASES["vg_gamma"])
    with pytest.raises(CompileError):
        resolve_free(low.names, ["nope"])
    with pytest.raises(CompileError):
        resolve_free(low.names, [0, 0])
    with pytest.raises(CompileError):
        resolve_free(low.names, [99])
    with pytest.raises(CompileError):
        ExponentKernel(low, fixed={"nope": 1.0})
    with pytest.raises(CompileError):
        ExponentKernel(low, free=["scale"], fixed={"scale": 1.0})
    k = ExponentKernel(low)
    with pytest.raises(ValueError):
        k.psi(U, np.ones(3))
    with pytest.raises(ValueError):
        k.psi(U, np.ones((2, 2, 6)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_branch_specialisation(backend: str) -> None:
    proc = L.CGMY(1.0, 5.0, 10.0, 0.5)
    k = ExponentKernel(lower(proc), backend)
    ys = np.array([0.5, 0.02, 0.98, 1.5])
    theta = np.tile(k.theta0, (4, 1))
    theta[:, 3] = ys
    # rows one at a time: each uses a specialised program
    rows = np.array([k.psi(UC, th) for th in theta])
    fam = k._families["psi"]
    assert len(fam.guards) == 2
    assert {(False, False), (True, False), (False, True)} <= set(fam.cache)
    # the batch straddles the bands: the general program
    both = k.psi(UC, theta)
    assert None in fam.cache
    _close(both, rows, 1e-15)
    for r, y in enumerate(ys):
        ref = L.CGMY(1.0, 5.0, 10.0, float(y))._char_exponent(UC)
        _close(rows[r], ref, 1e-11)
    # the specialised program computes one branch, the general one all three
    spec_src = k.source("psi", theta[0])
    gen_src = fam.cache[None].source
    assert len(gen_src) > 3 * len(spec_src)


def test_repr_and_program_access() -> None:
    k = ExponentKernel(lower(CASES["vg_gamma"]))
    assert "numpy" in repr(k) and "VarianceGamma.sigma" in repr(k)
    prog = k.program("grad")
    assert prog.n_outputs == 1 + k.n_free


def test_opaque_leaves_run_on_numpy() -> None:
    class MyGamma(L.GammaProcess):
        pass

    # an opaque summand: fixed, but the other parameters stay differentiable
    proc = L.BrownianMotion(0.0, 0.3) + MyGamma(2.0, 3.0)
    k = ExponentKernel(lower(proc))
    _close(k.psi(U), proc._char_exponent(U), 5e-14)
    psi, grad = k.psi_and_grad(U)
    assert grad.shape == U.shape + (2,)
    eps = 1e-6
    th = k.theta0.copy()
    th[1] += eps
    _close(grad[:, 1], (k.psi(U, th) - psi) / eps, 1e-5)
    # an opaque clock: its argument depends on the parameters, so no gradient
    sub = ExponentKernel(lower(L.BrownianMotion(0.0, 0.3) @ MyGamma(2.0, 3.0)))
    _close(sub.psi(U), (L.BrownianMotion(0.0, 0.3) @ MyGamma(2.0, 3.0))._char_exponent(U), 5e-14)
    with pytest.raises(CompileError, match="opaque"):
        sub.psi_and_grad(U)
