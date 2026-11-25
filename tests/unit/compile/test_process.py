"""The user-facing CompiledProcess and Process.compile()."""

from __future__ import annotations

import numpy as np
import pytest

import spxa
from spxa.compile import CompiledProcess, compile_process
from spxa.compile.ir import CompileError
from spxa.graph.autodiff.process import DiffProcess
from spxa.zoo import levy as L

U = np.linspace(-25.0, 25.0, 51)


def _proc():  # type: ignore[no-untyped-def]
    bm = L.BrownianMotion(0.05, 0.2)
    return (bm + L.VarianceGamma(0.2, 0.3, -0.1)) @ L.GammaProcess(4.0, 4.0) + 0.5 * L.NIG(
        10.0, -2.0, 0.5
    )


def _close(got: np.ndarray, ref: np.ndarray, rtol: float) -> None:
    ref = np.asarray(ref)
    err = np.abs(np.broadcast_to(got, ref.shape) - ref) / np.maximum(1.0, np.abs(ref))
    assert np.max(err) <= rtol, f"max scaled error {np.max(err):.3g}"


def test_process_compile_entry_point() -> None:
    proc = _proc()
    cp = proc.compile()
    assert isinstance(cp, CompiledProcess)
    assert cp.backend == "numpy" and cp.process is proc
    assert isinstance(compile_process(proc, free=[0]), CompiledProcess)
    assert spxa.compile.CompiledProcess is CompiledProcess


def test_star_import_keeps_the_builtin_compile() -> None:
    ns: dict = {}
    exec("from spxa import *", ns)  # noqa: S102
    assert "compile" not in ns or ns["compile"] is compile


def test_interface_matches_diff_process() -> None:
    proc = _proc()
    cp = proc.compile()
    dp = DiffProcess(proc)
    assert cp.names == tuple(dp.names) == cp.all_names
    np.testing.assert_array_equal(cp.values, np.asarray(dp.values, dtype=float))
    assert cp.n_params == len(dp.names) and cp.analytic
    _close(cp.char_exponent(U), dp.char_exponent(U), 5e-14)
    v, g = cp.char_exponent_and_grad(U)
    v0, g0 = dp.char_exponent_and_grad(U)
    _close(v, v0, 5e-14)
    _close(g, g0, 1e-12)
    _, _, h = cp.char_exponent_hessian(U[:5])
    _, _, h0 = dp.char_exponent_hessian(U[:5])
    _close(h, h0, 1e-10)
    _close(cp.log_char(U, 0.5), dp.log_char(U, 0.5), 5e-14)
    _close(cp.char_func(U, 2.0), dp.char_func(U, 2.0), 5e-14)
    z = np.array([-0.5, 0.0, 0.5])
    _close(cp.log_mgf(z), dp.log_mgf(z), 5e-14)
    k, j = cp.cumulants_and_jacobian(4)
    k0, j0 = dp.cumulants_and_jacobian(4)
    _close(k, k0, 1e-13)
    _close(j, j0, 1e-12)
    _close(cp.cumulants(4), k0, 1e-13)


def test_parameters_and_rebuild() -> None:
    proc = _proc()
    cp = proc.compile(free=["GammaProcess.a", "scale"], fixed={"NIG.delta": 0.7})
    assert cp.names == ("GammaProcess.a", "scale")
    full = cp.full_theta([5.0, 0.25])
    assert full[cp.all_names.index("NIG.delta")] == 0.7
    rebuilt = cp.to_process([5.0, 0.25])
    _close(cp.char_exponent(U, [5.0, 0.25]), rebuilt._char_exponent(U), 5e-14)
    moved = cp.with_values([5.0, 0.25])
    assert moved.names == cp.names
    np.testing.assert_array_equal(moved.values, [5.0, 0.25])
    _close(moved.char_exponent(U), cp.char_exponent(U, [5.0, 0.25]), 1e-15)
    with pytest.raises(ValueError):
        cp.to_process(np.ones((2, 2)))


def test_batched_calls() -> None:
    cp = _proc().compile()
    theta = np.vstack([cp.values, cp.values * 1.05])
    psi = cp.char_exponent(U, theta)
    assert psi.shape == (2,) + U.shape
    _close(psi[1], cp.char_exponent(U, theta[1]), 1e-15)
    assert cp.cumulants(4, theta).shape == (2, 4)


def test_introspection() -> None:
    cp = _proc().compile()
    src = cp.source("grad")
    assert src.startswith("def ") and "np.log" in src
    s = cp.story()
    assert "GammaProcess(a=4" in s and "@" in s
    assert "CompiledProcess(" in repr(cp) and "backend='numpy'" in repr(cp)


@pytest.mark.parametrize("backend", ["numba", "jax"])
def test_other_backends(backend: str) -> None:
    pytest.importorskip(backend)
    proc = _proc()
    cp = proc.compile(backend=backend)
    ref = proc.compile()
    _close(cp.char_exponent(U), ref.char_exponent(U), 1e-13)
    _close(cp.char_exponent_and_grad(U)[1], ref.char_exponent_and_grad(U)[1], 1e-12)


def test_unsupported_process_raises() -> None:
    from spxa.zoo.beyond.heston import Heston

    with pytest.raises(CompileError):
        Heston(sigma_v=0.3).compile()
