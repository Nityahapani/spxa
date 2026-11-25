"""Standalone source export of compiled exponents."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from spxa.compile import CompiledProcess, CompileError
from spxa.compile._numeric import sinc, sinc_d1
from spxa.compile.export import export_source
from spxa.zoo.levy import (
    CGMY,
    BrownianMotion,
    Drift,
    GammaProcess,
    GeneralizedHyperbolicProcess,
    KouJumpDiffusion,
)
from spxa.zoo.levy import compound_poisson as cp

PROC = (
    (BrownianMotion(0.05, 0.2) + 0.5 * KouJumpDiffusion(0.05, 0.2, 2.0, 0.4, 8.0, 5.0))
    @ GammaProcess(4.0, 4.0)
    + CGMY(1.0, 5.0, 6.0, 0.5)
    + cp.CompoundPoisson(2.0, cp.UniformJump(-0.2, 0.5))
    + GeneralizedHyperbolicProcess(-0.5, 10.0, -2.0, 0.5, 0.0)
)
U = np.linspace(-30.0, 30.0, 201)


@pytest.fixture(scope="module")
def compiled() -> CompiledProcess:
    names = CompiledProcess(PROC).all_names
    free = [n for n in names if n != "GeneralizedHyperbolicProcess.lambda_param"]
    return CompiledProcess(PROC, free=free)


@pytest.fixture(scope="module")
def module(compiled: CompiledProcess, tmp_path_factory: pytest.TempPathFactory) -> dict:
    path = tmp_path_factory.mktemp("export") / "exported.py"
    src = compiled.export(str(path))
    assert path.read_text(encoding="utf-8") == src
    return runpy.run_path(str(path))


@pytest.mark.parametrize("y", [0.5, 0.99, 1.0, 0.01, 1.5])
def test_bit_exact_across_branches(compiled: CompiledProcess, module: dict, y: float) -> None:
    th = compiled.values.copy()
    th[compiled.names.index("CGMY.Y")] = y  # each CGMY band (direct, series at 0 and 1)
    v, g = module["psi_and_grad"](U, th)
    rv, rg = compiled.char_exponent_and_grad(U, th)
    np.testing.assert_array_equal(v, rv)
    np.testing.assert_array_equal(g, rg)
    np.testing.assert_array_equal(module["psi"](U, th), compiled.char_exponent(U, th))


def test_defaults_and_metadata(compiled: CompiledProcess, module: dict) -> None:
    assert module["NAMES"] == compiled.names
    np.testing.assert_array_equal(module["THETA0"], compiled.values)
    np.testing.assert_array_equal(module["psi"](U), compiled.char_exponent(U))
    np.testing.assert_allclose(module["char_func"](U, 0.5), compiled.char_func(U, 0.5), rtol=1e-15)
    with pytest.raises(ValueError, match="theta needs"):
        module["psi"](U, [1.0])


def test_inlined_helpers_match_the_library() -> None:
    src = export_source(CompiledProcess(cp.CompoundPoisson(1.0, cp.UniformJump(-1.0, 2.0))))
    scope: dict = {}
    exec(compile(src, "<exported>", "exec"), scope)  # noqa: S102 - our own generated text
    z = np.concatenate([np.linspace(-3, 3, 601), [0.0, 1e-9, 0.999999, 1.0, 1.000001]])
    zc = z + 0.3j * z[::-1]
    for arr in (z, zc):
        np.testing.assert_array_equal(scope["_sinc"](arr), sinc(arr))
        np.testing.assert_array_equal(scope["_sinc_d1"](arr), sinc_d1(arr))


def test_runs_without_spxa(compiled: CompiledProcess, tmp_path: Path) -> None:
    path = tmp_path / "standalone.py"
    compiled.export(str(path))
    code = (
        "import sys; sys.modules['spxa'] = None\n"
        "import runpy, numpy as np\n"
        f"m = runpy.run_path({str(path)!r})\n"
        "v, g = m['psi_and_grad'](np.linspace(-5, 5, 11))\n"
        "print(v.shape, g.shape)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=tmp_path
    )
    assert out.stdout.strip() == f"(11,) (11, {compiled.n_params})"
    assert "spxa" not in "".join(
        ln for ln in path.read_text().splitlines() if ln.startswith(("import", "from"))
    )


def test_psi_only_and_one_parameter() -> None:
    src = CompiledProcess(Drift(0.25)).export(grad=False)
    scope: dict = {}
    exec(compile(src, "<exported>", "exec"), scope)  # noqa: S102 - our own generated text
    assert "psi_and_grad" not in scope and scope["THETA0"] == (0.25,)
    assert "scipy" not in src
    np.testing.assert_allclose(scope["psi"]([1.0, 2.0]), [0.25j, 0.5j], rtol=1e-15)


def test_unexportable() -> None:
    class MyGamma(GammaProcess):
        pass

    with pytest.raises(CompileError, match="opaque"):
        CompiledProcess(BrownianMotion(0.0, 0.3) @ MyGamma(2.0, 3.0)).export()
    with pytest.raises(CompileError, match="lambda_param"):
        CompiledProcess(GeneralizedHyperbolicProcess(-0.5, 10.0, -2.0, 0.5, 0.0)).export()
    # psi alone needs no order derivative
    src = CompiledProcess(GeneralizedHyperbolicProcess(-0.5, 10.0, -2.0, 0.5, 0.0)).export(
        grad=False
    )
    assert "sps.kve" in src


def test_literals_round_trip() -> None:
    from spxa.compile.export import _literal

    for v in [True, 0.1, -2.5e-300, float("inf"), float("-inf"), complex(1.5, -float("inf"))]:
        back = eval(_literal(v))  # noqa: S307 - literal text produced above
        assert back == v and type(back) is type(v)
    assert np.isnan(eval(_literal(float("nan"))))  # noqa: S307
