"""Tests for code generation from the compile IR (NumPy, Numba and JAX backends)."""

from __future__ import annotations

import traceback
import warnings
from typing import Any, Dict, List

import numpy as np
import pytest

from spxa.compile import ir as I
from spxa.compile.codegen import BackendUnsupported, Program, generate
from spxa.compile.diff import derivative
from spxa.compile.evaluate import evaluate
from spxa.compile.ir import CompileError

u = I.U
a = I.param(0, "a")
b = I.param(1, "b")
c = I.param(2, "c")


def _corpus() -> Dict[str, I.Expr]:
    s = I.sqrt(a * a - (b + 1j * u) ** 2.0)
    w = (a * u - 1j * b) / 2.0
    sg = I.where(I.ge(I.real(w), 0.0), 1.0, -1.0)
    return {
        "vg": -I.log(1.0 - 1j * u * b * a + 0.5 * c**2 * a * u * u) / a,
        "nig_shared": 1j * c * u + b * (I.sqrt(a * a - b * b) - s) + s * s,
        "logcosh": sg * w + I.log1p(I.exp(-2.0 * sg * w)) - float(np.log(2.0)),
        "jumps": c * I.expm1(1j * a * u - 0.5 * b**2 * u * u) + 1j * u * (a / (b - 1j * u)),
        "powers": (b - 1j * u) ** a - b**a + (a + u * u) ** 0.5 + u**3.0,
        "subordinated": -a * I.log(1.0 - 1j * (-1j * (1j * c * u - 0.5 * b * u * u)) / a),
        "sinc": I.exp(1j * a * u) * I.sinc(b * u) + I.sinc_d1(c * u),
        "trig": I.tan(0.1 * a * u) + I.sin(b * u) * I.cos(c),
        "params_only_heavy": I.gamma(-a) * u + I.digamma(b) * u * u + I.polygamma(1, c),
        "nonanalytic": I.abs_(u) * a + I.sign(I.real(u)) * b + I.imag(u * c),
        "neg_shared": (lambda n: (u - n) * n + 2.0 * n)(I.neg(a * u)),
    }


CORPUS = _corpus()
UU = np.array([-7.5, -1.0, -0.2, 0.0, 0.3, 1.0, 4.0, 25.0]) + np.array(
    [0, 0.1j, 0, 0, -0.2j, 0, 0.05j, 0]
)
P_SCALAR = [0.7, 1.3, 0.4]
P_BATCH = [np.array([0.7, 1.9, 0.35]), np.array([1.3, 0.2, 2.5]), np.array([0.4, -0.3, 1.1])]


def _ref(e: List[I.Expr], params: List[Any], batched: bool) -> List[np.ndarray]:
    if batched:
        out = evaluate(e, UU[None, :], [p[:, None] for p in params])
        shape = (len(params[0]),) + UU.shape
    else:
        out = evaluate(e, UU, params)
        shape = UU.shape
    return [np.broadcast_to(o, shape) for o in out]


def _outputs(name: str) -> List[I.Expr]:
    e = CORPUS[name]
    return [e] + [derivative(e, p) for p in (a, b, c)]


# ── NumPy: bit-exact against the reference interpreter ──────────────────────


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_numpy_is_bit_exact(name: str) -> None:
    outs = _outputs(name)
    prog = generate(outs, 3, "numpy", name)
    for got, ref in zip(prog(UU, *P_SCALAR), _ref(outs, P_SCALAR, False)):
        np.testing.assert_array_equal(np.broadcast_to(got, ref.shape), ref)
    got_b = prog(UU[None, :], *[p[:, None] for p in P_BATCH])
    for got, ref in zip(got_b, _ref(outs, P_BATCH, True)):
        np.testing.assert_array_equal(np.broadcast_to(got, ref.shape), ref)


def test_numpy_keeps_ieee_semantics_without_warnings() -> None:
    e_pow = (a**b) * u
    e_div = a / I.real(u) + 0j * u
    e_log = I.log(a * I.real(u)) + 0j * u
    prog = generate([e_pow, e_div, e_log], 2, "numpy")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p, d, lg = prog(np.array([0.0, 1.0]), -1.0, 0.5)
    assert np.all(np.isnan(p))  # (-1)^0.5 is nan for real operands, not complex
    assert d.real[0] == -np.inf and np.isnan(lg.real[1])  # log of a negative real is nan
    _, d, lg = prog(np.array([0.0]), 1.0, 0.5)
    assert d.real[0] == np.inf and lg.real[0] == -np.inf


def test_hoisting_and_sharing() -> None:
    s = I.sqrt(1.0 - 2j * u * a**2 / b)
    e = (b / a) * (1.0 - s) + I.gamma(-a) * s
    prog = generate([e, derivative(e, a)], 2, "numpy", "ig")
    assert prog.hoisted >= 2
    assert prog.source.count("np.sqrt") == 1
    assert prog.source.count("sps.gamma(") == 1
    # every hoisted statement precedes the first use of u
    lines = prog.source.splitlines()
    first_u = next(
        i for i, ln in enumerate(lines) if "u" in ln.split("=", 1)[-1] and "def" not in ln
    )
    assert all("u" not in ln.split("=", 1)[-1] for ln in lines[2:first_u])


def test_negated_temporaries_keep_their_sign() -> None:
    n = I.neg(a * u)
    e = (u - n) * n + 2.0 * n  # n is used three times, so it is a named temporary
    prog = generate([e], 1, "numpy")
    got = prog(UU, 0.8)[0]
    np.testing.assert_array_equal(got, evaluate(e, UU, [0.8]))


def test_deep_expressions_are_split() -> None:
    e = u
    for k in range(200):
        e = I.exp(-e * 0.01) + float(k)
    prog = generate([e], 0, "numpy")
    assert max(len(ln) for ln in prog.source.splitlines()) < 2000
    np.testing.assert_array_equal(prog(UU)[0], evaluate(e, UU, []))


def test_tracebacks_point_into_generated_source() -> None:
    class Boom:
        def _char_exponent(self, x: Any) -> Any:
            raise RuntimeError("boom")

    prog = generate([I.call(Boom(), a * u)], 1, "numpy", "tb_kernel")
    with pytest.raises(RuntimeError) as info:
        prog(UU, 1.0)
    frames = traceback.extract_tb(info.value.__traceback__)
    gen = [f for f in frames if "spxa.compile:tb_kernel" in f.filename]
    assert gen and gen[0].line and "_call" in gen[0].line


def test_opaque_calls_run_on_numpy() -> None:
    from spxa.zoo.levy import GammaProcess

    g = GammaProcess(2.0, 3.0)
    e = a * I.call(g, b * u)
    prog = generate([e], 2, "numpy")
    np.testing.assert_array_equal(prog(UU, 0.5, 2.0)[0], evaluate(e, UU, [0.5, 2.0]))


def test_constant_and_parameter_outputs() -> None:
    prog = generate([I.const(2.5), a * b, u], 2, "numpy")
    out = prog(UU, 2.0, 3.0)
    assert out[0] == 2.5 and out[1] == 6.0
    np.testing.assert_array_equal(out[2], UU)


def test_program_interface() -> None:
    prog = generate([a * u], 1, "numpy", "iface")
    assert isinstance(prog, Program)
    assert prog.n_params == 1 and prog.n_outputs == 1 and prog.backend == "numpy"
    assert "def _kernel_iface(u, p0):" in prog.source
    with pytest.raises(TypeError):
        prog(UU)


def test_generate_validates_its_arguments() -> None:
    with pytest.raises(CompileError):
        generate([], 0)
    with pytest.raises(CompileError):
        generate([a * u], 0)
    with pytest.raises(CompileError):
        generate([a * u], 1, name="not an identifier")
    with pytest.raises(CompileError):
        generate([a * u], 1, backend="fortran")


# ── Numba ───────────────────────────────────────────────────────────────────


def _assert_close(got: np.ndarray, ref: np.ndarray, rtol: float = 1e-13) -> None:
    got = np.broadcast_to(got, ref.shape)
    fin = np.isfinite(ref)
    assert np.all(np.isfinite(got[fin]))
    scale = np.maximum(1.0, np.abs(ref[fin]))
    assert np.max(np.abs(got[fin] - ref[fin]) / scale, initial=0.0) <= rtol
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))


NUMBA_CASES = [n for n in sorted(CORPUS) if n != "params_only_heavy"] + ["params_only_heavy"]


class TestNumba:
    @pytest.fixture(autouse=True)
    def _need_numba(self) -> None:
        pytest.importorskip("numba")

    @pytest.mark.parametrize("name", NUMBA_CASES)
    def test_matches_reference(self, name: str) -> None:
        outs = _outputs(name)
        prog = generate(outs, 3, "numba", name)
        for got, ref in zip(prog(UU, *P_SCALAR), _ref(outs, P_SCALAR, False)):
            _assert_close(got, ref)
        for got, ref in zip(prog(UU, *P_BATCH), _ref(outs, P_BATCH, True)):
            _assert_close(got, ref)

    def test_parallel_loop(self) -> None:
        outs = _outputs("vg")
        prog = generate(outs, 3, "numba", "vg_par", parallel=True)
        for got, ref in zip(prog(UU, *P_BATCH), _ref(outs, P_BATCH, True)):
            _assert_close(got, ref)

    def test_ieee_corners(self) -> None:
        e = a / (u - 1.0) + I.log1p(b * u) + I.exp(c * u)
        prog = generate([e], 3, "numba")
        uu = np.array([1.0, -2.0, 0.0])
        got = prog(uu, 1.0, 0.5, 1.0)[0]
        ref = evaluate(e, uu, [1.0, 0.5, 1.0])
        np.testing.assert_array_equal(np.isinf(got), np.isinf(ref))
        assert got[2] == ref[2]

    def test_shapes(self) -> None:
        prog = generate([a * u, a * b, I.const(1.0)], 2, "numba")
        out = prog(np.zeros((2, 3)), 2.0, 3.0)
        assert out[0].shape == (2, 3) and out[1].shape == (2, 3) and out[2].shape == (2, 3)
        out = prog(np.zeros(4), np.array([1.0, 2.0]), 3.0)
        assert out[0].shape == (2, 4)
        np.testing.assert_array_equal(out[1][:, 0], [3.0, 6.0])
        assert prog(np.zeros(0), 1.0, 1.0)[0].shape == (0,)
        with pytest.raises(CompileError):
            prog(UU, np.ones((2, 2)), 1.0)

    def test_unsupported_operations(self) -> None:
        with pytest.raises(BackendUnsupported):
            generate([I.kve(a, b + u)], 2, "numba")

        class Opaque:
            pass

        with pytest.raises(BackendUnsupported):
            generate([I.call(Opaque(), u)], 0, "numba")

    def test_parameter_only_special_functions_are_hoisted(self) -> None:
        e = I.kve(a, b) * u + I.gamma(-a) * u
        prog = generate([e], 2, "numba")
        _assert_close(prog(UU, 0.5, 2.0)[0], evaluate(e, UU, [0.5, 2.0]))


# ── JAX ─────────────────────────────────────────────────────────────────────


class TestJax:
    @pytest.fixture(autouse=True)
    def _need_jax(self) -> None:
        pytest.importorskip("jax")

    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_matches_reference(self, name: str) -> None:
        outs = _outputs(name)
        prog = generate(outs, 3, "jax", name)
        # the jax polygamma is accurate to a few 1e-15
        rtol = 1e-12 if name == "params_only_heavy" else 1e-13
        for got, ref in zip(prog(UU, *P_SCALAR), _ref(outs, P_SCALAR, False)):
            _assert_close(got, ref, rtol)
        got_b = prog(UU[None, :], *[p[:, None] for p in P_BATCH])
        for got, ref in zip(got_b, _ref(outs, P_BATCH, True)):
            _assert_close(got, ref, rtol)

    def test_float64_and_jit(self) -> None:
        prog = generate([a * u], 1, "jax")
        out = prog(UU, 0.1)[0]
        assert out.dtype == np.complex128
        assert hasattr(prog.fn, "jitted")

    def test_unsupported_operations(self) -> None:
        with pytest.raises(BackendUnsupported):
            generate([I.kve(a, b + u)], 2, "jax")

        class Opaque:
            pass

        with pytest.raises(BackendUnsupported):
            generate([I.call(Opaque(), u)], 0, "jax")

    def test_sinc_derivative_across_the_series_radius(self) -> None:
        z = np.array([0.0, 1e-8, 0.5, 0.999, 1.0, 1.001, 3.0]) + 0.0j
        prog = generate([I.sinc_d1(u)], 0, "jax")
        ref = evaluate(I.sinc_d1(u), z, [])
        np.testing.assert_allclose(prog(z)[0], ref, rtol=1e-15, atol=1e-17)
