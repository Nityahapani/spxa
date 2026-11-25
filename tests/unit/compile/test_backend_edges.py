"""IEEE edge cases where the Numba and JAX backends must follow the NumPy reference."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.compile import ir as I
from spxa.compile.codegen import generate
from spxa.compile.evaluate import evaluate

u = I.U
a = I.param(0, "a")
b = I.param(1, "b")

# zero base, complex exponents with Re > 0 (0), Re = 0 (nan), Re < 0 (nan), and 0 ** 0 = 1
EXPONENTS = [0.5 + 0.5j, 2.0 - 1.0j, 0.5j, -0.5 + 0.25j, 0j, 1.5 + 0j]


def _same(got: np.ndarray, ref: np.ndarray) -> None:
    got = np.asarray(got, dtype=complex)
    ref = np.broadcast_to(np.asarray(ref, dtype=complex), got.shape)  # constants fold to scalars
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    fin = ~np.isnan(ref)
    np.testing.assert_allclose(got[fin], ref[fin], rtol=1e-15, atol=0)


@pytest.mark.parametrize("backend", ["numba", "jax"])
@pytest.mark.parametrize("y", EXPONENTS)
def test_zero_to_a_complex_power(backend: str, y: complex) -> None:
    pytest.importorskip(backend)
    uu = np.array([0.0, 0.5, -2.0]) + 0j
    e_const = (a * u) ** I.const(y)  # constant exponent
    e_param = (a * u) ** (b + I.const(y - y.real))  # exponent through a parameter
    for e in (e_const, e_param):
        prog = generate([e], 2, backend, "zpow")
        _same(prog(uu, 1.3, y.real)[0], evaluate(e, uu, [1.3, y.real]))


@pytest.mark.parametrize("backend", ["numba", "jax"])
def test_expm1_overflow_keeps_the_infinite_real_part(backend: str) -> None:
    pytest.importorskip(backend)
    e = I.expm1(a * u)
    uu = np.array([700.0, 710.0, 1000.0, -1000.0]) + 0j
    got = np.asarray(generate([e], 1, backend, "ovf")(uu, 1.0)[0])
    ref = evaluate(e, uu, [1.0])
    np.testing.assert_array_equal(got.real, ref.real)
    # on the real axis the backends return the exact (real) value; NumPy's
    # complex expm1 forms inf * sin(0) = nan for the imaginary part
    assert np.all(got.imag == 0.0)
    off = np.array([710.0 + 1.0j, 800.0 + 0.5j])
    got_off = np.asarray(generate([e], 1, backend, "ovf2")(off, 1.0)[0])
    np.testing.assert_array_equal(got_off, evaluate(e, off, [1.0]))
