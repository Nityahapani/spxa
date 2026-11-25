"""Tests for the special functions behind the compile IR."""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from spxa.compile._numeric import NP_FUNCS, apply_op, kve_dnu, sinc, sinc_d1


def _mp_sinc_d1(z: complex) -> complex:
    zz = mp.mpc(z)
    return complex(mp.diff(lambda t: mp.sin(t) / t if t != 0 else mp.mpf(1), zz))


class TestSinc:
    def test_value_at_zero_and_symmetry(self) -> None:
        assert sinc(0.0) == 1.0
        z = np.array([-2.0, -0.1, 0.0, 0.1, 2.0])
        np.testing.assert_allclose(sinc(z), sinc(-z), rtol=0, atol=0)
        np.testing.assert_allclose(sinc(z[[0, 1, 3, 4]]), np.sin(z[[0, 1, 3, 4]]) / z[[0, 1, 3, 4]])

    def test_complex_argument(self) -> None:
        z = 0.7 - 1.3j
        assert abs(sinc(z) - np.sin(z) / z) < 1e-15

    @pytest.mark.parametrize(
        "z",
        [
            1e-12,
            1e-6,
            1e-3,
            0.1,
            0.25,
            0.4999,
            0.5,
            0.5001,
            1.0,
            3.0,
            30.0,
            0.1 + 0.2j,
            0.3 - 0.4j,
            2 - 1j,
        ],
    )
    def test_derivative_against_mpmath(self, z: complex) -> None:
        mp.mp.dps = 40
        ref = _mp_sinc_d1(z)
        got = complex(sinc_d1(z))
        # series inside |z| < 1, direct form outside: a few ulp either way
        assert abs(got - ref) <= 8 * np.finfo(float).eps * abs(ref)

    def test_derivative_at_zero_is_zero(self) -> None:
        assert sinc_d1(0.0) == 0.0

    def test_derivative_is_odd(self) -> None:
        z = np.linspace(-3, 3, 61)
        np.testing.assert_array_equal(sinc_d1(z), -sinc_d1(-z))


class TestBessel:
    @pytest.mark.parametrize("nu", [-1.3, -0.5, 0.0, 0.7, 2.5])
    @pytest.mark.parametrize("z", [0.3, 2.0, 15.0, 1.5 + 0.8j])
    def test_order_derivative_against_mpmath(self, nu: float, z: complex) -> None:
        mp.mp.dps = 40
        ref = complex(mp.diff(lambda n: mp.exp(mp.mpc(z)) * mp.besselk(n, mp.mpc(z)), nu))
        got = complex(kve_dnu(nu, z))
        assert abs(got - ref) <= 1e-12 * max(1.0, abs(ref))

    def test_order_derivative_vanishes_at_order_zero(self) -> None:
        # K_nu is even in nu, so the derivative is zero at nu = 0
        assert abs(kve_dnu(0.0, 1.7)) < 1e-14


class TestApply:
    def test_dispatch(self) -> None:
        assert apply_op("add", [1.0, 2.0]) == 3.0
        assert apply_op("polygamma", [1.0], 1) == pytest.approx(np.pi**2 / 6, rel=1e-15)
        assert apply_op("where", [np.array([True, False]), 1.0, 2.0]).tolist() == [1.0, 2.0]

    def test_unknown_op(self) -> None:
        with pytest.raises(KeyError):
            apply_op("frobnicate", [1.0])

    def test_table_covers_the_ir(self) -> None:
        from spxa.compile import ir

        ops = set(ir._REBUILD) - {"polygamma"}
        assert ops <= set(NP_FUNCS)
