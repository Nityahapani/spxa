"""Tests for the NumPy and Taylor-mode interpreters of the compile IR."""

from __future__ import annotations

from math import factorial

import numpy as np
import pytest

from spxa.compile import ir as I
from spxa.compile.evaluate import call_exponent, cumulants_from_ir, evaluate, taylor
from spxa.compile.ir import CompileError
from spxa.graph.autodiff.taylor import TaylorError

u = I.U
a = I.param(0, "a")
b = I.param(1, "b")


def gamma_psi() -> I.Expr:
    return -a * I.log(1.0 - 1j * u / b)


class TestEvaluate:
    def test_matches_handwritten_numpy(self) -> None:
        uu = np.linspace(-4, 4, 9)
        got = evaluate(gamma_psi(), uu, [2.0, 3.0])
        np.testing.assert_array_equal(got, -2.0 * np.log(1.0 - 1j * uu / 3.0))

    def test_parameters_broadcast_against_u(self) -> None:
        uu = np.linspace(-1, 1, 5)
        A = np.array([[1.0], [2.0]])
        got = evaluate(gamma_psi(), uu, [A, 3.0])
        assert got.shape == (2, 5)
        np.testing.assert_array_equal(got[1], evaluate(gamma_psi(), uu, [2.0, 3.0]))

    def test_multiple_roots_and_constants(self) -> None:
        out = evaluate([a * u, I.const(2.0), b], 1.5, [2.0, 7.0])
        assert out[0] == 3.0 + 0j and out[1] == 2.0 and out[2] == 7.0

    def test_where_and_special_functions(self) -> None:
        e = I.where(I.ge(I.real(u), 0.0), I.sinc(u), I.kve(a, b + u * u))
        uu = np.array([-1.0, 0.0, 2.0])
        got = evaluate(e, uu, [0.5, 2.0])
        import scipy.special as sps

        assert got[1] == 1.0
        assert got[2] == pytest.approx(np.sin(2.0) / 2.0, rel=1e-15)
        assert got[0] == pytest.approx(sps.kve(0.5, 3.0), rel=1e-15)

    def test_call_uses_the_process_exponent(self) -> None:
        from spxa.zoo.levy import GammaProcess

        g = GammaProcess(2.0, 3.0)
        e = I.call(g, a * u)
        uu = np.array([[0.5, -1.0], [2.0, 0.0]])
        got = evaluate(e, uu, [1.5])
        np.testing.assert_allclose(got, g._char_exponent((1.5 * uu).ravel()).reshape(2, 2))
        assert call_exponent(g, 0.25).shape == ()


class TestTaylor:
    @pytest.mark.parametrize("order", [1, 4, 9])
    def test_gamma_cumulants_are_exact(self, order: int) -> None:
        k = cumulants_from_ir(gamma_psi(), order, [2.0, 3.0])
        ref = [2.0 * factorial(n - 1) / 3.0**n for n in range(1, order + 1)]
        np.testing.assert_allclose(k, ref, rtol=1e-14)

    def test_batched_parameters(self) -> None:
        A = np.array([1.0, 2.0, 4.0])
        k = cumulants_from_ir(gamma_psi(), 3, [A, 3.0])
        assert k.shape == (3, 3)
        np.testing.assert_allclose(k[:, 1], A / 9.0, rtol=1e-15)

    def test_expansion_away_from_zero(self) -> None:
        e = I.exp(a * u)
        c = taylor(e, 3, [2.0], u0=0.5, du=1.0)
        # e^{a(u0+h)} = e^{a u0} sum (a h)^n / n!
        ref = [np.exp(1.0) * 2.0**n / factorial(n) for n in range(4)]
        np.testing.assert_allclose(np.array(c, dtype=complex), ref, rtol=1e-14)

    def test_constant_expression(self) -> None:
        c = taylor(a * b, 2, [2.0, 3.0])
        assert c[0] == 6.0 and c[1] == 0.0 and c[2] == 0.0

    def test_integer_powers_at_a_zero_constant_term(self) -> None:
        # u**2 has a zero constant term: the general power recurrence divides by it
        e = a * u**2.0 + u**3.0
        c = taylor(e, 4, [2.0])
        np.testing.assert_allclose(np.array(c, dtype=complex), [0, 0, 2.0, 1.0, 0], atol=0)

    def test_where_is_resolved_at_the_expansion_point(self) -> None:
        e = I.where(I.ge(I.real(u), 0.0), I.exp(u), I.exp(-u))
        c_pos = taylor(e, 2, [], u0=1.0)
        c_neg = taylor(e, 2, [], u0=-1.0)
        assert c_pos[1] == pytest.approx(np.e)
        assert c_neg[1] == pytest.approx(-np.e)

    def test_logcosh_style_guard_is_analytic(self) -> None:
        w = a * u
        s = I.where(I.ge(I.real(w), 0.0), 1.0, -1.0)
        e = s * w + I.log1p(I.exp(-2.0 * s * w)) - float(np.log(2.0))
        k = cumulants_from_ir(e, 4, [0.7])
        # log cosh(-i a z) = log cos(a z): kappa_2 = -a^2, kappa_4 = -2 a^4 (signs of K(z))
        np.testing.assert_allclose(k, [0.0, -(0.7**2), 0.0, -2 * 0.7**4], atol=1e-14)

    def test_sinc_series_at_zero(self) -> None:
        e = I.sinc(a * u)
        c = taylor(e, 4, [2.0])
        np.testing.assert_allclose(
            np.array(c, dtype=complex), [1.0, 0.0, -4.0 / 6.0, 0.0, 16.0 / 120.0], atol=1e-15
        )
        c1 = taylor(e, 2, [2.0], u0=0.5)
        assert c1[0] == pytest.approx(np.sin(1.0) / 1.0)

    def test_tan_series(self) -> None:
        c = taylor(I.tan(a * u), 3, [1.0])
        np.testing.assert_allclose(np.array(c, dtype=complex), [0, 1, 0, 1.0 / 3.0], atol=1e-15)

    def test_call_expands_through_cumulants(self) -> None:
        from spxa.zoo.levy import VarianceGamma

        vg = VarianceGamma(0.2, 0.3, -0.1)
        k = cumulants_from_ir(I.call(vg, u), 4, [])
        ref = vg.cumulants(4)
        np.testing.assert_allclose(k, [ref[n] for n in range(1, 5)], rtol=1e-13, atol=1e-16)
        with pytest.raises(TaylorError):
            taylor(I.call(vg, u), 2, [], u0=1.0)

    def test_non_analytic_values_raise(self) -> None:
        with pytest.raises(TaylorError):
            taylor(I.abs_(u) * a, 2, [1.0])
        with pytest.raises(TaylorError):
            taylor(I.gamma(I.real(u)), 2, [])

    def test_invalid_order(self) -> None:
        with pytest.raises(ValueError):
            taylor(u, 0, [])

    def test_sinc_with_mixed_expansion_points_raises(self) -> None:
        with pytest.raises(CompileError):
            taylor(I.sinc(a * u + b), 2, [1.0, np.array([0.0, 1.0])])
