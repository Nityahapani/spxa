"""Tests for the hash-consed expression IR of spxa.compile."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from spxa.compile import ir as I
from spxa.compile.ir import CompileError

u = I.U
a = I.param(0, "a")
b = I.param(1, "b")


class TestHashConsing:
    def test_structurally_equal_nodes_are_identical(self) -> None:
        e1 = I.log(1.0 - 1j * u / b) * a
        e2 = I.log(1.0 - 1j * u / b) * a
        assert e1 is e2

    def test_commutative_operands_are_canonicalised(self) -> None:
        assert a + b is b + a
        assert a * u is u * a

    def test_constants_distinguish_kinds_and_signed_zero(self) -> None:
        assert I.const(1.0) is I.const(1)
        assert I.const(1.0) is not I.const(1 + 0j)
        assert I.const(0.0) is not I.const(-0.0)
        assert I.const(True).kind == I.BOOL

    def test_dead_nodes_are_collected(self) -> None:
        def build() -> int:
            e = I.exp(I.sqrt(a * 12345.678 + u))
            assert e.depends_u
            return len(I._TABLE)

        n_live = build()
        gc.collect()
        assert len(I._TABLE) < n_live

    def test_sharing_counts_each_subexpression_once(self) -> None:
        s = I.sqrt(1.0 - 2j * u / b)
        e = a * (1.0 - s) + s * s
        # u, b, a, consts and the shared sqrt appear once each
        assert I.count_nodes(e) == len(I.postorder([e]))
        assert sum(1 for n in I.postorder([e]) if n.op == "sqrt") == 1


class TestKindsAndFolding:
    def test_kinds(self) -> None:
        assert a.kind == I.REAL
        assert u.kind == I.COMPLEX
        assert (a * u).kind == I.COMPLEX
        assert I.real(u).kind == I.REAL
        assert I.abs_(u).kind == I.REAL
        assert I.ge(a, 0.0).kind == I.BOOL

    def test_constant_folding_follows_numpy(self) -> None:
        assert I.const(2.0) * 3.0 is I.const(6.0)
        assert np.isnan(float(I.sqrt(-1.0)))  # real sqrt of a negative real is nan
        assert I.sqrt(I.const(-1 + 0j)).const_value() == 1j
        assert I.log1p(I.const(1e-20)).const_value() == 1e-20

    def test_identities(self) -> None:
        assert a + 0.0 is a
        assert a * 1.0 is a
        assert a * -1.0 is -a
        assert I.neg(I.neg(a)) is a
        assert (a * 0.0).is_const_equal(0)
        assert a / 1.0 is a
        assert a**1.0 is a
        assert (a**0.0).is_const_equal(1)

    def test_neutral_constants_keep_the_complex_kind(self) -> None:
        z = a + 0j
        assert z.kind == I.COMPLEX
        assert (a * (1 + 0j)).kind == I.COMPLEX

    def test_subtraction_is_addition_of_a_negation(self) -> None:
        e = u - a
        assert e.op == "add"
        assert any(x.op == "neg" for x in e.args)
        assert I.to_str(e) in ("(u - a)",)

    def test_dependency_tracking(self) -> None:
        e = I.exp(a * u) + b
        assert e.depends_u
        assert e.params == frozenset({0, 1})
        assert not (a * b).depends_u


class TestValidation:
    def test_boolean_misuse(self) -> None:
        c = I.ge(a, 0.0)
        with pytest.raises(CompileError):
            c + 1.0
        with pytest.raises(CompileError):
            I.where(a, a, b)
        with pytest.raises(CompileError):
            I.ge(u, 0.0)
        with pytest.raises(CompileError):
            bool(a)

    def test_real_only_functions(self) -> None:
        with pytest.raises(CompileError):
            I.gamma(u)
        with pytest.raises(CompileError):
            I.kve(u, a)
        with pytest.raises(CompileError):
            I.polygamma(0, a)

    def test_construction_errors(self) -> None:
        with pytest.raises(CompileError):
            I.param(-1, "x")
        with pytest.raises(CompileError):
            I.const("x")
        with pytest.raises(CompileError):
            float(a)
        with pytest.raises(CompileError):
            a.const_value()


class TestWhereAndRewriting:
    def test_where_with_constant_condition_selects(self) -> None:
        assert I.where(I.ge(1.0, 0.0), a, b) is a
        w = I.where(I.gt(0.0, 1.0), a, u)
        assert w is u
        promoted = I.where(I.ge(1.0, 0.0), a, u)
        assert promoted.kind == I.COMPLEX

    def test_where_with_equal_branches_collapses(self) -> None:
        assert I.where(I.ge(a, 0.0), b, b) is b

    def test_le_lt_are_flipped_ge_gt(self) -> None:
        assert I.le(a, b) is I.ge(b, a)
        assert I.lt(a, b) is I.gt(b, a)

    def test_substitute_resimplifies(self) -> None:
        e = a * I.exp(b * u)
        out = I.substitute(e, {b: 0.0})
        assert out is a * (1 + 0j)  # exp(0j) = 1 + 0j keeps the complex kind
        assert out.kind == I.COMPLEX
        both = I.substitute([e, b], {a: 2.0, b: 1.0})
        assert both[0] is 2.0 * I.exp(u)
        assert both[1] is I.const(1.0)

    def test_substitute_keeps_untouched_nodes(self) -> None:
        e = a * I.exp(b * u)
        assert I.substitute(e, {I.param(7, "z"): 1.0}) is e

    def test_rebuild_special_payloads(self) -> None:
        p = I.polygamma(2, a)
        assert I.rebuild(p, [b]) is I.polygamma(2, b)

    def test_postorder_puts_operands_first(self) -> None:
        e = I.log(a + I.exp(u)) * I.exp(u)
        order = I.postorder([e])
        pos = {id(n): i for i, n in enumerate(order)}
        for n in order:
            for x in n.args:
                assert pos[id(x)] < pos[id(n)]

    def test_postorder_is_iterative(self) -> None:
        e = u
        for k in range(5000):
            e = I.exp(e) + float(k)
        assert len(I.postorder([e])) > 5000

    def test_to_str(self) -> None:
        assert I.to_str(I.log(1.0 - u)) == "log((1.0 - u))"
        assert I.to_str(I.polygamma(1, a)) == "polygamma(1, a)"
        assert len(I.to_str(I.exp(u) * 0.123456789, max_len=10)) == 10
        assert "where" in repr(I.where(I.ge(a, 0.0), a, b))
