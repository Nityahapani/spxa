"""Shape bucketing of the jax backend: few compilations, unchanged results."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.compile import ir as I
from spxa.compile.codegen import _bucket, _pad_to_buckets, _unpad, generate

u = I.U
a = I.param(0, "a")
b = I.param(1, "b")


class TestBucketHelpers:
    @pytest.mark.parametrize(
        ("n", "want"), [(0, 0), (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (1000, 1024), (1024, 1024)]
    )
    def test_bucket(self, n: int, want: int) -> None:
        assert _bucket(n) == want

    def test_pad_repeats_edges_and_keeps_unit_axes(self) -> None:
        x = np.arange(5.0)
        col = np.array([[1.0], [2.0], [3.0]])
        padded, shape = _pad_to_buckets([x, col, np.float64(7.0)])
        assert shape == (3, 5)
        assert padded[0].shape == (8,)
        np.testing.assert_array_equal(padded[0], [0, 1, 2, 3, 4, 4, 4, 4])
        assert padded[1].shape == (4, 1)
        np.testing.assert_array_equal(padded[1][:, 0], [1, 2, 3, 3])
        assert padded[2].ndim == 0

    def test_already_bucketed_inputs_are_not_copied(self) -> None:
        x = np.arange(8.0)
        padded, shape = _pad_to_buckets([x, 2.0])
        assert shape == (8,)
        assert np.shares_memory(padded[0], x)

    def test_unpad_slices_back(self) -> None:
        full = np.arange(32.0).reshape(4, 8)
        np.testing.assert_array_equal(_unpad(full, (3, 5)), full[:3, :5])
        np.testing.assert_array_equal(_unpad(np.ones((4, 1)), (3, 5)), np.ones((3, 1)))
        assert _unpad(np.float64(2.0), (3, 5)) == 2.0


class TestJaxPrograms:
    @pytest.fixture(autouse=True)
    def _need_jax(self) -> None:
        pytest.importorskip("jax")

    def test_odd_shapes_match_numpy(self) -> None:
        outs = [-a * I.log(1.0 - 1j * u / b), a * u * u, I.const(3.0)]
        pj = generate(outs, 2, "jax", "bucket_odd")
        pn = generate(outs, 2, "numpy", "bucket_odd_np")
        rng = np.random.default_rng(5)
        for n_u, n_b in [(7, 1), (33, 5), (1000, 3), (1, 9)]:
            uu = rng.normal(size=n_u) * 4
            A = rng.uniform(0.5, 2.0, size=(n_b, 1))
            B = rng.uniform(1.0, 3.0, size=(n_b, 1))
            got, want = pj(uu, A, B), pn(uu, A, B)
            for g, w in zip(got, want):
                g, w = np.broadcast_arrays(g, w)
                np.testing.assert_allclose(g, w, rtol=1e-13, atol=1e-14)
            assert np.shape(got[0]) == (n_b, n_u)
            assert np.shape(got[1]) == (n_b, n_u)

    def test_sizes_in_one_bucket_share_a_compilation(self) -> None:
        prog = generate([a * I.exp(1j * u * b)], 2, "jax", "bucket_cache")
        jitted = prog.fn.jitted  # type: ignore[attr-defined]
        for n in range(17, 33):  # all round up to 32
            prog(np.linspace(-1, 1, n), 1.5, 0.5)
        assert jitted._cache_size() == 1
        prog(np.linspace(-1, 1, 40), 1.5, 0.5)  # 64
        assert jitted._cache_size() == 2

    def test_batched_cos_prices_reuse_compilations(self) -> None:
        from spxa.zoo.levy import BrownianMotion, GammaProcess

        cj = (BrownianMotion(0.05, 0.2) @ GammaProcess(4.0, 4.0)).compile(backend="jax")
        cn = (BrownianMotion(0.05, 0.2) @ GammaProcess(4.0, 4.0)).compile()
        K = np.linspace(80, 120, 7)
        rng = np.random.default_rng(1)
        for B in (5, 6, 7):
            th = cj.values * (1 + 0.05 * rng.uniform(-1, 1, (B, cj.n_params)))
            pj, gj = cj.cos_price(100.0, K, 0.5, theta=th, grad=True)
            pn, gn = cn.cos_price(100.0, K, 0.5, theta=th, grad=True)
            assert pj.shape == pn.shape and gj.shape == gn.shape
            np.testing.assert_allclose(pj, pn, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(gj, gn, rtol=1e-10, atol=1e-10)
        # batches 5, 6 and 7 all round up to 8: one compilation per program
        progs = [p for fam in cj.kernel._families.values() for p in fam.cache.values()]
        assert progs
        for prog in progs:
            assert prog.fn.jitted._cache_size() == 1  # type: ignore[attr-defined]
