"""
Tests for rough paths, tensor signatures, shuffle algebra, and Goursat signature kernels.
"""

import numpy as np
import pytest

from spxa.graph.signature.signature_nodes import (
    ExpectedSignatureNode,
    SignatureLinearPricerNode,
)
from spxa.signature.algebra import SignatureAlgebra
from spxa.signature.kernel import SignatureKernel
from spxa.signature.path_signature import PathSignature


def test_signature_dimension_and_chen_identity():
    # d=2, depth=3: 1 + 2 + 4 + 8 = 15
    dim = PathSignature.signature_dimension(d=2, depth=3)
    assert dim == 15

    # 2D path: Spiral path
    t = np.linspace(0.0, 2.0 * np.pi, 100)
    path = np.column_stack([t * np.cos(t), t * np.sin(t)])

    sig_calc = PathSignature(depth=3)
    sig = sig_calc.signature(path)
    flat = sig_calc.flatten_signature(sig)
    assert flat.shape == (15,)
    assert np.isclose(flat[0], 1.0)  # Level 0 is always 1

    # Verify Chen's identity splitting at index 50
    chen_err = sig_calc.verify_chen_identity(path, split_idx=50)
    assert chen_err < 1e-12


def test_shuffle_product_and_levy_area_circle():
    # Single step shuffle identity
    shuf_res = SignatureAlgebra.verify_shuffle_product_level2(np.array(1.5), np.array(-0.8))
    assert shuf_res["discrepancy"] < 1e-15

    # Circular path: X(t) = cos(t), Y(t) = sin(t) for t in [0, 2*pi]
    # Enclosed algebraic area is pi * R^2 = pi * 1^2 = pi
    n_pts = 500
    t = np.linspace(0.0, 2.0 * np.pi, n_pts)
    circle_path = np.column_stack([np.cos(t), np.sin(t)])

    sig_calc = PathSignature(depth=2)
    sig = sig_calc.signature(circle_path)
    log_sig = SignatureAlgebra.log_signature_2d(sig, depth=2)

    # Level 2 log-signature is Lévy area: Area = 0.5 * (S^{12} - S^{21})
    levy_area = log_sig[2][0]
    # For a circle traversed counter-clockwise, Area = pi ~ 3.14159
    assert np.isclose(levy_area, np.pi, rtol=0.01)


def test_log_signature_dimension_reduction():
    # Witt's formula for d=2, depth=3
    dim_witt = SignatureAlgebra.log_signature_dimension(d=2, depth=3)
    # L_2(1)=2, L_2(2)=1, L_2(3)=2 -> total = 5
    assert dim_witt == 5

    # Compare against full tensor signature dimension
    full_dim = PathSignature.signature_dimension(d=2, depth=3)
    assert full_dim == 15
    # 66.7% dimension reduction without information loss
    assert dim_witt < full_dim


def test_goursat_signature_kernel_evaluation():
    t = np.linspace(0.0, 1.0, 50)
    path_1 = np.column_stack([t, np.sin(2.0 * np.pi * t)])
    path_2 = np.column_stack([t, np.cos(2.0 * np.pi * t)])

    k_11 = SignatureKernel.evaluate(path_1, path_1)
    k_22 = SignatureKernel.evaluate(path_2, path_2)
    k_12 = SignatureKernel.evaluate(path_1, path_2)
    k_21 = SignatureKernel.evaluate(path_2, path_1)

    assert k_11 > 1.0
    assert k_22 > 1.0
    assert np.isclose(k_12, k_21, atol=1e-12)

    # Cauchy-Schwarz inequality for kernels: K(x, y)^2 <= K(x, x) K(y, y)
    assert k_12**2 <= k_11 * k_22 + 1e-10

    # Normalized distance: d(x, x) == 0, d(x, y) > 0
    d_same = SignatureKernel.normalized_distance(path_1, path_1)
    d_diff = SignatureKernel.normalized_distance(path_1, path_2)
    assert np.isclose(d_same, 0.0, atol=1e-10)
    assert d_diff > 0.1


def test_expected_signature_linear_pricing():
    # Simulate 50 sample paths of a 1D driftless BM + time channel (time-augmented path)
    rng = np.random.default_rng(42)
    n_paths = 60
    n_steps = 25
    dt = 1.0 / n_steps
    t_grid = np.linspace(0.0, 1.0, n_steps + 1)

    paths = np.zeros((n_paths, n_steps + 1, 2))
    for i in range(n_paths):
        dw = rng.normal(0.0, np.sqrt(dt), size=n_steps)
        bm = np.concatenate([[0.0], np.cumsum(dw)])
        paths[i, :, 0] = t_grid
        paths[i, :, 1] = bm

    # Asian call payoff on BM: max(mean(B_t) - 0.1, 0)
    payoffs = np.maximum(np.mean(paths[:, :, 1], axis=1) - 0.1, 0.0)

    # Train linear pricer
    pricer = SignatureLinearPricerNode(depth=2)
    weights = pricer.fit_weights(paths, payoffs)
    assert len(weights) == PathSignature.signature_dimension(d=2, depth=2)

    # Compute expected signature over the distribution
    exp_node = ExpectedSignatureNode(depth=2)
    exp_sig = exp_node.eval_expected_signature(paths)
    price = pricer.price(exp_sig)

    # Mean Monte Carlo payoff
    mc_price = float(np.mean(payoffs))
    assert np.isclose(price, mc_price, rtol=0.10)


def test_signature_kernel_converges_to_bessel_closed_form():
    from scipy.special import iv

    # For X_t = Y_t = t on [0, 1] (d = 1), K = sum_k 1 / (k!)^2 = I_0(2)
    t = np.linspace(0.0, 1.0, 101)
    assert SignatureKernel.evaluate(t, t) == pytest.approx(iv(0, 2.0), rel=1e-5)
    # Single linear step: second-order accurate, I_0(2 sqrt(M)) = 1 + M + M^2 / 4 + O(M^3)
    m = 0.01
    k_one = SignatureKernel.evaluate(np.array([0.0, np.sqrt(m)]), np.array([0.0, np.sqrt(m)]))
    assert k_one == pytest.approx(1.0 + m + m**2 / 4.0, abs=1e-6)


def test_signature_kernel_matches_truncated_signature_inner_product():

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0.0, 0.1, size=(30, 2)), axis=0)
    y = np.cumsum(rng.normal(0.0, 0.1, size=(40, 2)), axis=0)
    ps = PathSignature(depth=10)
    trunc = ps.flatten_signature(ps.signature(x)) @ ps.flatten_signature(ps.signature(y))
    assert SignatureKernel.evaluate(x, y) == pytest.approx(trunc, rel=1e-4)


def test_log_signature_level3_matches_tensor_logarithm():

    rng = np.random.default_rng(1)
    path = np.cumsum(rng.normal(0.0, 0.5, size=(6, 2)), axis=0)
    sig = PathSignature(depth=3).signature(path)
    s1, s2, s3 = sig[1], sig[2].reshape(2, 2), sig[3].reshape(2, 2, 2)
    l3 = (
        s3
        - 0.5 * (np.einsum("i,jk->ijk", s1, s2) + np.einsum("ij,k->ijk", s2, s1))
        + np.einsum("i,j,k->ijk", s1, s1, s1) / 3.0
    )
    # Hall basis: [[1,2],1] = 2*121 - 112 - 211, [[1,2],2] = 122 - 2*212 + 221
    p1 = np.zeros((2, 2, 2))
    p1[0, 1, 0], p1[0, 0, 1], p1[1, 0, 0] = 2.0, -1.0, -1.0
    p2 = np.zeros((2, 2, 2))
    p2[0, 1, 1], p2[1, 0, 1], p2[1, 1, 0] = 1.0, -2.0, 1.0

    c = SignatureAlgebra.log_signature_2d(sig, depth=3)[3]
    np.testing.assert_allclose(c[0] * p1 + c[1] * p2, l3, atol=1e-12)


def test_witt_dimension_and_mobius():
    assert [SignatureAlgebra.mobius(n) for n in range(1, 13)] == [
        1,
        -1,
        -1,
        0,
        -1,
        1,
        -1,
        0,
        0,
        1,
        -1,
        0,
    ]
    # Necklace polynomial values for d = 2: 2, 1, 2, 3, 6, 9, 18, 30, 56, 99
    assert SignatureAlgebra.log_signature_dimension(d=2, depth=10) == 226
    assert SignatureAlgebra.log_signature_dimension(d=3, depth=4) == 32
