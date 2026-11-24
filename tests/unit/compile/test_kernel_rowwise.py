"""dpsi/du and per-row frequency grids in ExponentKernel."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.compile.kernel import ExponentKernel
from spxa.compile.lower import lower
from spxa.zoo.levy import (
    CGMY,
    NIG,
    BrownianMotion,
    GammaProcess,
    KouJumpDiffusion,
    MeixnerProcess,
)

PROC = (
    BrownianMotion(0.05, 0.2) + 0.5 * KouJumpDiffusion(0.05, 0.2, 2.0, 0.4, 8.0, 5.0)
) @ GammaProcess(4.0, 4.0) + CGMY(1.0, 5.0, 6.0, 0.5)
U = np.linspace(-15.0, 15.0, 61) + 0j


@pytest.fixture(scope="module")
def kernel() -> ExponentKernel:
    return ExponentKernel(lower(PROC), "numpy")


def test_du_matches_complex_step_free_differences(kernel: ExponentKernel) -> None:
    psi, dpsi, grad = kernel.psi_du_and_grad(U)
    p2, g2 = kernel.psi_and_grad(U)
    np.testing.assert_array_equal(psi, p2)
    np.testing.assert_array_equal(grad, g2)
    # fourth-order central differences in u
    h = 1e-3
    fd = (
        -kernel.psi(U + 2 * h)
        + 8 * kernel.psi(U + h)
        - 8 * kernel.psi(U - h)
        + kernel.psi(U - 2 * h)
    ) / (12 * h)
    np.testing.assert_allclose(dpsi, fd, rtol=1e-8, atol=1e-9)


def test_du_closed_forms() -> None:
    # Brownian motion: psi' = i mu - sigma^2 u; NIG: i mu + delta (beta + iu) i / sqrt(...)
    k = ExponentKernel(lower(BrownianMotion(0.3, 0.4)))
    _, d, _ = k.psi_du_and_grad(U)
    np.testing.assert_allclose(d, 0.3j - 0.16 * U, rtol=0, atol=1e-15)
    a, b, dl, m = 10.0, -2.0, 0.5, 0.1
    k = ExponentKernel(lower(NIG(a, b, dl, m)))
    _, d, _ = k.psi_du_and_grad(U)
    exact = 1j * m + 1j * dl * (b + 1j * U) / np.sqrt(a**2 - (b + 1j * U) ** 2)
    np.testing.assert_allclose(d, exact, rtol=1e-13)


def test_du_through_the_meixner_sign_guard() -> None:
    k = ExponentKernel(lower(MeixnerProcess(0.3, -0.5, 1.0, 0.1)))
    _, d, _ = k.psi_du_and_grad(U)
    h = 1e-4
    fd = (k.psi(U + h) - k.psi(U - h)) / (2 * h)
    np.testing.assert_allclose(d, fd, rtol=1e-7, atol=1e-9)


def test_rowwise_equals_row_by_row(kernel: ExponentKernel) -> None:
    th = np.tile(kernel.theta0, (3, 1))
    th[:, 1] = [0.1, 0.2, 0.3]
    th[:, -1] = [0.3, 0.5, 1.4]
    u = np.stack([U, 2 * U, 0.5 * U])
    p, d, g = kernel.psi_du_and_grad(u, th, rowwise=True)
    pr = kernel.psi_rowwise(u, th)
    assert p.shape == u.shape and g.shape == u.shape + (kernel.n_free,)
    for b in range(3):
        rp, rd, rg = kernel.psi_du_and_grad(u[b], th[b])
        np.testing.assert_array_equal(p[b], rp)
        np.testing.assert_array_equal(pr[b], rp)
        np.testing.assert_array_equal(d[b], rd)
        np.testing.assert_array_equal(g[b], rg)


@pytest.mark.parametrize("backend", ["numba", "jax"])
def test_rowwise_backends(backend: str, kernel: ExponentKernel) -> None:
    pytest.importorskip(backend)
    proc = MeixnerProcess(0.3, -0.5, 1.0, 0.0) + NIG(10.0, -2.0, 0.5)
    ref = ExponentKernel(lower(proc), "numpy")
    k = ExponentKernel(lower(proc), backend)
    th = np.tile(k.theta0, (2, 1))
    th[:, 0] = [0.3, 0.6]
    u = np.stack([U, 3 * U])
    for a, b in zip(
        k.psi_du_and_grad(u, th, rowwise=True), ref.psi_du_and_grad(u, th, rowwise=True)
    ):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(k.psi_rowwise(u, th), ref.psi_rowwise(u, th), rtol=1e-12)


def test_rowwise_validation(kernel: ExponentKernel) -> None:
    with pytest.raises(ValueError, match="rowwise"):
        kernel.psi_rowwise(U, kernel.theta0)
    with pytest.raises(ValueError, match="rowwise"):
        kernel.psi_du_and_grad(np.stack([U, U]), np.tile(kernel.theta0, (3, 1)), rowwise=True)
