"""
Fractional Brownian motion, Riemann-Liouville kernels, and hybrid convolution schemes for rough volatility.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.special import gamma, hyp2f1


class FractionalKernel:
    r"""Evaluates fractional singular power kernels and simulates fractional Brownian motion :math:`B_t^H`."""

    def __init__(self, hurst: float = 0.1) -> None:
        if not (0.0 < hurst < 1.0):
            raise ValueError(f"Hurst exponent H must be in (0, 1), got {hurst}")
        self.H = float(hurst)
        self.alpha = self.H - 0.5  # In (-0.5, 0) for rough regime H < 0.5
        self.prefactor = np.sqrt(2.0 * self.H) / gamma(self.H + 0.5)

    def kernel_value(self, t: float) -> float:
        r"""Evaluates Riemann-Liouville fractional kernel :math:`K_H(t) = \frac{\sqrt{2H}}{\Gamma(H + 1/2)} t^{H - 1/2}`."""
        if t <= 0.0:
            return 0.0
        return float(self.prefactor * (t**self.alpha))

    def sample_fbm_cholesky(
        self,
        t_mat: float = 1.0,
        n_steps: int = 100,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Exact simulation of fractional Brownian motion :math:`B_t^H` via Cholesky decomposition of covariance:

        .. math::
            \mathbb{E}[B_t^H B_s^H] = \frac{1}{2} \left( t^{2H} + s^{2H} - |t - s|^{2H} \right)
        """
        rng = np.random.default_rng(seed)
        time_grid = np.linspace(0.0, t_mat, n_steps + 1)
        t_grid = time_grid[1:]  # Exclude t=0 where variance is 0
        n = len(t_grid)

        # Covariance matrix
        t_col = t_grid[:, np.newaxis]
        t_row = t_grid[np.newaxis, :]
        cov = 0.5 * (
            t_col ** (2.0 * self.H)
            + t_row ** (2.0 * self.H)
            - np.abs(t_col - t_row) ** (2.0 * self.H)
        )

        # Regularize diagonal slightly for numerical stability when H is very small
        cov += 1e-10 * np.eye(n)
        chol = np.linalg.cholesky(cov)

        z = rng.normal(0.0, 1.0, size=(n, n_paths))
        paths = chol @ z  # Shape (n, n_paths)

        full_paths = np.zeros((n_paths, n_steps + 1))
        full_paths[:, 1:] = paths.T

        return {
            "paths": full_paths,
            "time_grid": time_grid,
        }

    def volterra_joint_covariance(self, t_grid: np.ndarray) -> np.ndarray:
        r"""Exact joint covariance of the Riemann-Liouville Volterra process and its driver.

        For :math:`\tilde{W}_t = \int_0^t (t - s)^{H - 1/2} dZ_s` with :math:`a = H - 1/2`
        and grid points :math:`0 < t_1 < \dots < t_n`, returns the :math:`2n \times 2n`
        covariance of :math:`(\tilde{W}_{t_1}, \dots, \tilde{W}_{t_n}, Z_{t_1}, \dots, Z_{t_n})`:

        .. math::

            \mathrm{Cov}(\tilde{W}_s, \tilde{W}_t) &= \frac{s^{a + 1} t^{a}}{a + 1}
                \, {}_2F_1\!\left(-a, 1; a + 2; \tfrac{s}{t}\right), \quad s \le t, \\
            \mathrm{Var}(\tilde{W}_t) &= \frac{t^{2H}}{2H}, \\
            \mathrm{Cov}(\tilde{W}_t, Z_s) &= \frac{t^{a + 1} - (t - \min(s, t))^{a + 1}}{a + 1}, \\
            \mathrm{Cov}(Z_s, Z_t) &= \min(s, t).

        This is the exact Cholesky construction of Bayer, Friz & Gatheral (2016, Section 4).
        """
        t = np.asarray(t_grid, dtype=np.float64)
        if t.ndim != 1 or np.any(t <= 0.0):
            raise ValueError("t_grid must be a 1D array of strictly positive times")
        a = self.alpha
        n = len(t)
        cov = np.zeros((2 * n, 2 * n), dtype=np.float64)

        s_mat = np.minimum.outer(t, t)
        l_mat = np.maximum.outer(t, t)
        ratio = s_mat / l_mat
        cov_ww = (s_mat ** (a + 1.0)) * (l_mat**a) / (a + 1.0) * hyp2f1(-a, 1.0, a + 2.0, ratio)
        np.fill_diagonal(cov_ww, t ** (2.0 * self.H) / (2.0 * self.H))

        t_col = t[:, np.newaxis]
        m_mat = np.minimum(t_col, t[np.newaxis, :])
        cov_wz = (t_col ** (a + 1.0) - (t_col - m_mat) ** (a + 1.0)) / (a + 1.0)

        cov[:n, :n] = cov_ww
        cov[:n, n:] = cov_wz
        cov[n:, :n] = cov_wz.T
        cov[n:, n:] = s_mat
        return cov

    def sample_volterra_cholesky(
        self,
        t_mat: float = 1.0,
        n_steps: int = 100,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Exact joint simulation of :math:`(\tilde{W}_t, Z_t)` on a uniform grid.

        Returns arrays of shape ``(n_paths, n_steps + 1)`` with ``volterra_paths``
        :math:`\tilde{W}` and ``driver_paths`` :math:`Z`, both starting at 0.
        """
        rng = np.random.default_rng(seed)
        time_grid = np.linspace(0.0, t_mat, n_steps + 1)
        cov = self.volterra_joint_covariance(time_grid[1:])
        jitter = 1e-12 * np.max(np.diag(cov))
        chol = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
        z = rng.normal(0.0, 1.0, size=(cov.shape[0], n_paths))
        joint = (chol @ z).T

        volterra = np.zeros((n_paths, n_steps + 1))
        driver = np.zeros((n_paths, n_steps + 1))
        volterra[:, 1:] = joint[:, :n_steps]
        driver[:, 1:] = joint[:, n_steps:]
        return {
            "volterra_paths": volterra,
            "driver_paths": driver,
            "time_grid": time_grid,
        }

    def hybrid_convolution(
        self,
        dw: np.ndarray,
        dt: float,
        b_cutoff: int = 3,
    ) -> np.ndarray:
        r"""Bennedsen-Lunde-Pakkanen (2017) Hybrid scheme for evaluating the Volterra integral :math:`\int_0^t (t - s)^\alpha dW_s`.

        Splits into:
        1. Exact local integration for the singular part :math:`s \in [t - b \Delta t, t]`.
        2. Fast discrete convolution for the regular tail.
        """
        n_steps = len(dw)
        res = np.zeros(n_steps + 1)
        alpha = self.alpha

        # Local singular weights: exact integrals of (k dt - s)^alpha ds
        local_weights = np.zeros(b_cutoff)
        for k in range(1, b_cutoff + 1):
            w_k = ((k * dt) ** (alpha + 1.0) - ((k - 1) * dt) ** (alpha + 1.0)) / (alpha + 1.0)
            local_weights[k - 1] = w_k / dt

        # Convolution weights for tail
        tail_len = max(0, n_steps - b_cutoff)
        if tail_len > 0:
            k_tail = np.arange(b_cutoff, n_steps)
            tail_weights = (k_tail * dt) ** alpha
            weights = np.concatenate([local_weights, tail_weights])
        else:
            weights = local_weights[:n_steps]

        # Time-stepping convolution
        for i in range(1, n_steps + 1):
            w_slice = weights[:i][::-1]
            dw_slice = dw[:i]
            res[i] = np.sum(w_slice * dw_slice)

        return res
