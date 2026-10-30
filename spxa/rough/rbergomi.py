"""
Rough Bergomi (rBergomi) stochastic volatility model with power-law short-time skew.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from spxa.rough.kernel import FractionalKernel


class RoughBergomi:
    r"""Rough Bergomi (rBergomi) model of Bayer, Friz, and Gatheral (2016).

    .. math::
        S_t = S_0 \exp\left( \int_0^t \sqrt{v_s} dW_s - \frac{1}{2} \int_0^t v_s ds \right)
        v_t = \xi_0 \exp\left( \eta \sqrt{2H} \tilde{W}_t^H - \frac{1}{2} \eta^2 t^{2H} \right)
    where :math:`\tilde{W}_t^H = \int_0^t (t - s)^{H - 1/2} dZ_s` and :math:`d\langle W, Z \rangle_t = \rho dt`.

    Since :math:`\mathrm{Var}(\sqrt{2H}\tilde{W}_t^H) = t^{2H}`, the forward variance is flat:
    :math:`\mathbb{E}[v_t] = \xi_0` for all :math:`t`.

    References
    ----------
    Bayer, C., Friz, P. & Gatheral, J. (2016). Pricing under rough volatility.
    *Quantitative Finance*, 16(6), 887-904.
    Fukasawa, M. (2017). Short-time at-the-money skew and rough fractional volatility.
    *Quantitative Finance*, 17(2), 189-198.
    """

    def __init__(
        self,
        hurst: float = 0.1,
        eta: float = 1.9,
        rho: float = -0.7,
        xi0: float = 0.04,
        s0: float = 100.0,
    ) -> None:
        self.H = float(hurst)
        self.eta = float(eta)
        self.rho = float(rho)
        self.xi0 = float(xi0)
        self.s0 = float(s0)
        self.kernel = FractionalKernel(hurst=hurst)

    def theoretical_short_time_skew(self, tau: float) -> float:
        r"""Evaluates the leading-order short-time ATM implied volatility skew (absolute value):

        .. math::

            \left| \frac{\partial \sigma_{BS}}{\partial k} \right|_{k=0}
            \approx \frac{|\rho| \eta \sqrt{2H}}{2 (H + 1/2)(H + 3/2)} \tau^{H - 1/2}

        (Fukasawa 2017; Bayer, Friz & Gatheral 2016). The sign of the skew is that of :math:`\rho`.
        """
        if tau <= 0.0:
            return 0.0
        c = (abs(self.rho) * self.eta * np.sqrt(2.0 * self.H)) / (
            2.0 * (self.H + 0.5) * (self.H + 1.5)
        )
        return float(c * (tau ** (self.H - 0.5)))

    def simulate_paths(
        self,
        t_mat: float = 1.0,
        n_steps: int = 100,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Simulates joint paths of instantaneous variance :math:`v_t` and spot price :math:`S_t`.

        The Volterra driver :math:`\tilde{W}^H` and its Brownian driver :math:`Z` are sampled
        exactly and jointly by Cholesky factorisation of their covariance
        (:meth:`FractionalKernel.volterra_joint_covariance`). The spot Brownian increments are
        :math:`dW = \rho \, dZ + \sqrt{1 - \rho^2} \, dZ^\perp`, so the leverage correlation
        between returns and variance is preserved exactly; the log-price is advanced with a
        left-point (Itô) Euler step.
        """
        dt = t_mat / n_steps

        joint = self.kernel.sample_volterra_cholesky(
            t_mat=t_mat, n_steps=n_steps, n_paths=n_paths, seed=seed
        )
        w_tilde = joint["volterra_paths"]  # (n_paths, n_steps + 1)
        z_paths = joint["driver_paths"]
        time_grid = joint["time_grid"]

        # v_t = xi0 * exp(eta * sqrt(2H) * W_tilde_t - 0.5 * eta^2 * t^{2H})
        t_pow = time_grid ** (2.0 * self.H)
        exponent = self.eta * np.sqrt(2.0 * self.H) * w_tilde - 0.5 * (self.eta**2) * t_pow
        v_paths = self.xi0 * np.exp(exponent)

        # Spot Brownian increments correlated with the variance driver Z
        dz = np.diff(z_paths, axis=1)
        # Independent stream (spawned child seed) for the orthogonal component
        rng_perp = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        dz_perp = rng_perp.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
        dw = self.rho * dz + np.sqrt(1.0 - self.rho**2) * dz_perp

        d_log_s = -0.5 * v_paths[:, :-1] * dt + np.sqrt(v_paths[:, :-1]) * dw
        log_s = np.log(self.s0) + np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(d_log_s, axis=1)], axis=1
        )
        s_paths = np.exp(log_s)

        return {
            "spot_paths": s_paths,
            "variance_paths": v_paths,
            "time_grid": time_grid,
        }
