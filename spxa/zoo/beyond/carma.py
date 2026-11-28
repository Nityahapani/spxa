r"""
Lévy-driven continuous-time ARMA processes, CARMA(p, q).

A CARMA(p, q) process is the continuous-time analogue of an ARMA model
(Brockwell 2001). Formally, :math:`a(D) Y_t = b(D) DL_t`, with :math:`D`
the derivative, :math:`L` a Lévy process, and

.. math::
    a(z) = z^p + a_1 z^{p-1} + \dots + a_p, \qquad
    b(z) = b_0 + b_1 z + \dots + b_q z^q, \qquad q < p .

It is defined through the state-space representation

.. math::
    dX_t = A X_t\,dt + e_p\,dL_t, \qquad Y_t = b^\top X_t,

where :math:`A` is the companion matrix of :math:`a` (ones on the
superdiagonal, last row :math:`(-a_p, \dots, -a_1)`), :math:`e_p` is the last
unit vector, and :math:`b = (b_0, \dots, b_q, 0, \dots, 0)`. CARMA(1, 0) is
the Lévy-driven Ornstein–Uhlenbeck process.

When every zero of :math:`a` has negative real part (causality), the process
has a stationary version

.. math::
    Y_t = \int_{-\infty}^t g(t-s)\,dL_s, \qquad
    g(t) = b^\top e^{At} e_p = \sum_i \frac{b(\lambda_i)}{a'(\lambda_i)}\,e^{\lambda_i t},

the sum running over the (distinct) zeros :math:`\lambda_i` of :math:`a`.
Several stationary quantities follow exactly from this kernel:

- mean: :math:`E[Y] = \kappa_1(L_1)\,b_0/a_p`;
- autocovariance: :math:`\gamma(h) = \kappa_2(L_1)\,b^\top e^{A|h|}\Sigma b`, where
  :math:`A\Sigma + \Sigma A^\top + e_pe_p^\top = 0`;
- spectral density: :math:`f(\omega) = \frac{\kappa_2(L_1)}{2\pi}\frac{|b(i\omega)|^2}{|a(i\omega)|^2}`,
  normalised so that :math:`\int f\,d\omega = \gamma(0)`;
- cumulants: :math:`\kappa_n(Y) = \kappa_n(L_1)\int_0^\infty g(s)^n\,ds`;
- characteristic function: :math:`\log E[e^{iuY}] = \int_0^\infty \psi_L(u\,g(s))\,ds`.

The second-order structure depends only on :math:`\kappa_2(L_1)`. Jumps of
the driver show up in the higher cumulants and in the paths: each jump of
:math:`L` adds a copy of the kernel :math:`g` to :math:`Y`.

References
----------
Brockwell, P. J. (2001). Lévy-driven CARMA processes. *Ann. Inst. Statist.
Math.* 53, 113–124.

Brockwell, P. J., Davis, R. A. & Yang, Y. (2011). Estimation for non-negative
Lévy-driven CARMA processes. *J. Bus. Econ. Statist.* 29, 250–259.
"""

from __future__ import annotations

import itertools
from typing import Optional, Sequence

import numpy as np
from scipy import integrate, linalg

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind
from spxa.zoo.levy.brownian import BrownianMotion

__all__ = ["CARMA", "CARMAProcess"]


def _polyval_ascending(coef: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate sum_k coef[k] z^k."""
    return np.polyval(coef[::-1], z)


class CARMAProcess(Process):
    r"""
    Lévy-driven CARMA(p, q) process (see the module docstring).

    Parameters
    ----------
    ar :
        Coefficients :math:`(a_1, \dots, a_p)` of
        :math:`a(z) = z^p + a_1 z^{p-1} + \dots + a_p`. All zeros of
        :math:`a` must have negative real parts.
    ma :
        Coefficients :math:`(b_0, \dots, b_q)` of
        :math:`b(z) = b_0 + b_1 z + \dots + b_q z^q`, with :math:`q < p`.
        Default ``(1.0,)``, the CAR(p) process.
    driver :
        The driving Lévy process :math:`L`. The default is a standard Brownian
        motion, which gives a Gaussian CARMA process.
    x0 :
        Initial state :math:`X_0` (length ``p``) for :meth:`simulate`.
        ``None`` (the default) starts from the stationary law.
    substeps :
        Driver increments per output step used by :meth:`simulate` when the
        driver is not a Brownian motion (see there).

    Notes
    -----
    ExactnessLevel: MOMENT_PROPAGATION. The stationary moments, cumulants,
    autocovariance and spectral density are exact. With a Brownian driver,
    simulation is exact as well. ``triplet`` raises, because the process
    does not have independent increments.
    """

    def __init__(
        self,
        ar: Sequence[float],
        ma: Sequence[float] = (1.0,),
        driver: Optional[Process] = None,
        x0: Optional[Sequence[float]] = None,
        substeps: int = 16,
    ) -> None:
        a = np.atleast_1d(np.asarray(ar, dtype=float))
        b = np.atleast_1d(np.asarray(ma, dtype=float))
        if a.ndim != 1 or a.size < 1:
            raise ValueError("ar must hold at least one coefficient")
        if b.ndim != 1 or b.size < 1:
            raise ValueError("ma must hold at least one coefficient")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("coefficients must be finite")
        p, q = a.size, b.size - 1
        if q >= p:
            raise ValueError(f"CARMA(p, q) needs q < p, got p={p}, q={q}")
        if not np.any(b != 0.0):
            raise ValueError("the MA polynomial b must not vanish")
        roots = np.roots(np.concatenate([[1.0], a]))
        if not np.all(roots.real < 0):
            raise ValueError(
                "the zeros of the AR polynomial must have negative real parts "
                f"(causality); got {np.round(roots, 6)}"
            )
        if int(substeps) < 1:
            raise ValueError("substeps must be at least 1")
        self.ar = a
        self.ma = b
        self.p, self.q = p, q
        self.driver = BrownianMotion(0.0, 1.0) if driver is None else driver
        self.substeps = int(substeps)

        A = np.zeros((p, p))
        A[np.arange(p - 1), np.arange(1, p)] = 1.0
        A[-1, :] = -a[::-1]
        self._A = A
        self._e = np.zeros(p)
        self._e[-1] = 1.0
        self._b = np.zeros(p)
        self._b[: q + 1] = b
        # stationary state covariance for a unit-variance driver
        self._sigma = linalg.solve_continuous_lyapunov(A, -np.outer(self._e, self._e))
        self._sigma = 0.5 * (self._sigma + self._sigma.T)

        kap = self.driver.cumulants(2)
        self._k1 = float(kap.get(1, np.nan))
        self._k2 = float(kap.get(2, np.nan))

        if x0 is not None:
            x = np.asarray(x0, dtype=float).reshape(-1)
            if x.size != p:
                raise ValueError(f"x0 must have length p={p}")
            self.x0: Optional[np.ndarray] = x
        else:
            self.x0 = None

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CARMA({p},{q})(ar={a.tolist()}, ma={b.tolist()}, driver={self.driver!r})",
            math_note=(
                "a(D)Y = b(D)DL; dX = AX dt + e_p dL, Y = b'X; "
                f"AR zeros {np.round(roots, 4).tolist()}; "
                f"E[Y] = κ1 b0/a_p = {self.stationary_mean:.6g}; Var(Y) = κ2 b'Σb = {self.stationary_variance:.6g}"
            ),
            reference="Brockwell (2001), Ann. Inst. Statist. Math. 53",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_roots(
        cls,
        ar_roots: Sequence[complex],
        ma_roots: Sequence[complex] = (),
        *,
        scale: float = 1.0,
        driver: Optional[Process] = None,
        x0: Optional[Sequence[float]] = None,
        substeps: int = 16,
    ) -> CARMAProcess:
        r"""Build from the zeros of :math:`a` and of :math:`b`, with :math:`b(z) = \text{scale}\prod_j (z - \mu_j)`.

        Complex zeros must come in conjugate pairs.
        """
        ar_poly = np.poly(np.asarray(ar_roots, dtype=complex))
        ma_poly = scale * np.poly(np.asarray(ma_roots, dtype=complex))
        for poly in (ar_poly, ma_poly):
            if np.max(np.abs(np.imag(poly))) > 1e-9 * max(1.0, float(np.max(np.abs(poly)))):
                raise ValueError("complex zeros must come in conjugate pairs")
        return cls(
            np.real(ar_poly[1:]),
            np.real(ma_poly[::-1]),
            driver=driver,
            x0=x0,
            substeps=substeps,
        )

    # -- structure ----------------------------------------------------------

    @property
    def state_matrix(self) -> np.ndarray:
        """The companion matrix :math:`A`."""
        return self._A.copy()

    @property
    def output_vector(self) -> np.ndarray:
        """:math:`b`, padded with zeros to length ``p``."""
        return self._b.copy()

    @property
    def ar_roots(self) -> np.ndarray:
        """Zeros of :math:`a`; all have negative real parts."""
        return np.roots(np.concatenate([[1.0], self.ar]))

    @property
    def ma_roots(self) -> np.ndarray:
        """Zeros of :math:`b` (empty for a CAR process)."""
        b = np.trim_zeros(self.ma, "b")
        return np.roots(b[::-1]) if b.size > 1 else np.array([], dtype=complex)

    @property
    def stationary_state_covariance(self) -> np.ndarray:
        r""":math:`\kappa_2(L_1)\,\Sigma`, the covariance of the stationary state :math:`X`."""
        return self._k2 * self._sigma

    @property
    def stationary_state_mean(self) -> np.ndarray:
        r""":math:`\kappa_1(L_1)\,(-A)^{-1} e_p`, the mean of the stationary state."""
        return self._k1 * np.linalg.solve(-self._A, self._e)

    def _residues(self) -> Optional[np.ndarray]:
        """b(λ)/a'(λ) at the zeros λ of a, or None when zeros nearly coincide."""
        lam = self.ar_roots
        if lam.size > 1:
            gaps = np.abs(lam[:, None] - lam[None, :])
            np.fill_diagonal(gaps, np.inf)
            if np.min(gaps) < 1e-6 * max(1.0, float(np.max(np.abs(lam)))):
                return None
        dpoly = np.polyder(np.concatenate([[1.0], self.ar]))
        return _polyval_ascending(self.ma.astype(complex), lam) / np.polyval(dpoly, lam)

    def kernel(self, t: float | np.ndarray) -> np.ndarray:
        r"""The kernel :math:`g(t) = b^\top e^{At}e_p` for :math:`t \ge 0` (zero for :math:`t < 0`)."""
        t_arr = np.asarray(t, dtype=float)
        flat = t_arr.reshape(-1)
        res = self._residues()
        if res is not None:
            lam = self.ar_roots
            out = np.real(np.exp(np.outer(np.maximum(flat, 0.0), lam)) @ res)
        else:
            out = np.array([self._b @ linalg.expm(self._A * max(s, 0.0)) @ self._e for s in flat])
        out = np.where(flat < 0, 0.0, out)
        return out.reshape(t_arr.shape)

    # -- stationary second-order structure ---------------------------------------

    @property
    def stationary_mean(self) -> float:
        r""":math:`E[Y] = \kappa_1(L_1)\,b_0/a_p`."""
        return float(self._k1 * self.ma[0] / self.ar[-1])

    @property
    def stationary_variance(self) -> float:
        r""":math:`\mathrm{Var}(Y) = \kappa_2(L_1)\,b^\top\Sigma b`."""
        return float(self._k2 * self._b @ self._sigma @ self._b)

    def autocovariance(self, h: float | np.ndarray) -> np.ndarray | float:
        r""":math:`\gamma(h) = \kappa_2(L_1)\,b^\top e^{A|h|}\Sigma b` in stationarity."""
        h_arr = np.abs(np.asarray(h, dtype=float))
        sb = self._sigma @ self._b
        flat = h_arr.reshape(-1)
        vals = np.array([self._b @ linalg.expm(self._A * s) @ sb for s in flat]) * self._k2
        return float(vals[0]) if h_arr.ndim == 0 else vals.reshape(h_arr.shape)

    def autocorrelation(self, h: float | np.ndarray) -> np.ndarray | float:
        r""":math:`\rho(h) = \gamma(h)/\gamma(0)`."""
        g = self.autocovariance(h)
        return g / self.stationary_variance

    def spectral_density(self, omega: float | np.ndarray) -> np.ndarray | float:
        r""":math:`f(\omega) = \frac{\kappa_2(L_1)}{2\pi}|b(i\omega)|^2/|a(i\omega)|^2`, with :math:`\int f = \gamma(0)`."""
        w = np.asarray(omega, dtype=float)
        z = 1j * w
        num = np.abs(_polyval_ascending(self.ma.astype(complex), z)) ** 2
        den = np.abs(np.polyval(np.concatenate([[1.0], self.ar]), z)) ** 2
        out = self._k2 / (2 * np.pi) * num / den
        return float(out) if w.ndim == 0 else out

    def sampled_autocovariance(self, dt: float, max_lag: int) -> np.ndarray:
        r""":math:`\gamma(k\,\Delta t)` for :math:`k = 0, \dots,` ``max_lag``, by powers of :math:`e^{A\Delta t}`."""
        if dt <= 0 or max_lag < 0:
            raise ValueError("dt must be positive and max_lag non-negative")
        F = linalg.expm(self._A * dt)
        v = self._sigma @ self._b
        out = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            out[k] = self._b @ v
            v = F @ v
        return self._k2 * out

    # -- higher-order stationary structure --------------------------------------

    def kernel_power_integral(self, n: int) -> float:
        r""":math:`\int_0^\infty g(s)^n\,ds`, by residues when the AR zeros are distinct."""
        if n < 1:
            raise ValueError("n must be at least 1")
        if n == 1:
            return float(self.ma[0] / self.ar[-1])
        if n == 2:
            return float(self._b @ self._sigma @ self._b)
        res = self._residues()
        lam = self.ar_roots
        if res is not None and self.p**n <= 200_000:
            total = 0.0 + 0.0j
            for idx in itertools.product(range(self.p), repeat=n):
                total += np.prod(res[list(idx)]) / (-np.sum(lam[list(idx)]))
            return float(total.real)
        decay = float(-np.max(lam.real))
        val, _ = integrate.quad(lambda s: float(self.kernel(s)) ** n, 0.0, 60.0 / decay, limit=500)
        return float(val)

    def stationary_cumulant(self, n: int) -> float:
        r""":math:`\kappa_n(Y) = \kappa_n(L_1)\int_0^\infty g(s)^n\,ds`."""
        kap = self.driver.cumulants(n).get(n, np.nan)
        return float(kap * self.kernel_power_integral(n))

    def stationary_char_func(self, u: float | np.ndarray) -> np.ndarray | complex:
        r""":math:`E[e^{iuY}] = \exp\int_0^\infty \psi_L(u\,g(s))\,ds` in stationarity.

        Needs a driver with an exact characteristic exponent. The integral is
        computed by adaptive quadrature over the range where :math:`g` is above
        :math:`10^{-16}` of its peak (beyond it, :math:`\psi_L(ug) = O(ug)`
        contributes below rounding).
        """
        if self.driver.exactness != ExactnessLevel.EXACT:
            raise ExactnessError(
                method="stationary_char_func",
                required="EXACT driver",
                actual=self.driver.exactness.name,
                reason="the stationary law needs the driver's characteristic exponent",
            )
        u_arr = np.atleast_1d(np.asarray(u, dtype=float))
        decay = float(-np.max(self.ar_roots.real))
        upper = 40.0 / decay

        def integrand(s: float) -> np.ndarray:
            psi = self.driver._char_exponent(u_arr * float(self.kernel(s)))
            return np.concatenate([psi.real, psi.imag])

        val, _ = integrate.quad_vec(integrand, 0.0, upper, epsabs=1e-12, epsrel=1e-10, limit=400)
        k = u_arr.size
        out = np.exp(val[:k] + 1j * val[k:])
        return complex(out[0]) if np.ndim(u) == 0 else out.reshape(np.shape(u))

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        kap = self.driver.cumulants(order)
        return {
            n: float(kap.get(n, np.nan) * self.kernel_power_integral(n))
            for n in range(1, order + 1)
        }

    # -- Process interface ---------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "a CARMA process has serially dependent increments and no Lévy–Khintchine triplet; "
                "use stationary_cumulant, autocovariance, spectral_density or simulate"
            ),
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=bool(np.isfinite(self._k2)),
            has_finite_mean=bool(np.isfinite(self._k1)),
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                f"CARMA({self.p},{self.q}) driven by {self.driver!r}.",
                f"AR zeros {np.round(self.ar_roots, 4).tolist()} (all with negative real part).",
                "Stationary autocovariance is a sum of damped exponentials over the AR zeros.",
            ],
        )

    def _is_gaussian_driver(self) -> bool:
        return isinstance(self.driver, BrownianMotion)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""Paths of :math:`Y` on an even grid, shape ``(n_paths, n_steps + 1)``.

        With a Brownian driver, the state recursion
        :math:`X_{k+1} = e^{A\Delta}X_k + m_\Delta + \varepsilon_k` is exact. Here
        :math:`m_\Delta = \mu A^{-1}(e^{A\Delta}-I)e_p` and
        :math:`\varepsilon_k \sim N(0, \sigma^2(\Sigma - e^{A\Delta}\Sigma e^{A^\top\Delta}))`.
        With ``x0=None``, the start is drawn from the stationary law.

        Any other driver is simulated on ``substeps`` sub-intervals of length
        :math:`\delta` per step. Each driver increment enters through the
        kernel averaged over its sub-interval,
        :math:`\delta^{-1}\int_0^\delta e^{As}e_p\,ds = A^{-1}(e^{A\delta}-I)e_p/\delta`.
        This keeps the mean exact, and the covariance error is :math:`O(\delta^2)`.
        With ``x0=None``, the path starts from a draw with the stationary mean
        and covariance. It is then run for 5 relaxation times,
        :math:`5/\min_i|\mathrm{Re}\,\lambda_i|`, so that the higher cumulants
        settle too.
        """
        rng = rng or np.random.default_rng()
        if n_steps < 1 or n_paths < 1 or not T > 0:
            raise ValueError("n_steps and n_paths must be positive, and T > 0")
        dt = T / n_steps
        A, e, p = self._A, self._e, self.p
        Ainv = np.linalg.inv(A)
        out = np.empty((n_paths, n_steps + 1))

        if self._is_gaussian_driver():
            mu = float(self.driver.mu)  # type: ignore[attr-defined]
            sig = float(self.driver.sigma)  # type: ignore[attr-defined]
            F = linalg.expm(A * dt)
            m = mu * Ainv @ (F - np.eye(p)) @ e
            Q = sig**2 * (self._sigma - F @ self._sigma @ F.T)
            root = _psd_sqrt(Q)
            if self.x0 is None:
                x = (
                    self.stationary_state_mean
                    + rng.standard_normal((n_paths, p))
                    @ _psd_sqrt(self.stationary_state_covariance).T
                )
            else:
                x = np.broadcast_to(self.x0, (n_paths, p)).copy()
            out[:, 0] = x @ self._b
            for k in range(n_steps):
                x = x @ F.T + m + rng.standard_normal((n_paths, p)) @ root.T
                out[:, k + 1] = x @ self._b
            return out

        m_sub = self.substeps
        delta = dt / m_sub
        F = linalg.expm(A * delta)
        w = Ainv @ (F - np.eye(p)) @ e / delta
        # the m sub-steps of one output step: x -> F^m x + sum_j F^(m-1-j) w dL_j
        weights = np.empty((m_sub, p))
        v = w.copy()
        for j in range(m_sub - 1, -1, -1):
            weights[j] = v
            v = F @ v
        F_step = np.linalg.matrix_power(F, m_sub)

        def advance(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
            path = np.empty((x.shape[0], n))
            done = 0
            while done < n:
                chunk = min(n - done, max(1, 2_000_000 // (m_sub * x.shape[0])))
                dl = np.diff(
                    self.driver.simulate(chunk * m_sub, x.shape[0], chunk * dt, rng), axis=1
                )
                dl = dl.reshape(x.shape[0], chunk, m_sub)
                for k in range(chunk):
                    x = x @ F_step.T + dl[:, k, :] @ weights
                    path[:, done + k] = x @ self._b
                done += chunk
            return path, x

        if self.x0 is None:
            x = (
                self.stationary_state_mean
                + rng.standard_normal((n_paths, p)) @ _psd_sqrt(self.stationary_state_covariance).T
            )
            decay = float(-np.max(self.ar_roots.real))
            n_burn = int(min(np.ceil(5.0 / decay / dt), 100_000))
            _, x = advance(x, n_burn)
        else:
            x = np.broadcast_to(self.x0, (n_paths, p)).copy()
        out[:, 0] = x @ self._b
        out[:, 1:] = advance(x, n_steps)[0]
        return out

    def __repr__(self) -> str:
        return f"CARMAProcess(ar={self.ar.tolist()}, ma={self.ma.tolist()}, driver={self.driver!r})"


def _psd_sqrt(S: np.ndarray) -> np.ndarray:
    """A square root R with R R^T = S for a symmetric positive semi-definite S."""
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    return np.asarray(vecs * np.sqrt(np.clip(vals, 0.0, None)))


CARMA = CARMAProcess
