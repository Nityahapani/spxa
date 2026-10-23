"""
Heston stochastic volatility model.

The Heston (1993) model couples a continuous log-price diffusion with a
Cox–Ingersoll–Ross (CIR) square-root instantaneous variance process:

    dS_t = μ S_t dt + √V_t S_t dW_t^S
    dV_t = κ(θ - V_t) dt + σ_v √V_t dW_t^V

with correlation:
    E[dW_t^S dW_t^V] = ρ dt

Under log-price coordinates X_t = ln(S_t / S_0) with initial value X_0 = 0:
    dX_t = (μ - q - ½ V_t) dt + √V_t dW_t^S
    dV_t = κ(θ - V_t) dt + σ_v √V_t dW_t^V

The model is affine and belongs to the beyond-Lévy family (MOMENT_PROPAGATION).
The characteristic function of X_t is known in closed form via the continuous,
numerically stable formulation of Albrecher et al. (2007) that avoids the
complex logarithm branch-cut discontinuity ("The Little Heston Trap").

References
----------
Heston, S. L. (1993). A closed-form solution for options with stochastic
volatility with applications to bond and currency options.
*The Review of Financial Studies*, 6(2), 327–343.

Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).
The little Heston trap. *Wilmott Magazine*, 83–92.

Lord, R., Koekkoek, R., & van Dijk, D. (2010). A comparison of biased simulation
schemes for stochastic volatility models. *Quantitative Finance*, 10(2), 177–194.

Andersen, L. B., & Piterbarg, V. V. (2007). Moment explosions in stochastic
volatility models. *Finance and Stochastics*, 11(1), 29–53.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.stats import norm

from spxa.core._validation import validate_order, validate_time
from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.sim.exact import sample_heston
from spxa.story.narrator import CompositionNode, NodeKind


@lru_cache(maxsize=128)
def _heston_cumulants(
    v0: float, kappa: float, theta: float, sigma: float, rho: float, mu: float, t: float, order: int
) -> tuple[float, ...]:
    def rhs(_time: float, values: np.ndarray) -> np.ndarray:
        derivatives = np.zeros_like(values)
        for j in range(order):
            n = j + 1
            derivatives[j] = -kappa * values[j]
            if n > 1:
                derivatives[j] += rho * sigma * n * values[j - 1]
            derivatives[j] += (
                0.5
                * sigma
                * sigma
                * sum(math.comb(n, k) * values[k - 1] * values[n - k - 1] for k in range(1, n))
            )
            if n == 1:
                derivatives[j] -= 0.5
            elif n == 2:
                derivatives[j] += 1
            derivatives[order + j] = kappa * theta * values[j]
        return derivatives

    solution = solve_ivp(rhs, (0, t), np.zeros(2 * order), method="DOP853", rtol=2e-12, atol=2e-14)
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ArithmeticError("Heston cumulant ODE failed to converge")
    D, C = solution.y[:order, -1], solution.y[order:, -1]
    result = C + v0 * D
    result[0] += mu * t
    return tuple(float(x) for x in result)


class HestonProcess(Process):
    r"""
    Heston stochastic volatility model.

    Coupled two-dimensional diffusion for log-price :math:`X_t = \ln(S_t / S_0)`
    and instantaneous variance :math:`V_t`:

    .. math::

        dX_t = \left(\mu - \frac{1}{2} V_t\right) dt + \sqrt{V_t} \, dW_t^S, \quad X_0 = 0 \\
        dV_t = \kappa (\theta - V_t) dt + \sigma_v \sqrt{V_t} \, dW_t^V, \quad V_0 = v_0

    with correlation :math:`\mathbb{E}[dW_t^S dW_t^V] = \rho \, dt`.

    Parameters
    ----------
    v0 :
        Initial variance :math:`V_0 = v_0 > 0`. Default 0.04 (:math:`\sqrt{v_0} = 20\%`).
    kappa :
        Rate of mean reversion for variance :math:`\kappa > 0`.
    theta :
        Long-term variance level :math:`\theta > 0`.
    sigma_v :
        Volatility of variance :math:`\sigma_v > 0`. May also be supplied as
        ``sigma`` or ``xi``.
    rho :
        Correlation between asset and variance Brownian motions :math:`\rho \in [-1, 1]`.
    mu :
        Continuous drift of asset log-return :math:`\mu \in \mathbb{R}` (e.g. :math:`r - q`).
        Default 0.0.
    s0 :
        Initial asset spot price :math:`S_0 > 0`. Default 1.0.
    """

    def __init__(
        self,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float | None = None,
        rho: float = -0.7,
        mu: float = 0.0,
        s0: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if sigma_v is None:
            if "sigma" in kwargs:
                sigma_v = kwargs.pop("sigma")
            elif "xi" in kwargs:
                sigma_v = kwargs.pop("xi")
            else:
                raise ValueError("Must provide 'sigma_v' (volatility of variance).")

        if kwargs:
            unexpected = ", ".join(repr(k) for k in kwargs)
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        v0_f = float(v0)
        kappa_f = float(kappa)
        theta_f = float(theta)
        sigma_v_f = float(sigma_v)
        rho_f = float(rho)
        mu_f = float(mu)
        s0_f = float(s0)

        if not np.all(np.isfinite([v0_f, kappa_f, theta_f, sigma_v_f, rho_f, mu_f, s0_f])):
            raise ValueError("Heston parameters must be finite")
        if v0_f <= 0.0:
            raise ValueError(f"Initial variance v0 must be strictly positive, got {v0_f}")
        if kappa_f <= 0.0:
            raise ValueError(f"Mean reversion kappa must be strictly positive, got {kappa_f}")
        if theta_f <= 0.0:
            raise ValueError(f"Long-term variance theta must be strictly positive, got {theta_f}")
        if sigma_v_f <= 0.0:
            raise ValueError(
                f"Volatility of variance sigma_v must be strictly positive, got {sigma_v_f}"
            )
        if not (-1.0 <= rho_f <= 1.0):
            raise ValueError(f"Correlation rho must be in [-1, 1], got {rho_f}")
        if s0_f <= 0.0:
            raise ValueError(f"Spot price s0 must be strictly positive, got {s0_f}")

        self._v0 = v0_f
        self._kappa = kappa_f
        self._theta = theta_f
        self._sigma_v = sigma_v_f
        self._rho = rho_f
        self._mu = mu_f
        self._s0 = s0_f

        feller_ratio = (2.0 * kappa_f * theta_f) / (sigma_v_f**2)
        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=(
                "Heston(v0="
                f"{v0_f}"
                ", kappa="
                f"{kappa_f}"
                ", theta="
                f"{theta_f}"
                ", sigma_v="
                f"{sigma_v_f}"
                ", rho="
                f"{rho_f}"
                ", mu="
                f"{mu_f}"
                ", s0="
                f"{s0_f}"
                ")"
            ),
            math_note=(
                f"Heston stochastic volatility model: dX_t = (μ - ½V_t)dt + √V_t dW_t^S; "
                f"dV_t = κ(θ - V_t)dt + σ_v √V_t dW_t^V; ρ={rho_f}; Feller ratio={feller_ratio:.3f}"
            ),
            reference="Heston (1993); Albrecher et al. (2007)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @property
    def v0(self) -> float:
        """Initial instantaneous variance :math:`V_0`."""
        return self._v0

    @property
    def kappa(self) -> float:
        r"""Speed of mean reversion :math:`\kappa`."""
        return self._kappa

    @property
    def theta(self) -> float:
        r"""Long-term variance level :math:`\theta`."""
        return self._theta

    @property
    def sigma_v(self) -> float:
        r"""Volatility of variance :math:`\sigma_v`."""
        return self._sigma_v

    @property
    def rho(self) -> float:
        r"""Correlation :math:`\rho` between asset and variance Brownian motions."""
        return self._rho

    @property
    def mu(self) -> float:
        r"""Continuous drift rate :math:`\mu` of the asset log-return."""
        return self._mu

    @property
    def s0(self) -> float:
        """Initial asset spot price :math:`S_0`."""
        return self._s0

    # ------------------------------------------------------------------
    # Feller condition
    # ------------------------------------------------------------------

    @property
    def feller_ratio(self) -> float:
        r"""The Feller condition ratio :math:`\frac{2\kappa\theta}{\sigma_v^2}`."""
        return (2.0 * self._kappa * self._theta) / (self._sigma_v**2)

    @property
    def is_feller_satisfied(self) -> bool:
        r"""True if :math:`2\kappa\theta > \sigma_v^2` (variance strictly positive)."""
        return self.feller_ratio >= 1.0

    # ------------------------------------------------------------------
    # Variance Process Moments (CIR variance)
    # ------------------------------------------------------------------

    def variance_mean(self, t: float) -> float:
        r"""Expected variance :math:`\mathbb{E}[V_t \mid V_0 = v_0]`."""
        if not np.isfinite(t) or t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float(
            self._v0 * np.exp(-self._kappa * t) + self._theta * (-np.expm1(-self._kappa * t))
        )

    def variance_var(self, t: float) -> float:
        r"""Variance of variance :math:`\mathrm{Var}(V_t \mid V_0 = v_0)`."""
        if not np.isfinite(t) or t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        exp_kt = np.exp(-self._kappa * t)
        c1 = self._v0 * (self._sigma_v**2 / self._kappa) * (exp_kt * (-np.expm1(-self._kappa * t)))
        c2 = (
            self._theta
            * (self._sigma_v**2 / (2.0 * self._kappa))
            * ((-np.expm1(-self._kappa * t)) ** 2)
        )
        return float(c1 + c2)

    @property
    def stationary_variance_mean(self) -> float:
        r"""Stationary mean of variance :math:`\theta`."""
        return self._theta

    @property
    def stationary_variance_var(self) -> float:
        r"""Stationary variance of variance :math:`\frac{\theta \sigma_v^2}{2\kappa}`."""
        return float((self._theta * self._sigma_v**2) / (2.0 * self._kappa))

    def expected_integrated_variance(self, t: float) -> float:
        r"""
        Expected integrated variance :math:`\mathbb{E}\left[\int_0^t V_s \, ds\right]`.

        .. math::

            \mathbb{E}\left[\int_0^t V_s \, ds\right] = \theta t + (v_0 - \theta) \frac{1 -
            e^{-\kappa t}}{\kappa}
        """
        if not np.isfinite(t) or t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return 0.0
        return float(
            self._theta * t + (self._v0 - self._theta) * (-np.expm1(-self._kappa * t)) / self._kappa
        )

    # ------------------------------------------------------------------
    # Characteristic Function (Albrecher et al. 2007)
    # ------------------------------------------------------------------

    def char_func(self, u: complex | float | np.ndarray, t: float = 1.0) -> complex | np.ndarray:
        """Affine CF, including the finite complex-moment strip.

        A divided-exponential Riccati form avoids the g-ratio singularity at
        u=-i when kappa-rho*sigma_v <= 0 and cancellation as sigma_v tends to zero.
        Heston (1993); Albrecher et al. (2007). This is not a path-sampler claim.
        """
        validate_time(t)
        z = np.asarray(u, dtype=complex)
        if not np.all(np.isfinite(z)):
            raise ValueError("Frequencies must be finite")
        if t == 0:
            result = np.ones_like(z)
            return complex(result) if z.ndim == 0 else result
        # Riccati blow-up time for E[exp(p X_t)], p=-Im(u).
        for moment in np.unique(-z.imag):
            if 0 <= moment <= 1:
                continue
            b = self._rho * self._sigma_v * moment - self._kappa
            discriminant = b * b - self._sigma_v**2 * (moment * moment - moment)
            critical = np.inf
            if discriminant < 0:
                d_imag = math.sqrt(-discriminant)
                critical = 2 / d_imag * (math.pi / 2 - math.atan(b / d_imag))
            elif b > 0:
                d_real = math.sqrt(discriminant)
                critical = 2 / b if d_real == 0 else math.log1p(2 * d_real / (b - d_real)) / d_real
            if t >= critical:
                raise ValueError("Heston exponential moment does not exist at this horizon")
        # At 0 and -i the solution D=C=0 is known without a limiting division.
        safe = np.where((z == 0) | (z == -1j), 0j, z)
        q = safe * safe + 1j * safe
        a = self._kappa - 1j * self._rho * self._sigma_v * safe
        sig2 = self._sigma_v**2
        d = np.sqrt(a * a + sig2 * q)
        difference = np.asarray(a - d)
        np.divide(
            -sig2 * q,
            a + d,
            out=difference,
            where=np.abs(a + d) > 1e-12 * (1 + np.abs(a) + np.abs(d)),
        )
        h = np.full_like(d, t)
        np.divide(-np.expm1(-d * t), d, out=h, where=d != 0)
        denominator = 2 + difference * h
        D = -q * h / denominator
        w = difference * h / 2
        logarithm = np.asarray(np.log1p(w))
        small = np.abs(w) < 0.01
        ws = w[small]
        logarithm[small] = 0.5 * np.log1p(2 * ws.real + np.abs(ws) ** 2) + 1j * np.arctan2(
            ws.imag, 1 + ws.real
        )
        C = self._kappa * self._theta / sig2 * (difference * t - 2 * logarithm)
        result = np.exp(C + D * self._v0 + 1j * z * self._mu * t)
        if not np.all(np.isfinite(result)):
            raise ValueError("Heston CF exceeds numerical range or moment domain")
        return complex(result) if z.ndim == 0 else np.asarray(result)

    def asset_char_func(
        self, u: complex | float | np.ndarray, t: float = 1.0
    ) -> complex | np.ndarray:
        r"""
        Characteristic function of log-price :math:`\ln S_t = \ln S_0 + X_t`.

        .. math::

            \mathbb{E}\left[e^{i u \ln S_t}\right] = e^{i u \ln S_0} \mathbb{E}\left[e^{i u
            X_t}\right]
        """
        u_arr = np.asarray(u)
        phase = np.exp(1j * u_arr * np.log(self._s0))
        cf_x = self.char_func(u, t=t)
        return np.asarray(phase * cf_x)

    # ------------------------------------------------------------------
    # Moments & Cumulants of Log-Return X_t
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float = 1.0) -> float:
        r"""
        Expected log-return :math:`\mathbb{E}[X_t] = \mu t -
        \frac{1}{2}\mathbb{E}\left[\int_0^t V_s \, ds\right]`.
        """
        if not np.isfinite(t) or t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float(self._mu * t - 0.5 * self.expected_integrated_variance(t))

    def conditional_cumulants(self, t: float = 1.0, order: int = 4) -> dict[int, float]:
        """Finite-time log-return cumulants from differentiated affine Riccati ODEs.

        No unstable finite differences or fabricated NaNs above order four.
        DOP853 integration is numerical; MOMENT_PROPAGATION is retained.
        The explicit order budget is twelve.
        """
        validate_order(order)
        validate_time(t)
        if order > 12:
            raise NotImplementedError("Heston cumulant ODE order budget is twelve")
        if t == 0:
            return {n: 0.0 for n in range(1, order + 1)}
        values = _heston_cumulants(
            self._v0, self._kappa, self._theta, self._sigma_v, self._rho, self._mu, t, order
        )
        return {n: values[n - 1] for n in range(1, order + 1)}

    def conditional_variance(self, t: float = 1.0) -> float:
        r"""Variance :math:`\mathrm{Var}(X_t)` of the log-return."""
        return self.conditional_cumulants(t=t, order=2)[2]

    def cumulants(self, order: int) -> dict[int, float]:
        r"""Cumulants of :math:`X_1` (at horizon :math:`t = 1.0`)."""
        return self.conditional_cumulants(t=1.0, order=order)

    # ------------------------------------------------------------------
    # Option Pricing via Gil-Pelaez Fourier Inversion
    # ------------------------------------------------------------------

    def call_price(
        self,
        K: float | np.ndarray,
        T: float = 1.0,
        r: float | None = None,
        q: float = 0.0,
    ) -> float | np.ndarray:
        r"""
        European call option price :math:`C(S_0, K, T)` via Gil-Pelaez Fourier inversion.

        .. math::

            C = S_0 e^{-q T} P_1 - K e^{-r T} P_2

        where :math:`P_1, P_2` are risk-neutral probabilities:

        .. math::

            P_j = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \operatorname{Re}\left(
            \frac{e^{-i u \ln(K / S_0)} \psi_j(u, T)}{i u} \right) du

        with :math:`\psi_2(u) = \phi(u)` and :math:`\psi_1(u) = e^{-\mu T} \phi(u - i)`.

        Parameters
        ----------
        K :
            Strike price or array of strikes (:math:`K > 0`).
        T :
            Time to maturity in years (:math:`T > 0`). Default 1.0.
        r :
            Risk-free rate. If None, uses drift :math:`\mu`. Default None.
        q :
            Continuous dividend yield. Default 0.0.

        Returns
        -------
        float or np.ndarray
        """
        if T <= 0.0:
            raise ValueError(f"Maturity T must be strictly positive, got {T}")
        r_val = float(self._mu) if r is None else float(r)
        q_val = float(q)
        if not np.all(np.isfinite([T, r_val, q_val])):
            raise ValueError("Pricing time and rates must be finite")

        def pricing_cf(u: complex) -> complex:
            shift = r_val - q_val - self._mu
            return complex(np.exp(1j * u * shift * T) * self.char_func(u, t=T))

        def _price_single(strike: float) -> float:
            if not np.isfinite(strike) or strike <= 0.0:
                raise ValueError(f"Strike K must be strictly positive, got {strike}")

            ln_k_s0 = np.log(strike / self._s0)

            def integrand1(u: float) -> float:
                if u < 1e-12:
                    return 0.0
                cf1 = np.exp(-(r_val - q_val) * T) * pricing_cf(u - 1j)
                val = np.exp(-1j * u * ln_k_s0) * cf1 / (1j * u)
                return float(np.real(val))

            def integrand2(u: float) -> float:
                if u < 1e-12:
                    return 0.0
                cf2 = pricing_cf(complex(u))
                val = np.exp(-1j * u * ln_k_s0) * cf2 / (1j * u)
                return float(np.real(val))

            p1_int, _ = quad(integrand1, 0.0, np.inf, limit=400, epsabs=1e-9)
            p2_int, _ = quad(integrand2, 0.0, np.inf, limit=400, epsabs=1e-9)

            P1 = float(np.clip(0.5 + (1.0 / np.pi) * p1_int, 0.0, 1.0))
            P2 = float(np.clip(0.5 + (1.0 / np.pi) * p2_int, 0.0, 1.0))

            price = self._s0 * np.exp(-q_val * T) * P1 - strike * np.exp(-r_val * T) * P2
            intrinsic = max(0.0, self._s0 * np.exp(-q_val * T) - strike * np.exp(-r_val * T))
            return float(max(price, intrinsic))

        K_arr = np.asarray(K)
        if K_arr.ndim == 0:
            return _price_single(float(K_arr))
        return np.array([_price_single(float(k)) for k in K_arr], dtype=float)

    def put_price(
        self,
        K: float | np.ndarray,
        T: float = 1.0,
        r: float | None = None,
        q: float = 0.0,
    ) -> float | np.ndarray:
        r"""
        European put option price :math:`P(S_0, K, T)` via Put-Call parity.

        .. math::

            P = C - S_0 e^{-q T} + K e^{-r T}
        """
        r_val = float(self._mu) if r is None else float(r)
        calls = self.call_price(K, T=T, r=r_val, q=q)
        K_arr = np.asarray(K, dtype=float)
        puts = calls - self._s0 * np.exp(-q * T) + K_arr * np.exp(-r_val * T)
        intrinsic = np.maximum(0.0, K_arr * np.exp(-r_val * T) - self._s0 * np.exp(-q * T))
        return np.asarray(np.maximum(puts, intrinsic))

    def implied_volatility(
        self,
        K: float | np.ndarray,
        T: float = 1.0,
        r: float | None = None,
        q: float = 0.0,
    ) -> float | np.ndarray:
        r"""
        Black-Scholes implied volatility :math:`\sigma_{\mathrm{BS}}(K, T)` inverted from
        Heston call price.
        """
        r_val = float(self._mu) if r is None else float(r)
        q_val = float(q)

        def _bs_call(s: float, k: float, t: float, rate: float, div: float, vol: float) -> float:
            if vol <= 1e-8:
                return float(max(0.0, s * np.exp(-div * t) - k * np.exp(-rate * t)))
            d1 = (np.log(s / k) + (rate - div + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
            d2 = d1 - vol * np.sqrt(t)
            return float(s * np.exp(-div * t) * norm.cdf(d1) - k * np.exp(-rate * t) * norm.cdf(d2))

        def _iv_single(strike: float) -> float:
            call_mkt = float(self.call_price(strike, T=T, r=r_val, q=q_val))
            intrinsic = float(max(0.0, self._s0 * np.exp(-q_val * T) - strike * np.exp(-r_val * T)))
            upper_bound = float(self._s0 * np.exp(-q_val * T))
            if call_mkt <= intrinsic + 1e-10:
                return 0.0
            if call_mkt >= upper_bound - 1e-10:
                return 5.0

            def obj(v: float) -> float:
                return _bs_call(self._s0, strike, T, r_val, q_val, v) - call_mkt

            try:
                return float(brentq(obj, 1e-5, 5.0, xtol=1e-6))
            except Exception:
                return float(np.nan)

        K_arr = np.asarray(K)
        if K_arr.ndim == 0:
            return _iv_single(float(K_arr))
        return np.array([_iv_single(float(k)) for k in K_arr], dtype=float)

    def implied_vol_smile(
        self,
        strikes: list[float] | np.ndarray,
        T: float = 1.0,
        r: float | None = None,
        q: float = 0.0,
    ) -> np.ndarray:
        r"""Compute the Black-Scholes implied volatility smile across an array of strikes."""
        return np.asarray(self.implied_volatility(np.asarray(strikes, dtype=float), T=T, r=r, q=q))

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
        method: str = "full_truncation",
        return_price: bool = False,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of log-return :math:`X_t = \ln(S_t / S_0)`.

        Uses the Full Truncation scheme (Lord et al. 2010) to preserve
        variance non-negativity and minimize discretization bias.

        Parameters
        ----------
        n_steps :
            Number of time steps.
        n_paths :
            Number of independent trajectories. Default 1.
        T :
            Terminal horizon. Default 1.0.
        rng :
            Random number generator instance.
        method :
            Simulation algorithm: ``"full_truncation"`` or ``"euler"``.
        return_price :
            If True, returns asset prices :math:`S_t = S_0 e^{X_t}`; if False,
            returns log-returns :math:`X_t` starting at 0.0. Default False.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_paths, n_steps + 1)``.
        """
        paths_x = sample_heston(
            v0=self._v0,
            kappa=self._kappa,
            theta=self._theta,
            sigma_v=self._sigma_v,
            rho=self._rho,
            mu=self._mu,
            s0=self._s0,
            n_steps=n_steps,
            n_paths=n_paths,
            T=T,
            rng=rng,
            method=method,
            return_variance=False,
        )
        assert isinstance(paths_x, np.ndarray)
        if return_price:
            return np.asarray(self._s0 * np.exp(paths_x))
        return paths_x

    def simulate_joint(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
        method: str = "full_truncation",
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Simulate joint sample paths of asset price :math:`S_t` and variance :math:`V_t`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(S_paths, V_paths)`` each of shape ``(n_paths, n_steps + 1)``.
        """
        joint = sample_heston(
            v0=self._v0,
            kappa=self._kappa,
            theta=self._theta,
            sigma_v=self._sigma_v,
            rho=self._rho,
            mu=self._mu,
            s0=self._s0,
            n_steps=n_steps,
            n_paths=n_paths,
            T=T,
            rng=rng,
            method=method,
            return_variance=True,
        )
        assert isinstance(joint, tuple)
        paths_x, paths_v = joint
        paths_s = self._s0 * np.exp(paths_x)
        return paths_s, paths_v

    # ------------------------------------------------------------------
    # Abstract Base Class implementations
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual=self._exactness.name,
            reason=(
                "The Heston stochastic volatility process has serially correlated increments "
                "driven by a persistent CIR variance state; it is not a Lévy process and has "
                "no Lévy–Khintchine triplet."
            ),
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_mean=True,
            has_finite_variance=True,
            notes=[
                (
                    f"Heston stochastic volatility model: dX_t = (μ - ½V_t)dt + √V_t dW_t^S; "
                    f"dV_t = κ(θ - V_t)dt + σ_v √V_t dW_t^V; ρ={self._rho}."
                ),
                (
                    f"Parameters: v0={self._v0}, κ={self._kappa}, θ={self._theta}, "
                    f"σ_v={self._sigma_v}, ρ={self._rho}, μ={self._mu}, S0={self._s0}."
                ),
                (
                    f"Feller ratio 2κθ/σ_v² = {self.feller_ratio:.4f} "
                    f"({'satisfied' if self.is_feller_satisfied else 'violated'})."
                ),
                "Affine characteristic function is exact via Albrecher et al. (2007) formulation.",
                (
                    "HestonProcess has state-dependent stochastic volatility and "
                    "serially correlated increments."
                ),
            ],
        )

    def __repr__(self) -> str:
        return (
            f"HestonProcess(v0={self._v0}, kappa={self._kappa}, theta={self._theta}, "
            f"sigma_v={self._sigma_v}, rho={self._rho}, mu={self._mu}, s0={self._s0})"
        )

    def __str__(self) -> str:
        return (
            "Heston(v0="
            f"{self._v0}"
            ", kappa="
            f"{self._kappa}"
            ", theta="
            f"{self._theta}"
            ", sigma_v="
            f"{self._sigma_v}"
            ", rho="
            f"{self._rho}"
            ")"
        )

    def cumulants_at(self, order: int, t: float = 1.0) -> dict[int, float]:
        return self.conditional_cumulants(t=t, order=order)


# Canonical alias
Heston = HestonProcess
