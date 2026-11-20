r"""
Normal Tempered Stable (NTS) Lévy process.

The Normal Tempered Stable (NTS) process is a six-parameter Lévy process constructed
by subordinating a Brownian motion with drift by an independent Tempered Stable
subordinator:
    X_t = θ T_t + σ W(T_t)

where:
  T_t : Tempered Stable subordinator with parameters (α, λ, δ)
  α ∈ (0, 1)  : Stability index of the subordinator (α = 0.5 recovers NIG)
  λ > 0       : Tempering rate parameter
  δ > 0       : Scale / intensity parameter
  θ ∈ ℝ       : Drift / skewness parameter
  σ > 0       : Volatility parameter
  W_t         : Standard Brownian motion independent of T_t

Theoretical Properties
----------------------
1. Characteristic Exponent & Transform:
   The Laplace exponent of the subordinator is:
       ψ_T(u) = δ (Γ(1 - α) / α) [ (λ + u)^α - λ^α ]
   The characteristic function of X_t has the exact closed form:
       E[e^{i u X_t}] = exp( -t ψ_T(-i θ u + 0.5 σ² u²) )
                      = exp( -t δ (Γ(1 - α) / α) [ (λ - i θ u + 0.5 σ² u²)^α - λ^α ] )

2. Exact Cumulants via Subordination:
   The subordinator has cumulants:
       κ_1(T_1) = δ Γ(1 - α) λ^{α - 1}
       κ_2(T_1) = δ Γ(2 - α) λ^{α - 2}
       κ_3(T_1) = δ Γ(3 - α) λ^{α - 3}
       κ_4(T_1) = δ Γ(4 - α) λ^{α - 4}

   By the law of total cumulants:
       c_1(X_1) = θ κ_1(T_1)
       c_2(X_1) = σ² κ_1(T_1) + θ² κ_2(T_1)
       c_3(X_1) = 3 θ σ² κ_2(T_1) + θ³ κ_3(T_1)
       c_4(X_1) = 3 σ⁴ κ_2(T_1) + 6 θ² σ² κ_3(T_1) + θ⁴ κ_4(T_1)

3. Exactness Level:
   EXACT (closed-form characteristic function, exact cumulant generating function,
   and exact Lévy process structure).

References
----------
Barndorff-Nielsen, O.E. & Levendorskii, S.Z. (2001). Feller processes of normal
tempered stable type. *Quantitative Finance*, 1(3), 318–331.

Rachev, S.T., Stoyanov, S.V. & Fabozzi, F.J. (2011). *A Probability Metrics Approach
to Financial Risk Measures*, Wiley, Chapter 8.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.special import gamma

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class NormalTemperedStableProcess(Process):
    r"""
    Normal Tempered Stable (NTS) Lévy process.

    X_t = θ T_t + σ W(T_t),   T_t ~ TemperedStable(α, λ, δ).

    Parameters
    ----------
    alpha : float
        Stability index α ∈ (0, 1).
    lambd : float
        Tempering parameter λ > 0.
    delta : float
        Scale parameter δ > 0.
    theta : float
        Brownian drift / skewness parameter θ ∈ ℝ (default 0.0).
    sigma : float
        Brownian volatility parameter σ > 0 (default 1.0).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        lambd: float = 1.0,
        delta: float = 1.0,
        theta: float = 0.0,
        sigma: float = 1.0,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if lambd <= 0:
            raise ValueError(f"lambd must be strictly positive, got {lambd}")
        if delta <= 0:
            raise ValueError(f"delta must be strictly positive, got {delta}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")

        self.alpha = float(alpha)
        self.lambd = float(lambd)
        self.delta = float(delta)
        self.theta = float(theta)
        self.sigma = float(sigma)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=(
                f"NormalTemperedStableProcess(alpha={self.alpha}, lambd={self.lambd}, "
                f"delta={self.delta}, theta={self.theta}, sigma={self.sigma})"
            ),
            math_note=(
                f"X_t = {self.theta} T_t + {self.sigma} W(T_t); "
                f"T_t ~ TS(α={self.alpha}, λ={self.lambd}, δ={self.delta})"
            ),
            reference="Barndorff-Nielsen & Levendorskii (2001), Quantitative Finance 1(3), 318–331",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    # ------------------------------------------------------------------
    # Subordinator Cumulants
    # ------------------------------------------------------------------

    def subordinator_cumulant(self, order: int) -> float:
        r"""Exact n-th cumulant of the unit-time subordinator T_1."""
        if order < 1:
            raise ValueError("Cumulant order must be >= 1")
        # κ_n(T_1) = δ * Γ(n - α) * λ^{α - n}
        val = self.delta * gamma(order - self.alpha) * (self.lambd ** (self.alpha - order))
        return float(val)

    # ------------------------------------------------------------------
    # Process Cumulants
    # ------------------------------------------------------------------

    def cumulant(self, order: int) -> float:
        r"""Exact n-th cumulant of X_1."""
        if order == 1:
            return float(self.theta * self.subordinator_cumulant(1))
        elif order == 2:
            k1 = self.subordinator_cumulant(1)
            k2 = self.subordinator_cumulant(2)
            return float(self.sigma**2 * k1 + self.theta**2 * k2)
        elif order == 3:
            k2 = self.subordinator_cumulant(2)
            k3 = self.subordinator_cumulant(3)
            return float(3.0 * self.theta * self.sigma**2 * k2 + self.theta**3 * k3)
        elif order == 4:
            k2 = self.subordinator_cumulant(2)
            k3 = self.subordinator_cumulant(3)
            k4 = self.subordinator_cumulant(4)
            return float(
                3.0 * (self.sigma**4) * k2
                + 6.0 * (self.theta**2) * (self.sigma**2) * k3
                + (self.theta**4) * k4
            )
        else:
            raise NotImplementedError("Cumulants for order > 4 not available in closed form")

    def cumulants(self, order: int) -> dict[int, float]:
        r"""Return exact cumulants up to order."""
        if order > 4:
            raise NotImplementedError("Cumulants for order > 4 not available in closed form")
        return {n: self.cumulant(n) for n in range(1, order + 1)}

    @property
    def mean(self) -> float:
        r"""Mean E[X_1]."""
        return self.cumulant(1)

    @property
    def variance(self) -> float:
        r"""Variance Var(X_1)."""
        return self.cumulant(2)

    @property
    def skewness(self) -> float:
        r"""Skewness γ_1 = c_3 / c_2^{3/2}."""
        c2 = self.cumulant(2)
        c3 = self.cumulant(3)
        return float(c3 / (c2**1.5))

    @property
    def excess_kurtosis(self) -> float:
        r"""Excess kurtosis γ_2 = c_4 / c_2^2."""
        c2 = self.cumulant(2)
        c4 = self.cumulant(4)
        return float(c4 / (c2**2))

    # ------------------------------------------------------------------
    # Characteristic Function
    # ------------------------------------------------------------------

    def char_func(
        self,
        u: Union[float, np.ndarray],
        t: float = 1.0,
    ) -> Union[complex, np.ndarray]:
        r"""
        Characteristic function E[exp(i u X_t)].

        E[exp(i u X_t)] = exp( -t δ (Γ(1 - α) / α) [ (λ - i θ u + 0.5 σ² u²)^α - λ^α ] ).
        """
        if t <= 0:
            raise ValueError("t must be strictly positive")

        u_arr = np.asarray(u, dtype=np.float64)
        c = self.delta * gamma(1.0 - self.alpha) / self.alpha

        # Argument inside exponent: z = lambda - 1j * theta * u + 0.5 * sigma^2 * u^2
        z = self.lambd - 1j * self.theta * u_arr + 0.5 * (self.sigma**2) * (u_arr**2)

        # Complex power z^alpha
        psi = c * (z**self.alpha - (self.lambd**self.alpha))
        cf = np.exp(-t * psi)

        if np.isscalar(u):
            return complex(cf)
        return cf

    # ------------------------------------------------------------------
    # Process Overrides
    # ------------------------------------------------------------------

    def _levy_density(self, x: np.ndarray) -> np.ndarray:
        r"""
        Lévy density of :math:`X`, mixing Gaussian jumps over the clock's measure.

        .. math::
            k(x) = \int_0^\infty \frac{e^{-(x-\theta s)^2/(2\sigma^2 s)}}{\sigma\sqrt{2\pi s}}
                   \,\delta s^{-1-\alpha}e^{-\lambda s}\,ds
                 = \frac{2\delta}{\sigma\sqrt{2\pi}}\,e^{\theta x/\sigma^2}
                   \Big(\frac{|x|}{\sigma\sqrt{2b}}\Big)^{-(\alpha+1/2)}
                   K_{\alpha+1/2}\Big(\frac{|x|\sqrt{2b}}{\sigma}\Big),

        with :math:`b = \theta^2/(2\sigma^2) + \lambda`. This uses
        :math:`\int_0^\infty s^{\nu-1}e^{-a/s-bs}ds = 2(a/b)^{\nu/2}K_\nu(2\sqrt{ab})`.
        It is evaluated with ``kve``, so the two exponentials combine into
        :math:`\exp(\theta x/\sigma^2 - |x|\sqrt{2b}/\sigma)`, which decays on
        both sides.
        """
        from scipy.special import kve  # type: ignore[import-untyped]

        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        nz = x != 0.0
        ax = np.abs(x[nz])
        rate = np.sqrt(2.0 * (self.theta**2 / (2.0 * self.sigma**2) + self.lambd)) / self.sigma
        order = self.alpha + 0.5
        out[nz] = (
            2.0
            * self.delta
            / (self.sigma * np.sqrt(2.0 * np.pi))
            * (ax / (self.sigma**2 * rate)) ** (-order)  # sigma*sqrt(2b) = sigma^2 * rate
            * kve(order, ax * rate)
            * np.exp(self.theta * x[nz] / self.sigma**2 - ax * rate)
        )
        return out

    def _triplet(self) -> LevyTriplet:
        r"""
        :math:`(b, 0, \nu)`, with :math:`\nu` from :meth:`_levy_density`.

        The drift is :math:`b = \kappa_1 - \int_{|x|>1}x\,\nu(dx)` for the
        truncation :math:`\mathbb 1_{|x|\le1}`. The measure's moments of
        order 2 to 4 are the exact cumulants. Its jump exponent is the closed
        form :math:`\psi(u) - ibu`.
        """
        measure = DensityLevyMeasure(
            _density_fn=self._levy_density,
            _total_mass=np.inf,
            _moment_fns={n: (lambda n=n: self.cumulant(n)) for n in (2, 3, 4)},
        )
        b = self.cumulant(1) - measure.first_moment_between(1.0, np.inf)
        measure._jump_exponent_fn = lambda u: complex(
            self._char_exponent(np.array([u]))[0] - 1j * b * u
        )
        return LevyTriplet(b=b, sigma_sq=0.0, nu=measure)

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=bool(self.theta == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=2.0 * self.alpha,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "Normal Tempered Stable (NTS) Lévy process.",
                f"Stability index α={self.alpha}, tempering λ={self.lambd}.",
            ],
        )

    def _char_exponent(self, u: np.ndarray) -> np.ndarray:
        r"""
        :math:`\psi(u) = \delta\Gamma(-\alpha)\big[(\lambda - \psi_{BM}(u))^\alpha - \lambda^\alpha\big]`.

        Here :math:`\psi_{BM}(u) = i\theta u - \tfrac12\sigma^2u^2`, since NTS is
        Brownian motion on a tempered-stable clock. Where
        :math:`E[e^{iuX_1}]` converges, :math:`\lambda - \psi_{BM}(u)` has a
        positive real part, so the principal power is analytic there.
        """
        u = np.asarray(u, dtype=complex)
        z = self.lambd - 1j * self.theta * u + 0.5 * self.sigma**2 * u**2
        c = self.delta * gamma(1.0 - self.alpha) / self.alpha  # = -delta * Gamma(-alpha)
        return -c * (z**self.alpha - self.lambd**self.alpha)

    def simulate(
        self,
        n_steps: Optional[int | np.ndarray] = None,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        *,
        t_grid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""
        Simulate NTS paths: :math:`X = \theta\,\Delta T + \sigma\sqrt{\Delta T}\,Z` per step.

        The subordinator increments :math:`\Delta T` are drawn from
        TemperedStable(α, δΔt, λ), the law of the NTS clock, by cf inversion
        (:mod:`spxa.sim.cf_sampling`). Previously they came from a Gamma
        matched only to the first two moments.

        The positional order ``(n_steps, n_paths, T, rng)`` is the one shared by
        every :class:`~spxa.core.process.Process`. Pass a time grid as
        ``t_grid=``. For backward compatibility, an array in first position is
        also read as a grid.
        """
        if isinstance(n_steps, np.ndarray) and t_grid is None:
            t_grid, n_steps = n_steps, None
        if t_grid is None:
            steps = n_steps if n_steps is not None else 100
            t_grid = np.linspace(0.0, T, steps + 1)
        elif isinstance(t_grid, (int, np.integer)):
            steps = int(t_grid)
            t_grid = np.linspace(0.0, T, steps + 1)
        else:
            t_grid = np.asarray(t_grid, dtype=np.float64)
        if t_grid.ndim != 1 or len(t_grid) < 2:
            raise ValueError("t_grid must be a 1-D array with at least 2 points")
        if rng is None:
            rng = np.random.default_rng()

        n_steps = len(t_grid)
        paths = np.zeros((n_paths, n_steps), dtype=np.float64)

        from spxa.zoo.levy.tempered_stable import TemperedStable

        clock = TemperedStable(self.alpha, self.delta, self.lambd)

        for i in range(1, n_steps):
            dt = t_grid[i] - t_grid[i - 1]
            if dt <= 0:
                raise ValueError("t_grid must be strictly increasing")

            delta_T = clock.simulate(1, n_paths, T=float(dt), rng=rng)[:, 1]

            # Normal draw given delta_T
            z = rng.standard_normal(size=n_paths)
            dx = self.theta * delta_T + self.sigma * np.sqrt(delta_T) * z
            paths[:, i] = paths[:, i - 1] + dx

        return paths

    def __repr__(self) -> str:
        return (
            f"NormalTemperedStableProcess(alpha={self.alpha}, lambd={self.lambd}, "
            f"delta={self.delta}, theta={self.theta}, sigma={self.sigma})"
        )


# Aliases
NTSProcess = NormalTemperedStableProcess
NormalTemperedStable = NormalTemperedStableProcess
