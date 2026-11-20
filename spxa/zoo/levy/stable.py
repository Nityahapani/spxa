"""
α-stable Lévy process.

The only class of Lévy processes that is closed under addition and scaling
(stable distributions). Generalises the Gaussian (α=2) and Cauchy (α=1).
Power-law tails: moments of order ≥ α do not exist.

Lévy measure (Samorodnitsky & Taqqu 1994, Property 1.2.15):
  ν(dx) = (c₊ x^{-α-1} 𝟙_{x>0} + c₋ |x|^{-α-1} 𝟙_{x<0}) dx

where c₊, c₋ ≥ 0, not both zero, and α ∈ (0, 2).

The characteristic exponent (Zolotarev parametrisation, type S):
  ψ(u) = -σ^α |u|^α (1 - iβ sign(u) tan(πα/2)) + iμu   α ≠ 1
  ψ(u) = -σ|u|(1 + iβ (2/π) sign(u) log|u|) + iμu        α = 1

This is the S₁ parametrisation (Nolan 2020 calls it S₁): for α > 1 the
location μ is the mean, and the process is a martingale iff μ = 0.

References
----------
Samorodnitsky, G. & Taqqu, M.S. (1994). *Stable Non-Gaussian Random Processes*.
Chapman & Hall.

Nolan, J.P. (2020). *Univariate Stable Distributions*. Springer.

Chambers, J.M., Mallows, C.L. & Stuck, B.W. (1976). A method for simulating
stable random variables. *Journal of the American Statistical Association*,
71(354), 340–344.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import LevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class _StableLevyMeasure(LevyMeasure):
    r"""
    Lévy measure of the α-stable law with the exponent used by :class:`AlphaStable`.

    .. math::
        \nu(dx) = c_+x^{-1-\alpha}\mathbb 1_{x>0}\,dx + c_-|x|^{-1-\alpha}\mathbb 1_{x<0}\,dx,
        \qquad c_\pm = \frac{\sigma^\alpha\Gamma(1+\alpha)\sin(\pi\alpha/2)}{\pi}(1 \pm \beta).

    This normalisation reproduces :math:`-\sigma^\alpha|u|^\alpha`. For
    :math:`\beta = 0`, :math:`2c\int_0^\infty(\cos ux - 1)x^{-1-\alpha}dx =
    2c\,\Gamma(-\alpha)\cos(\pi\alpha/2)|u|^\alpha`, and the reflection
    formula turns :math:`-1/(2\Gamma(-\alpha)\cos(\pi\alpha/2))` into the
    constant above (Sato 1999, Thm 14.10; Samorodnitsky & Taqqu 1994,
    Prop. 1.2.15). At :math:`\alpha = 1` it gives :math:`c_\pm = \sigma(1\pm\beta)/\pi`.

    The jump exponent is exact: :math:`J(u) = \Phi(u) - iu\,\delta`, where
    :math:`\Phi` is the closed-form exponent without the location term,
    and :math:`\delta` is the drift that the truncation
    :math:`\mathbb 1_{|x|\le1}` moves into :math:`b` (see :meth:`drift_shift`).
    """

    def __init__(self, alpha: float, beta: float, sigma: float) -> None:
        from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

        self.alpha, self.beta, self.sigma = alpha, beta, sigma
        factor = sigma**alpha * gamma_fn(alpha + 1) * np.sin(np.pi * alpha / 2) / np.pi
        self.c_plus = float(factor * (1 + beta))
        self.c_minus = float(factor * (1 - beta))

    def density(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        pos, neg = x > 0, x < 0
        out[pos] = self.c_plus * x[pos] ** (-self.alpha - 1)
        out[neg] = self.c_minus * np.abs(x[neg]) ** (-self.alpha - 1)
        return out

    def total_mass(self) -> float:
        return float("inf")

    def tail(self, x: float) -> float:
        if x <= 0:
            raise ValueError("tail(x) needs x > 0")
        return self.c_plus * x ** (-self.alpha) / self.alpha

    def moment(self, n: int) -> float:
        """Diverges for every n ≥ α: +inf for even n, NaN (undefined) for odd n."""
        if n < 1:
            raise ValueError("Moment order must be >= 1")
        if n < self.alpha:  # only n = 1 with α > 1, where the integral converges at ∞
            raise ValueError("the first moment of a stable measure diverges at 0")
        return float("inf") if n % 2 == 0 else float("nan")

    def first_moment_between(self, a: float, b: float) -> float:
        r""":math:`(c_+ - c_-)\int_a^b x^{-\alpha}dx`; infinite :math:`b` needs :math:`\alpha > 1`."""
        if not 0 < a < b:
            raise ValueError(f"need 0 < a < b, got a={a}, b={b}")
        diff = self.c_plus - self.c_minus
        if diff == 0:
            return 0.0 if np.isfinite(b) or self.alpha > 1 else float("nan")
        al = self.alpha
        if np.isinf(b):
            return float(diff * a ** (1 - al) / (al - 1)) if al > 1 else float("nan")
        if abs(al - 1) <= 1e-12:
            return float(diff * np.log(b / a))
        return float(diff * (b ** (1 - al) - a ** (1 - al)) / (1 - al))

    def drift_shift(self) -> float:
        r"""
        :math:`\delta = b - \mu`, the drift that the truncation moves into :math:`b`.

        - :math:`\alpha < 1`: :math:`\Phi = \int(e^{iux}-1)\,\nu`, so
          :math:`\delta = \int_{|x|\le1}x\,\nu = (c_+-c_-)/(1-\alpha)`.
        - :math:`1 < \alpha < 2`: :math:`\Phi = \int(e^{iux}-1-iux)\,\nu`, so
          :math:`\delta = -\int_{|x|>1}x\,\nu = -(c_+-c_-)/(\alpha-1)`.
        - :math:`\alpha = 1`: :math:`\int(e^{iux}-1-iux\mathbb 1_{|x|\le1})\nu
          = -\sigma|u| - i\tfrac2\pi\beta\sigma u\log|u| + iu(c_+-c_-)(1-\gamma_E)`,
          using :math:`\int_0^\infty(\sin x - x\mathbb 1_{x\le1})x^{-2}dx = 1-\gamma_E`.
          So :math:`\delta = -(c_+-c_-)(1-\gamma_E)`.
        """
        diff = self.c_plus - self.c_minus
        if abs(self.alpha - 1) <= 1e-12:
            return float(-diff * (1 - np.euler_gamma))
        if self.alpha < 1:
            return float(diff / (1 - self.alpha))
        return float(-diff / (self.alpha - 1))

    def jump_exponent(self, u: complex) -> complex:
        u = complex(u)
        if u.imag != 0:
            raise ValueError("a stable measure has no exponential moments: u must be real")
        x = u.real
        if x == 0:
            return 0j
        al, be, sg = self.alpha, self.beta, self.sigma
        if abs(al - 1) <= 1e-12:
            phi = -sg * abs(x) * (1 + 1j * be * (2 / np.pi) * np.sign(x) * np.log(abs(x)))
        else:
            phi = -(sg**al) * abs(x) ** al * (1 - 1j * be * np.sign(x) * np.tan(np.pi * al / 2))
        return complex(phi - 1j * u * self.drift_shift())


class AlphaStable(Process):
    """
    α-stable Lévy process (Samorodnitsky–Taqqu S₁ parametrisation).

    The only non-trivial class of Lévy processes stable under addition and
    scaling. For α < 2, moments of order ≥ α do not exist; for α = 2 this
    reduces to Brownian motion.

    Parameters
    ----------
    alpha :
        Stability index α ∈ (0, 2]. α=2 → Gaussian, α=1 → Cauchy.
        Controls tail heaviness: smaller α → heavier tails.
    beta :
        Skewness parameter β ∈ [-1, 1].
        β=0 → symmetric, β=1 → maximally right-skewed, β=-1 → maximally left-skewed.
    sigma :
        Scale parameter σ > 0.
    mu :
        Location (shift) parameter μ ∈ ℝ. Default 0.

    Notes
    -----
    Moments: E[|X_1|^p] < ∞ iff p < α.
    So for α ≤ 1, the mean does not exist; for α ≤ 2, the variance does not
    exist (except at α=2 exactly).

    Characteristic exponent (Nolan 2020, Definition 1.6):
      ψ(u) = -σ^α|u|^α(1 - iβ sign(u) tan(πα/2)) + iμu   (α ≠ 1)
      ψ(u) = -σ|u|(1 + iβ(2/π)sign(u)log|u|) + iμu        (α = 1)

    Simulation via Chambers–Mallows–Stuck (1976) algorithm — exact in
    distribution for unit-time increments.

    References
    ----------
    Samorodnitsky, G. & Taqqu, M.S. (1994). *Stable Non-Gaussian Random
    Processes*. Chapman & Hall. Property 1.2.15.

    Nolan, J.P. (2020). *Univariate Stable Distributions*. Springer.
    Definition 1.6.

    Chambers, J.M., Mallows, C.L. & Stuck, B.W. (1976). A method for
    simulating stable random variables. *Journal of the American Statistical
    Association*, 71(354), 340–344.
    """

    def __init__(
        self,
        alpha: float,
        beta: float = 0.0,
        sigma: float = 1.0,
        mu: float = 0.0,
    ) -> None:
        if not (0 < alpha <= 2):
            raise ValueError(f"alpha must be in (0, 2], got {alpha}")
        if not (-1 <= beta <= 1):
            raise ValueError(f"beta must be in [-1, 1], got {beta}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.mu = mu

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"AlphaStable(alpha={alpha}, beta={beta}, sigma={sigma}, mu={mu})",
            math_note=(
                f"ψ(u) = -σ^α|u|^α(1-iβ·sign(u)·tan(πα/2)) + iμu; "
                f"moments of order < α={alpha} exist"
            ),
            reference="Samorodnitsky & Taqqu (1994); Nolan (2020) Def. 1.6; CMS (1976)",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        if self.alpha == 2.0:
            from spxa.zoo.levy.brownian import _ZeroLevyMeasure

            return LevyTriplet(b=self.mu, sigma_sq=2 * self.sigma**2, nu=_ZeroLevyMeasure())

        nu = _StableLevyMeasure(self.alpha, self.beta, self.sigma)
        return LevyTriplet(b=self.mu + nu.drift_shift(), sigma_sq=0.0, nu=nu)

    def cumulants(self, order: int) -> dict[int, float]:
        """
        κ_1 = μ when α > 1 (NaN for α ≤ 1, where the mean does not exist).

        For α < 2 every higher cumulant diverges. Even orders are reported as
        +inf and odd orders as NaN, matching the measure's moments. α = 2 is
        Gaussian with variance 2σ².
        """
        if self.alpha == 2.0:
            gauss = {1: self.mu, 2: 2 * self.sigma**2}
            return {n: gauss.get(n, 0.0) for n in range(1, order + 1)}
        out: dict[int, float] = {}
        for n in range(1, order + 1):
            if n == 1:
                out[n] = float(self.mu) if self.alpha > 1 else float("nan")
            else:
                out[n] = float("inf") if n % 2 == 0 else float("nan")
        return out

    def _strictly_stable(self) -> bool:
        r"""
        Whether :math:`X_{ct} \sim c^{1/\alpha}X_t`.

        With the exponent :math:`-\sigma^\alpha|u|^\alpha(1 - i\beta\,\mathrm{sgn}(u)
        \tan\frac{\pi\alpha}2) + i\mu u`, the drift scales as :math:`c` and
        the rest as :math:`c^{1/\alpha}`, so :math:`\mu` must vanish for
        :math:`\alpha \ne 1`. For :math:`\alpha = 1`, the drift scales like
        the rest, but the :math:`\beta\,|u|\log|u|` term gives an extra
        :math:`\frac2\pi\beta\sigma\,c\log c`, so :math:`\beta` must vanish
        (Samorodnitsky & Taqqu 1994, Prop. 1.2.1 and Def. 1.1.6).
        """
        if abs(self.alpha - 1.0) <= 1e-12:
            return self.beta == 0.0
        return self.mu == 0.0

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            # a Lévy process with E|X_1| < ∞ is a martingale iff E[X_1] = μ = 0
            is_martingale=(self.alpha > 1.0 and self.mu == 0.0),
            has_finite_variance=(self.alpha == 2.0),
            has_finite_mean=(self.alpha > 1.0),
            self_similarity_index=1.0 / self.alpha if self._strictly_stable() else None,
            tail_index=self.alpha if self.alpha < 2 else None,
            is_subordinator=False,
        )

    def char_func_exact(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Exact characteristic function of α-stable process.

        For α ≠ 1 (Nolan 2020, Definition 1.6):
          φ(u; t) = exp(t·(-σ^α|u|^α(1 - iβ·sign(u)·tan(πα/2)) + iμu))

        For α = 1:
          φ(u; t) = exp(t·(-σ|u|(1 + iβ·(2/π)·sign(u)·log|u|) + iμu))

        Parameters
        ----------
        u :
            Frequency argument(s).
        t :
            Time.
        """
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        abs_u = np.abs(u_arr.real)
        sign_u = np.sign(u_arr.real)

        if abs(self.alpha - 1.0) > 1e-12:
            exponent = (
                -(self.sigma**self.alpha)
                * (abs_u**self.alpha)
                * (1 - 1j * self.beta * sign_u * np.tan(np.pi * self.alpha / 2))
                + 1j * self.mu * u_arr
            )
        else:
            log_abs_u = np.where(abs_u > 0, np.log(abs_u), 0.0)
            exponent = (
                -self.sigma * abs_u * (1 + 1j * self.beta * (2 / np.pi) * sign_u * log_abs_u)
                + 1j * self.mu * u_arr
            )

        result = np.exp(t * exponent)
        return result if result.shape != (1,) else result[0]

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Exact simulation via Chambers–Mallows–Stuck (1976) algorithm.

        Each increment X_{t_{k+1}} - X_{t_k} is an independent α-stable
        random variable scaled to the interval length Δt^{1/α}.

        Algorithm (CMS 1976, equations 2.1–2.3):
          1. U ~ Uniform(-π/2, π/2)
          2. E ~ Exponential(1)
          3. For α ≠ 1:
             X = S · sin(α(U+B)) / cos(U)^{1/α} · (cos(U-α(U+B))/E)^{(1-α)/α}
          4. Scale by σ·Δt^{1/α} and shift by μ·Δt

        Parameters
        ----------
        n_steps, n_paths, T, rng :
            Standard simulation parameters.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1).
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        size = (n_paths, n_steps)

        U = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
        E = rng.exponential(1.0, size=size)

        if abs(self.alpha - 1.0) > 1e-12:
            B = np.arctan(self.beta * np.tan(np.pi * self.alpha / 2)) / self.alpha
            S = (1 + (self.beta * np.tan(np.pi * self.alpha / 2)) ** 2) ** (1 / (2 * self.alpha))
            increments = (
                S
                * np.sin(self.alpha * (U + B))
                / np.cos(U) ** (1 / self.alpha)
                * (np.cos(U - self.alpha * (U + B)) / E) ** ((1 - self.alpha) / self.alpha)
            )
        else:
            increments = (2 / np.pi) * (
                (np.pi / 2 + self.beta * U) * np.tan(U)
                - self.beta * np.log(np.pi / 2 * E * np.cos(U) / (np.pi / 2 + self.beta * U))
            )

        increments = self.sigma * (dt ** (1 / self.alpha)) * increments + self.mu * dt

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return (
            f"AlphaStable(alpha={self.alpha}, beta={self.beta}, sigma={self.sigma}, mu={self.mu})"
        )
