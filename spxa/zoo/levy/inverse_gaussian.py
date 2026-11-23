"""
Inverse Gaussian (Wald) process.

A subordinator with independent IG-distributed increments. Fundamental
in NIG construction (NIG = BM subordinated by IG process) and in
reliability and first-passage-time modelling.

Lévy density (Cont & Tankov 2004, Table 4.1):
  k(x) = (λ/(2π))^{1/2} · x^{-3/2} · exp(-λ(x-μ)²/(2μ²x))  for x > 0

This is the density of the IG(μ, λ) distribution scaled to unit time,
generalised to arbitrary t via the self-similarity of the IG process:
  X_t ~ IG(μt, λt²)

Bernstein function: φ(s) = λ/μ · (1 - sqrt(1 - 2μ²s/λ))

References
----------
Tweedie, M.C.K. (1957). Statistical properties of inverse Gaussian
distributions I. *Annals of Mathematical Statistics*, 28(2), 362–377.

Barndorff-Nielsen, O.E. & Shephard, N. (2001). Non-Gaussian OU-based
models. *Journal of the Royal Statistical Society B*, 63(2), 167–241.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapter 4, Table 4.1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def _ig_levy_measure(mu: float, lam: float) -> DensityLevyMeasure:
    r"""
    Lévy measure of the IG process with :math:`X_1 \sim IG(\mu, \lambda)`.

    .. math:: k(x) = \sqrt{\lambda / (2\pi)}\; x^{-3/2} e^{-\eta x}, \qquad
              \eta = \lambda / (2\mu^2), \quad x > 0.

    This is the one-sided tempered stable measure with index 1/2. It is
    determined by the Laplace exponent
    :math:`(\lambda/\mu)(1 - \sqrt{1 + 2\mu^2 s/\lambda}) = \int (e^{-sx} - 1) k(x)\,dx`
    (Cont & Tankov 2004, Table 4.4). Moments of every order are
    :math:`\int x^n k = c\,\Gamma(n - 1/2)\,\eta^{1/2 - n} = (2n-3)!!\,\mu^{2n-1}/\lambda^{n-1}`.
    """
    from scipy.special import erfc, gammaln  # type: ignore[import-untyped]

    c = np.sqrt(lam / (2 * np.pi))
    eta = lam / (2 * mu**2)

    def density(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        out = np.zeros_like(x_arr)
        pos = x_arr > 0
        xp = x_arr[pos]
        out[pos] = c * xp ** (-1.5) * np.exp(-eta * xp)
        return out

    def tail(xv: float) -> float:
        # c ∫_x^∞ t^{-3/2} e^{-ηt} dt = c η^{1/2} Γ(-1/2, ηx)
        #                             = c [2 e^{-ηx}/√x − 2√(πη) erfc(√(ηx))]
        if xv <= 0:
            return float(np.inf)
        return float(
            c
            * (
                2 * np.exp(-eta * xv) / np.sqrt(xv)
                - 2 * np.sqrt(np.pi * eta) * erfc(np.sqrt(eta * xv))
            )
        )

    def make_moment(n: int):  # type: ignore[no-untyped-def]
        return lambda: float(c * np.exp(gammaln(n - 0.5)) * eta ** (0.5 - n))

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=np.inf,
        _tail_fn=tail,
        _moment_fns={n: make_moment(n) for n in range(1, 21)},
    )


class InverseGaussianProcess(Process):
    """
    Inverse Gaussian (Wald) process — a subordinator.

    X_t has independent increments distributed as IG(μt, λt²):
      X_t - X_s ~ IG(μ(t-s), λ(t-s)²)

    The IG(μ, λ) distribution has density:
      f(x; μ, λ) = sqrt(λ/(2πx³)) exp(-λ(x-μ)²/(2μ²x))  for x > 0

    Parameters
    ----------
    mu :
        Mean of X_1. μ > 0.
    lam :
        Shape parameter. λ > 0. Controls the variance: Var(X_1) = μ³/λ.

    Notes
    -----
    Cumulants (Tweedie 1957; Johnson, Kotz & Balakrishnan 1994):
      κ_1(X_t) = t · μ
      κ_n(X_t) = t · (2n-3)!! · μ^{2n-1} / λ^{n-1}  for n ≥ 2
    where (2n-3)!! = 1·3·5·...·(2n-3).
    Explicitly:
      κ₁ = μ
      κ₂ = μ³/λ
      κ₃ = 3μ⁵/λ²
      κ₄ = 15μ⁷/λ³

    Bernstein function (Schilling et al. 2012):
      φ(s) = (λ/μ)(1 - sqrt(1 - 2μ²s/λ))

    is_subordinator = True (non-decreasing paths).

    References
    ----------
    Tweedie, M.C.K. (1957). *Annals of Mathematical Statistics*, 28(2),
    362–377.

    Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
    Table 4.1.

    Schilling, R., Song, R. & Vondraček, Z. (2012). *Bernstein Functions*,
    2nd ed. De Gruyter. Example 3.12.
    """

    def __init__(self, mu: float, lam: float) -> None:
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        self.mu = mu
        self.lam = lam

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"InverseGaussianProcess(mu={mu}, lam={lam})",
            math_note=(
                f"Lévy density: sqrt(λ/2π)·x^{{-3/2}}·exp(-λ(x-μ)²/(2μ²x)); "
                f"φ(s) = (λ/μ)(1-sqrt(1-2μ²s/λ)); "
                f"κ₁={mu}, κ₂={mu**3 / lam:.4f}"
            ),
            reference="Tweedie (1957); Cont & Tankov (2004) Table 4.1; Schilling et al. (2012) Ex. 3.12",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        from scipy.special import erf  # type: ignore[import-untyped]

        nu = _ig_levy_measure(self.mu, self.lam)
        # b = ∫_0^1 x k(x) dx = c η^{-1/2} γ(1/2, η) = μ erf(√η), with η = λ/(2μ²);
        # the large-jump mean is then μ erfc(√η), and κ_1 = μ.
        b_drift = self.mu * float(erf(np.sqrt(self.lam / (2 * self.mu**2))))
        return LevyTriplet(b=b_drift, sigma_sq=0.0, nu=nu)

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=True,
        )

    def bernstein_function(self, lam: float | np.ndarray) -> np.ndarray:
        """
        Exact Bernstein function: φ(s) = (λ/μ)(1 - sqrt(1 - 2μ²s/λ)).

        Schilling, Song & Vondraček (2012), Example 3.12.
        """
        s = np.atleast_1d(np.asarray(lam, dtype=float))
        inner = 1.0 - 2.0 * self.mu**2 * s / self.lam
        inner = np.maximum(inner, 0.0)
        result = (self.lam / self.mu) * (1.0 - np.sqrt(inner))
        return result if result.shape != (1,) else result[0]

    def cumulant_exact(self, n: int, t: float = 1.0) -> float:
        """
        Exact closed-form cumulant.

        κ_1(X_t) = t · μ
        κ_n(X_t) = t · (2n-3)!! · μ^{2n-1} / λ^{n-1}  for n >= 2
        where (2n-3)!! = 1·3·5·...·(2n-3).

        Explicitly:
          κ_1 = μ
          κ_2 = μ³/λ
          κ_3 = 3μ⁵/λ²
          κ_4 = 15μ⁷/λ³

        References: Tweedie (1957); Johnson, Kotz & Balakrishnan (1994).
        """
        if n < 1:
            raise ValueError(f"order n must be >= 1, got {n}")
        if n == 1:
            return t * self.mu
        from math import prod

        double_factorial = prod(range(1, 2 * n - 2, 2))
        return t * double_factorial * (self.mu ** (2 * n - 1)) / (self.lam ** (n - 1))

    def cumulants(self, order: int) -> dict[int, float]:
        return {n: self.cumulant_exact(n) for n in range(1, order + 1)}

    def char_func_exact(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Exact characteristic function.

        φ(u; t) = exp(t · λ/μ · (1 - sqrt(1 - 2iuμ²/λ - u²μ²/λ)))
                = exp(t · φ_Bernstein(iu ... ))

        More precisely for the IG(μt, λt²) distribution:
          φ(u) = exp(λt/μ · (1 - sqrt(1 - 2iuμ²/λ + u²μ²... )))

        Using the known MGF result (Tweedie 1957):
          φ(u; t) = exp(tλ/μ · (1 - sqrt(1 - 2iuμ²/λ)))

        Reference: Barndorff-Nielsen & Shephard (2001), equation (A.2).
        """
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        inner = 1.0 - 2j * u_arr * self.mu**2 / self.lam
        exponent = t * (self.lam / self.mu) * (1.0 - np.sqrt(inner))
        result = np.exp(exponent)
        return result if result.shape != (1,) else result[0]

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def _char_exponent(self, u: np.ndarray) -> np.ndarray:
        r"""
        :math:`\psi(u) = \frac{\lambda}{\mu}\big(1 - \sqrt{1 - 2i\mu^2u/\lambda}\big)`, the closed form.

        The radicand has a positive real part in the convergence strip
        :math:`\operatorname{Im}u > -\lambda/(2\mu^2)`, where the principal
        square root is analytic.
        """
        u = np.asarray(u, dtype=complex)
        return (self.lam / self.mu) * (1.0 - np.sqrt(1.0 - 2j * self.mu**2 * u / self.lam))

    _vector_sampler = True

    def sample_at(self, times: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Independent draws of X_t for each entry of ``times``: IG(mean mu*t, shape lam*t^2)."""
        rng = rng or np.random.default_rng()
        t = np.asarray(times, dtype=float)
        out = np.zeros(t.shape)
        pos = t > 0
        out[pos] = rng.wald(self.mu * t[pos], self.lam * t[pos] ** 2)
        return out

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Exact simulation: increments X_{t_{k+1}} - X_{t_k} ~ IG(μΔt, λΔt²).

        Uses numpy's wald sampler: wald(mean=μΔt, scale=λΔt²).

        Devroye (1986), Chapter 4.
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        increments = rng.wald(
            mean=self.mu * dt,
            scale=self.lam * dt**2,
            size=(n_paths, n_steps),
        )
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"InverseGaussianProcess(mu={self.mu}, lam={self.lam})"
