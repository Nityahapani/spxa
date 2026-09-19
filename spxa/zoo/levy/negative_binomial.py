"""
Negative Binomial process.

A pure-jump finite-activity Lévy process with negative binomially distributed
increments. Arises naturally as a Poisson process with a Gamma-distributed
rate (i.e. a Gamma-mixture of Poisson), making it the discrete analogue of
the Variance Gamma process.

Lévy measure: ν = ∑_{k=1}^∞ (p^k / k) · r · δ_k
where the mass at each positive integer k is r · p^k / k.

This is a discrete, finite-activity measure (countably many point masses at
positive integers, total mass r·log(1/(1-p)) < ∞).

Characteristic function (Quenouille 1949):
  φ(u; t) = ((1-p) / (1-p·e^{iu}))^{rt}

Parameters: r > 0 (shape), p ∈ (0,1) (success probability).

Interpretation: X_t counts events in [0,t] where the number of events in
each unit interval follows NegBin(r, p). Equivalently X_t is a Poisson
process with rate λ time-changed by a Gamma process, giving it overdispersion
relative to the Poisson.

References
----------
Quenouille, M.H. (1949). A relation between the logarithmic, Poisson and
negative binomial series. *Biometrics*, 5(2), 162–164.

Johnson, N.L., Kemp, A.W. & Kotz, S. (2005). *Univariate Discrete
Distributions*, 3rd ed. Wiley. Chapter 5.

Kozubowski, T.J. & Podgórski, K. (2009). Distributional properties of the
negative binomial Lévy process. *Probability and Mathematical Statistics*,
29(1), 43–71.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import LevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class _NegBinLevyMeasure(LevyMeasure):
    r"""
    Lévy measure of the Negative Binomial process.

    ν = ∑_{k=1}^∞ (r · p^k / k) · δ_k

    This is a discrete measure on {1, 2, 3, ...} with total mass
    nu(R\{0}) = r · log(1/(1-p)) < ∞ (finite activity).

    Quenouille (1949); Kozubowski & Podgórski (2009), Section 2.
    """

    def __init__(self, r: float, p: float) -> None:
        self.r = r
        self.p = p

    def density(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "NegativeBinomial Lévy measure is discrete (point masses at "
            "positive integers) and has no density w.r.t. Lebesgue measure. "
            "Use total_mass(), tail(), or moment() instead."
        )

    def total_mass(self) -> float:
        # ∑_{k=1}^∞ r·p^k/k = r·log(1/(1-p))  (Taylor series of -log(1-p))
        return self.r * np.log(1.0 / (1.0 - self.p))

    def tail(self, x: float) -> float:
        # ν((x, ∞)) = ∑_{k=ceil(x)+1}^∞ r·p^k/k
        # Computed as: total_mass - ∑_{k=1}^{floor(x)} r·p^k/k
        k_max = int(np.floor(x))
        if k_max < 1:
            return self.total_mass()
        partial = sum(self.r * self.p**k / k for k in range(1, k_max + 1))
        return max(self.total_mass() - partial, 0.0)

    def moment(self, n: int) -> float:
        # ∫ x^n ν(dx) = ∑_{k=1}^∞ k^n · r·p^k/k = r · ∑_{k=1}^∞ k^{n-1} · p^k
        # This is r times the polylogarithm Li_{1-n}(p).
        # For small n, use exact formulas; for large n, truncate the series.
        if n < 1:
            raise ValueError("Moment order must be >= 1")
        # Truncate at k_max where p^k < 1e-15
        k_max = int(np.ceil(-15 * np.log(10) / np.log(self.p))) + 10
        return self.r * sum(k ** (n - 1) * self.p**k for k in range(1, k_max + 1))


class NegativeBinomialProcess(Process):
    """
    Negative Binomial process.

    A finite-activity pure-jump Lévy process where X_t - X_s ~ NegBin(r(t-s), p)
    for all 0 ≤ s < t, meaning increments take non-negative integer values.

    Equivalent representations:
    - Poisson process with Gamma-distributed rate: if N|Λ ~ Poisson(Λ) and
      Λ ~ Gamma(r, p/(1-p)), then N ~ NegBin(r, p)
    - Gamma-subordinated Poisson: X_t = N(G_t) where N is Poisson(1) and
      G_t ~ Gamma(rt, (1-p)/p)

    Parameters
    ----------
    r :
        Shape parameter r > 0. Controls the dispersion.
        As r → ∞ with rp fixed, converges to Poisson.
    p :
        Success probability p ∈ (0, 1).
        Mean of X_1 = rp/(1-p). Larger p → larger mean.

    Notes
    -----
    Cumulants (Johnson et al. 2005, Chapter 5):
      κ_n(X_t) = t · r · p · d^{n-1}/dp^{n-1} [p/(1-p)^n]  (via cumulant GF)

    Explicitly:
      κ₁ = r·p/(1-p)
      κ₂ = r·p/(1-p)²
      κ₃ = r·p·(1+p)/(1-p)³
      κ₄ = r·p·(1 + 4p + p²)/(1-p)⁴

    Characteristic function (Quenouille 1949):
      φ(u; t) = ((1-p)/(1-p·e^{iu}))^{rt}

    All moments are finite. is_subordinator = True (non-negative integer
    increments, non-decreasing paths).

    References
    ----------
    Quenouille, M.H. (1949). *Biometrics*, 5(2), 162–164.

    Johnson, N.L., Kemp, A.W. & Kotz, S. (2005). *Univariate Discrete
    Distributions*, 3rd ed. Wiley. Chapter 5.

    Kozubowski, T.J. & Podgórski, K. (2009). *Probability and Mathematical
    Statistics*, 29(1), 43–71.
    """

    def __init__(self, r: float, p: float) -> None:
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")
        if not (0 < p < 1):
            raise ValueError(f"p must be in (0, 1), got {p}")

        self.r = r
        self.p = p

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"NegativeBinomialProcess(r={r}, p={p})",
            math_note=(
                f"φ(u;t) = ((1-p)/(1-p·e^{{iu}}))^{{rt}}; "
                f"κ₁={r*p/(1-p):.4f}, κ₂={r*p/(1-p)**2:.4f}; "
                f"finite activity: nu(R\\{{0}})=r*log(1/(1-p))={r*np.log(1/(1-p)):.4f}"
            ),
            reference="Quenouille (1949); Johnson et al. (2005) Ch. 5",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        nu = _NegBinLevyMeasure(self.r, self.p)
        # All jumps are positive integers ≥ 1, so b = ∫_{|x|≤1} x ν(dx) = 0
        # (no mass in (0,1) since jumps are integers ≥ 1, and 1 is included
        # in the truncation but the mass at k=1 gives a drift contribution)
        # Using truncation h(x) = 𝟙_{|x|≤1}: only k=1 falls in [-1,1]
        # b = ∫_{|x|≤1} x ν(dx) = 1 · r·p  (mass at k=1 is r·p/1 = r·p)
        b = self.r * self.p
        return LevyTriplet(b=b, sigma_sq=0.0, nu=nu)

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

    def char_func_exact(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Exact characteristic function.

        φ(u; t) = ((1-p) / (1-p·e^{iu}))^{rt}

        Quenouille (1949); Johnson et al. (2005), equation (5.4).

        Parameters
        ----------
        u :
            Frequency argument(s).
        t :
            Time.
        """
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        result = ((1 - self.p) / (1 - self.p * np.exp(1j * u_arr))) ** (self.r * t)
        return result if result.shape != (1,) else result[0]

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def cumulant_exact(self, n: int, t: float = 1.0) -> float:
        """
        Exact cumulants from the cumulant generating function.

        The CGF of NegBin(r, p) is K(s) = r·log((1-p)/(1-p·e^s)).
        Cumulants are obtained by differentiating K(s) at s=0.

        Formulas (Johnson et al. 2005, Table 5.2):
          κ₁ = r·p/(1-p)
          κ₂ = r·p/(1-p)²
          κ₃ = r·p·(1+p)/(1-p)³
          κ₄ = r·p·(1+4p+p²)/(1-p)⁴

        Higher orders computed numerically via the polylogarithm series.

        Parameters
        ----------
        n :
            Cumulant order ≥ 1.
        t :
            Time (cumulants scale linearly with t).
        """
        q = 1.0 - self.p
        if n == 1:
            return t * self.r * self.p / q
        if n == 2:
            return t * self.r * self.p / q**2
        if n == 3:
            return t * self.r * self.p * (1 + self.p) / q**3
        if n == 4:
            return t * self.r * self.p * (1 + 4*self.p + self.p**2) / q**4
        # Higher orders: κ_n = t·r·p · (d/dp)^{n-1}[p/(1-p)^n] evaluated at p
        # Use the Lévy measure moment: κ_n = t · ∫ x^n ν(dx) for n ≥ 2,
        # κ_1 = t·(b + ∫_{|x|>1} x ν(dx)) = t·mean
        nu = _NegBinLevyMeasure(self.r, self.p)
        return t * nu.moment(n)

    def cumulants(self, order: int) -> dict[int, float]:
        return {n: self.cumulant_exact(n) for n in range(1, order + 1)}

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Exact simulation: increments X_{t+Δt} - X_t ~ NegBin(rΔt, p).

        Uses numpy's negative_binomial(n=rΔt, p=1-p) sampler.
        Note: numpy parametrises as number of failures before rΔt successes,
        with success probability 1-p, giving mean rΔt·p/(1-p) as required.

        Parameters
        ----------
        n_steps, n_paths, T, rng :
            Standard simulation parameters.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). Integer-valued, non-decreasing.
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        # numpy's negative_binomial(n, p_np): number of failures before n successes
        # with success prob p_np. Mean = n*(1-p_np)/p_np.
        # We want mean = r*dt*p/(1-p) → set n=r*dt, p_np=1-p.
        increments = rng.negative_binomial(
            n=self.r * dt,
            p=1.0 - self.p,
            size=(n_paths, n_steps),
        ).astype(float)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"NegativeBinomialProcess(r={self.r}, p={self.p})"
