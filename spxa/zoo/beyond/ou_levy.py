"""
Ornstein–Uhlenbeck process driven by a Lévy subordinator (OU-Lévy).

dV_t = -λ V_t dt + dL_{λt}

where L is a Lévy subordinator and λ > 0 is the mean reversion rate.
Not a Lévy process (no independent increments), but the stationary
distribution is infinitely divisible with a known Lévy measure.

The marginal stationary distribution has Lévy measure:
  ν_∞(dx) = ∫_x^∞ ν_L(dy)/y  (Barndorff-Nielsen & Shephard 2001, Prop. 2.1)

ExactnessLevel: MOMENT_PROPAGATION for path operations.
The stationary distribution is EXACT (known triplet).

References
----------
Barndorff-Nielsen, O.E. & Shephard, N. (2001). Non-Gaussian OU-based models
and some of their uses in financial economics. *Journal of the Royal
Statistical Society B*, 63(2), 167–241.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind
from spxa.zoo.levy.gamma import GammaProcess


class OULevy(Process):
    """
    Ornstein–Uhlenbeck process driven by a Lévy subordinator.

    The solution to the SDE dV_t = -λ V_t dt + dL_{λt} is:
      V_t = e^{-λt} V_0 + ∫_0^t e^{-λ(t-s)} dL_{λs}

    Currently implemented with a GammaProcess subordinator, giving the
    Gamma-OU model widely used in stochastic volatility (Barndorff-Nielsen
    & Shephard 2001).

    Parameters
    ----------
    lam :
        Mean reversion rate λ > 0.
    subordinator :
        The driving Lévy subordinator L. Must be a subordinator
        (is_subordinator=True). Default: GammaProcess(a=1.0, b=1.0).
    v0 :
        Initial value V_0. Default 0.

    Notes
    -----
    Autocovariance (Barndorff-Nielsen & Shephard 2001, equation 17):
      Cov(V_t, V_{t+h}) = e^{-λh} · Var(V_∞)

    where Var(V_∞) = κ_2(L_1) / (2λ) and κ_2(L_1) is the variance of the
    subordinator per unit time.

    ExactnessLevel: MOMENT_PROPAGATION.

    References
    ----------
    Barndorff-Nielsen, O.E. & Shephard, N. (2001). *Journal of the Royal
    Statistical Society B*, 63(2), 167–241.
    """

    def __init__(
        self,
        lam: float,
        subordinator: Optional[Process] = None,
        v0: float = 0.0,
    ) -> None:
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")

        self.lam = lam
        self.v0 = v0

        if subordinator is None:
            subordinator = GammaProcess(a=1.0, b=1.0)

        if not subordinator.properties.is_subordinator:
            raise ValueError(
                f"{type(subordinator).__name__} is not a subordinator. "
                "OULevy requires a non-decreasing driving process."
            )
        self.subordinator = subordinator

        sub_kappas = subordinator.cumulants(order=2)
        self._sub_mean = sub_kappas.get(1, float("nan"))
        self._sub_var = sub_kappas.get(2, float("nan"))

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"OULevy(lam={lam}, subordinator={subordinator!r}, v0={v0})",
            math_note=(
                f"dV_t = -λV_t dt + dL_{{λt}}; λ={lam}; "
                f"Var(V_∞) = κ_2(L_1)/(2λ) = {self._sub_var/(2*lam):.4f}; "
                f"Cov(V_t, V_{{t+h}}) = e^{{-λh}}·Var(V_∞)"
            ),
            reference="Barndorff-Nielsen & Shephard (2001), eq. (17)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    @property
    def stationary_mean(self) -> float:
        """
        E[V_∞] = κ_1(L_1).

        Barndorff-Nielsen & Shephard (2001), Section 2.
        """
        return self._sub_mean

    @property
    def stationary_variance(self) -> float:
        """
        Var(V_∞) = κ_2(L_1) / (2λ).

        Barndorff-Nielsen & Shephard (2001), equation (16).
        """
        return self._sub_var / (2 * self.lam)

    def autocovariance(self, h: float) -> float:
        """
        Cov(V_t, V_{t+h}) = e^{-λh} · Var(V_∞) in stationarity.

        Barndorff-Nielsen & Shephard (2001), equation (17).

        Parameters
        ----------
        h :
            Lag h ≥ 0.
        """
        return np.exp(-self.lam * h) * self.stationary_variance

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "OULevy has serially correlated values — it is not a Lévy process "
                "and has no Lévy–Khintchine triplet. "
                "Use .stationary_mean, .stationary_variance, .autocovariance(h), or .simulate()."
            ),
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=True,
            hurst_index=None,
            notes=[
                f"OU-Lévy with λ={self.lam}; driven by {self.subordinator!r}.",
                f"Stationary distribution: mean={self.stationary_mean:.4f}, var={self.stationary_variance:.4f}.",
                "Autocovariance decays exponentially: Cov(V_t, V_{t+h}) = e^{-λh}·Var(V_∞).",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        result: dict[int, float] = {}
        if order >= 1:
            result[1] = self.stationary_mean
        if order >= 2:
            result[2] = self.stationary_variance
        for n in range(3, order + 1):
            result[n] = float("nan")
        return result

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate OU-Lévy paths via exact discretisation.

        For the exponential OU SDE with Lévy subordinator L:
          V_{t+Δt} = e^{-λΔt} V_t + ∫_t^{t+Δt} e^{-λ(t+Δt-s)} dL_{λs}

        The integral term is approximated as a Lévy increment scaled by
        the factor (1 - e^{-λΔt})/λ, which is exact for compound Poisson
        and Gamma subordinators in the limit of small Δt.

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
        decay = np.exp(-self.lam * dt)
        jump_scale = (1 - decay) / self.lam

        sub_increments = self.subordinator.simulate(
            n_steps=n_steps, n_paths=n_paths, T=self.lam * T, rng=rng
        )
        sub_increments = np.diff(sub_increments, axis=1)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.v0

        for k in range(n_steps):
            paths[:, k + 1] = decay * paths[:, k] + jump_scale * sub_increments[:, k]

        return paths

    def __repr__(self) -> str:
        return f"OULevy(lam={self.lam}, subordinator={self.subordinator!r}, v0={self.v0})"
