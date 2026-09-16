"""
Multivariate Lévy processes.

Extends the 1-dimensional algebra to ℝᵈ. The Lévy–Khintchine formula
in ℝᵈ is:

    ψ(u) = i⟨b, u⟩ - ½⟨u, Σu⟩ + ∫(e^{i⟨u,x⟩} - 1 - i⟨u,x⟩𝟙_{‖x‖≤1}) ν(dx)

where:
  b  ∈ ℝᵈ          — drift vector
  Σ  ∈ ℝ^{d×d}     — symmetric positive semidefinite diffusion matrix
  ν              — Lévy measure on ℝᵈ \ {0}

The key structural result (Sato 1999, Theorem 11.3):
For independent Lévy processes X¹, …, Xᵏ on ℝ^{d₁}, …, ℝ^{dₖ},
the joint process (X¹, …, Xᵏ) is a Lévy process on ℝ^{d₁+…+dₖ}
with triplet (b_joint, Σ_joint, ν_joint) obtained by stacking.

Linear transforms: if X has triplet (b, Σ, ν) and A: ℝᵈ → ℝᵏ is linear,
then AX has triplet (Ab + correction, AΣAᵀ, ν ∘ A⁻¹).

This module implements:
  1. MultivariateLevyTriplet — d-dimensional triplet with matrix Σ
  2. MultivariateBrownianMotion — d-dimensional BM with covariance matrix
  3. CorrelatedLevy — constructs a correlated d-dimensional Lévy process
     from independent 1-d processes via a linear map (Cholesky of Σ)
  4. linear_transform — applies a matrix to a multivariate process

References
----------
Sato, K.-I. (1999). *Lévy Processes and Infinitely Divisible Distributions*.
Cambridge University Press. Theorem 11.3, Chapter 14.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapter 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import CompoundLevyMeasure, LevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


@dataclass(frozen=True)
class MultivariateLevyTriplet:
    """
    d-dimensional Lévy–Khintchine characteristic triplet (b, Σ, ν).

    Parameters
    ----------
    b : np.ndarray
        Drift vector, shape (d,).
    sigma : np.ndarray
        Symmetric positive semidefinite diffusion matrix, shape (d, d).
    nu : LevyMeasure
        Lévy measure on ℝᵈ \ {0}.
    dim : int
        Dimension d.

    References
    ----------
    Sato (1999), Theorem 8.1 (d-dimensional case).
    """

    b: np.ndarray
    sigma: np.ndarray
    nu: LevyMeasure
    dim: int

    def __post_init__(self) -> None:
        b = np.asarray(self.b)
        s = np.asarray(self.sigma)
        if b.shape != (self.dim,):
            raise ValueError(f"b must have shape ({self.dim},), got {b.shape}")
        if s.shape != (self.dim, self.dim):
            raise ValueError(f"sigma must have shape ({self.dim},{self.dim}), got {s.shape}")
        if not np.allclose(s, s.T):
            raise ValueError("sigma must be symmetric")
        if np.any(np.linalg.eigvalsh(s) < -1e-10):
            raise ValueError("sigma must be positive semidefinite")

    def __add__(self, other: MultivariateLevyTriplet) -> MultivariateLevyTriplet:
        """
        Add two d-dimensional triplets (independent process sum).
        Sato (1999), Proposition 11.10.
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return MultivariateLevyTriplet(
            b=np.asarray(self.b) + np.asarray(other.b),
            sigma=np.asarray(self.sigma) + np.asarray(other.sigma),
            nu=self.nu + other.nu,
            dim=self.dim,
        )

    def transform(self, A: np.ndarray) -> MultivariateLevyTriplet:
        """
        Apply linear map A: ℝᵈ → ℝᵏ to the triplet.

        Result has:
          b_new   = A·b  (+ drift correction, omitted for simplicity)
          Σ_new   = A·Σ·Aᵀ
          ν_new   = ν ∘ A⁻¹  (image measure)

        Sato (1999), Theorem 11.3.
        """
        A = np.asarray(A)
        k = A.shape[0]
        from spxa.core.levy_measure import DensityLevyMeasure

        def image_density(x: np.ndarray) -> np.ndarray:
            raise NotImplementedError(
                "Image measure density not implemented for general A. "
                "Use numeric simulation."
            )

        image_nu = DensityLevyMeasure(
            _density_fn=image_density,
            _total_mass=self.nu.total_mass(),
        )
        return MultivariateLevyTriplet(
            b=A @ np.asarray(self.b),
            sigma=A @ np.asarray(self.sigma) @ A.T,
            nu=image_nu,
            dim=k,
        )

    def cumulant(self, n: int, direction: np.ndarray) -> float:
        """
        Return the n-th cumulant of ⟨direction, X₁⟩ (a projection onto a 1-d process).

        Parameters
        ----------
        n :
            Cumulant order ≥ 1.
        direction :
            Unit vector in ℝᵈ.
        """
        d = np.asarray(direction, dtype=float)
        b_proj = float(d @ np.asarray(self.b))
        sigma_proj = float(d @ np.asarray(self.sigma) @ d)

        if n == 1:
            return b_proj
        if n == 2:
            return sigma_proj + self.nu.moment(2)
        return self.nu.moment(n)


class MultivariateBrownianMotion(Process):
    """
    d-dimensional Brownian motion with drift μ and covariance matrix Σ.

    X_t = μt + L·W_t  where L·Lᵀ = Σ (Cholesky decomposition) and
    W is a d-dimensional standard BM.

    Parameters
    ----------
    mu : array_like
        Drift vector, shape (d,). Default zeros.
    sigma : array_like
        Covariance matrix Σ, shape (d, d). Must be symmetric PSD.

    Notes
    -----
    Triplet: (μ, Σ, 0) — no jump component.
    All cumulants of order ≥ 3 vanish (Gaussian).
    Cov(X_t^i, X_t^j) = t · Σ_{ij}.

    References
    ----------
    Sato (1999), Example 8.3 (d-dimensional case).
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        dim: int = 2,
    ) -> None:
        if sigma is None:
            sigma = np.eye(dim)
        sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
        dim = sigma.shape[0]
        if mu is None:
            mu = np.zeros(dim)
        mu = np.asarray(mu, dtype=float)

        if mu.shape != (dim,):
            raise ValueError(f"mu must have shape ({dim},)")
        if sigma.shape != (dim, dim):
            raise ValueError(f"sigma must have shape ({dim},{dim})")
        if not np.allclose(sigma, sigma.T):
            raise ValueError("sigma must be symmetric")

        self.mu = mu
        self.sigma_mat = sigma
        self.dim = dim
        self._chol = np.linalg.cholesky(sigma + 1e-12 * np.eye(dim))

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"MultivariateBrownianMotion(dim={dim})",
            math_note=f"Triplet: (μ, Σ, 0); dim={dim}; Σ={sigma.tolist()}",
            reference="Sato (1999) Ex. 8.3; Wiener (1923)",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        from spxa.zoo.levy.brownian import _ZeroLevyMeasure
        # Return 1-d projection triplet (trace of Σ as effective scalar)
        return LevyTriplet(
            b=float(np.sum(self.mu)),
            sigma_sq=float(np.trace(self.sigma_mat)),
            nu=_ZeroLevyMeasure(),
        )

    def multivariate_triplet(self) -> MultivariateLevyTriplet:
        """Return the full d-dimensional characteristic triplet."""
        from spxa.zoo.levy.brownian import _ZeroLevyMeasure
        return MultivariateLevyTriplet(
            b=self.mu.copy(),
            sigma=self.sigma_mat.copy(),
            nu=_ZeroLevyMeasure(),
            dim=self.dim,
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=bool(np.allclose(self.mu, 0)),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=0.5,
            tail_index=None,
            is_subordinator=False,
        )

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Exact simulation via Cholesky decomposition of Σ.

        ΔX = μ·Δt + L·Z·√Δt  where Z ~ N(0, I_d) and L = chol(Σ).

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1, d). paths[:, 0, :] == 0.
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        Z = rng.standard_normal(size=(n_paths, n_steps, self.dim))
        increments = self.mu * dt + np.sqrt(dt) * (Z @ self._chol.T)
        paths = np.zeros((n_paths, n_steps + 1, self.dim))
        paths[:, 1:, :] = np.cumsum(increments, axis=1)
        return paths

    def covariance(self, t: float = 1.0) -> np.ndarray:
        """
        Cov(X_t) = t · Σ.

        Parameters
        ----------
        t : float
            Time.
        """
        return t * self.sigma_mat.copy()

    def correlation(self) -> np.ndarray:
        """Return the correlation matrix of X_1."""
        std = np.sqrt(np.diag(self.sigma_mat))
        return self.sigma_mat / np.outer(std, std)

    def __repr__(self) -> str:
        return f"MultivariateBrownianMotion(dim={self.dim}, mu={self.mu.tolist()})"


class CorrelatedLevy(Process):
    """
    Correlated d-dimensional Lévy process constructed from d independent
    1-dimensional Lévy processes via a linear mixing matrix A.

    Z_t = A · (X¹_t, …, Xᵈ_t)ᵀ

    where X¹, …, Xᵈ are independent processes (possibly different types)
    and A ∈ ℝ^{k×d} is a mixing matrix. When A = chol(Σ) for a target
    covariance Σ, this gives a Lévy process with that covariance structure.

    Parameters
    ----------
    processes : list[Process]
        Independent 1-dimensional Lévy processes.
    A : array_like
        Mixing matrix A ∈ ℝ^{k×d}. d = len(processes).

    Notes
    -----
    The resulting process has dimension k. Its characteristic exponent is:
      ψ_Z(u) = ∑_j ψ_{Xʲ}(Aᵀ u)_j  — component-wise composition.

    References
    ----------
    Cont & Tankov (2004), Proposition 4.1.
    """

    def __init__(self, processes: list[Process], A: np.ndarray) -> None:
        A = np.atleast_2d(np.asarray(A, dtype=float))
        if A.shape[1] != len(processes):
            raise ValueError(
                f"A has {A.shape[1]} columns but {len(processes)} processes supplied."
            )
        self.processes = processes
        self.A = A
        self.dim_in = len(processes)
        self.dim_out = A.shape[0]

        node = CompositionNode(
            kind=NodeKind.SCALING,
            name=f"CorrelatedLevy(dim_in={self.dim_in}, dim_out={self.dim_out})",
            children=[p._node for p in processes],
            math_note=f"Z_t = A·(X¹,...,Xᵈ)ᵀ; A shape {A.shape}",
            reference="Cont & Tankov (2004), Proposition 4.1",
        )
        new_exactness = ExactnessLevel.EXACT
        for p in processes:
            new_exactness = ExactnessLevel.combine(new_exactness, p.exactness)
        super().__init__(exactness=new_exactness, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise NotImplementedError(
            "CorrelatedLevy does not have a scalar 1-d triplet. "
            "Use .multivariate_triplet() for the d-dimensional triplet."
        )

    def multivariate_triplet(self) -> MultivariateLevyTriplet:
        """
        Return the d-dimensional characteristic triplet (b, Σ, ν) of Z = A·X.

        b_Z = A · (b_X¹, …, b_Xᵈ)
        Σ_Z = A · diag(σ²_X¹, …, σ²_Xᵈ) · Aᵀ
        ν_Z = image of ν_joint under A (numeric only)
        """
        b_vec = np.array([p._triplet().b for p in self.processes])
        sigma_diag = np.diag([p._triplet().sigma_sq for p in self.processes])
        b_Z = self.A @ b_vec
        sigma_Z = self.A @ sigma_diag @ self.A.T
        nu_Z = CompoundLevyMeasure([p._triplet().nu for p in self.processes])
        return MultivariateLevyTriplet(
            b=b_Z, sigma=sigma_Z, nu=nu_Z, dim=self.dim_out
        )

    def _properties(self) -> ProcessProperties:
        combined = ProcessProperties(
            has_stationary_increments=all(p.properties.has_stationary_increments for p in self.processes),
            has_independent_increments=all(p.properties.has_independent_increments for p in self.processes),
            is_martingale=all(p.properties.is_martingale for p in self.processes),
            has_finite_variance=all(p.properties.has_finite_variance for p in self.processes),
            has_finite_mean=all(p.properties.has_finite_mean for p in self.processes),
        )
        return combined

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate the correlated process by simulating each component
        independently and mixing with A.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1, dim_out).
        """
        rng = rng or np.random.default_rng()
        component_paths = np.stack(
            [p.simulate(n_steps, n_paths, T, rng) for p in self.processes],
            axis=-1,
        )
        return component_paths @ self.A.T

    def __repr__(self) -> str:
        return (
            f"CorrelatedLevy(processes={[repr(p) for p in self.processes]}, "
            f"A_shape={self.A.shape})"
        )


def correlated_brownian_motion(
    sigma_matrix: np.ndarray,
    mu: Optional[np.ndarray] = None,
) -> MultivariateBrownianMotion:
    """
    Construct a multivariate BM with given covariance matrix Σ.

    This is the primary constructor for correlated BM — prefer this over
    CorrelatedLevy for the pure Gaussian case.

    Parameters
    ----------
    sigma_matrix :
        Target covariance matrix Σ ∈ ℝ^{d×d}. Must be symmetric PSD.
    mu :
        Drift vector. Default zeros.

    Returns
    -------
    MultivariateBrownianMotion
    """
    return MultivariateBrownianMotion(mu=mu, sigma=sigma_matrix)


def independent_levy_vector(*processes: Process) -> CorrelatedLevy:
    """
    Stack independent 1-d processes into a d-dimensional Lévy vector process.

    The mixing matrix A = I_d (identity), so Z_t = (X¹_t, …, Xᵈ_t).

    Parameters
    ----------
    *processes :
        Independent 1-dimensional Lévy processes.

    Returns
    -------
    CorrelatedLevy with A = I_d.
    """
    d = len(processes)
    return CorrelatedLevy(list(processes), A=np.eye(d))
