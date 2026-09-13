"""
Property lattice for stochastic processes.

Each process carries a ProcessProperties instance that tracks which mathematical
properties hold and how they propagate under operations. Properties are typed,
not strings, so propagation rules are encoded as methods rather than inferred
from metadata.

Property propagation rules under composition are derived from:
  Sato (1999), Chapters 11 and 30.
  Cont & Tankov (2004), Chapter 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProcessProperties:
    """
    Structured set of analytical properties of a stochastic process.

    Attributes
    ----------
    has_stationary_increments :
        X_{t+s} - X_s ∼ X_t in distribution for all s, t ≥ 0.
    has_independent_increments :
        X_t - X_s ⊥ X_s - X_r for all r < s < t.
    is_martingale :
        E[X_t | F_s] = X_s a.s. for s ≤ t.
    has_finite_variance :
        Var(X_t) < ∞ for all t.
    has_finite_mean :
        E[|X_t|] < ∞ for all t.
    self_similarity_index :
        H such that (X_{ct})_t ∼ (c^H X_t)_t. None if not self-similar.
    tail_index :
        α such that ν((x,∞)) ~ x^{-α} as x → ∞. Governs moment existence.
        None if the tail decays faster than any power (e.g. finite variance with
        exponential tails).
    is_subordinator :
        True if X is non-decreasing a.s. (used for the @ operator).
    hurst_index :
        Relevant only for fBM and related processes. None for Lévy processes.
    notes :
        Free-form list of human-readable notes appended during composition,
        used by .__story__() to explain property changes.
    """

    has_stationary_increments: bool = True
    has_independent_increments: bool = True
    is_martingale: bool = False
    has_finite_variance: bool = True
    has_finite_mean: bool = True
    self_similarity_index: Optional[float] = None
    tail_index: Optional[float] = None
    is_subordinator: bool = False
    hurst_index: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def add_note(self, note: str) -> ProcessProperties:
        """Return a copy with an additional note appended."""
        return ProcessProperties(
            has_stationary_increments=self.has_stationary_increments,
            has_independent_increments=self.has_independent_increments,
            is_martingale=self.is_martingale,
            has_finite_variance=self.has_finite_variance,
            has_finite_mean=self.has_finite_mean,
            self_similarity_index=self.self_similarity_index,
            tail_index=self.tail_index,
            is_subordinator=self.is_subordinator,
            hurst_index=self.hurst_index,
            notes=self.notes + [note],
        )

    @staticmethod
    def combine_addition(p: ProcessProperties, q: ProcessProperties) -> ProcessProperties:
        """
        Propagate properties through Z = X + Y for independent X, Y.

        Rules (Sato 1999, Proposition 11.10; Cont & Tankov 2004, Proposition 4.1):
        - Stationary increments: preserved (both have it)
        - Independent increments: preserved
        - Martingale: preserved if both are martingales
        - Finite variance: preserved if both have it
        - Finite mean: preserved if both have it
        - Self-similarity: lost unless both have same index
        - Tail index: min(α_X, α_Y) — heavier tail dominates
        - Subordinator: preserved only if both are subordinators
        """
        notes: list[str] = []

        self_sim: Optional[float] = None
        if (p.self_similarity_index is not None
                and q.self_similarity_index is not None
                and p.self_similarity_index == q.self_similarity_index):
            self_sim = p.self_similarity_index
        elif p.self_similarity_index is not None or q.self_similarity_index is not None:
            notes.append(
                "Self-similarity index lost under addition: operands have different "
                "or asymmetric self-similarity."
            )

        tail: Optional[float] = None
        if p.tail_index is not None and q.tail_index is not None:
            tail = min(p.tail_index, q.tail_index)
            if tail != p.tail_index or tail != q.tail_index:
                notes.append(
                    f"Tail index is min({p.tail_index}, {q.tail_index}) = {tail}: "
                    "the heavier-tailed component dominates."
                )
        elif p.tail_index is not None:
            tail = p.tail_index
        elif q.tail_index is not None:
            tail = q.tail_index

        if not (p.is_martingale and q.is_martingale) and (p.is_martingale or q.is_martingale):
            notes.append("Martingale property lost: only one operand is a martingale.")

        return ProcessProperties(
            has_stationary_increments=(
                p.has_stationary_increments and q.has_stationary_increments
            ),
            has_independent_increments=(
                p.has_independent_increments and q.has_independent_increments
            ),
            is_martingale=p.is_martingale and q.is_martingale,
            has_finite_variance=p.has_finite_variance and q.has_finite_variance,
            has_finite_mean=p.has_finite_mean and q.has_finite_mean,
            self_similarity_index=self_sim,
            tail_index=tail,
            is_subordinator=p.is_subordinator and q.is_subordinator,
            hurst_index=None,
            notes=p.notes + q.notes + notes,
        )

    @staticmethod
    def combine_scaling(p: ProcessProperties, c: float) -> ProcessProperties:
        """
        Propagate properties through Z = cX.

        Rules:
        - All increment properties preserved.
        - Martingale preserved iff c ≠ 0 (scaling does not break martingale).
        - Self-similarity index preserved (scale changes, index does not).
        - Tail index preserved.
        - Subordinator: only if c > 0 (negative scale reverses sign, not monotone).
        """
        notes = list(p.notes)
        if c < 0 and p.is_subordinator:
            notes.append(f"Subordinator property lost: scale factor c={c} < 0 reverses sign.")

        return ProcessProperties(
            has_stationary_increments=p.has_stationary_increments,
            has_independent_increments=p.has_independent_increments,
            is_martingale=p.is_martingale and c != 0,
            has_finite_variance=p.has_finite_variance,
            has_finite_mean=p.has_finite_mean,
            self_similarity_index=p.self_similarity_index,
            tail_index=p.tail_index,
            is_subordinator=p.is_subordinator and c > 0,
            hurst_index=p.hurst_index,
            notes=notes,
        )

    @staticmethod
    def combine_subordination(
        parent: ProcessProperties,
        subordinator: ProcessProperties,
    ) -> ProcessProperties:
        """
        Propagate properties through Z_t = X_{T_t} (subordination).

        The result is a Lévy process (stationary and independent increments
        are preserved). The martingale and variance properties depend on the
        specific processes.

        Reference: Sato (1999), Theorem 30.1.
        """
        notes: list[str] = []

        if not subordinator.is_subordinator:
            raise ValueError(
                "The right operand of @ must be a subordinator "
                "(non-decreasing Lévy process)."
            )

        if parent.is_martingale:
            notes.append(
                "Martingale property after subordination depends on the specific "
                "processes; not propagated automatically."
            )

        finite_var = parent.has_finite_variance and subordinator.has_finite_mean

        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=False,
            has_finite_variance=finite_var,
            has_finite_mean=parent.has_finite_mean and subordinator.has_finite_mean,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=notes,
        )
