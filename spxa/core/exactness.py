"""Exactness levels for stochastic process operations."""

from __future__ import annotations

from enum import Enum, auto


class ExactnessLevel(Enum):
    """
    Describes the analytical guarantee attached to a process or operation result.

    Levels are ordered: EXACT > MOMENT_PROPAGATION > SIMULATION_ONLY.
    Operations between two processes degrade to the minimum level of the operands.

    Attributes
    ----------
    EXACT :
        The Lévy–Khintchine triplet is known in closed form.
        All arithmetic is exact: triplet addition, cumulant derivation, and
        characteristic function evaluation are analytically correct.
        Applies only to independent Lévy processes.
    MOMENT_PROPAGATION :
        A closed-form triplet is unavailable or inapplicable (e.g. fBM, Hawkes,
        OU-Lévy, or compositions that leave the Lévy class).
        Finite moments are tracked exactly where computable; the full distribution
        is not available in closed form.
    SIMULATION_ONLY :
        No analytical handle exists. Sample paths can be generated but no
        distributional quantities are computed.
    """

    EXACT = auto()
    MOMENT_PROPAGATION = auto()
    SIMULATION_ONLY = auto()

    def __le__(self, other: ExactnessLevel) -> bool:
        order = [ExactnessLevel.EXACT, ExactnessLevel.MOMENT_PROPAGATION, ExactnessLevel.SIMULATION_ONLY]
        return order.index(self) <= order.index(other)

    def __lt__(self, other: ExactnessLevel) -> bool:
        return self != other and self <= other

    @staticmethod
    def combine(a: ExactnessLevel, b: ExactnessLevel) -> ExactnessLevel:
        """Return the minimum (most degraded) of two exactness levels."""
        order = [ExactnessLevel.EXACT, ExactnessLevel.MOMENT_PROPAGATION, ExactnessLevel.SIMULATION_ONLY]
        return order[max(order.index(a), order.index(b))]
