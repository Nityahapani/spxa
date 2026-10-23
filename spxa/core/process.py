"""
Base Process class.

All stochastic processes in spxa inherit from Process. The class is immutable:
every operator returns a new Process instance. This makes composition trees
inspectable and reproducible, and is what makes .__story__() possible.

Design invariants:
  1. Process instances are immutable after construction.
  2. Every operation produces a new Process with updated triplet, properties,
     exactness level, and composition node.
  3. ExactnessLevel degrades to the minimum of operands under any operation.
  4. SpxaDegradationWarning is emitted whenever exactness degrades, with a
     mathematical reason.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from math import comb

import numpy as np

from spxa.core._immutable import Immutable, ImmutableMeta
from spxa.core._validation import validate_order
from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError, SpxaDegradationWarning
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind, narrate


class Process(Immutable, ABC, metaclass=ImmutableMeta):
    """
    Abstract base class for all stochastic processes in spxa.

    Subclasses must implement `_triplet()`, `_properties()`, and `simulate()`.
    Operator overloading (`+`, `*`, `@`) is implemented here and returns
    composed Process instances.

    Parameters
    ----------
    exactness :
        Analytical guarantee for this process.
    _node :
        Composition tree node. Set automatically by operators; primitive
        processes set this in their own __init__.
    """

    def __init__(
        self,
        exactness: ExactnessLevel = ExactnessLevel.EXACT,
        _node: CompositionNode | None = None,
    ) -> None:
        self._exactness = exactness
        self._node = _node or CompositionNode(kind=NodeKind.PRIMITIVE, name=type(self).__name__)

    @property
    def exactness(self) -> ExactnessLevel:
        """The ExactnessLevel of this process."""
        return self._exactness

    @property
    def triplet(self) -> LevyTriplet:
        """
        The Lévy–Khintchine characteristic triplet (b, σ², ν).

        Only available when exactness == ExactnessLevel.EXACT.

        Raises
        ------
        ExactnessError
            If the process does not have an exact triplet.
        """
        if self._exactness != ExactnessLevel.EXACT:
            raise ExactnessError(
                method="triplet",
                required="EXACT",
                actual=self._exactness.name,
                reason=self._degradation_reason(),
            )
        return self._triplet()

    @property
    def properties(self) -> ProcessProperties:
        """The property lattice of this process."""
        return deepcopy(self._properties())

    @abstractmethod
    def _triplet(self) -> LevyTriplet:
        """Return the LevyTriplet. Called only when exactness == EXACT."""

    @abstractmethod
    def _properties(self) -> ProcessProperties:
        """Return the ProcessProperties for this process."""

    @abstractmethod
    def simulate(
        self,
        n_steps: int,
        *,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulate sample paths.

        Parameters
        ----------
        n_steps :
            Number of time steps per path.
        n_paths :
            Number of independent paths.
        T :
            Terminal time. Steps are equally spaced on [0, T].
        rng :
            Random number generator. If None, uses numpy default.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). The first column is the configured
            initial state (zero for the Lévy increment processes). Pass
            n_paths, T and rng by keyword for portability across subclasses.
        """

    def cumulants(self, order: int) -> dict[int, float]:
        """
        Return cumulants κ_1, …, κ_order of X_1 (per unit time).

        Available for EXACT and MOMENT_PROPAGATION processes up to the
        highest order with a finite moment.

        Parameters
        ----------
        order :
            Highest cumulant order to compute.
        """
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("order must be a positive integer")
        if self._exactness == ExactnessLevel.SIMULATION_ONLY:
            raise ExactnessError(
                method="cumulants",
                required="MOMENT_PROPAGATION or EXACT",
                actual="SIMULATION_ONLY",
                reason=self._degradation_reason(),
            )
        if self._exactness == ExactnessLevel.EXACT:
            t = self._triplet()
            return {n: t.cumulant(n) for n in range(1, order + 1)}
        return self._moment_propagation_cumulants(order)

    def char_func(self, u: complex | float | np.ndarray, t: float = 1.0) -> complex | np.ndarray:
        """
        Evaluate the characteristic function E[e^{iuX_t}] at frequency u.

        Exact for EXACT processes (via Lévy–Khintchine). Not available for
        SIMULATION_ONLY.

        Parameters
        ----------
        u :
            Frequency or array of frequencies.
        t :
            Time. Default 1.
        """
        if self._exactness == ExactnessLevel.SIMULATION_ONLY:
            raise ExactnessError(
                method="char_func",
                required="MOMENT_PROPAGATION or EXACT",
                actual="SIMULATION_ONLY",
                reason=self._degradation_reason(),
            )
        if self._exactness == ExactnessLevel.EXACT:
            return self._triplet().char_exp(u, t)
        raise ExactnessError(
            method="char_func",
            required="EXACT",
            actual=self._exactness.name,
            reason="Characteristic function requires an exact triplet.",
        )

    def __story__(self) -> str:
        """
        Return a human-readable derivation of this process's distributional
        properties, tracing how the current triplet and property lattice
        were obtained through composition.

        The output is designed to be readable in a terminal or rendered as
        a Markdown/LaTeX block in a notebook.
        """
        lines = ["=" * 60]
        lines.append(f"Story of: {self!r}")
        lines.append(f"Exactness: {self._exactness.name}")
        lines.append("-" * 60)
        lines.append("Composition trace:")
        lines.append(narrate(self._node))
        lines.append("-" * 60)
        lines.append("Properties:")
        props = self._properties()
        lines.append(f"  Stationary increments : {props.has_stationary_increments}")
        lines.append(f"  Independent increments: {props.has_independent_increments}")
        lines.append(f"  Martingale            : {props.is_martingale}")
        lines.append(f"  Finite mean           : {props.has_finite_mean}")
        lines.append(f"  Finite variance       : {props.has_finite_variance}")
        if props.tail_index is not None:
            lines.append(f"  Tail index α          : {props.tail_index}")
        if props.self_similarity_index is not None:
            lines.append(f"  Self-similarity H     : {props.self_similarity_index}")
        if props.hurst_index is not None:
            lines.append(f"  Hurst index H         : {props.hurst_index}")
        if props.notes:
            lines.append("  Notes:")
            for note in props.notes:
                lines.append(f"    - {note}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __story_latex__(self) -> str:
        """
        Return a LaTeX derivation of this process's composition and properties.

        Output is a ``\\begin{aligned}...\\end{aligned}`` block suitable for
        display in a Jupyter notebook:

        .. code-block:: python

            from IPython.display import display, Math
            display(Math(Z.__story_latex__()))

        See Also
        --------
        __story__ : Plain-text version.
        spxa.story.story_latex : Standalone function.
        """
        from spxa.story.narrator import story_latex

        return story_latex(self)

    def _degradation_reason(self) -> str:
        """Return the reason for exactness degradation from the composition notes."""
        notes = self._properties().notes
        if notes:
            return notes[-1]
        return (
            f"This process has ExactnessLevel.{self._exactness.name}. "
            "Check .__story__() for the composition history."
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        """
        Fallback cumulant computation for MOMENT_PROPAGATION processes.
        Subclasses should override with exact formulas where available.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement moment propagation cumulants. "
            "Override _moment_propagation_cumulants() or use simulate() and compute empirically."
        )

    def __add__(self, other: Process) -> _ComposedProcess:
        """
        Return the process Z = X + Y (sum of independent processes).

        If both are EXACT Lévy processes, the result is EXACT with triplet
        (b_X+b_Y, σ²_X+σ²_Y, ν_X+ν_Y).

        Otherwise degrades to the minimum ExactnessLevel with a warning.
        """
        new_exactness = ExactnessLevel.combine(self._exactness, other._exactness)

        if new_exactness != ExactnessLevel.EXACT:
            reason = (
                f"Adding {type(self).__name__} (ExactnessLevel.{self._exactness.name}) "
                f"and {type(other).__name__} (ExactnessLevel.{other._exactness.name}): "
                "result is not an exact Lévy process."
            )
            warnings.warn(reason, SpxaDegradationWarning, stacklevel=2)

        new_props = ProcessProperties.combine_addition(self._properties(), other._properties())
        node = CompositionNode(
            kind=NodeKind.ADDITION,
            name=f"{self!r} + {other!r}",
            children=[self._node, other._node],
            math_note="Triplet addition: (b,σ²,ν) = (b_X+b_Y, σ²_X+σ²_Y, ν_X+ν_Y)",
            reference="Sato (1999), Prop. 11.10",
        )
        return _ComposedProcess(
            left=self,
            right=other,
            operation="+",
            exactness=new_exactness,
            props=new_props,
            node=node,
        )

    def __radd__(self, other: object) -> Process:
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, c: float) -> _ScaledProcess:
        """Return the process Z = c * X."""
        if not isinstance(c, (int, float)):
            return NotImplemented
        return _make_scaled(self, float(c))

    def __rmul__(self, c: float) -> _ScaledProcess:
        """Return the process Z = c * X (right multiplication)."""
        if not isinstance(c, (int, float)):
            return NotImplemented
        return _make_scaled(self, float(c))

    def __neg__(self) -> _ScaledProcess:
        return _make_scaled(self, -1.0)

    def __sub__(self, other: Process) -> Process:
        return self.__add__(_make_scaled(other, -1.0))

    def __matmul__(self, subordinator: Process) -> _ComposedProcess:
        """
        Return the subordinated process Z_t = X_{T_t}.

        The left operand X is the parent process, the right operand T must
        be a subordinator (non-decreasing Lévy process).

        Reference: Sato (1999), Theorem 30.1.
        """
        if (
            not subordinator.properties.is_subordinator
            or not subordinator.properties.has_independent_increments
            or not subordinator.properties.has_stationary_increments
            or subordinator.exactness != ExactnessLevel.EXACT
            or self.exactness != ExactnessLevel.EXACT
        ):
            raise ValueError(
                f"{type(subordinator).__name__} is not a subordinator. "
                "The right operand of @ must be a non-decreasing Lévy process."
            )

        analytic = _canonical_subordination(self, subordinator)
        new_exactness = (
            ExactnessLevel.EXACT if analytic is not None else ExactnessLevel.MOMENT_PROPAGATION
        )
        if analytic is None:
            warnings.warn(
                "Generic subordination has an exact CF and moment composition, but an explicit "
                "triplet is not implemented; retaining MOMENT_PROPAGATION capability.",
                SpxaDegradationWarning,
                stacklevel=2,
            )
        new_props = ProcessProperties.combine_subordination(
            self._properties(), subordinator._properties()
        )
        node = CompositionNode(
            kind=NodeKind.SUBORDINATION,
            name=f"{self!r} @ {subordinator!r}",
            children=[self._node, subordinator._node],
            math_note="ψ_Z(u) = -φ(-ψ_X(u)) where φ is the Laplace exponent of T",
            reference="Sato (1999), Thm. 30.1",
        )
        return _ComposedProcess(
            left=self,
            right=subordinator,
            operation="@",
            exactness=new_exactness,
            props=new_props,
            node=node,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(exactness={self._exactness.name})"

    def characteristic_exponent(self, u: complex | float | np.ndarray) -> np.ndarray:
        """Continuous unit-time log CF for a Levy process (Sato, Theorem 8.1)."""
        return self.triplet.characteristic_exponent(u)

    def cumulants_at(self, order: int, t: float = 1.0) -> dict[int, float]:
        """Cumulants at a specified horizon; non-Levy subclasses must override.

        Linear time scaling is valid only for Levy processes. Unknown
        time dependence is rejected rather than silently assumed linear.
        """
        validate_order(order)
        if not np.isfinite(t) or t < 0:
            raise ValueError("Positive order and finite nonnegative time required")
        if self.exactness == ExactnessLevel.EXACT:
            if t == 0:
                return {n: 0.0 for n in range(1, order + 1)}
            return {n: t * k for n, k in self.cumulants(order).items()}
        if t == 1:
            return self.cumulants(order)
        raise ExactnessError(
            method="cumulants_at",
            required="time-aware moment implementation",
            actual=self.exactness.name,
            reason="Non-Levy cumulants need an explicit time law",
        )


class _ScaledProcess(Process):
    """Process Z = c * X. Internal — users get this via X * c or c * X."""

    def __init__(
        self,
        base: Process,
        scale: float,
        exactness: ExactnessLevel,
        props: ProcessProperties,
        node: CompositionNode,
    ) -> None:
        super().__init__(exactness=exactness, _node=node)
        self._base = base
        self._scale = scale
        self._props = props

    def _triplet(self) -> LevyTriplet:
        return self._base._triplet().scale(self._scale)

    def _properties(self) -> ProcessProperties:
        return self._props

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        paths = self._base.simulate(n_steps, n_paths=n_paths, T=T, rng=rng)
        return self._scale * paths

    def __repr__(self) -> str:
        return f"{self._scale} * {self._base!r}"

    def char_func(self, u: complex | float | np.ndarray, t: float = 1.0) -> complex | np.ndarray:
        return self._base.char_func(np.asarray(u) * self._scale, t)

    def characteristic_exponent(self, u: complex | float | np.ndarray) -> np.ndarray:
        return self._base.characteristic_exponent(np.asarray(u) * self._scale)

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        return {n: self._scale**n * k for n, k in self._base.cumulants(order).items()}

    def cumulants_at(self, order: int, t: float = 1.0) -> dict[int, float]:
        return {n: self._scale**n * k for n, k in self._base.cumulants_at(order, t).items()}


class _ComposedProcess(Process):
    """Process produced by + or @. Internal."""

    def __init__(
        self,
        left: Process,
        right: Process,
        operation: str,
        exactness: ExactnessLevel,
        props: ProcessProperties,
        node: CompositionNode,
    ) -> None:
        super().__init__(exactness=exactness, _node=node)
        self._left = left
        self._right = right
        self._operation = operation
        self._props = props
        self._analytic = _canonical_subordination(left, right) if operation == "@" else None

    def _triplet(self) -> LevyTriplet:
        if self._operation == "+":
            return self._left.triplet + self._right.triplet
        if self._analytic is not None:
            return self._analytic.triplet
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual=self.exactness.name,
            reason="Generic subordination exposes CFs and cumulants, not an explicit triplet",
        )

    def _properties(self) -> ProcessProperties:
        return self._props

    def simulate(
        self, n_steps: int, n_paths: int = 1, T: float = 1.0, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if n_steps < 1 or n_paths < 1 or not np.isfinite(T) or T < 0:
            raise ValueError("Positive grid/path counts and finite nonnegative horizon required")
        if self._operation == "+":
            return np.asarray(
                self._left.simulate(n_steps, n_paths=n_paths, T=T, rng=rng)
                + self._right.simulate(n_steps, n_paths=n_paths, T=T, rng=rng)
            )
        if self._analytic is not None:
            return self._analytic.simulate(n_steps, n_paths=n_paths, T=T, rng=rng)
        # Draw independent parent increments at the actual random clock lengths.
        # No interpolated fine-grid approximation or artificial clock horizon.
        clock = self._right.simulate(n_steps, n_paths=n_paths, T=T, rng=rng)
        delta = np.diff(clock, axis=1)
        if np.any(delta < 0):
            raise ValueError("Clock produced negative increments")
        increments = np.zeros_like(delta)
        for i, k in np.ndindex(delta.shape):
            if delta[i, k] > 0:
                increments[i, k] = self._left.simulate(1, n_paths=1, T=float(delta[i, k]), rng=rng)[
                    0, -1
                ]
        return np.column_stack((np.zeros(n_paths), np.cumsum(increments, axis=1)))

    def __repr__(self) -> str:
        return f"({self._left!r} {self._operation} {self._right!r})"

    def characteristic_exponent(self, u: complex | float | np.ndarray) -> np.ndarray:
        if self._operation == "+":
            return np.asarray(
                self._left.characteristic_exponent(u) + self._right.characteristic_exponent(u)
            )
        return self._right.characteristic_exponent(-1j * self._left.characteristic_exponent(u))

    def char_func(self, u: complex | float | np.ndarray, t: float = 1.0) -> complex | np.ndarray:
        if not np.isfinite(t) or t < 0:
            raise ValueError("Time must be finite and nonnegative")
        if self._operation == "+":
            return np.asarray(self._left.char_func(u, t) * self._right.char_func(u, t))
        return np.exp(t * self.characteristic_exponent(u))

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        return self.cumulants_at(order, 1.0)

    def cumulants(self, order: int) -> dict[int, float]:
        if self.exactness == ExactnessLevel.SIMULATION_ONLY:
            return super().cumulants(order)
        return self.cumulants_at(order, 1.0)

    def cumulants_at(self, order: int, t: float = 1.0) -> dict[int, float]:
        validate_order(order)
        if not np.isfinite(t) or t < 0:
            raise ValueError("Positive order and finite nonnegative time required")
        if self._operation == "+":
            a, b = self._left.cumulants_at(order, t), self._right.cumulants_at(order, t)
            return {n: a[n] + b[n] for n in range(1, order + 1)}
        if t == 0:
            return {n: 0.0 for n in range(1, order + 1)}
        # Partial exponential Bell polynomials: K_Z = K_T composed with K_X.
        a, b = self._left.cumulants(order), self._right.cumulants_at(order, t)
        bell = {(0, 0): 1.0}
        for n in range(1, order + 1):
            for k in range(1, n + 1):
                bell[n, k] = sum(
                    comb(n - 1, j - 1) * a[j] * bell.get((n - j, k - 1), 0.0)
                    for j in range(1, n - k + 2)
                )
        return {n: sum(b[k] * bell[n, k] for k in range(1, n + 1)) for n in range(1, order + 1)}


def _make_scaled(base: Process, c: float) -> _ScaledProcess:
    new_exactness = base._exactness
    new_props = ProcessProperties.combine_scaling(base._properties(), c)
    node = CompositionNode(
        kind=NodeKind.SCALING,
        name=f"{c} * {base!r}",
        children=[base._node],
        math_note=f"Triplet scaling by c={c}: (cb+Δb, c²σ², ν(·/c))",
        reference="Sato (1999), Prop. 11.10",
    )
    return _ScaledProcess(
        base=base,
        scale=c,
        exactness=new_exactness,
        props=new_props,
        node=node,
    )


def _canonical_subordination(parent: Process, clock: Process) -> Process | None:
    """Known Bochner identities (Cont & Tankov, Chapter 4)."""
    from spxa.zoo.levy.brownian import BrownianMotion
    from spxa.zoo.levy.gamma import GammaProcess
    from spxa.zoo.levy.inverse_gaussian import InverseGaussianProcess
    from spxa.zoo.levy.nig import NIG
    from spxa.zoo.levy.vg import VarianceGamma

    if not isinstance(parent, BrownianMotion):
        return None
    if parent.sigma == 0:
        return parent if parent.mu == 0 else parent.mu * clock
    if isinstance(clock, BrownianMotion) and clock.sigma == 0:
        return BrownianMotion(mu=parent.mu * clock.mu, sigma=parent.sigma * np.sqrt(clock.mu))
    if isinstance(clock, GammaProcess):
        return VarianceGamma(
            sigma=parent.sigma * np.sqrt(clock.a / clock.b),
            nu=1 / clock.a,
            theta=parent.mu * clock.a / clock.b,
        )
    if isinstance(clock, InverseGaussianProcess):
        beta = parent.mu / parent.sigma**2
        return NIG(
            alpha=np.sqrt(clock.lam / (clock.mu**2 * parent.sigma**2) + beta**2),
            beta=beta,
            delta=parent.sigma * np.sqrt(clock.lam),
        )
    return None
