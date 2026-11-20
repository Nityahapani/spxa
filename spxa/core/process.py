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
from typing import Any, Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError, SpxaDegradationWarning, TripletUnavailableError
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind, narrate


class Process(ABC):
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
        _node: Optional[CompositionNode] = None,
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
        return self._properties()

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
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
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
            Shape (n_paths, n_steps + 1). paths[:, 0] == 0 always.
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
        if self._exactness == ExactnessLevel.SIMULATION_ONLY:
            raise ExactnessError(
                method="cumulants",
                required="MOMENT_PROPAGATION or EXACT",
                actual="SIMULATION_ONLY",
                reason=self._degradation_reason(),
            )
        if self._exactness == ExactnessLevel.EXACT:
            return self._exact_cumulants(order)
        return self._moment_propagation_cumulants(order)

    def _exact_cumulants(self, order: int) -> dict[int, float]:
        """Cumulants of an EXACT process; from the triplet unless overridden."""
        t = self._triplet()
        return {n: t.cumulant(n) for n in range(1, order + 1)}

    def _char_exponent(self, u: np.ndarray) -> np.ndarray:
        r"""
        :math:`\psi(u)` with :math:`E[e^{iuX_t}] = e^{t\psi(u)}`, elementwise.

        Complex ``u`` is allowed where the exponent converges. This is how
        subordination evaluates the Laplace exponent of the clock. The
        default uses the Lévy–Khintchine triplet. It never takes the
        logarithm of the characteristic function, whose branch would be
        ambiguous.
        """
        trip = self._triplet()
        u = np.asarray(u, dtype=complex)
        out = np.empty(u.shape, dtype=complex)
        for i, ui in enumerate(u.flat):
            out.flat[i] = (
                1j * trip.b * ui - 0.5 * trip.sigma_sq * ui**2 + trip.nu.jump_exponent(complex(ui))
            )
        return out

    def char_func(self, u: float | np.ndarray, t: float = 1.0) -> np.ndarray:
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
            u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
            result = np.exp(t * self._char_exponent(u_arr))
            return result if result.shape != (1,) else result[0]
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
        if not subordinator._properties().is_subordinator:
            raise ValueError(
                f"{type(subordinator).__name__} is not a subordinator. "
                "The right operand of @ must be a non-decreasing Lévy process."
            )

        new_exactness = ExactnessLevel.combine(self._exactness, subordinator._exactness)
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

    def morph(
        self,
        other: Process,
        alpha: float = 0.5,
        method: str = "triplet",
        coupling: str = "independent",
        **kwargs: Any,
    ) -> Any:
        r"""
        Morph this process into ``other`` via homotopy parameter :math:`\alpha \in [0, 1]`.

        .. math::
            P_\alpha = \text{Morph}(P_0, P_1; \alpha)

        Parameters
        ----------
        other :
            Target process corresponding to :math:`\alpha = 1`.
        alpha :
            Homotopy parameter in :math:`[0, 1]`. Default 0.5.
        method :
            Morphing scheme: ``"triplet"``, ``"generator"``, ``"geodesic"``,
            ``"wasserstein"``, or ``"convex_combination"``.
        coupling :
            Simulation coupling: ``"independent"`` or ``"common"``.

        Returns
        -------
        MorphedProcess
            A new MorphedProcess instance.
        """
        from spxa.ops.morph import morph

        return morph(self, other, alpha=alpha, method=method, coupling=coupling, **kwargs)

    def to_pde(self, **kwargs: Any) -> Any:
        r"""
        Pricing / Kolmogorov equation of this process.

        Delegates to :func:`spxa.pde.to_pde`. Diffusions (Brownian motion,
        Ornstein–Uhlenbeck, CIR, CEV) give a
        :class:`~spxa.pde.generator.DiffusionPDE` for the backward equation
        :math:`u_\tau = \mathcal{A}u` and its Fokker–Planck adjoint; Lévy
        processes with jumps give the exponential-Lévy pricing PIDE
        :class:`~spxa.pde.pide.LevyPIDE` and require ``r`` (and optionally
        ``q``).

        Raises
        ------
        NotImplementedError
            When no PDE representation is registered for this process.
        """
        from spxa.pde.generator import to_pde

        return to_pde(self, **kwargs)

    def to_vol_surface(self, spot: float, rate: float, expiries: Any, **kwargs: Any) -> Any:
        r"""
        Implied-volatility surface of the exponential-Lévy model on this process.

        Delegates to :func:`spxa.vol.process_vol_surface`. The model is
        :math:`S_T = S_0 e^{(r - q)T + X_T}`, where :math:`X` is this process
        normalised to be a martingale in the exponent. The result is a
        :class:`~spxa.vol.process_surface.ProcessSurface`, i.e. a
        :class:`~spxa.vol.surface.VolSurface` together with the model implied
        vols it was fitted to.
        """
        from spxa.vol.process_surface import process_vol_surface

        return process_vol_surface(self, spot, rate, expiries, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(exactness={self._exactness.name})"


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

    def _char_exponent(self, u: np.ndarray) -> np.ndarray:
        try:
            return super()._char_exponent(u)
        except TripletUnavailableError:
            # ψ_{cX}(u) = ψ_X(cu)
            return self._base._char_exponent(self._scale * np.asarray(u, dtype=complex))

    def _exact_cumulants(self, order: int) -> dict[int, float]:
        try:
            return super()._exact_cumulants(order)
        except TripletUnavailableError:
            base = self._base.cumulants(order)
            return {n: self._scale**n * base[n] for n in range(1, order + 1)}

    def _properties(self) -> ProcessProperties:
        return self._props

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        paths = self._base.simulate(n_steps, n_paths, T, rng)
        return self._scale * paths

    def __repr__(self) -> str:
        return f"{self._scale} * {self._base!r}"


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
        self._refined_props: Optional[ProcessProperties] = None

    def _triplet(self) -> LevyTriplet:
        if self._operation == "+":
            return self._left._triplet() + self._right._triplet()
        raise TripletUnavailableError(
            "the triplet of a subordinated process X(T) involves the law of X_s for "
            "every s (Sato 1999, Thm 30.1) and is not implemented."
        )

    def _char_exponent(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        if self._operation == "@":
            # ψ_Z(u) = ψ_T(-i ψ_X(u)): E[e^{iuX_{T_t}}] = E[e^{T_t ψ_X(u)}], Re ψ_X ≤ 0
            return self._right._char_exponent(-1j * self._left._char_exponent(u))
        try:
            return super()._char_exponent(u)
        except TripletUnavailableError:
            return self._left._char_exponent(u) + self._right._char_exponent(u)

    def _exact_cumulants(self, order: int) -> dict[int, float]:
        if self._operation == "@":
            return _subordinated_cumulants(
                self._left.cumulants(order), self._right.cumulants(order), order
            )
        try:
            return super()._exact_cumulants(order)
        except TripletUnavailableError:
            left, right = self._left.cumulants(order), self._right.cumulants(order)
            return {n: left[n] + right[n] for n in range(1, order + 1)}

    def _properties(self) -> ProcessProperties:
        if self._refined_props is None:
            self._refined_props = self._refine_martingale(self._props)
        return self._refined_props

    def _refine_martingale(self, props: ProcessProperties) -> ProcessProperties:
        r"""
        Flag a sum of non-martingales whose means cancel as a martingale.

        The rule "martingale iff both summands are" is exact when only one
        summand is a martingale, but misses :math:`E[X_1] + E[Y_1] = 0`, e.g. a
        compensated process :math:`X - E[X_1]t`. A Lévy process with a finite
        mean is a martingale iff that mean is zero. For EXACT sums with
        finite means the two means are compared, and cancellation within
        rounding (:math:`10^{-12}` of their sizes) counts as zero. The check
        runs once, on first access.
        """
        if (
            self._operation != "+"
            or props.is_martingale
            or not props.has_finite_mean
            or self._exactness != ExactnessLevel.EXACT
        ):
            return props
        try:
            left = float(self._left.cumulants(1)[1])
            right = float(self._right.cumulants(1)[1])
        except Exception:
            return props
        if not (np.isfinite(left) and np.isfinite(right)):
            return props
        if abs(left + right) > 1e-12 * (abs(left) + abs(right)):
            return props
        refined = props.add_note(
            f"The means cancel (E[X_1] = {left!r}, E[Y_1] = {right!r}), so the sum is a martingale."
        )
        refined.is_martingale = True
        return refined

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if self._operation == "+":
            return self._left.simulate(n_steps, n_paths, T, rng) + self._right.simulate(
                n_steps, n_paths, T, rng
            )
        if self._operation == "@":
            # Subordination: Z_t = X_{T_t}
            # Simulate the subordinator to get random time changes,
            # then simulate the parent process at those random times.
            # Uses a path-by-path approach: for each path, simulate the
            # subordinator time grid, then interpolate the parent process.
            rng = rng or np.random.default_rng()

            # Simulate subordinator paths on a fine grid
            sub_paths = self._right.simulate(n_steps, n_paths, T, rng)
            # sub_paths[:, k] = T_k (random time at step k)

            # For each path, simulate parent at fine resolution and interpolate
            # We simulate the parent on a finer grid covering [0, max(T_T)]
            max_time = float(sub_paths[:, -1].max()) * 1.5 + 1e-6
            fine_steps = max(n_steps * 4, 500)
            parent_paths = self._left.simulate(fine_steps, n_paths, max_time, rng)
            fine_grid = np.linspace(0, max_time, fine_steps + 1)

            paths = np.zeros((n_paths, n_steps + 1))
            for i in range(n_paths):
                paths[i] = np.interp(sub_paths[i], fine_grid, parent_paths[i])

            return paths
        raise NotImplementedError(f"Simulation not implemented for operation '{self._operation}'.")

    def __repr__(self) -> str:
        return f"({self._left!r} {self._operation} {self._right!r})"


def _subordinated_cumulants(
    parent: dict[int, float], clock: dict[int, float], order: int
) -> dict[int, float]:
    r"""
    Cumulants of :math:`X_{T_1}` from those of :math:`X_1` and :math:`T_1`.

    The cumulant generating functions compose, :math:`K_Z = K_T\circ K_X`,
    because :math:`E[e^{\theta X_{T}}] = E[e^{T K_X(\theta)}]`. With
    :math:`K(\theta) = \sum_n \kappa_n\theta^n/n!` the coefficients of the
    composed series give, for example,
    :math:`\kappa_1^Z = \kappa_1^T\kappa_1^X` and
    :math:`\kappa_2^Z = \kappa_1^T\kappa_2^X + \kappa_2^T(\kappa_1^X)^2`.
    """
    from math import factorial

    a = np.zeros(order + 1)
    for n in range(1, order + 1):
        a[n] = parent[n] / factorial(n)
    result = np.zeros(order + 1)
    power = np.zeros(order + 1)
    power[0] = 1.0
    for m in range(1, order + 1):
        power = np.convolve(power, a)[: order + 1]
        result += clock[m] / factorial(m) * power
    return {n: float(result[n] * factorial(n)) for n in range(1, order + 1)}


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
