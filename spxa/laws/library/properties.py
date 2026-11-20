r"""
The property lattice: flags against the quantities they claim.

Every process carries :class:`~spxa.core.properties.ProcessProperties`,
and the composition operators propagate them by rules. A flag is a
claim about the distribution, so it can be checked against the
quantities. For a Lévy process :math:`X`:

==================================  =============================================================
``martingale_flag_is_sound``        ``is_martingale`` :math:`\Rightarrow E|X_1| < \infty` and :math:`\kappa_1 = 0`
``martingale_flag_is_complete``     :math:`\kappa_1 = 0` with a finite mean :math:`\Rightarrow` ``is_martingale``,
                                    tested on :math:`X` and on the centred :math:`X - E[X_1]t`
``subordinator_flag_is_sound``      ``is_subordinator`` :math:`\Rightarrow K(z) \le 0` for :math:`z \le 0`
                                    and the strip is unbounded below (since :math:`X_1 \ge 0`)
``moment_flags_are_sound``          ``has_finite_mean`` / ``has_finite_variance`` :math:`\Rightarrow`
                                    finite :math:`\kappa_1` / :math:`\kappa_2`
``self_similarity_is_sound``        index :math:`H` :math:`\Rightarrow \varphi_t(u) = \varphi_1(t^Hu)`
==================================  =============================================================

A Lévy process is a martingale exactly when :math:`E|X_1| < \infty` and
:math:`E[X_1] = 0`. The completeness law matters because a ``False`` flag
is also a claim: it says the process is *not* a martingale.

Closure laws, stated as implications between flags of operands and
results, follow the user-facing style
``martingale(X) => martingale(cX)``:

==========================================  ===========================================
``martingales_closed_under_sums``           both martingales :math:`\Rightarrow X+Y` is one
``martingales_closed_under_scaling``        :math:`X` martingale :math:`\Rightarrow cX` is one
``martingales_survive_time_change``         :math:`X` martingale, :math:`E[T_1] < \infty \Rightarrow X\circ T` is one
``subordinators_closed``                    sums, positive multiples and compositions of subordinators
==========================================  ===========================================

The martingale laws draw half their processes from
:func:`~spxa.laws.processes.martingales`. Generic draws almost never have
:math:`E[X_1] = 0` exactly, so an implication with a martingale hypothesis
would otherwise hold vacuously.

``martingale(X) => martingale(X + Drift(c))`` is **not** a law: adding
:math:`ct` gives mean :math:`ct`. Running it shows the shrunk
counterexample at once, which is the intended use of a law checker.
"""

from __future__ import annotations

import numpy as np

from spxa.core.process import Process
from spxa.laws.expect import assume, expect, expect_close, implies
from spxa.laws.law import Law, law
from spxa.laws.library._common import RTOL, frequencies
from spxa.laws.processes import compositions, martingales, subordinators
from spxa.laws.quantities import char_func, cumulants, exponential_moment_strip, laplace_exponent
from spxa.laws.strategies import Strategy, floats, one_of
from spxa.zoo.levy import Drift

_X = compositions(max_depth=1, operations=("+", "*", "-", "@"))
_T = subordinators()
# half generic draws, half centred ones, so martingale hypotheses are not vacuous
_M = one_of(_X, martingales())


def _kappa1_is_small(X: Process) -> bool:
    """|kappa_1| within rounding of zero, relative to the standard deviation."""
    k = cumulants(X, 2)
    return bool(abs(k[0]) <= 1e-12 * (1.0 + np.sqrt(abs(k[1]))))


def _self_similar_leaves() -> Strategy[Process]:
    from spxa.laws.strategies import builds, just
    from spxa.zoo.levy import BrownianMotion

    centred_bm = builds(BrownianMotion, mu=just(0.0), sigma=floats(0.05, 1.0, simplest=1.0))
    return one_of(centred_bm, builds(Drift, mu=floats(-1.0, 1.0, simplest=1.0)))


def martingale(X: Process) -> bool:
    """The process's own claim: ``X.properties.is_martingale``."""
    return bool(X.properties.is_martingale)


def subordinator(X: Process) -> bool:
    """The process's own claim: ``X.properties.is_subordinator``."""
    return bool(X.properties.is_subordinator)


@law(X=_M, tags=("properties", "martingale"))
def martingale_flag_is_sound(X: Process) -> None:
    """is_martingale => finite mean and kappa_1 = 0"""
    implies(
        martingale(X),
        lambda: X.properties.has_finite_mean and _kappa1_is_small(X),
        f"flagged a martingale but kappa_1 = {cumulants(X, 1)[0]!r}",
    )


@law(X=_X, tags=("properties", "martingale"))
def martingale_flag_is_complete(X: Process) -> None:
    """finite mean and kappa_1 = 0 => is_martingale, also for X - E[X_1] t"""
    implies(
        X.properties.has_finite_mean and cumulants(X, 1)[0] == 0.0,
        lambda: martingale(X),
        "kappa_1 = 0 with a finite mean, but not flagged a martingale",
    )
    if X.properties.has_finite_mean:
        centred = X + Drift(-cumulants(X, 1)[0])
        expect(
            martingale(centred),
            "X + Drift(-E[X_1]) has mean zero but is not flagged a martingale",
            centred=centred,
        )


@law(
    X=one_of(_X, compositions(_T, max_depth=1, operations=("+", "*", "@"))),
    tags=("properties", "subordinator"),
)
def subordinator_flag_is_sound(X: Process) -> None:
    """is_subordinator => K(z) <= 0 for z <= 0, strip unbounded below"""
    assume(subordinator(X))
    lo, _ = exponential_moment_strip(X)
    expect(lo == -np.inf, f"subordinator with exponential moments only above {lo!r}")
    k = laplace_exponent(X, np.array([-0.5, -2.0, -10.0])).real
    expect(bool(np.all(k <= 1e-10)), f"K(z) > 0 at negative z: {k}", K=k)
    expect(bool(np.all(np.diff(k) <= 1e-10)), f"K not monotone on z < 0: {k}", K=k)


@law(X=_X, tags=("properties", "moments"))
def moment_flags_are_sound(X: Process) -> None:
    """finite mean / variance flags => finite kappa_1 / kappa_2"""
    p = X.properties
    k = cumulants(X, 2)
    implies(p.has_finite_mean, bool(np.isfinite(k[0])), f"kappa_1 = {k[0]!r}")
    implies(p.has_finite_variance, bool(np.isfinite(k[1])), f"kappa_2 = {k[1]!r}")


@law(
    X=one_of(_X, compositions(_self_similar_leaves(), max_depth=1, operations=("+", "*", "-"))),
    u=frequencies(),
    t=floats(0.1, 4.0, simplest=2.0),
    tags=("properties", "scaling"),
)
def self_similarity_is_sound(X: Process, u: float, t: float) -> None:
    """self_similarity_index H => phi_t(u) = phi_1(t^H u)"""
    H = X.properties.self_similarity_index
    assume(H is not None)
    expect_close(
        char_func(X, u, t),
        char_func(X, t**H * u, 1.0),
        rtol=RTOL,
        atol=1e-9,
        what=f"phi_t(u) vs phi_1(t^H u), H = {H}",
    )


@law(X=_M, Y=_M, tags=("properties", "martingale", "closure"))
def martingales_closed_under_sums(X: Process, Y: Process) -> None:
    """martingale(X) and martingale(Y) => martingale(X + Y)"""
    implies(martingale(X) and martingale(Y), lambda: martingale(X + Y))


@law(
    X=_M,
    c=floats(-3.0, 3.0, simplest=2.0).filter(lambda c: c != 0),
    tags=("properties", "martingale", "closure"),
)
def martingales_closed_under_scaling(X: Process, c: float) -> None:
    """martingale(X) => martingale(cX)"""
    implies(martingale(X), lambda: martingale(c * X))


@law(X=_M, T=_T, tags=("properties", "martingale", "closure"))
def martingales_survive_time_change(X: Process, T: Process) -> None:
    """martingale(X) and E[T_1] < inf => martingale(X @ T)"""
    implies(
        martingale(X) and T.properties.has_finite_mean,
        lambda: martingale(X @ T),
        "a martingale run on an integrable clock lost its martingale flag",
    )


@law(T=_T, S=_T, c=floats(0.1, 3.0, simplest=2.0), tags=("properties", "subordinator", "closure"))
def subordinators_closed(T: Process, S: Process, c: float) -> None:
    """subordinators are closed under +, positive scaling and @"""
    expect(subordinator(T + S), "T + S is not flagged a subordinator")
    expect(subordinator(c * T), "cT is not flagged a subordinator")
    expect(subordinator(T @ S), "T @ S is not flagged a subordinator")


PROPERTIES: tuple[Law, ...] = (
    martingale_flag_is_sound,
    martingale_flag_is_complete,
    subordinator_flag_is_sound,
    moment_flags_are_sound,
    self_similarity_is_sound,
    martingales_closed_under_sums,
    martingales_closed_under_scaling,
    martingales_survive_time_change,
    subordinators_closed,
)
